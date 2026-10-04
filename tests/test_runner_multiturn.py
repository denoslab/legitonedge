"""TDD for the live per-turn agent loop (run_multiturn_cell) + tool-call parser.

The runner DRIVES a model turn-by-turn over the bespoke multi-step tool workload and
produces the ``trace`` that ``MultiTurnToolWorkload.score_detailed`` consumes. A
``ScriptedAdapter`` test double feeds deterministic model text so each loop branch
(correct calls, non-termination, wrong call) is exercised without a real model.

Instances are built with the same self-consistent state-diff fixtures as the 2.2a
workload tests (expected_states replayed through a fresh ToolSandbox).
"""
from __future__ import annotations

import json

from legit_edge.adapter import AdapterResponse, ModelMetadata
from legit_edge.tool_sandbox import (
    ToolSandbox,
    build_tool_system_prompt,
    parse_tool_calls,
)
from legit_edge.workload import Instance, MultiTurnToolWorkload
from legit_edge.runner import run_multiturn_cell, CellResult
from legit_edge.metrics import MetricResult


# --------------------------------------------------------------------------- fixtures
def _expected_states(initial_state: dict, per_turn_calls: list[list[dict]]) -> list[dict]:
    sb = ToolSandbox(initial_state=initial_state)
    out = []
    for calls in per_turn_calls:
        for c in calls:
            sb.dispatch(c)
        out.append(sb.state())
    return out


def _make_instance(
    task_id: str, turns: list[str], initial_state: dict, per_turn_calls: list[list[dict]]
) -> Instance:
    expected = _expected_states(initial_state, per_turn_calls)
    return Instance(
        input=json.dumps({"id": task_id, "turns": turns, "initial_state": initial_state}),
        target=json.dumps(
            {"id": task_id, "initial_state": initial_state, "expected_states": expected}
        ),
        raw={},
    )


TURN1_CALLS = [
    {"name": "set_note", "arguments": {"key": "payee", "value": "alice"}},
    {"name": "ledger_add", "arguments": {"amount": 100}},
]
TURN2_CALLS = [
    {"name": "ledger_subtract", "arguments": {"amount": 40}},
]
TWO_TURN_USER = ["note payee=alice and add 100", "now subtract 40"]


def _two_turn_instance() -> Instance:
    return _make_instance("t_basic", TWO_TURN_USER, {}, [TURN1_CALLS, TURN2_CALLS])


def _calls_json(calls: list[dict]) -> str:
    """Serialize a call list the way a model would emit it (a bare JSON array)."""
    return json.dumps(calls)


# --------------------------------------------------------------------------- test double
class ScriptedAdapter:
    """Deterministic adapter: ``generate`` pops the next queued response text.

    The queue is consumed in order across every turn / inner round of the whole cell.
    """

    def __init__(self, scripted: list[str]):
        self._queue = list(scripted)
        self.calls_seen: list[dict] = []

    def generate(self, system, user, *, max_tokens=512, temperature=0.0,
                 expected=None, **kw) -> AdapterResponse:
        self.calls_seen.append({"system": system, "user": user})
        text = self._queue.pop(0) if self._queue else "done"
        return AdapterResponse(text=text, latency_s=0.01, input_tokens=5, output_tokens=3)

    def metadata(self) -> ModelMetadata:
        return ModelMetadata(name="m", tier="mock", quantization="none", runtime="mock")


def _run(adapter, instance, **kw) -> CellResult:
    """Drive run_multiturn_cell over a single-instance synthetic workload."""
    w = MultiTurnToolWorkload(jsonl_path=None)
    w.instances = lambda: [instance]  # type: ignore[method-assign]
    defaults = dict(
        adapter=adapter, workload=w, n_instances=1,
        mode="megaquick", thermal="MAXN",
    )
    defaults.update(kw)
    return run_multiturn_cell(**defaults)


# --------------------------------------------------------------------------- parse tests
def test_parse_tool_calls():
    # Clean JSON array.
    clean = '[{"name": "ledger_add", "arguments": {"amount": 5}}]'
    assert parse_tool_calls(clean) == [
        {"name": "ledger_add", "arguments": {"amount": 5}}
    ]

    # Array embedded in prose.
    prose = 'Sure, I will do that.\n[{"name": "list_notes", "arguments": {}}]\nDone.'
    assert parse_tool_calls(prose) == [{"name": "list_notes", "arguments": {}}]

    # Array inside a fenced code block.
    fenced = '```json\n[{"name": "ledger_balance", "arguments": {}}]\n```'
    assert parse_tool_calls(fenced) == [{"name": "ledger_balance", "arguments": {}}]

    # Plain prose with no array -> [].
    assert parse_tool_calls("The task is complete.") == []

    # Not JSON at all -> [].
    assert parse_tool_calls("[this is not json]") == []

    # arguments defaulted to {} when omitted.
    assert parse_tool_calls('[{"name": "list_notes"}]') == [
        {"name": "list_notes", "arguments": {}}
    ]

    # A list with one malformed entry (no string name) -> that entry dropped.
    mixed = '[{"name": "ledger_add", "arguments": {"amount": 1}}, {"foo": "bar"}]'
    assert parse_tool_calls(mixed) == [
        {"name": "ledger_add", "arguments": {"amount": 1}}
    ]

    # A bare JSON object (not a list) -> [].
    assert parse_tool_calls('{"name": "ledger_add", "arguments": {}}') == []


def test_build_tool_system_prompt_is_deterministic_and_lists_tools():
    p1 = build_tool_system_prompt()
    p2 = build_tool_system_prompt()
    assert p1 == p2  # deterministic (no timestamps)
    # Mentions every tool name and the protocol.
    for name in ("set_note", "ledger_add", "web_search"):
        assert name in p1
    assert "JSON array" in p1


# --------------------------------------------------------------------------- loop tests
def test_all_turns_correct_task_success():
    inst = _two_turn_instance()
    # turn1: emit correct calls, then (next round) a plain final answer.
    # turn2: emit correct calls, then a plain final answer.
    adapter = ScriptedAdapter([
        _calls_json(TURN1_CALLS), "All set.",
        _calls_json(TURN2_CALLS), "Subtracted 40.",
    ])
    res = _run(adapter, inst)
    p = res.per_instance[0]
    detail = p["score_metadata"]
    assert detail["task_success"] == 1.0
    assert detail["nonterminated"] is False
    assert p["score"] == 1.0
    assert res.capability == 1.0
    # Trace recorded exactly the executed calls per turn.
    trace = json.loads(p["output"])
    assert [c["name"] for c in trace["turns"][0]["calls"]] == ["set_note", "ledger_add"]
    assert [c["name"] for c in trace["turns"][1]["calls"]] == ["ledger_subtract"]
    assert all(t["terminated"] for t in trace["turns"])


def test_nontermination_when_model_loops():
    inst = _two_turn_instance()
    # Model ALWAYS emits a (correct turn1) tool call -> never gives a final answer.
    looping = _calls_json(TURN1_CALLS)
    adapter = ScriptedAdapter([looping] * 40)  # more than enough to exhaust the budget
    res = _run(adapter, inst, step_budget=5)
    p = res.per_instance[0]
    detail = p["score_metadata"]
    assert detail["nonterminated"] is True
    assert detail["task_success"] == 0.0
    trace = json.loads(p["output"])
    # Each graded turn hit the step budget without terminating.
    assert trace["turns"][0]["terminated"] is False


def test_wrong_call_fails_task():
    inst = _two_turn_instance()
    wrong_turn2 = [{"name": "ledger_subtract", "arguments": {"amount": 999}}]
    adapter = ScriptedAdapter([
        _calls_json(TURN1_CALLS), "ok",
        _calls_json(wrong_turn2), "done",
    ])
    res = _run(adapter, inst)
    p = res.per_instance[0]
    assert p["score_metadata"]["task_success"] == 0.0
    assert p["score"] == 0.0


def test_run_multiturn_cell_returns_cellresult_schema():
    inst = _two_turn_instance()
    adapter = ScriptedAdapter([
        _calls_json(TURN1_CALLS), "ok",
        _calls_json(TURN2_CALLS), "done",
    ])
    res = _run(adapter, inst)
    assert isinstance(res, CellResult)
    assert 0.0 <= res.capability <= 1.0
    assert isinstance(res.latency, MetricResult)
    assert isinstance(res.throughput, MetricResult)
    assert isinstance(res.energy, MetricResult)
    assert res.cell_id == "m|mock|tooluse_mt|megaquick|MAXN"
    assert res.mode == "megaquick"
    assert res.thermal == "MAXN"
    p = res.per_instance[0]
    for key in (
        "instance_index", "input", "target", "output", "score", "score_metadata",
        "latency_s", "output_tokens", "input_tokens", "timestamp",
    ):
        assert key in p, f"missing per-instance key: {key}"
    detail = p["score_metadata"]
    for key in ("task_success", "turn_failures", "nonterminated", "n_turns", "failed_turns"):
        assert key in detail, f"missing score_metadata key: {key}"
    # Per-task latency is the sum of the per-call latencies (4 generate calls @ 0.01).
    assert p["latency_s"] > 0.0
    assert p["output_tokens"] > 0


def test_per_task_latency_and_tokens_are_summed():
    inst = _two_turn_instance()
    # 4 generate calls total (2 rounds x 2 turns) @ latency 0.01, output_tokens 3 each.
    adapter = ScriptedAdapter([
        _calls_json(TURN1_CALLS), "ok",
        _calls_json(TURN2_CALLS), "done",
    ])
    res = _run(adapter, inst)
    p = res.per_instance[0]
    assert abs(p["latency_s"] - 0.04) < 1e-9   # 4 * 0.01
    assert p["output_tokens"] == 12            # 4 * 3
    assert p["input_tokens"] == 20             # 4 * 5 (summed)
