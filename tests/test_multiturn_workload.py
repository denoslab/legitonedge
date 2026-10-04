"""TDD for MultiTurnToolWorkload (Path B bespoke multi-step tool grader).

Fixtures build expected_states by replaying a reference call sequence through a
fresh ToolSandbox, so the state-diff oracle is self-consistent by construction.
"""
from __future__ import annotations

import json

from legit_edge.tool_sandbox import ToolSandbox
from legit_edge.workload import MultiTurnToolWorkload, Instance


def _expected_states(initial_state: dict, per_turn_calls: list[list[dict]]) -> list[dict]:
    """Replay reference calls turn-by-turn; snapshot state after each turn."""
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


# A canonical 2-turn task: turn1 sets a note + opens a ledger; turn2 spends from the ledger
# (turn2 depends on the state turn1 established).
TURN1_CALLS = [
    {"name": "set_note", "arguments": {"key": "payee", "value": "alice"}},
    {"name": "ledger_add", "arguments": {"amount": 100}},
]
TURN2_CALLS = [
    {"name": "ledger_subtract", "arguments": {"amount": 40}},
]
TWO_TURN_REF = [TURN1_CALLS, TURN2_CALLS]
TWO_TURN_USER = ["note payee=alice and add 100", "now subtract 40"]


def _two_turn_instance() -> Instance:
    return _make_instance("t_basic", TWO_TURN_USER, {}, TWO_TURN_REF)


def _trace(per_turn_calls: list[list[dict]], terminated: list[bool] | None = None) -> dict:
    if terminated is None:
        terminated = [True] * len(per_turn_calls)
    return {
        "turns": [
            {"calls": calls, "terminated": term}
            for calls, term in zip(per_turn_calls, terminated)
        ]
    }


def test_multiturn_all_turns_correct_scores_success():
    inst = _two_turn_instance()
    w = MultiTurnToolWorkload(jsonl_path="unused")
    trace = _trace([TURN1_CALLS, TURN2_CALLS])
    res = w.score_detailed(inst, trace)
    assert res["task_success"] == 1.0
    assert res["turn_failures"] == 0.0
    assert res["nonterminated"] is False


def test_multiturn_one_bad_turn_fails_task_and_counts_turn():
    inst = _two_turn_instance()
    w = MultiTurnToolWorkload(jsonl_path="unused")
    # turn2 subtracts the WRONG amount -> ledger != expected
    bad_turn2 = [{"name": "ledger_subtract", "arguments": {"amount": 999}}]
    trace = _trace([TURN1_CALLS, bad_turn2])
    res = w.score_detailed(inst, trace)
    assert res["task_success"] == 0.0
    assert res["turn_failures"] == 0.5  # 1 of 2 turns failed
    assert res["nonterminated"] is False


def test_multiturn_nontermination_flagged():
    inst = _two_turn_instance()
    w = MultiTurnToolWorkload(jsonl_path="unused")
    # Both turns have the right calls, but turn2's inner loop never terminated.
    trace = _trace([TURN1_CALLS, TURN2_CALLS], terminated=[True, False])
    res = w.score_detailed(inst, trace)
    assert res["nonterminated"] is True
    assert res["task_success"] == 0.0


def test_score_wrapper_matches_task_success():
    inst = _two_turn_instance()
    w = MultiTurnToolWorkload(jsonl_path="unused")
    trace = _trace([TURN1_CALLS, TURN2_CALLS])
    assert w.score(inst, json.dumps(trace)) == w.score_detailed(inst, trace)["task_success"]
    # And for a failing trace:
    bad = _trace([TURN1_CALLS, [{"name": "ledger_subtract", "arguments": {"amount": 1}}]])
    assert w.score(inst, json.dumps(bad)) == w.score_detailed(inst, bad)["task_success"]


def test_score_detailed_accepts_json_string_trace():
    inst = _two_turn_instance()
    w = MultiTurnToolWorkload(jsonl_path="unused")
    trace = _trace([TURN1_CALLS, TURN2_CALLS])
    res = w.score_detailed(inst, json.dumps(trace))
    assert res["task_success"] == 1.0


def test_missing_function_turn_expects_unchanged_state():
    # A 2-turn task where turn2 asks for something NO tool can do (missing-function);
    # its expected post-turn state equals the prior state (unchanged).
    turn1 = [{"name": "ledger_add", "arguments": {"amount": 50}}]
    turn2_noop: list[dict] = []  # no state-changing call is the correct behaviour
    inst = _make_instance(
        "t_missing_fn",
        ["add 50 to the ledger", "please book me a flight to mars"],
        {},
        [turn1, turn2_noop],
    )
    w = MultiTurnToolWorkload(jsonl_path="unused")

    # Correct: model makes no graded-state change in turn2 (maybe only a decoy call).
    good = _trace([turn1, [{"name": "web_search", "arguments": {"query": "mars flights"}}]])
    assert w.score_detailed(inst, good)["task_success"] == 1.0

    # Wrong: model hallucinates a state mutation in turn2.
    bad = _trace([turn1, [{"name": "ledger_add", "arguments": {"amount": 1}}]])
    res = w.score_detailed(inst, bad)
    assert res["task_success"] == 0.0
    assert res["turn_failures"] == 0.5


def test_missing_turns_count_as_failed():
    # Model gave up after turn1: trace has only 1 turn, task expects 2.
    inst = _two_turn_instance()
    w = MultiTurnToolWorkload(jsonl_path="unused")
    trace = _trace([TURN1_CALLS])  # only turn1 present
    res = w.score_detailed(inst, trace)
    assert res["task_success"] == 0.0
    assert res["turn_failures"] == 0.5  # the missing turn2 counts as failed


def test_instances_from_jsonl(tmp_path):
    initial = {"notes": {}, "ledger": 0}
    expected = _expected_states(initial, TWO_TURN_REF)
    row = {
        "id": "t1",
        "turns": TWO_TURN_USER,
        "initial_state": initial,
        "expected_states": expected,
    }
    p = tmp_path / "tooluse_mt_megaquick.jsonl"
    p.write_text(json.dumps(row) + "\n", encoding="utf-8")
    w = MultiTurnToolWorkload(jsonl_path=p)
    items = w.instances()
    assert len(items) == 1
    inp = json.loads(items[0].input)
    tgt = json.loads(items[0].target)
    assert inp["id"] == "t1"
    assert inp["turns"] == TWO_TURN_USER
    assert tgt["expected_states"] == expected
    # The instance produced from the JSONL grades a correct trace as success.
    trace = _trace([TURN1_CALLS, TURN2_CALLS])
    assert w.score_detailed(items[0], trace)["task_success"] == 1.0


def test_score_detailed_exposes_raw_counts_for_microaverage():
    # Cell-level turn_failure_rate is a micro-average over all tasks, so score_detailed must
    # expose raw counts (not just the per-task fraction) for the analysis to sum correctly.
    inst = _two_turn_instance()
    w = MultiTurnToolWorkload(jsonl_path=None)
    bad_turn2 = [{"name": "ledger_subtract", "arguments": {"amount": 999}}]
    res = w.score_detailed(inst, _trace([TURN1_CALLS, bad_turn2]))
    assert res["n_turns"] == 2
    assert res["failed_turns"] == 1
    assert res["turn_failures"] == 0.5


def test_extra_trace_turns_beyond_expected_are_ignored():
    # A 1-turn task; the trace has a spurious 2nd turn that never terminated. Non-termination
    # is scoped to the task's graded turns, so the extra turn does not flip the verdict.
    inst = _make_instance("t_one", ["set note + add 100"], {}, [TURN1_CALLS])
    w = MultiTurnToolWorkload(jsonl_path=None)
    trace = _trace(
        [TURN1_CALLS, [{"name": "ledger_add", "arguments": {"amount": 1}}]],
        terminated=[True, False],
    )
    res = w.score_detailed(inst, trace)
    assert res["task_success"] == 1.0
    assert res["nonterminated"] is False
    assert res["n_turns"] == 1


def test_empty_expected_states_is_not_a_success():
    inst = Instance(
        input=json.dumps({"id": "t_empty", "turns": [], "initial_state": {}}),
        target=json.dumps({"id": "t_empty", "initial_state": {}, "expected_states": []}),
        raw={},
    )
    w = MultiTurnToolWorkload(jsonl_path=None)
    res = w.score_detailed(inst, _trace([]))
    assert res["task_success"] == 0.0
    assert res["n_turns"] == 0
    assert res["turn_failures"] == 0.0


def test_instances_returns_empty_when_no_path():
    assert MultiTurnToolWorkload(jsonl_path=None).instances() == []
