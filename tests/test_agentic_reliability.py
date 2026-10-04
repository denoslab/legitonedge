"""Tests for the separate agentic (multi-turn tool-use) reliability analysis.

Builds a synthetic results dir of ``{model}__{tier}__tooluse_mt.traces.jsonl`` files whose
rows are constructed exactly as the runner writes them — ``score_metadata`` is produced by
``MultiTurnToolWorkload.score_detailed`` on the SAME (instance, trace) pair, so the fidelity
re-grade matches by construction. We then assert:
  * per-cell task_success_rate / MICRO-averaged turn_failure_rate / nontermination_rate;
  * the fidelity check passes on consistent rows and RAISES on a corrupted score_metadata row;
  * the script skips dirs with no tooluse_mt cells.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str((Path(__file__).parent.parent / "scripts" / "analysis").resolve()))

from legit_edge.tool_sandbox import ToolSandbox  # noqa: E402
from legit_edge.workload import Instance, MultiTurnToolWorkload  # noqa: E402


# --------------------------------------------------------------------------- fixtures
def _expected_states(initial_state: dict, per_turn_calls: list[list[dict]]) -> list[dict]:
    sb = ToolSandbox(initial_state=initial_state)
    out = []
    for calls in per_turn_calls:
        for c in calls:
            sb.dispatch(c)
        out.append(sb.state())
    return out


def _instance(task_id: str, turns: list[str], initial_state: dict,
              ref_calls: list[list[dict]]) -> Instance:
    expected = _expected_states(initial_state, ref_calls)
    return Instance(
        input=json.dumps({"id": task_id, "turns": turns, "initial_state": initial_state}),
        target=json.dumps(
            {"id": task_id, "initial_state": initial_state, "expected_states": expected}
        ),
        raw={},
    )


def _trace(per_turn_calls: list[list[dict]], terminated: list[bool] | None = None) -> dict:
    if terminated is None:
        terminated = [True] * len(per_turn_calls)
    return {"turns": [{"calls": c, "terminated": t}
                      for c, t in zip(per_turn_calls, terminated)]}


def _row(inst: Instance, trace: dict) -> dict:
    """A trace row exactly as run_multiturn_cell writes it (score_metadata from score_detailed)."""
    detail = MultiTurnToolWorkload(jsonl_path=None).score_detailed(inst, trace)
    return {
        "input": inst.input,
        "target": inst.target,
        "output": json.dumps(trace),
        "score": detail["task_success"],
        "score_metadata": detail,
        "latency_s": 0.1,
        "output_tokens": 10,
        "input_tokens": 20,
    }


# Reusable building blocks.
SET_ADD = [{"name": "set_note", "arguments": {"key": "payee", "value": "alice"}},
           {"name": "ledger_add", "arguments": {"amount": 100}}]
SUB40 = [{"name": "ledger_subtract", "arguments": {"amount": 40}}]
ADD5 = [{"name": "ledger_add", "arguments": {"amount": 5}}]
WRONG_SUB = [{"name": "ledger_subtract", "arguments": {"amount": 999}}]


def _four_task_rows() -> list[dict]:
    """4 tasks: 2 success, 1 wrong-turn fail, 1 nontermination fail.

    Hand-computed cell aggregates:
      task_success_rate   = 2/4 = 0.5
      sum failed_turns    = 0(A) + 0(B) + 1(C) + 0(D) = 1
      sum n_turns         = 2(A) + 3(B) + 2(C) + 2(D) = 9
      turn_failure_rate   = 1/9   (MICRO-average, not mean of per-task fractions)
      nontermination_rate = 1/4 = 0.25
    """
    rows = []
    # A: 2 turns, all correct, terminated -> success
    instA = _instance("A", ["t1", "t2"], {}, [SET_ADD, SUB40])
    rows.append(_row(instA, _trace([SET_ADD, SUB40])))
    # B: 3 turns, all correct, terminated -> success
    instB = _instance("B", ["t1", "t2", "t3"], {}, [SET_ADD, SUB40, ADD5])
    rows.append(_row(instB, _trace([SET_ADD, SUB40, ADD5])))
    # C: 2 turns, turn2 wrong -> fail (1 failed turn)
    instC = _instance("C", ["t1", "t2"], {}, [SET_ADD, SUB40])
    rows.append(_row(instC, _trace([SET_ADD, WRONG_SUB])))
    # D: 2 turns, correct calls but turn2 never terminated -> fail (nonterminated, 0 failed turns)
    instD = _instance("D", ["t1", "t2"], {}, [SET_ADD, SUB40])
    rows.append(_row(instD, _trace([SET_ADD, SUB40], terminated=[True, False])))
    return rows


def _write_cell(d: Path, cell_base: str, rows: list[dict]) -> None:
    (d / f"{cell_base}.traces.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )


# --------------------------------------------------------------------------- aggregation
def test_aggregate_cell_metrics_match_hand_computation():
    import agentic_reliability as ar
    rows = _four_task_rows()
    agg = ar.aggregate_cell(rows, b=200, seed=1)
    assert agg["n_tasks"] == 4
    assert agg["task_success_rate"] == pytest.approx(0.5)
    assert agg["nontermination_rate"] == pytest.approx(0.25)
    # MICRO-average: sum(failed)/sum(turns) = 1/9, NOT the mean of per-task fractions
    # (which would be mean(0,0,0.5,0)=0.125).
    assert agg["sum_failed_turns"] == 1
    assert agg["sum_turns"] == 9
    assert agg["turn_failure_rate"] == pytest.approx(1 / 9)
    assert agg["turn_failure_rate"] != pytest.approx(0.125)  # guards against macro-average
    lo, hi = agg["task_success_ci"]
    assert lo <= agg["task_success_rate"] <= hi


def test_micro_average_differs_from_macro_on_uneven_turn_counts():
    import agentic_reliability as ar
    # Task1: 4 turns, 2 failed (macro fraction 0.5). Task2: 2 turns, 0 failed (macro 0.0).
    # Macro mean = 0.25; micro = (2+0)/(4+2) = 2/6 = 0.333... -> they must differ.
    four = [SET_ADD, SUB40, ADD5, SUB40]
    inst1 = _instance("u1", ["a", "b", "c", "d"], {}, four)
    # corrupt turns 3 and 4 so exactly 2 of 4 fail
    bad = _trace([SET_ADD, SUB40, WRONG_SUB, WRONG_SUB])
    inst2 = _instance("u2", ["a", "b"], {}, [SET_ADD, SUB40])
    good2 = _trace([SET_ADD, SUB40])
    rows = [_row(inst1, bad), _row(inst2, good2)]
    agg = ar.aggregate_cell(rows, b=200, seed=1)
    assert agg["sum_turns"] == 6
    assert agg["sum_failed_turns"] == 2
    assert agg["turn_failure_rate"] == pytest.approx(2 / 6)
    macro = (2 / 4 + 0 / 2) / 2  # 0.25
    assert agg["turn_failure_rate"] != pytest.approx(macro)


# --------------------------------------------------------------------------- fidelity
def test_fidelity_check_passes_on_consistent_rows():
    import agentic_reliability as ar
    ar.fidelity_check(_four_task_rows())  # must not raise


def test_fidelity_check_raises_on_corrupted_metadata():
    import agentic_reliability as ar
    rows = _four_task_rows()
    # Flip a stored task_success so it no longer matches the re-graded trace.
    rows[0]["score_metadata"] = dict(rows[0]["score_metadata"])
    rows[0]["score_metadata"]["task_success"] = 0.0  # row A was a genuine success
    with pytest.raises(AssertionError):
        ar.fidelity_check(rows)


def test_fidelity_check_raises_on_corrupted_turn_counts():
    import agentic_reliability as ar
    rows = _four_task_rows()
    rows[1]["score_metadata"] = dict(rows[1]["score_metadata"])
    rows[1]["score_metadata"]["failed_turns"] = 99  # bogus
    with pytest.raises(AssertionError):
        ar.fidelity_check(rows)


# --------------------------------------------------------------------------- discovery + e2e
def test_analyze_finds_cells_and_carries_model_tier(tmp_path):
    import agentic_reliability as ar
    _write_cell(tmp_path, "hermes3_8b__jetson__tooluse_mt", _four_task_rows())
    _write_cell(tmp_path, "llama_3_1_8b__spark__tooluse_mt", _four_task_rows())
    rows = ar.analyze([str(tmp_path)], b=200)
    assert len(rows) == 2
    by_cell = {r["cell"]: r for r in rows}
    h = by_cell["hermes3_8b__jetson__tooluse_mt"]
    assert h["model"] == "hermes3_8b" and h["tier"] == "jetson"
    assert h["agentic_tuned"] is True
    assert h["task_success_rate"] == pytest.approx(0.5)
    assert by_cell["llama_3_1_8b__spark__tooluse_mt"]["agentic_tuned"] is False


def test_run_shards_are_pooled(tmp_path):
    import agentic_reliability as ar
    # k=2 repeat: two __run shards for one cell should pool into a single cell of 8 tasks.
    rows = _four_task_rows()
    (tmp_path / "hermes3_8b__jetson__tooluse_mt__run0.traces.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    (tmp_path / "hermes3_8b__jetson__tooluse_mt__run1.traces.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    out = ar.analyze([str(tmp_path)], b=200)
    assert len(out) == 1
    assert out[0]["n_tasks"] == 8
    assert out[0]["sum_turns"] == 18  # 2 x 9


def test_mode_thermal_recovered_from_summary_json(tmp_path):
    import agentic_reliability as ar
    _write_cell(tmp_path, "hermes3_8b__jetson__tooluse_mt", _four_task_rows())
    # sibling summary carries the cell_id with mode|thermal
    (tmp_path / "hermes3_8b__jetson__tooluse_mt.json").write_text(
        json.dumps({"cell_id": "hermes3_8b|jetson|tooluse_mt|standard|throttled"}),
        encoding="utf-8")
    rows = ar.analyze([str(tmp_path)], b=100)
    assert rows[0]["mode"] == "standard"
    assert rows[0]["thermal"] == "throttled"


def test_skips_dir_with_no_tooluse_mt(tmp_path):
    import agentic_reliability as ar
    # A non-agentic cell present, but no tooluse_mt -> analyze returns nothing, render is graceful.
    (tmp_path / "llama_3_1_8b__spark__math.traces.jsonl").write_text(
        json.dumps({"input": "q", "target": "4", "output": "4", "score": 1.0}) + "\n",
        encoding="utf-8")
    rows = ar.analyze([str(tmp_path)], b=100)
    assert rows == []
    md = ar.render_markdown(rows)
    assert "No `tooluse_mt` cells found" in md


def test_render_markdown_table_has_metrics(tmp_path):
    import agentic_reliability as ar
    _write_cell(tmp_path, "hermes3_8b__jetson__tooluse_mt", _four_task_rows())
    rows = ar.analyze([str(tmp_path)], b=200)
    md = ar.render_markdown(rows)
    assert "task_success" in md
    assert "turn_failure" in md
    assert "nontermination" in md
    assert "hermes3_8b__jetson__tooluse_mt" in md
    # the agentic-tuned mean line should fire because hermes3_8b is tuned
    assert "Agentic-tuned models" in md


def test_main_writes_file(tmp_path):
    import agentic_reliability as ar
    _write_cell(tmp_path, "hermes3_8b__jetson__tooluse_mt", _four_task_rows())
    out = tmp_path / "agentic.md"
    rc = ar.main([str(tmp_path), "--out", str(out), "--b", "100"])
    assert rc == 0
    assert out.exists()
    assert "Agentic multi-turn tool-use reliability" in out.read_text(encoding="utf-8")
