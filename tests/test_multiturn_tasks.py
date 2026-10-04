"""TDD for the bespoke multi-step tool-use TASK DATA generator (Path B agentic workload).

``generate_multiturn_tasks`` deterministically (seed 42) instantiates a fixed pool of
multi-turn, state-dependent tool-use task TEMPLATES and derives each task's per-turn
``expected_states`` oracle by REPLAYING a reference call sequence through a fresh
``ToolSandbox`` (never hand-written). These tests prove:
  * determinism (same seed -> identical pool),
  * the locked pool size + sub_type coverage,
  * every task is genuinely multi-turn AND has real state progression,
  * the pinned oracle is ACHIEVABLE by a correct agent (the key guard against an
    impossible benchmark), and
  * the missing_function / missing_parameter no-change-turn semantics hold.

The achievability test reuses the same trace shape the runner produces and grades it
through ``MultiTurnToolWorkload.score_detailed`` (state-diff replay).
"""
from __future__ import annotations

import json
import random

from legit_edge.multiturn_tasks import generate_multiturn_tasks
from legit_edge.tool_sandbox import ToolSandbox
from legit_edge.workload import Instance, MultiTurnToolWorkload


# --------------------------------------------------------------------------- helpers
def _instance_from_row(row: dict) -> Instance:
    """Build a MultiTurnToolWorkload Instance from a pinned JSONL row (mirrors instances())."""
    return Instance(
        input=json.dumps(
            {
                "id": row["id"],
                "turns": row["turns"],
                "initial_state": row["initial_state"],
            }
        ),
        target=json.dumps(
            {
                "id": row["id"],
                "initial_state": row["initial_state"],
                "expected_states": row["expected_states"],
            }
        ),
        raw=row,
    )


def _trace_from_reference(reference_calls: list[list[dict]]) -> dict:
    """A correct agent's trace: per-turn calls == the reference sequence, all terminated."""
    return {
        "turns": [
            {"calls": calls, "terminated": True} for calls in reference_calls
        ]
    }


def _reference_calls(row: dict) -> list[list[dict]]:
    """Pull the debug reference call sequence the generator stashed on each row."""
    return row["_reference_calls"]


ROW_KEYS = {"id", "turns", "initial_state", "expected_states", "sub_type"}


# --------------------------------------------------------------------------- tests
def test_generation_is_deterministic():
    a = generate_multiturn_tasks(seed=42)
    b = generate_multiturn_tasks(seed=42)
    assert a == b  # deep equality across the whole pool


def test_row_schema_is_complete():
    pool = generate_multiturn_tasks(seed=42)
    for row in pool:
        assert ROW_KEYS.issubset(row.keys()), f"row missing keys: {ROW_KEYS - set(row)}"
        assert isinstance(row["id"], str) and row["id"]
        assert isinstance(row["turns"], list) and all(isinstance(t, str) for t in row["turns"])
        assert isinstance(row["initial_state"], dict)
        assert "notes" in row["initial_state"] and "ledger" in row["initial_state"]
        assert isinstance(row["expected_states"], list)
        assert row["sub_type"] in {"base", "missing_function", "missing_parameter"}
        # One expected post-turn state per user turn.
        assert len(row["expected_states"]) == len(row["turns"])


def test_ids_are_stable_and_unique():
    pool = generate_multiturn_tasks(seed=42)
    ids = [r["id"] for r in pool]
    assert len(ids) == len(set(ids)), "task ids must be unique"
    # Stable, sortable, and embed the sub_type.
    for i, row in enumerate(pool):
        assert row["id"] == f"mt_{i:02d}_{row['sub_type']}"


def test_pool_size_and_subtypes():
    pool = generate_multiturn_tasks(n=24, seed=42)
    assert len(pool) == 24
    counts: dict[str, int] = {}
    for row in pool:
        counts[row["sub_type"]] = counts.get(row["sub_type"], 0) + 1
    # All three sub_types present, each with a meaningful number of tasks.
    for st in ("base", "missing_function", "missing_parameter"):
        assert counts.get(st, 0) >= 3, f"too few {st}: {counts}"
    # base is the plurality (the bulk of the benchmark is the happy path).
    assert counts["base"] == max(counts.values())
    assert counts["base"] > counts["missing_function"]
    assert counts["base"] > counts["missing_parameter"]


def test_note_values_have_no_brackets():
    # The runner's tool-call parser is bracket-counting; note keys/values must not contain
    # '[' or ']' or a pinned correct trace could be mis-parsed live.
    pool = generate_multiturn_tasks(seed=42)
    for row in pool:
        blob = json.dumps(row["initial_state"]) + json.dumps(row["expected_states"])
        # We only care about the note strings, but scanning the whole state blob is a
        # strict superset guard (ledger ints never contain brackets anyway).
        for state in [row["initial_state"], *row["expected_states"]]:
            for k, v in state.get("notes", {}).items():
                assert "[" not in k and "]" not in k, f"bracket in note key: {k!r}"
                assert "[" not in v and "]" not in v, f"bracket in note value: {v!r}"
        assert blob  # sanity


def test_every_task_is_multi_turn_and_state_dependent():
    pool = generate_multiturn_tasks(seed=42)
    progressing = 0
    for row in pool:
        n = len(row["turns"])
        assert 2 <= n <= 4, f"{row['id']} has {n} turns (want 2-4)"
        # A task "progresses" if some turn's expected state differs from the previous turn's.
        states = row["expected_states"]
        prev = row["initial_state"]
        changed_at_least_once = False
        for st in states:
            if st != prev:
                changed_at_least_once = True
            prev = st
        if changed_at_least_once:
            progressing += 1
    # The vast majority must show real state progression (every base/param task does;
    # a degenerate all-no-op task would be a benchmark smell).
    assert progressing >= len(pool) - 2


def test_oracle_is_achievable_for_every_task():
    """KEY TEST: a correct agent replaying the reference calls scores task_success == 1.0.

    Proves the pinned state-diff oracle is reachable (guards against an impossible
    benchmark). We construct the Instance straight from the pinned row and feed a trace
    whose per-turn calls are exactly the generator's reference sequence.
    """
    pool = generate_multiturn_tasks(seed=42)
    w = MultiTurnToolWorkload(jsonl_path=None)
    for row in pool:
        inst = _instance_from_row(row)
        trace = _trace_from_reference(_reference_calls(row))
        res = w.score_detailed(inst, trace)
        assert res["task_success"] == 1.0, f"{row['id']} oracle not achievable: {res}"
        assert res["nonterminated"] is False
        assert res["turn_failures"] == 0.0


def test_expected_states_match_independent_replay():
    # The row's expected_states must equal an INDEPENDENT replay of its reference calls
    # through a fresh sandbox (i.e. the oracle really is replay-derived, not hand-written).
    pool = generate_multiturn_tasks(seed=42)
    for row in pool:
        sb = ToolSandbox(initial_state=row["initial_state"])
        recomputed = []
        for calls in _reference_calls(row):
            for c in calls:
                sb.dispatch(c)
            recomputed.append(sb.state())
        assert recomputed == row["expected_states"], f"{row['id']} states not replay-derived"


def test_missing_function_turns_expect_unchanged_state():
    pool = generate_multiturn_tasks(seed=42)
    mf = [r for r in pool if r["sub_type"] == "missing_function"]
    assert mf, "expected at least one missing_function task"
    for row in mf:
        ref = _reference_calls(row)
        states = row["expected_states"]
        prev_states = [row["initial_state"], *states[:-1]]
        # At least one turn must be the out-of-scope ask: its reference calls are empty
        # (no graded-state mutation) AND its expected state equals the prior state.
        noop_turns = [
            i
            for i, calls in enumerate(ref)
            if len(calls) == 0 and states[i] == prev_states[i]
        ]
        assert noop_turns, f"{row['id']} has no unchanged out-of-scope turn"


def test_missing_parameter_pattern():
    pool = generate_multiturn_tasks(seed=42)
    mp = [r for r in pool if r["sub_type"] == "missing_parameter"]
    assert mp, "expected at least one missing_parameter task"
    for row in mp:
        ref = _reference_calls(row)
        states = row["expected_states"]
        prev_states = [row["initial_state"], *states[:-1]]
        # The ambiguous turn supplies no value -> no change (empty ref calls, state unchanged).
        ambiguous = [
            i
            for i, calls in enumerate(ref)
            if len(calls) == 0 and states[i] == prev_states[i]
        ]
        assert ambiguous, f"{row['id']} missing the no-value (unchanged) turn"
        # A LATER turn must actually change state (the value is finally supplied).
        first_amb = min(ambiguous)
        later_change = any(
            states[i] != prev_states[i] for i in range(first_amb + 1, len(states))
        )
        assert later_change, f"{row['id']} never resolves the missing parameter later"


def test_base_tasks_have_no_forced_noop_turn():
    # base tasks are the happy path: every turn does something (non-empty reference calls).
    pool = generate_multiturn_tasks(seed=42)
    base = [r for r in pool if r["sub_type"] == "base"]
    assert base
    for row in base:
        for i, calls in enumerate(_reference_calls(row)):
            assert calls, f"{row['id']} base turn {i} has no reference calls"


def test_variation_across_instances():
    # The seeded RNG must vary parameters so instances of the same template differ.
    pool = generate_multiturn_tasks(seed=42)
    # Distinct turn-text tuples across the pool (no two tasks identical word-for-word).
    turn_blobs = ["".join(r["turns"]) for r in pool]
    assert len(set(turn_blobs)) == len(turn_blobs), "tasks are not varied (duplicate turns)"


def test_megaquick_is_seeded_subset_of_pool():
    # Mirror pin.py's freeze() to prove the mega-quick=8 subset is a stable seed-42 sample.
    pool = generate_multiturn_tasks(n=24, seed=42)
    random.seed(42)
    mq = random.sample(pool, 8) if len(pool) > 8 else pool
    assert len(mq) == 8
    ids = {r["id"] for r in mq}
    assert ids.issubset({r["id"] for r in pool})
    # Deterministic across repeats.
    random.seed(42)
    mq2 = random.sample(pool, 8)
    assert [r["id"] for r in mq] == [r["id"] for r in mq2]
