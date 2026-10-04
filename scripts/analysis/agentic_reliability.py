"""Agentic (multi-turn tool-use) reliability — the SEPARATE pre-registered result.

Per pre-registration amendment E.5, the multi-step agentic workload is reported
as a STANDALONE reliability characterization and is **deliberately NOT folded into the
reliability composite** (capability there is single-turn; the agentic metric
is a distinct task-completion axis). This script aggregates the three pre-registered
agentic metrics per cell, over the committed ``*__*__tooluse_mt*.traces.jsonl`` traces:

  * **task_success_rate** = mean over tasks of per-task ``task_success`` (0/1), with a
    B=10,000 percentile bootstrap 95% CI resampling tasks (the framework convention).
  * **turn_failure_rate** = MICRO-average = sum(failed_turns) / sum(n_turns) over tasks.
    This weights by turn count (a 4-turn task contributes 4 turns), and is NOT the mean of
    the per-task ``failed_turns / n_turns`` fractions — the pre-registration fixes the
    micro form so a few long tasks are not down-weighted to a short task's level.
  * **nontermination_rate** = mean over tasks of the per-task ``nonterminated`` flag — the
    fraction of tasks where the agent kept emitting tool calls until the step budget on at
    least one turn (never gave a final answer).

A **fidelity re-grade** (``fidelity_check``) reconstructs each trace row's ``Instance`` and
trace and re-runs ``MultiTurnToolWorkload.score_detailed`` (the same state-diff replay oracle
the runner used), asserting the recomputed ``{task_success, turn_failures, nonterminated}``
matches the stored ``score_metadata``. This guards against drift between run-time scoring and
this post-hoc analysis. It is a function so the test can drive it directly.

Post-hoc over committed traces only (0 new compute). Skips gracefully if a results dir has no
``tooluse_mt`` cells.

Usage:
  python agentic_reliability.py <results-dir> [<results-dir> ...] [--out report.md] [--b 10000]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Allow ``import legit_edge`` when run as a bare script from scripts/analysis/.
sys.path.insert(0, str((Path(__file__).resolve().parent.parent.parent / "src")))
from legit_edge.workload import Instance, MultiTurnToolWorkload  # noqa: E402

WORKLOAD_NAME = "tooluse_mt"
B = 10_000
SEED = 42
# Models whose presence we explicitly surface (the agentic-tuned pair), so the reader can
# see at a glance whether tool-tuned models actually score on the agentic axis.
AGENTIC_TUNED = {"hermes3_8b", "llama3_groq_tooluse_8b"}


# --------------------------------------------------------------------------- discovery
def cell_files(run_dir: Path, cell_base: str) -> list[Path]:
    """Trace files for a tooluse_mt cell: the k>1 ``__run{j}`` shards else the single file.

    Mirrors ``reshoot_calibration._cell_files`` (the calibration script) so k-repeat runs pool correctly.
    """
    run_dir = Path(run_dir)
    files = sorted(run_dir.glob(f"{cell_base}__run*.traces.jsonl"))
    if not files:
        single = run_dir / f"{cell_base}.traces.jsonl"
        files = [single] if single.exists() else []
    return files


def find_cells(run_dir: Path) -> list[str]:
    """Distinct ``{model}__{tier}__tooluse_mt`` cell bases in a results dir (run-shards pooled)."""
    d = Path(run_dir)
    bases: set[str] = set()
    for f in d.glob(f"*__{WORKLOAD_NAME}*.traces.jsonl"):
        name = f.name.replace(".traces.jsonl", "")
        base = name.split("__run")[0]
        # keep only true tooluse_mt cells (suffix exactly the workload name)
        if base.endswith(f"__{WORKLOAD_NAME}"):
            bases.add(base)
    return sorted(bases)


def cell_mode_thermal(run_dir: Path, cell_base: str) -> tuple[str, str]:
    """Recover (mode, thermal) from the sibling summary JSON's cell_id, else ('-', '-').

    The trace rows do not carry mode/thermal; the per-cell ``.json`` summary does, as
    ``cell_id = model|tier|wl|mode|thermal``. We read whichever summary shard exists.
    """
    d = Path(run_dir)
    cands = sorted(d.glob(f"{cell_base}__run*.json")) + [d / f"{cell_base}.json"]
    for p in cands:
        if p.exists() and not p.name.endswith(".traces.jsonl"):
            try:
                cid = json.loads(p.read_text(encoding="utf-8")).get("cell_id", "")
            except Exception:
                continue
            parts = cid.split("|")
            if len(parts) >= 5:
                return parts[3], parts[4]
    return "-", "-"


def _read_rows(run_dir: Path, cell_base: str) -> list[dict]:
    rows: list[dict] = []
    for f in cell_files(run_dir, cell_base):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


# --------------------------------------------------------------------------- fidelity
def fidelity_check(rows: list[dict]) -> None:
    """Re-grade every row's trace and assert it matches the stored ``score_metadata``.

    Reconstructs ``Instance(input, target, raw={})`` and ``trace = json.loads(row["output"])``,
    then re-runs ``MultiTurnToolWorkload(jsonl_path=None).score_detailed``. Raises
    ``AssertionError`` on any drift in task_success / turn_failures / nonterminated /
    n_turns / failed_turns. This is the run-time-vs-analysis consistency guard.
    """
    wl = MultiTurnToolWorkload(jsonl_path=None)
    for i, r in enumerate(rows):
        stored = r.get("score_metadata")
        if stored is None:
            raise AssertionError(f"row {i}: missing score_metadata")
        inst = Instance(input=r["input"], target=r["target"], raw={})
        trace = json.loads(r["output"])
        regraded = wl.score_detailed(inst, trace)
        for key in ("task_success", "turn_failures", "nonterminated", "n_turns", "failed_turns"):
            a, b = regraded[key], stored.get(key)
            if isinstance(a, bool) or isinstance(b, bool):
                ok = bool(a) == bool(b)
            elif isinstance(a, float) or isinstance(b, float):
                ok = abs(float(a) - float(b)) < 1e-9
            else:
                ok = a == b
            if not ok:
                raise AssertionError(
                    f"row {i} fidelity mismatch on {key!r}: regraded={a!r} stored={b!r}"
                )


# --------------------------------------------------------------------------- metrics
def _bootstrap_mean_ci(values: np.ndarray, b: int = B, seed: int = SEED) -> tuple[float, float]:
    """Percentile bootstrap 95% CI on the mean, resampling the unit (tasks) with replacement."""
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = np.empty(b)
    for k in range(b):
        boots[k] = values[rng.integers(0, n, size=n)].mean()
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def aggregate_cell(rows: list[dict], b: int = B, seed: int = SEED) -> dict:
    """Compute the three pre-registered agentic metrics for one cell from its task rows.

    Reads each task's ``score_metadata`` (task_success / nonterminated / n_turns /
    failed_turns). turn_failure_rate is the micro-average over turns; task_success_rate
    and nontermination_rate are means over tasks (task_success_rate carries a bootstrap CI).
    """
    succ, nonterm = [], []
    sum_failed = sum_turns = 0
    for r in rows:
        meta = r.get("score_metadata", {})
        succ.append(float(meta.get("task_success", 0.0)))
        nonterm.append(1.0 if meta.get("nonterminated") else 0.0)
        sum_failed += int(meta.get("failed_turns", 0))
        sum_turns += int(meta.get("n_turns", 0))
    succ_arr = np.array(succ, dtype=float)
    nonterm_arr = np.array(nonterm, dtype=float)
    lo, hi = _bootstrap_mean_ci(succ_arr, b=b, seed=seed)
    return {
        "n_tasks": len(rows),
        "task_success_rate": float(succ_arr.mean()) if len(succ_arr) else float("nan"),
        "task_success_ci": (lo, hi),
        "turn_failure_rate": (sum_failed / sum_turns) if sum_turns else float("nan"),
        "nontermination_rate": float(nonterm_arr.mean()) if len(nonterm_arr) else float("nan"),
        "sum_failed_turns": sum_failed,
        "sum_turns": sum_turns,
    }


def analyze(run_dirs: list[str], b: int = B, seed: int = SEED) -> list[dict]:
    """Per-cell agentic metrics across one or more results dirs (each cell fidelity-checked)."""
    out: list[dict] = []
    for rd in run_dirs:
        d = Path(rd)
        for base in find_cells(d):
            rows = _read_rows(d, base)
            if not rows:
                continue
            fidelity_check(rows)   # guard: run-time vs analysis scoring must agree
            agg = aggregate_cell(rows, b=b, seed=seed)
            model, tier, _wl = base.split("__")[:3]
            mode, thermal = cell_mode_thermal(d, base)
            agg.update({
                "cell": base, "model": model, "tier": tier,
                "mode": mode, "thermal": thermal,
                "agentic_tuned": model in AGENTIC_TUNED,
            })
            out.append(agg)
    return out


# --------------------------------------------------------------------------- render
def render_markdown(rows: list[dict], b: int = B) -> str:
    if not rows:
        return (
            "# Agentic multi-turn tool-use reliability (separate from the reliability composite)\n\n"
            "_No `tooluse_mt` cells found in the supplied results dir(s)._\n"
        )
    L = [
        "# Agentic multi-turn tool-use reliability (separate from the reliability composite)\n",
        "_Pre-registered as a STANDALONE result — NOT folded into the "
        "reliability composite. Per cell (model x tier x mode x regime): task_success_rate "
        f"(mean over tasks, bootstrap 95% CI B={b:,}, seed {SEED}); turn_failure_rate = "
        "MICRO-average = sum(failed_turns)/sum(n_turns); nontermination_rate = mean over "
        "tasks of the per-task nonterminated flag. ✓ marks the agentic-tuned models._\n",
        "| cell | tier | mode | regime | tuned? | n_tasks | task_success [95% CI] | "
        "turn_failure (micro) | nontermination |",
        "|---|---|---|---|:--:|--:|:--:|--:|--:|",
    ]
    for r in sorted(rows, key=lambda z: (z["tier"], z["cell"])):
        lo, hi = r["task_success_ci"]
        ci = f"[{lo:.3f}, {hi:.3f}]" if lo == lo else "[n/a]"
        L.append(
            f"| {r['cell']} | {r['tier']} | {r['mode']} | {r['thermal']} | "
            f"{'✓' if r['agentic_tuned'] else ''} | {r['n_tasks']} | "
            f"{r['task_success_rate']:.3f} {ci} | "
            f"{r['turn_failure_rate']:.3f} | {r['nontermination_rate']:.3f} |"
        )
    # Compact aggregate so the headline reads at a glance.
    succ = [r["task_success_rate"] for r in rows if r["task_success_rate"] == r["task_success_rate"]]
    tuned_succ = [r["task_success_rate"] for r in rows
                  if r["agentic_tuned"] and r["task_success_rate"] == r["task_success_rate"]]
    L.append("")
    if succ:
        L.append(
            f"**{len(rows)} cells.** Mean task_success_rate across cells "
            f"**{float(np.mean(succ)):.3f}** (range {min(succ):.3f}–{max(succ):.3f})."
        )
    if tuned_succ:
        L.append(
            f"Agentic-tuned models ({', '.join(sorted(AGENTIC_TUNED))}): mean task_success_rate "
            f"**{float(np.mean(tuned_succ)):.3f}** over {len(tuned_succ)} cell(s)."
        )
    else:
        L.append(
            "_No agentic-tuned-model cells found in the supplied run dirs; "
            "the table holds the general-purpose models only._"
        )
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dirs", nargs="+", help="results dir(s) with *__*__tooluse_mt traces")
    ap.add_argument("--out", default="-", help="output Markdown path, or - for stdout")
    ap.add_argument("--b", type=int, default=B, help="bootstrap replicates (default 10,000)")
    args = ap.parse_args(argv)
    rows = analyze(args.run_dirs, b=args.b)
    md = render_markdown(rows, b=args.b)
    if args.out != "-":
        Path(args.out).write_text(md, encoding="utf-8")
        print(f"wrote {args.out} ({len(rows)} cells)")
    else:
        print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
