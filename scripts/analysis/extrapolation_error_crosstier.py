"""Compute |megaquick - standard| / standard per scalar per cell, BOTH tiers.

Mega-quick vs standard agreement over the full grid (both tiers). Pre-registered
H2 is defined over the 5 *reliability* dims; this script compares the latency tail
(p50, p99), which is directly comparable between the two modes. `capability` is
reported as a CONVENIENCE column -- it is NOT one of the pre-registered H2
reliability dims (PRE_REGISTRATION.md lists it separately); treat any capability
row as exploratory, not an H2 test.

Pre-registered H2 (PRE_REGISTRATION.md): median |mq-std|/std <= 0.15 across the
5 reliability dimensions, computed per-cell. seed=42 is irrelevant here (no
resampling; this reads committed point estimates).

Usage:
    uv run python scripts/analysis/extrapolation_error_crosstier.py DIR [DIR ...]

Pass every run dir (any mix of mega-quick / standard, any tiers). Each dir's
cells are classified by their `mode` field and keyed by (model, tier, workload)
from cell_id (`model|tier|workload|mode|thermal`). mega-quick is paired against
standard within each (model, tier, workload).
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path


def _load(dirs: list[str]) -> tuple[dict, dict]:
    """Return ({(model,tier,wl): cell} megaquick, ...standard)."""
    mq: dict[tuple[str, str, str], dict] = {}
    st: dict[tuple[str, str, str], dict] = {}
    for d in dirs:
        for f in sorted(Path(d).glob("*.json")):
            c = json.loads(f.read_text(encoding="utf-8"))
            parts = c["cell_id"].split("|")  # model|tier|workload|mode|thermal
            key = (parts[0], parts[1], parts[2])
            mode = c.get("mode") or parts[3]
            (mq if mode == "megaquick" else st)[key] = c
    return mq, st


def _rel_err(mq_val: float, st_val: float) -> float:
    if st_val == 0 or st_val != st_val:  # zero or NaN
        return float("nan")
    return abs(mq_val - st_val) / abs(st_val)


def _cell_errs(m: dict, s: dict) -> tuple[float, float, float]:
    return (
        _rel_err(m["capability"], s["capability"]),
        _rel_err(m["metrics"]["latency"]["p50"], s["metrics"]["latency"]["p50"]),
        _rel_err(m["metrics"]["latency"]["p99"], s["metrics"]["latency"]["p99"]),
    )


def _verdict(rows: list[tuple], label: str) -> None:
    cap = statistics.median([r[3] for r in rows])
    p50 = statistics.median([r[4] for r in rows])
    p99 = statistics.median([r[5] for r in rows])
    worst = max(cap, p50, p99)
    tag = "WITHIN" if worst <= 0.15 else "EXCEEDS"
    print(
        f"- **{label}** (n={len(rows)}): median |Δ| cap={cap:.3f} "
        f"p50={p50:.3f} p99={p99:.3f}; worst-of-three={worst:.3f} "
        f"-> **{tag}** the 0.15 pre-registered band."
    )


def main(dirs: list[str]) -> None:
    mq, st = _load(dirs)
    keys = sorted(set(mq) & set(st))
    if not keys:
        print("No paired (mega-quick, standard) cells found.", file=sys.stderr)
        sys.exit(1)
    print("# Mega-quick -> Standard extrapolation error -- full cross-tier grid")
    print()
    print(
        "Scalars: capability, latency p50, latency p99. Throughput, energy, "
        "output stability and calibration are not compared here. "
        "Pre-reg H2 band: median |mq-std|/std <= 0.15."
    )
    print()
    print("| Model | Tier | Workload | |Δ cap|/std | |Δ p50|/std | |Δ p99|/std |")
    print("|---|---|---|---:|---:|---:|")
    rows: list[tuple] = []
    for k in keys:
        ce, p50e, p99e = _cell_errs(mq[k], st[k])
        rows.append((k[0], k[1], k[2], ce, p50e, p99e))
        print(f"| {k[0]} | {k[1]} | {k[2]} | {ce:.3f} | {p50e:.3f} | {p99e:.3f} |")
    print()
    n_tool = sum(1 for r in rows if r[2] == "tooluse")
    print(
        f"dagger {n_tool} `tooluse` rows: single-call BFCL capability is near zero "
        "for every model on BOTH tiers, so the capability column is degenerate "
        "there, not real agreement. The latency columns on tooluse cells are "
        "valid (scoring-independent)."
    )
    print()
    print("## Medians & verdict")
    print()
    _verdict(rows, "Overall, all cells (incl. tooluse cap)")
    _verdict([r for r in rows if r[2] != "tooluse"], "Overall, tooluse dropped")
    for tier in sorted({r[1] for r in rows}):
        _verdict([r for r in rows if r[1] == tier], f"{tier} only (incl. tooluse)")
        _verdict(
            [r for r in rows if r[1] == tier and r[2] != "tooluse"],
            f"{tier} only, tooluse dropped",
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1:])
