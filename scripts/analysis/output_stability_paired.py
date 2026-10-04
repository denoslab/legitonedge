"""Paired-Δ analysis (MAXN vs throttled) over per-instance .traces.jsonl files.

Usage:
  python output_stability_paired.py <maxn-run-dir> <throt-run-dir> [--out report.md] [--bootstrap-b 10000]

Computes per-cell paired Δ on capability, p50 latency, p95 latency. Paired
bootstrap 95% CI on Δ_cap (B configurable, defaults to 10 000 per pre-reg).
Holm-Bonferroni MT correction applied across the non-tool-use cells (tooluse
rows are dropped from the capability rollup: single-call tool-use capability is
near zero for every model, so it gives no paired signal).

Outputs Markdown to stdout or --out path.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Callable
import numpy as np


def _load_traces(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def paired_delta_cell(maxn_path: Path, throt_path: Path) -> dict:
    """Per-cell paired Δ over capability + latency percentiles.

    Pairs by truncation to min(len) — both runs use the same deterministic
    seed-42 sampler so instance_index 0 of MAXN is "the same prompt" as
    instance_index 0 of throttled, ordering preserved.
    """
    m = _load_traces(maxn_path)
    t = _load_traces(throt_path)
    n = min(len(m), len(t))
    m = m[:n]; t = t[:n]
    scores_m = np.array([r["score"] for r in m], dtype=float)
    scores_t = np.array([r["score"] for r in t], dtype=float)
    lat_m = np.array([r["latency_s"] for r in m], dtype=float)
    lat_t = np.array([r["latency_s"] for r in t], dtype=float)
    return {
        "n": n,
        "cap_M": float(scores_m.mean()),
        "cap_T": float(scores_t.mean()),
        "delta_cap": float(scores_t.mean() - scores_m.mean()),
        "p50_M": float(np.percentile(lat_m, 50)),
        "p50_T": float(np.percentile(lat_t, 50)),
        "delta_p50": float(np.percentile(lat_t, 50) - np.percentile(lat_m, 50)),
        "p95_M": float(np.percentile(lat_m, 95)),
        "p95_T": float(np.percentile(lat_t, 95)),
        "delta_p95": float(np.percentile(lat_t, 95) - np.percentile(lat_m, 95)),
        "scores_m": scores_m,  # carried for later bootstrap calls
        "scores_t": scores_t,
    }


def paired_bootstrap_ci(values_a: np.ndarray, values_b: np.ndarray,
                         stat_fn: Callable[[np.ndarray, np.ndarray], float],
                         B: int = 10_000, seed: int = 42) -> tuple[float, float]:
    """Paired bootstrap: resample paired indices, apply stat_fn(a_resamp, b_resamp)."""
    rng = np.random.default_rng(seed)
    n = len(values_a)
    if n == 0:
        return (float("nan"), float("nan"))
    boots = np.empty(B, dtype=float)
    for i in range(B):
        idx = rng.integers(0, n, size=n)
        boots[i] = stat_fn(values_a[idx], values_b[idx])
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def paired_bootstrap_pvalue(values_a: np.ndarray, values_b: np.ndarray,
                              stat_fn: Callable[[np.ndarray, np.ndarray], float],
                              B: int = 10_000, seed: int = 42) -> float:
    """Two-sided p-value: fraction of bootstrap replicates that flipped sign
    relative to the point estimate. Cheap nonparametric test for Δ ≠ 0."""
    rng = np.random.default_rng(seed)
    n = len(values_a)
    if n == 0:
        return float("nan")
    point = stat_fn(values_a, values_b)
    boots = np.empty(B, dtype=float)
    for i in range(B):
        idx = rng.integers(0, n, size=n)
        boots[i] = stat_fn(values_a[idx], values_b[idx])
    if point >= 0:
        p = (boots <= 0).mean()
    else:
        p = (boots >= 0).mean()
    return float(min(1.0, 2 * p))


def holm_bonferroni(pvalues: list[float], alpha: float = 0.05) -> list[bool]:
    """Holm step-down. Returns per-input decisions (True = reject H0).

    Implementation: sort indices by p, walk threshold a/(k-i) where k = total,
    i = position (0-based). Stop on first non-reject; remaining accept.
    Result is mapped back to input order.
    """
    k = len(pvalues)
    if k == 0:
        return []
    order = sorted(range(k), key=lambda i: pvalues[i])
    decisions = [False] * k
    for pos, idx in enumerate(order):
        thresh = alpha / (k - pos)
        if pvalues[idx] < thresh:
            decisions[idx] = True
        else:
            break  # step-down stop
    return decisions


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("maxn_dir")
    ap.add_argument("throt_dir")
    ap.add_argument("--out", default="-")
    ap.add_argument("--bootstrap-b", type=int, default=10_000)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args(argv)

    maxn_dir = Path(args.maxn_dir)
    throt_dir = Path(args.throt_dir)
    cells: list[dict] = []
    for traces_path in sorted(maxn_dir.glob("*.traces.jsonl")):
        throt_path = throt_dir / traces_path.name
        if not throt_path.exists():
            continue
        d = paired_delta_cell(traces_path, throt_path)
        d["cell"] = traces_path.name.replace(".traces.jsonl", "")
        cells.append(d)

    # Add bootstrap CIs + p-values on Δ_cap for every cell
    for c in cells:
        ci = paired_bootstrap_ci(
            c["scores_m"], c["scores_t"],
            lambda a, b: float(b.mean() - a.mean()),
            B=args.bootstrap_b,
        )
        c["delta_cap_ci"] = ci
        c["delta_cap_p"] = paired_bootstrap_pvalue(
            c["scores_m"], c["scores_t"],
            lambda a, b: float(b.mean() - a.mean()),
            B=args.bootstrap_b,
        )

    # Non-tool-use aggregate (drop tooluse cells from capability rollup; latency
    # aggregates can include tooluse cells without artifact)
    g6_clean = [c for c in cells if "tooluse" not in c["cell"]]
    p_g6 = [c["delta_cap_p"] for c in g6_clean]
    holm = holm_bonferroni(p_g6, alpha=args.alpha)
    for c, decision in zip(g6_clean, holm):
        c["holm_reject"] = decision
    for c in cells:
        if "tooluse" in c["cell"]:
            c["holm_reject"] = None  # excluded from rollup

    # Render markdown
    out_lines = []
    out_lines.append("# Paired-Δ MAXN vs throttled\n")
    out_lines.append(f"Bootstrap B = {args.bootstrap_b}, Holm-Bonferroni α = {args.alpha} "
                     f"across {len(g6_clean)} non-tool-use cells.\n")
    out_lines.append("| cell | n | cap_M | cap_T | Δcap | Δcap CI95 | p | Holm reject? | Δp50 | Δp95 |")
    out_lines.append("|---|--:|--:|--:|--:|--:|--:|:--:|--:|--:|")
    for c in cells:
        ci = c["delta_cap_ci"]
        holm_str = ("—" if c["holm_reject"] is None
                    else ("✓" if c["holm_reject"] else "✗"))
        out_lines.append(
            f"| {c['cell']} | {c['n']} | "
            f"{c['cap_M']:.3f} | {c['cap_T']:.3f} | "
            f"{c['delta_cap']:+.3f} | [{ci[0]:+.3f}, {ci[1]:+.3f}] | "
            f"{c['delta_cap_p']:.3f} | {holm_str} | "
            f"{c['delta_p50']:+.2f} s | {c['delta_p95']:+.2f} s |"
        )
    out_lines.append(f"\n## Non-tool-use aggregate (n_cells = {len(g6_clean)})\n")
    if g6_clean:
        deltas_cap = np.array([c["delta_cap"] for c in g6_clean])
        deltas_p50 = np.array([c["delta_p50"] for c in g6_clean])
        out_lines.append(
            f"- mean Δcap: {float(deltas_cap.mean()):+.4f} "
            f"(range [{float(deltas_cap.min()):+.3f}, {float(deltas_cap.max()):+.3f}])"
        )
        out_lines.append(
            f"- mean Δp50: {float(deltas_p50.mean()):+.2f} s "
            f"(range [{float(deltas_p50.min()):+.2f}, {float(deltas_p50.max()):+.2f}] s)"
        )
        n_holm_reject = sum(1 for c in g6_clean if c["holm_reject"])
        out_lines.append(f"- Holm rejections at α={args.alpha}: {n_holm_reject} / {len(g6_clean)}")

    md = "\n".join(out_lines) + "\n"
    if args.out == "-":
        print(md)
    else:
        Path(args.out).write_text(md, encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv[1:]))
