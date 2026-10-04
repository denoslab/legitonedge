"""Empirical reliability function R(τ) from per-prompt latency traces.

For each (tier, model, workload) cell we derive:
  R(τ) = Pr(latency ≤ τ) = fraction of prompts completing within τ seconds

This is the empirical reliability function in the classical sense: the probability
that a system completes its required function (answer a prompt) without failure
(latency budget violation) within a given time budget τ.  Treating latency > τ_lat
as a failure converts our per-prompt measurements into a classical reliability
characterization.

We report:
  - R(τ) at the pre-registered SLA τ_lat = 60 s (the latency normalization bound)
  - R(τ) at operationally motivated tighter budgets: 10, 30, 60 s
  - Bootstrap 95% CI for each R(τ) estimate (B=10,000, seed 42)
  - The latency p50/p99 for context

Post-hoc over standard-MAXN traces; no new hardware runs.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

# 5-model study (3 general-purpose + 2 agentic-tuned). Cells with no `.traces.jsonl` on
# disk are skipped in the loop below, so this works on partial 3-model data
# as well as the full 5-model grid.
MODELS   = ["llama_3_1_8b", "qwen_2_5_7b", "phi_3_5_mini", "hermes3_8b", "llama3_groq_tooluse_8b"]
WORKLOADS = ["math", "reasoning"]
TAU_VALUES = [10.0, 30.0, 60.0]       # seconds; 60 s = pre-registered SLA
B = 10_000
SEED = 42


def load_latencies(traces: Path) -> np.ndarray:
    lats = []
    for line in traces.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        lat = r.get("latency_s")
        if lat is not None:
            lats.append(float(lat))
    return np.array(lats, dtype=float)


def bootstrap_reliability(lats: np.ndarray, tau: float, b: int = B, seed: int = SEED):
    rng = np.random.default_rng(seed)
    n = len(lats)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(np.mean(lats <= tau))
    boots = np.empty(b)
    for k in range(b):
        sample = rng.choice(lats, size=n, replace=True)
        boots[k] = np.mean(sample <= tau)
    lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    return point, lo, hi


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--jetson", required=True, help="jetson standard-MAXN run dir")
    ap.add_argument("--spark",  required=True, help="spark standard run dir")
    ap.add_argument("--out", default="-")
    args = ap.parse_args(argv)

    tau_cols = "  ".join(f"R({int(t)}s) [95% CI]" for t in TAU_VALUES)
    L = [
        "# Empirical reliability function R(τ) from per-prompt latency\n",
        "_R(τ) = Pr(latency ≤ τ) — fraction of prompts completing within τ s. "
        "Latency > τ is treated as a failure event; R(τ) is the per-cell empirical "
        "reliability at each time budget. Pre-registered SLA τ_lat = 60 s (the latency normalization bound). "
        "Bootstrap 95% CI (B=10,000, seed 42, i.i.d. over prompts)._\n",
        f"| cell | tier | n | p50 (s) | p99 (s) | "
        + " | ".join(f"R({int(t)}s)" for t in TAU_VALUES)
        + " | "
        + " | ".join(f"CI95 R({int(t)}s)" for t in TAU_VALUES)
        + " |",
        "|" + "---|" * (5 + 2*len(TAU_VALUES)),
    ]

    all_rows = []
    for tier, run_dir in [("jetson", args.jetson), ("spark", args.spark)]:
        for model in MODELS:
            for wl in WORKLOADS:
                cell = f"{model}__{tier}__{wl}"
                tpath = Path(run_dir) / f"{cell}.traces.jsonl"
                if not tpath.exists():
                    continue
                lats = load_latencies(tpath)
                if len(lats) == 0:
                    continue
                p50 = float(np.percentile(lats, 50))
                p99 = float(np.percentile(lats, 99))
                rt_vals = []
                ci_vals = []
                for tau in TAU_VALUES:
                    pt, lo, hi = bootstrap_reliability(lats, tau)
                    rt_vals.append(f"{pt:.4f}")
                    ci_vals.append(f"[{lo:.4f}, {hi:.4f}]")
                row = [cell, tier, str(len(lats)), f"{p50:.2f}", f"{p99:.2f}"] + rt_vals + ci_vals
                L.append("| " + " | ".join(row) + " |")
                all_rows.append({
                    "cell": cell, "tier": tier, "n": len(lats),
                    "p50": p50, "p99": p99,
                    "R60": float(np.mean(lats <= 60.0)),
                })

    # Summary: cross-tier mean R(60s) per tier
    for tier in ["jetson", "spark"]:
        r60_vals = [r["R60"] for r in all_rows if r["tier"] == tier]
        if r60_vals:
            L.append("")
            L.append(f"**{tier.title()} mean R(60s) = {np.mean(r60_vals):.4f}** "
                     f"(range {min(r60_vals):.4f}–{max(r60_vals):.4f}, n={len(r60_vals)} cells)")

    L += [
        "",
        "_Note: R(60s) = Pr(latency ≤ τ_lat) is the complement of the tail-exceedance "
        "probability that the latency score penalises. A cell with R(60s) near 1 meets the SLA on nearly "
        "all prompts; one near 0 fails it on most. Cross-tier differences in R(τ) "
        "directly quantify the substrate's effect on task-completion reliability._"
    ]

    md = "\n".join(L) + "\n"
    if args.out != "-":
        Path(args.out).write_text(md, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(md)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
