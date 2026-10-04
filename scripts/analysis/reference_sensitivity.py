"""Reference-value sensitivity of the 3-dim systems reliability score.

The latency, throughput, and energy scores are normalized against reference values
(60 s p99 latency, 0.05 tokens/s/min throughput slope, 2000 J per correct answer).
This rescales those reference values and reports, per deployment scenario:
  - how many of the 20 math/reasoning cells change rank position against the
    unscaled ranking, and Kendall's tau between the two rankings;
  - the Pearson r between capability and the rescaled score, with a bootstrap
    95% CI (B=10,000, seed 42), as in h3_systems_composite.py.
Rescalings: all three reference values together at x0.5, x1.5, x2, and each
reference value alone at x0.5 and x2. Post-hoc over committed standard runs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

from legit_edge.normalize import (
    J_PER_CORRECT_SLA, P99_LATENCY_SLA_S, THROUGHPUT_SLOPE_REF,
    energy_score, latency_score, throughput_score,
)
from legit_edge.scoring import reliability_composite

from h3_systems_composite import MODELS, SCENARIOS, WORKLOADS, pearson_ci

BASE = {"lat": P99_LATENCY_SLA_S, "thr": THROUGHPUT_SLOPE_REF, "ener": J_PER_CORRECT_SLA}
RESCALINGS = [
    ("all x0.5", {"lat": 0.5, "thr": 0.5, "ener": 0.5}),
    ("all x1.5", {"lat": 1.5, "thr": 1.5, "ener": 1.5}),
    ("all x2", {"lat": 2.0, "thr": 2.0, "ener": 2.0}),
    ("latency x0.5", {"lat": 0.5}), ("latency x2", {"lat": 2.0}),
    ("throughput x0.5", {"thr": 0.5}), ("throughput x2", {"thr": 2.0}),
    ("energy x0.5", {"ener": 0.5}), ("energy x2", {"ener": 2.0}),
]


def load_raw(spark_std: str, jetson_std: str) -> list[dict]:
    rows = []
    for tier, d in [("spark", Path(spark_std)), ("jetson", Path(jetson_std))]:
        for model in MODELS:
            for wl in WORKLOADS:
                p = d / f"{model}__{tier}__{wl}.json"
                if not p.exists():
                    continue
                s = json.loads(p.read_text(encoding="utf-8"))
                m = s["metrics"]
                rows.append({
                    "cell": p.stem, "capability": float(s["capability"]),
                    "p99": m["latency"]["p99"],
                    "slope": m["throughput_stability"]["slope_tps_per_min"],
                    "jpc": m["energy_per_correct"]["j_per_correct"],
                })
    return rows


def scores(rows: list[dict], scale: dict, weights: dict) -> np.ndarray:
    ref = {k: BASE[k] * scale.get(k, 1.0) for k in BASE}
    out = []
    for r in rows:
        out.append(reliability_composite(
            latency_score=latency_score(r["p99"], ref_s=ref["lat"]),
            throughput_score=throughput_score(r["slope"], ref=ref["thr"]),
            energy_score=energy_score(r["jpc"], ref_j=ref["ener"]),
            output_stability=1.0, calibration_r=1.0, weights=weights,
        ))
    return np.array(out)


def rank_vector(v: np.ndarray) -> np.ndarray:
    order = np.argsort(-v, kind="stable")
    ranks = np.empty(len(v), dtype=int)
    ranks[order] = np.arange(len(v)) + 1
    return ranks


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spark-std", required=True)
    ap.add_argument("--jetson-std", required=True)
    ap.add_argument("--out", default="-")
    args = ap.parse_args(argv)
    rows = load_raw(args.spark_std, args.jetson_std)
    cap = np.array([r["capability"] for r in rows])

    L = ["# Reference-value sensitivity of the 3-dim systems reliability score\n",
         f"{len(rows)} cross-tier math and reasoning cells, standard runs. "
         "Rank changes count cells whose rank position differs from the unscaled ranking. "
         "Pearson r against capability with bootstrap CI95 (B=10,000, seed 42).\n"]
    for name, w in SCENARIOS.items():
        base = scores(rows, {}, w)
        r0, lo0, hi0 = pearson_ci(cap, base)
        L += [f"## {name}", f"Unscaled: r = {r0:+.3f} [{lo0:+.3f}, {hi0:+.3f}]\n",
              "| rescaling | rank changes /20 | Kendall tau vs unscaled | r | CI95 |",
              "|---|--:|--:|--:|:--|"]
        for label, sc in RESCALINGS:
            v = scores(rows, sc, w)
            flips = int(np.sum(rank_vector(base) != rank_vector(v)))
            tau = kendalltau(base, v).statistic
            r, lo, hi = pearson_ci(cap, v)
            L.append(f"| {label} | {flips} | {tau:.3f} | {r:+.3f} | [{lo:+.3f}, {hi:+.3f}] |")
        L.append("")
    md = "\n".join(L) + "\n"
    if args.out != "-":
        Path(args.out).write_text(md, encoding="utf-8")
        print(f"wrote {args.out}")
    print(md)
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv[1:]))
