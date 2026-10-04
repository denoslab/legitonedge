"""H3 + the 3-dim SYSTEMS reliability composite {latency, throughput, energy}.

The reliability composite is the three cleanly
measured systems dimensions only. Calibration and output-stability are reported as
SEPARATE axes -- calibration is an accuracy-coupled trustworthiness reading (1-ECE
approximates accuracy under uniform overconfidence), output-stability was not activated
under this workload (the device never throttled; the Spark has one operating point) --
so neither enters the composite. This recomputes, per scenario, the per-cell 3-dim
composite and the capability-composite Pearson r (expected near zero, i.e. the systems
composite is statistically independent of capability). Standard-run summaries only; no
calibration or throttled inputs are needed because the composite skips zero-weight dims.
Post-hoc over committed standard traces; 0 new compute.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from legit_edge.normalize import latency_score, energy_score, throughput_score
from legit_edge.scoring import reliability_composite

MODELS = ["llama_3_1_8b", "qwen_2_5_7b", "phi_3_5_mini", "hermes3_8b", "llama3_groq_tooluse_8b"]
WORKLOADS = ["math", "reasoning"]   # tool-use excluded (degenerate single-call axis)

# 3-dim deployment scenarios (weights over the systems composite only).
SCENARIOS = {
    "balanced":           {"latency": 1 / 3, "throughput": 1 / 3, "energy": 1 / 3},
    "latency_critical":   {"latency": 0.6, "throughput": 0.2, "energy": 0.2},
    "energy_constrained": {"latency": 0.2, "throughput": 0.2, "energy": 0.6},
}


def build_cells(spark_std: str, jetson_std: str) -> list[dict]:
    rows = []
    for tier, d in [("spark", Path(spark_std)), ("jetson", Path(jetson_std))]:
        for model in MODELS:
            for wl in WORKLOADS:
                cell = f"{model}__{tier}__{wl}"
                p = d / f"{cell}.json"
                if not p.exists():
                    continue
                s = json.loads(p.read_text(encoding="utf-8"))
                m = s["metrics"]
                ls = latency_score(m["latency"]["p99"])
                ts = throughput_score(m["throughput_stability"]["slope_tps_per_min"])
                es = energy_score(m["energy_per_correct"]["j_per_correct"])
                row = {"cell": cell, "tier": tier, "capability": float(s["capability"]),
                       "ls": ls, "ts": ts, "es": es}
                for name, w in SCENARIOS.items():
                    # output_stability / calibration carry weight 0 in every scenario,
                    # so reliability_composite skips them -> a pure 3-dim composite.
                    row[name] = reliability_composite(
                        latency_score=ls, throughput_score=ts, energy_score=es,
                        output_stability=1.0, calibration_r=1.0, weights=w,
                    )
                rows.append(row)
    return rows


def pearson_ci(x: np.ndarray, y: np.ndarray, b: int = 10_000, seed: int = 42):
    def _r(a, c):
        if np.std(a) == 0 or np.std(c) == 0:
            return 0.0
        return float(np.corrcoef(a, c)[0, 1])
    r = _r(x, y)
    rng = np.random.default_rng(seed)
    n = len(x)
    bs = np.empty(b)
    for i in range(b):
        idx = rng.integers(0, n, size=n)
        bs[i] = _r(x[idx], y[idx])
    return r, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spark-std", required=True)
    ap.add_argument("--jetson-std", required=True)
    ap.add_argument("--out", default="-")
    args = ap.parse_args(argv)
    rows = build_cells(args.spark_std, args.jetson_std)

    L = [
        "# H3 + 3-dim systems reliability composite {latency, throughput, energy}\n",
        "_Calibration and output-stability are reported as separate axes and EXCLUDED from "
        "the composite. Composite = scenario-weighted geometric mean of the three systems "
        "dimension scores. Standard MAXN summaries, cross-tier._\n",
        f"Cross-tier, **{len(rows)} cells** ({len(rows) // 4 if rows else 0} models x 2 "
        "workloads x 2 tiers).\n",
        "## Capability vs systems composite (Pearson r, B=10,000 bootstrap)",
    ]
    for name in SCENARIOS:
        x = np.array([r["capability"] for r in rows])
        y = np.array([r[name] for r in rows])
        r, lo, hi = pearson_ci(x, y)
        verdict = "orthogonal (CI upper < 0.5)" if hi < 0.5 else "inconclusive"
        L.append(f"- **{name}**: r = {r:+.3f}  CI95 [{lo:+.3f}, {hi:+.3f}]  ({verdict})")
    L += [
        "",
        "| cell | tier | capability | s_lat | s_thr | s_ener | balanced | latency_critical | energy_constrained |",
        "|---|---|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for r in sorted(rows, key=lambda z: (z["tier"], z["cell"])):
        L.append(
            f"| {r['cell']} | {r['tier']} | {r['capability']:.3f} | {r['ls']:.3f} | "
            f"{r['ts']:.3f} | {r['es']:.3f} | {r['balanced']:.3f} | "
            f"{r['latency_critical']:.3f} | {r['energy_constrained']:.3f} |"
        )
    md = "\n".join(L) + "\n"
    if args.out != "-":
        Path(args.out).write_text(md, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(md)
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv[1:]))
