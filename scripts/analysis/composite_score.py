"""Per-cell LegitOnEdge composite + Reliability Cards + capability-reliability scatter.

Reads standard-run summary JSONs (+ optional paired throttled dir for the dim-3
drift d, + optional confidence in traces for dim-5 r). Emits:
  - per-cell reliability_composite (Y) and capability (X)
  - per-cell legit_edge_score (the Reliability Card headline number)
  - Pearson r(capability, reliability_composite) + bootstrap CI
  - per-profile re-ranking table
Usage:
  python composite_score.py <maxn_std_dir> [--throttled <dir>] [--profile balanced]
                            [--out report.md]

NOTE: output stability (dim 3) and calibration (dim 5) are held NEUTRAL here
(d=0 -> output_stability=1.0; calibration_r=1.0), so the reliability_composite below
is effectively the three systems dimensions (latency/throughput/energy). Calibration
and output stability are reported as separate axes by their own scripts
(reshoot_calibration.py, output_stability_paired.py). --throttled is reserved and
currently unused.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from legit_edge.normalize import latency_score, energy_score, throughput_score
from legit_edge.scoring import reliability_composite, legit_edge_score
from legit_edge.profiles import load_profiles


def _load(p: Path) -> dict:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def cell_composite(summary_path: Path, *, d_drift: float = 0.0, calib_r: float = 1.0,
                   weights: dict | None = None) -> dict:
    c = _load(summary_path)
    m = c["metrics"]
    ls = latency_score(m["latency"]["p99"])
    ts = throughput_score(m["throughput_stability"]["slope_tps_per_min"])
    es = energy_score(m["energy_per_correct"]["j_per_correct"])
    out_stab = max(0.0, 1.0 - abs(d_drift))
    rc = reliability_composite(latency_score=ls, throughput_score=ts, energy_score=es,
                               output_stability=out_stab, calibration_r=calib_r, weights=weights)
    card = legit_edge_score(capability=c["capability"], latency_score=ls, throughput_score=ts,
                            energy_score=es, output_stability=out_stab,
                            calibration_r=max(0.85, calib_r))
    return {"cell": summary_path.name.replace(".json", ""), "capability": c["capability"],
            "reliability_composite": rc, "legit_edge_score": card,
            "latency_score": ls, "throughput_score": ts, "energy_score": es,
            "output_stability": out_stab, "calibration_r": calib_r}


def pearson_with_ci(x: np.ndarray, y: np.ndarray, b: int = 10_000, seed: int = 42):
    def _r(a, c):
        if np.std(a) == 0 or np.std(c) == 0:
            return 0.0
        return float(np.corrcoef(a, c)[0, 1])
    r = _r(x, y)
    rng = np.random.default_rng(seed)
    n = len(x)
    boots = np.empty(b)
    for i in range(b):
        idx = rng.integers(0, n, size=n)
        boots[i] = _r(x[idx], y[idx])
    return r, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("maxn_dir")
    ap.add_argument("--throttled", default=None)
    ap.add_argument("--profile", default="balanced")
    ap.add_argument("--out", default="-")
    args = ap.parse_args(argv)
    profs = load_profiles()
    w = profs[args.profile]
    maxn = Path(args.maxn_dir)
    rows = []
    for sp in sorted(maxn.glob("*.json")):
        if sp.name.endswith(".traces.jsonl"):
            continue
        rows.append(cell_composite(sp, d_drift=0.0, calib_r=1.0, weights=w))
    scatter = [r for r in rows if "tooluse" not in r["cell"]]
    x = np.array([r["capability"] for r in scatter])
    y = np.array([r["reliability_composite"] for r in scatter])
    r, lo, hi = pearson_with_ci(x, y)
    L = [f"# Composite + capability-reliability scatter (profile={args.profile})\n",
         f"Pearson r(capability, reliability_composite) = {r:+.3f}  CI95 [{lo:+.3f}, {hi:+.3f}]  "
         f"(n={len(scatter)}; pre-registered threshold r<0.5; CI-upper {'<' if hi < 0.5 else '>='} 0.5)\n",
         "_Output stability (dim 3) and calibration (dim 5) held neutral; latency/throughput/energy from measured standard-run data._\n",
         "| cell | capability | reliability | LegitOnEdge card |", "|---|--:|--:|--:|"]
    for r0 in rows:
        L.append(f"| {r0['cell']} | {r0['capability']:.3f} | {r0['reliability_composite']:.3f} | {r0['legit_edge_score']:.1f} |")
    L.append("\n† tooluse cells are excluded from the scatter (BFCL live_simple is a degenerate axis — all models ~0); their Reliability Cards are shown for completeness only.")
    md = "\n".join(L) + "\n"
    if args.out == "-":
        print(md)
    else:
        Path(args.out).write_text(md, encoding="utf-8"); print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv[1:]))
