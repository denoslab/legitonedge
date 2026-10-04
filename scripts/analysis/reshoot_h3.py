"""Cross-tier H3 — capability vs the 5-dim LegitOnEdge reliability composite.

This 5-dim variant (systems dims + output stability + calibration) is the
"+calibration" contrast. The paper's headline composite is the 3-dim systems
composite computed by h3_systems_composite.py.

Joins, per (tier, model, workload) cell [tooluse excluded — single-call tool-use
capability is near zero for every model]:
  X (scatter)  = capability                 <- standard MAXN run (clean, non-confidence)
  reliability dims (Y inputs):
    latency / throughput / energy           <- standard MAXN summary
    d (output-stability drift)              <- standard MAXN-vs-throttled paired |Δcap|
                                               (Jetson only; Spark has no throttled regime -> d=0)
    r = 1 - ECE (calibration)               <- five-repeat mega-quick verbalized-confidence
                                               (confidence-stripped re-score)
  Y            = reliability_composite(...)  (weighted geomean; capability EXCLUDED)
Then Pearson r(X, Y) over the cross-tier cells + bootstrap CI; profile-conditioned.

Disclosed scale-mix: latency/throughput/energy/d at standard scale (n~200);
calibration r at mega-quick scale (n~55-85 pooled over k=5). Spark d=0 (the DGX is
active-cooled / regime-invariant).
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from legit_edge.normalize import latency_score, energy_score, throughput_score
from legit_edge.scoring import reliability_composite, legit_edge_score
from legit_edge.profiles import load_profiles
from reshoot_calibration import cell_calibration
from output_stability_paired import paired_delta_cell

# 5-model study (3 general-purpose + 2 agentic-tuned). build_cells skips any cell whose
# standard-mode summary JSON is absent on disk (e.g. a model
# not yet run), so this runs on a partial 3-model grid as well as the full 5-model one.
MODELS = ["llama_3_1_8b", "qwen_2_5_7b", "phi_3_5_mini", "hermes3_8b", "llama3_groq_tooluse_8b"]
WORKLOADS = ["math", "reasoning"]   # tooluse excluded (degenerate BFCL axis)


def _load(p: Path) -> dict:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def _calib_dirs(spec: str) -> list[Path]:
    """Accept one calib dir OR a comma-separated list (the 5-model k=5 calibration is
    split across two mega-quick dirs per tier: one for the 3 general-purpose models and one
    for the 2 agentic-tuned models). We try each dir in
    order and use the first that yields confidence pairs for the cell."""
    return [Path(s) for s in str(spec).split(",") if s.strip()]


def _cell_calibration_multi(calib_dirs: list[Path], cell: str) -> dict:
    """cell_calibration over a list of dirs; first dir with non-empty pairs (n>0) wins."""
    last = None
    for d in calib_dirs:
        cal = cell_calibration(d, cell)
        last = cal
        if cal["n"] > 0:
            return cal
    return last if last is not None else {"cell": cell, "n": 0, "ece": float("nan")}


def build_cells(spark_std, jetson_std_maxn, jetson_std_throt,
                spark_calib, jetson_calib, weights) -> list[dict]:
    specs = [
        ("spark", Path(spark_std), None, _calib_dirs(spark_calib)),
        ("jetson", Path(jetson_std_maxn), Path(jetson_std_throt), _calib_dirs(jetson_calib)),
    ]
    rows = []
    for tier, std_dir, throt_dir, calib_dir in specs:
        for model in MODELS:
            for wl in WORKLOADS:
                cell = f"{model}__{tier}__{wl}"
                summ_path = std_dir / f"{cell}.json"
                if not summ_path.exists():
                    continue   # skip-missing: model with no committed data yet
                summ = _load(summ_path)
                m = summ["metrics"]
                ls = latency_score(m["latency"]["p99"])
                ts = throughput_score(m["throughput_stability"]["slope_tps_per_min"])
                es = energy_score(m["energy_per_correct"]["j_per_correct"])
                if throt_dir is not None:
                    d = abs(paired_delta_cell(
                        std_dir / f"{cell}.traces.jsonl",
                        throt_dir / f"{cell}.traces.jsonl",
                    )["delta_cap"])
                else:
                    d = 0.0
                out_stab = max(0.0, 1.0 - d)
                cal = _cell_calibration_multi(calib_dir, cell)
                ece = cal["ece"]
                r = (1.0 - ece) if ece == ece else 1.0   # nan -> neutral
                rc = reliability_composite(
                    latency_score=ls, throughput_score=ts, energy_score=es,
                    output_stability=out_stab, calibration_r=r, weights=weights,
                )
                card = legit_edge_score(
                    capability=summ["capability"], latency_score=ls, throughput_score=ts,
                    energy_score=es, output_stability=out_stab, calibration_r=max(0.85, r),
                )
                rows.append({
                    "cell": cell, "tier": tier, "capability": summ["capability"],
                    "reliability": rc, "card": card, "ECE": ece, "d": d,
                    "ls": ls, "ts": ts, "es": es, "r": r,
                })
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
    ap.add_argument("--jetson-std-maxn", required=True)
    ap.add_argument("--jetson-std-throttled", required=True)
    ap.add_argument("--spark-calib", required=True)
    ap.add_argument("--jetson-calib", required=True)
    ap.add_argument("--profile", default="balanced")
    ap.add_argument("--out", default="-")
    args = ap.parse_args(argv)
    w = load_profiles()[args.profile]
    rows = build_cells(args.spark_std, args.jetson_std_maxn, args.jetson_std_throttled,
                       args.spark_calib, args.jetson_calib, w)
    x = np.array([r["capability"] for r in rows])
    y = np.array([r["reliability"] for r in rows])
    r, lo, hi = pearson_ci(x, y)
    verdict = "WITHIN (orthogonal)" if hi < 0.5 else "INCONCLUSIVE (CI-upper >= 0.5)"
    L = [
        f"# H3 — capability vs 5-dim reliability composite (profile={args.profile})\n",
        f"Cross-tier, **{len(rows)} cells** "
        f"({len(rows)//4 if len(rows) else 0} models x 2 workloads x 2 tiers; tooluse excluded).",
        f"**Pearson r = {r:+.3f}**  CI95 [{lo:+.3f}, {hi:+.3f}]  "
        f"(pre-registered threshold r < 0.5 -> {verdict}).",
        "_5-dim: latency/throughput/energy/d at standard scale; r=1-ECE at mega-quick "
        "(disclosed scale-mix). Spark d=0 (no throttled regime — active-cooled)._\n",
        "| cell | tier | capability | reliability | ECE | d | LegitOnEdge card |",
        "|---|---|--:|--:|--:|--:|--:|",
    ]
    for r0 in sorted(rows, key=lambda z: (z["tier"], z["cell"])):
        L.append(
            f"| {r0['cell']} | {r0['tier']} | {r0['capability']:.3f} | "
            f"{r0['reliability']:.3f} | {r0['ECE']:.3f} | {r0['d']:.3f} | {r0['card']:.1f} |"
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
