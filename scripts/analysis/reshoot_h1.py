"""H1 — paired MAXN-vs-throttled calibration (ECE) test.

Pre-registered **H1**: ECE measured at MAXN over-predicts the throttled regime's
calibration by >= 5 pp (two-sided, alpha = 0.10 after Holm-Bonferroni across the
math+reasoning cells).

Per cell: gap = ECE_MAXN - ECE_throttled (positive => MAXN looks worse-calibrated,
i.e. a larger MAXN ECE). MAXN and throttled are independent mega-quick runs (same
seeded prompts, different regime), so we bootstrap each regime's pooled
(confidence, correct) pairs INDEPENDENTLY and recompute ECE (B = 10,000). A
two-sided bootstrap p-value per cell, Holm-Bonferroni across the 6 cells, plus an
aggregate mean-gap CI.

Expectation (the thermal-stress pilot did not reach throttling): the Orin Nano cannot be thermally throttled by
inference (peaks ~70 C << ~83 C throttle point) and decoding is deterministic, so
MAXN ~ naturally-throttled at this workload => gap < 5 pp and H1 most likely NOT
supported. That is a valid, reportable result consistent with the thermal finding.

Inputs: the MAXN and throttled mega-quick confidence-elicitation run dirs; pairs come from
reshoot_calibration.cell_pairs (confidence-stripped re-score). Post-hoc, 0 compute.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from legit_edge.metrics import CalibrationMetric
from reshoot_calibration import cell_pairs, _cell_files

# 5-model study (3 general-purpose + 2 agentic-tuned). Cells lacking trace files in EITHER
# regime are skipped (e.g. a model not yet run in that regime), so Holm-Bonferroni
# only ever corrects across cells that were actually measured.
MODELS = ["llama_3_1_8b", "qwen_2_5_7b", "phi_3_5_mini", "hermes3_8b", "llama3_groq_tooluse_8b"]
WORKLOADS = ["math", "reasoning"]   # tooluse excluded (degenerate BFCL axis)
THRESHOLD = 0.05                    # 5 pp pre-registered effect size
ALPHA = 0.10


def ece(pairs) -> float:
    """Expected Calibration Error (10-bin) over (confidence, correct) pairs."""
    return float(CalibrationMetric().compute(list(pairs)).value["ece"])


def _resample(rng, pairs):
    idx = rng.integers(0, len(pairs), size=len(pairs))
    return [pairs[i] for i in idx]


def gap_bootstrap(maxn_pairs, throt_pairs, B: int = 10_000, seed: int = 42) -> dict:
    """Point gap = ECE(MAXN) - ECE(throttled) with an independent bootstrap of each pool."""
    g0 = ece(maxn_pairs) - ece(throt_pairs)
    rng = np.random.default_rng(seed)
    boot = np.empty(B)
    for i in range(B):
        boot[i] = ece(_resample(rng, maxn_pairs)) - ece(_resample(rng, throt_pairs))
    lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    p = 2.0 * min(float((boot <= 0).mean()), float((boot >= 0).mean()))
    return {"gap": float(g0), "ci_lo": lo, "ci_hi": hi, "p_two_sided": float(min(1.0, p))}


def holm_bonferroni(pvals, alpha: float = ALPHA):
    """Holm step-down. Returns (reject_flags, adjusted_pvals), both in input order."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    reject = [False] * m
    adj = [0.0] * m
    prev = 0.0
    still = True
    for rank, idx in enumerate(order):
        crit = alpha / (m - rank)
        ap = max(min(1.0, (m - rank) * pvals[idx]), prev)   # enforce monotonicity
        adj[idx] = ap
        prev = ap
        if still and pvals[idx] <= crit:
            reject[idx] = True
        else:
            still = False
    return reject, adj


def cell_gap(maxn_dir, throt_dir, cell_base, B: int = 10_000, seed: int = 42) -> dict:
    mp = cell_pairs(Path(maxn_dir), cell_base)
    tp = cell_pairs(Path(throt_dir), cell_base)
    res = gap_bootstrap(mp, tp, B=B, seed=seed)
    res.update({"cell": cell_base, "ece_maxn": ece(mp), "ece_throt": ece(tp),
                "n_maxn": len(mp), "n_throt": len(tp)})
    return res


def mean_gap_bootstrap(cell_pair_lists, B: int = 10_000, seed: int = 42) -> dict:
    """Aggregate: mean over cells of (ECE_MAXN - ECE_throttled), paired bootstrap CI."""
    g0 = float(np.mean([ece(m) - ece(t) for m, t in cell_pair_lists]))
    rng = np.random.default_rng(seed)
    boot = np.empty(B)
    for i in range(B):
        boot[i] = float(np.mean([ece(_resample(rng, m)) - ece(_resample(rng, t))
                                 for m, t in cell_pair_lists]))
    return {"mean_gap": g0, "ci_lo": float(np.percentile(boot, 2.5)),
            "ci_hi": float(np.percentile(boot, 97.5))}


def analyze(maxn_dir, throt_dir, B: int = 10_000, seed: int = 42):
    candidate = [f"{m}__jetson__{w}" for m in MODELS for w in WORKLOADS]
    # Only retain cells with trace files present in BOTH regimes (skip-missing): a
    # paired gap is undefined otherwise, and absent models must not inject nan p-values.
    cells = [
        c for c in candidate
        if _cell_files(Path(maxn_dir), c) and _cell_files(Path(throt_dir), c)
    ]
    rows = [cell_gap(maxn_dir, throt_dir, c, B=B, seed=seed) for c in cells]
    reject, adj = holm_bonferroni([r["p_two_sided"] for r in rows], ALPHA)
    for r, rj, ap in zip(rows, reject, adj):
        r["reject_h0"] = rj
        r["p_holm"] = ap
        # H1 is directional ("MAXN over-predicts"): a significant gap of >= +5 pp.
        r["supports_h1"] = bool(rj and r["gap"] >= THRESHOLD)
    pair_lists = [(cell_pairs(Path(maxn_dir), c), cell_pairs(Path(throt_dir), c)) for c in cells]
    agg = mean_gap_bootstrap(pair_lists, B=B, seed=seed)
    return rows, agg


def _fmt(rows, agg, B: int) -> str:
    any_h1 = any(r["supports_h1"] for r in rows)
    L = [
        "# H1 — paired MAXN-vs-throttled calibration (ECE)\n",
        f"Pre-registered H1: MAXN ECE over-predicts throttled calibration by "
        f">= {THRESHOLD * 100:.0f} pp (two-sided, alpha = {ALPHA} after Holm-Bonferroni). "
        f"gap = ECE_MAXN - ECE_throttled; independent bootstrap per regime (B = {B:,}). "
        "6 math+reasoning cells (tooluse excluded).\n",
        f"**Mean gap = {agg['mean_gap']:+.3f}**  CI95 [{agg['ci_lo']:+.3f}, {agg['ci_hi']:+.3f}]  "
        f"(threshold {THRESHOLD:+.2f}).",
        f"**Verdict: H1 {'SUPPORTED' if any_h1 else 'NOT supported'}** "
        f"({'>= 1' if any_h1 else 'no'} cell with a Holm-significant gap >= {THRESHOLD * 100:.0f} pp).\n",
        "| cell | n MAXN/throt | ECE_MAXN | ECE_throt | gap | CI95 | p | p_Holm | H1? |",
        "|---|--:|--:|--:|--:|---|--:|--:|:--:|",
    ]
    for r in rows:
        L.append(
            f"| {r['cell']} | {r['n_maxn']}/{r['n_throt']} | {r['ece_maxn']:.3f} | "
            f"{r['ece_throt']:.3f} | {r['gap']:+.3f} | "
            f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] | {r['p_two_sided']:.3f} | "
            f"{r['p_holm']:.3f} | {'yes' if r['supports_h1'] else 'no'} |"
        )
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--maxn-dir", required=True)
    ap.add_argument("--throttled-dir", required=True)
    ap.add_argument("--b", type=int, default=10_000)
    ap.add_argument("--out", default="-")
    args = ap.parse_args(argv)
    rows, agg = analyze(args.maxn_dir, args.throttled_dir, B=args.b)
    md = _fmt(rows, agg, args.b)
    if args.out != "-":
        Path(args.out).write_text(md, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(md)
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv[1:]))
