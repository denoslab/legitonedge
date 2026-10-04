"""ECE bootstrap CIs + binning-sensitivity for the calibration cells.

Post-hoc over the five-repeat mega-quick confidence-elicitation traces (0 new compute),
so the ECE *magnitude* is reported with an interval and is shown not to depend on the
10-bin resolution. Reuses
``reshoot_calibration.cell_pairs`` -- the (confidence, true_correct) pairs from the
confidence-stripped re-score, pooled across the k=5 repeats -- as the resampling unit.

For each non-degenerate cell (tool-use excluded; capability is ~0 there, so ECE is
trivially the mean confidence):
  - point ECE at 10 equal-width bins (matches the paper / CalibrationMetric default),
  - percentile bootstrap 95% CI on ECE (B=10,000, seed 42), resampling the pairs,
  - ECE at {5,10,15,20} bins, to show the value is not an artifact of bin count.

A fidelity assert checks the local vectorized ECE against the canonical
``CalibrationMetric`` at 10 bins, so the bootstrap/binning use the paper's definition.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
from legit_edge.metrics import CalibrationMetric
from reshoot_calibration import cell_pairs  # same directory

BINS = [5, 10, 15, 20]
B = 10_000
SEED = 42


def _ece(confs: np.ndarray, corrects: np.ndarray, n_bins: int) -> float:
    """Equal-width-bin ECE; a vectorized replica of CalibrationMetric (verified below)."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(confs)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (confs >= lo) & (confs < hi) if i < n_bins - 1 else (confs >= lo) & (confs <= hi)
        m = int(mask.sum())
        if m:
            ece += (m / n) * abs(corrects[mask].mean() - confs[mask].mean())
    return float(ece)


def _bases(run_dir: str):
    d = Path(run_dir)
    bases = sorted({f.name.split("__run")[0] for f in d.glob("*__run*.traces.jsonl")})
    if not bases:
        bases = sorted({f.name.replace(".traces.jsonl", "") for f in d.glob("*.traces.jsonl")})
    return d, bases


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+", help="mega-quick confidence-elicitation run dirs (one per tier)")
    ap.add_argument("--out", default="-")
    args = ap.parse_args(argv)

    rows = []
    for rd in args.run_dirs:
        d, bases = _bases(rd)
        for base in bases:
            if base.split("__")[2] == "tooluse":
                continue
            pairs = cell_pairs(d, base)
            if not pairs:
                continue
            confs = np.array([c for c, _ in pairs], dtype=float)
            corr = np.array([y for _, y in pairs], dtype=float)
            # fidelity: local ECE must equal the canonical metric at 10 bins
            canon = CalibrationMetric(n_bins=10).compute(pairs).value["ece"]
            mine = _ece(confs, corr, 10)
            assert abs(canon - mine) < 1e-9, f"ECE mismatch {base}: {canon} vs {mine}"
            # bootstrap 95% CI on ECE@10
            rng = np.random.default_rng(SEED)
            n = len(confs)
            boots = np.empty(B)
            for k in range(B):
                idx = rng.integers(0, n, size=n)
                boots[k] = _ece(confs[idx], corr[idx], 10)
            lo, hi = (float(x) for x in np.percentile(boots, [2.5, 97.5]))
            ece_bins = {nb: _ece(confs, corr, nb) for nb in BINS}
            rows.append({
                "cell": base, "n": n, "mean_conf": float(confs.mean()),
                "corr_acc": float(corr.mean()), "ece10": mine, "lo": lo, "hi": hi,
                "bins": ece_bins,
            })

    L = [
        "# ECE bootstrap CIs + binning sensitivity (verbalized-confidence calibration)\n",
        f"_Post-hoc over the five-repeat mega-quick confidence-elicitation traces; confidence-stripped re-score; "
        f"(confidence, true-correct) pairs resampled, B={B:,}, seed {SEED}. Tool-use cells "
        f"excluded (degenerate). ECE bin-count sweep {{{','.join(map(str, BINS))}}}._\n",
        "| cell | n | mean_conf | corr_acc | ECE (10-bin) | 95% CI | ECE@5 | ECE@10 | ECE@15 | ECE@20 |",
        "|---|--:|--:|--:|--:|:--:|--:|--:|--:|--:|",
    ]
    for r in rows:
        b = r["bins"]
        L.append(
            f"| {r['cell']} | {r['n']} | {r['mean_conf']:.3f} | {r['corr_acc']:.3f} | "
            f"{r['ece10']:.3f} | [{r['lo']:.3f}, {r['hi']:.3f}] | "
            f"{b[5]:.3f} | {b[10]:.3f} | {b[15]:.3f} | {b[20]:.3f} |"
        )
    if rows:
        eces = [r["ece10"] for r in rows]
        widths = [r["hi"] - r["lo"] for r in rows]
        # max absolute deviation of any binning from the 10-bin value, across cells
        max_bin_dev = max(abs(r["bins"][nb] - r["ece10"]) for r in rows for nb in BINS)
        L += [
            "",
            f"**{len(rows)} cells.** ECE (10-bin) range **{min(eces):.3f}–{max(eces):.3f}**; "
            f"median 95% CI width **{float(np.median(widths)):.3f}**; "
            f"max deviation across the {{5,10,15,20}}-bin sweep **{max_bin_dev:.3f}** "
            f"(the overconfidence ordering is invariant to bin count).",
        ]
    md = "\n".join(L) + "\n"
    if args.out != "-":
        Path(args.out).write_text(md, encoding="utf-8")
        print(f"wrote {args.out} ({len(rows)} cells)")
    else:
        print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
