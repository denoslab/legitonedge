"""H2 bootstrap CIs -- 95% CI on the median |mq-std|/std per dimension per cut.

Adds confidence intervals to the H2 extrapolation-error medians in the frozen
table results/h2-crosstier-v2-20260530-235036.md (output of
extrapolation_error_crosstier.py on our runs; it gives per-cell point ratios and
medians but no CIs).

Statistic: the median over cells of the per-cell |mq-std|/std. CI: a percentile
bootstrap that resamples cells with replacement (B=10,000, seed=42), matching the
framework's bootstrap convention (PRE_REGISTRATION.md). This captures the
cell-to-cell variability of the median, which is the dominant uncertainty for a
"median over cells" statistic.

Tool-use cells are dropped (single-call tool-use capability is near zero for every
model, so it carries no signal), per the clean cuts in the source table. Per-cell |Δ|/std values are taken verbatim from that frozen table
(capability, latency p50, latency p99 -- the 3 scalars wired identically in both
modes).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

# (model, tier, workload, |Dcap|/std, |Dp50|/std, |Dp99|/std)
# Source: h2-crosstier-v2-20260530-235036.md  (math+reasoning rows; tool-use dropped)
CELLS = [
    ("llama", "jetson", "math",      0.183, 0.110, 0.812),
    ("llama", "jetson", "reasoning", 0.075, 0.001, 0.006),
    ("llama", "spark",  "math",      0.147, 0.101, 0.262),
    ("llama", "spark",  "reasoning", 0.201, 0.075, 0.008),
    ("phi",   "jetson", "math",      0.233, 0.035, 0.811),
    ("phi",   "jetson", "reasoning", 0.232, 0.105, 0.003),
    ("phi",   "spark",  "math",      0.038, 0.080, 0.294),
    ("phi",   "spark",  "reasoning", 0.155, 0.052, 0.007),
    ("qwen",  "jetson", "math",      0.238, 0.048, 0.488),
    ("qwen",  "jetson", "reasoning", 0.063, 0.006, 0.002),
    ("qwen",  "spark",  "math",      0.189, 0.009, 0.004),
    ("qwen",  "spark",  "reasoning", 0.004, 0.059, 0.004),
]
DIM_IDX = {"cap": 3, "p50": 4, "p99": 5}
DIMS = ["cap", "p50", "p99"]
BAND = 0.15
B = 10_000
SEED = 42


def boot_median_ci(vals, rng, b=B):
    vals = np.asarray(vals, dtype=float)
    n = len(vals)
    med = float(np.median(vals))
    boots = np.empty(b)
    for i in range(b):
        boots[i] = np.median(vals[rng.integers(0, n, n)])
    return med, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main() -> int:
    rng = np.random.default_rng(SEED)
    cuts = {
        "cross-tier": CELLS,
        "jetson": [c for c in CELLS if c[1] == "jetson"],
        "spark": [c for c in CELLS if c[1] == "spark"],
    }
    L = [
        "# H2 extrapolation error -- bootstrap 95% CIs on the median |mq-std|/std",
        "",
        f"Percentile bootstrap over cells (B={B:,}, seed={SEED}); tool-use dropped.",
        f"Pre-registered band: median |mq-std|/std <= {BAND}.",
        "Per-cell source: h2-crosstier-v2-20260530-235036.md (math+reasoning).",
        "",
    ]
    for name, cells in cuts.items():
        L += [f"## {name} (n={len(cells)})", "",
              "| dim | median | CI95 | within band? |", "|---|--:|---|:--:|"]
        worst = None
        for d in DIMS:
            vals = [c[DIM_IDX[d]] for c in cells]
            med, lo, hi = boot_median_ci(vals, rng)
            L.append(f"| {d} | {med:.3f} | [{lo:.3f}, {hi:.3f}] | {'yes' if med <= BAND else 'no'} |")
            if worst is None or med > worst[1]:
                worst = (d, med, lo, hi)
        L += ["", f"worst-of-three: {worst[0]} = {worst[1]:.3f} CI95 "
              f"[{worst[2]:.3f}, {worst[3]:.3f}] -> "
              f"{'WITHIN' if worst[1] <= BAND else 'EXCEEDS'} the {BAND} band.", ""]
    out = "\n".join(L) + "\n"
    outpath = Path(__file__).resolve().parent.parent.parent / "results" / "h2-bootstrap-ci-20260602.md"
    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text(out, encoding="utf-8")
    print(out)
    print(f"wrote {outpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
