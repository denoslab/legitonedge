"""H3 influence sensitivity: leave-groq-out Pearson r.

The 5-dim "+calibration" composite estimate (r = +0.649, n = 20, balanced profile;
reshoot_h3.py on the full 5-model grid) is partly anchored by
the degenerate llama3-groq-tooluse-8b cells (ECE = 1.00 floors calibration in
two math cells; the throughput-slope anchor floors the Spark reasoning cell).
This script quantifies that influence from the frozen per-cell table
(results/v3-h3-20260608.md, output of reshoot_h3.py on our runs; not included): Pearson r with percentile-bootstrap CI95
(B = 10,000, seed 42, i.i.d. over cells — mirroring reshoot_h3.py) for
  (a) all 20 cells (fidelity check vs the committed +0.649),
  (b) the 16 cells excluding the groq-tool-use model entirely,
  (c) the 17 cells excluding only the three floored cells (reliability < 0.1).

Output: results/v3-h3-sensitivity-groq-<date>.md
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

RES = Path(__file__).resolve().parents[2] / "results"
SRC = RES / "v3-h3-20260608.md"
OUT = RES / "v3-h3-sensitivity-groq-20260610.md"
B = 10_000
SEED = 42

rows = []  # (cell, capability, reliability)
for line in SRC.read_text(encoding="utf-8").splitlines():
    m = re.match(r"\| (\S+__(?:jetson|spark)__\S+) \| (?:jetson|spark) \| ([0-9.]+) \| ([0-9.]+) \|", line)
    if m:
        rows.append((m.group(1), float(m.group(2)), float(m.group(3))))
assert len(rows) == 20, f"expected 20 cells in {SRC.name}, got {len(rows)}"


def pearson(c: np.ndarray, r: np.ndarray) -> float:
    return float(np.corrcoef(c, r)[0, 1])


def boot_ci(c: np.ndarray, r: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    n = len(c)
    boots = []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        if np.std(c[idx]) > 0 and np.std(r[idx]) > 0:
            boots.append(pearson(c[idx], r[idx]))
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


subsets = {
    "all 20 cells (fidelity check)": rows,
    "excl. groq-tool-use model (16 cells)": [t for t in rows if "groq" not in t[0]],
    "excl. floored cells rel<0.1 (17 cells)": [t for t in rows if t[2] >= 0.1],
}

lines = [
    "# H3 sensitivity — influence of the degenerate groq-tool-use cells",
    "",
    "_Pearson r between capability and the 5-dim balanced reliability composite,",
    f"recomputed from the canonical committed table ({SRC.name}); percentile",
    f"bootstrap CI95, B={B:,}, seed {SEED}, i.i.d. over cells. The three floored",
    "cells are: groq math x2 (ECE=1.00 -> calibration=0) and groq Spark-reasoning",
    "(throughput-slope -0.132 tok/s/min vs the 0.05 anchor -> dimension=0; the",
    "geometric mean then collapses the composite)._",
    "",
    "| subset | n | Pearson r | CI95 |",
    "|---|--:|--:|:--|",
]
for name, sub in subsets.items():
    c = np.array([t[1] for t in sub])
    r = np.array([t[2] for t in sub])
    pr = pearson(c, r)
    lo, hi = boot_ci(c, r)
    lines.append(f"| {name} | {len(sub)} | {pr:+.3f} | [{lo:+.3f}, {hi:+.3f}] |")
    print(f"{name:42s} n={len(sub):2d}  r={pr:+.3f}  CI95=[{lo:+.3f}, {hi:+.3f}]")

lines += [
    "",
    "Removing the groq-tool-use model returns the point estimate to the",
    "3-model value (~0.5): the higher full-composite correlation on the 5-model",
    "grid is substantially anchored by one degenerate model. The systems composite",
    "without calibration (h3_systems_composite.py) is uncorrelated with capability",
    "regardless of the groq cells.",
    "",
]
OUT.write_text("\n".join(lines), encoding="utf-8")
print(f"wrote {OUT}")
