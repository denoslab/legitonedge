"""Tests for the H1 paired MAXN-vs-throttled calibration test (reshoot_h1).

gap = ECE_MAXN - ECE_throttled, per cell, with an independent bootstrap of each
regime's pooled (confidence, correct) pairs; Holm-Bonferroni across cells.
"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str((Path(__file__).parent.parent / "scripts" / "analysis").resolve()))


def test_holm_bonferroni_rejects_in_order():
    from reshoot_h1 import holm_bonferroni
    reject, adj = holm_bonferroni([0.20, 0.01, 0.04], alpha=0.10)
    assert reject == [False, True, True]
    # step-down adjusted p-values (input order): idx1->0.03, idx2->0.08, idx0->0.20
    assert abs(adj[1] - 0.03) < 1e-9
    assert abs(adj[2] - 0.08) < 1e-9
    assert abs(adj[0] - 0.20) < 1e-9


def test_holm_bonferroni_all_and_none():
    from reshoot_h1 import holm_bonferroni
    assert holm_bonferroni([0.001, 0.002], 0.10)[0] == [True, True]
    assert holm_bonferroni([0.9, 0.8], 0.10)[0] == [False, False]


def test_ece_calibrated_vs_overconfident():
    from reshoot_h1 import ece
    well = [(0.5, 1), (0.5, 0)] * 20      # conf .5, acc .5  -> ECE ~ 0
    over = [(0.9, 0)] * 40                # conf .9, acc 0   -> ECE = 0.9
    assert ece(well) < 0.05
    assert abs(ece(over) - 0.9) < 1e-9


def test_gap_bootstrap_detects_real_difference():
    from reshoot_h1 import gap_bootstrap
    over = [(0.9, 0)] * 40                 # high ECE (~0.9)
    well = [(0.5, 1), (0.5, 0)] * 20       # low ECE (~0)
    res = gap_bootstrap(over, well, B=2000, seed=1)   # gap = ECE_maxn - ECE_throt ~ +0.9
    assert res["gap"] > 0.7
    assert res["ci_lo"] > 0.0              # CI excludes 0
    assert res["p_two_sided"] < 0.05


def test_gap_bootstrap_null_when_pools_identical():
    from reshoot_h1 import gap_bootstrap
    a = [(0.9, 1), (0.9, 0), (0.8, 1), (0.7, 0)] * 15
    res = gap_bootstrap(a, list(a), B=2000, seed=1)   # identical pools -> point gap = 0
    assert abs(res["gap"]) < 1e-9
    assert res["ci_lo"] <= 0.0 <= res["ci_hi"]        # CI spans 0
    assert res["p_two_sided"] > 0.10                  # not significant


def test_cell_gap_reads_two_dirs(tmp_path):
    from reshoot_h1 import cell_gap
    maxn = tmp_path / "maxn"
    throt = tmp_path / "throt"
    maxn.mkdir()
    throt.mkdir()
    rows = [{"input": "q", "target": "4",
             "output": "The answer is 5\nConfidence: 95%", "score": 0.0,
             "confidence": 0.95} for _ in range(20)]
    body = "\n".join(json.dumps(r) for r in rows) + "\n"
    (maxn / "m__jetson__math__run0.traces.jsonl").write_text(body, encoding="utf-8")
    (throt / "m__jetson__math__run0.traces.jsonl").write_text(body, encoding="utf-8")
    row = cell_gap(maxn, throt, "m__jetson__math", B=500, seed=1)
    assert row["cell"] == "m__jetson__math"
    assert abs(row["ece_maxn"] - 0.95) < 1e-9          # conf .95, acc 0 -> ECE 0.95
    assert abs(row["ece_throt"] - 0.95) < 1e-9
    assert abs(row["gap"]) < 1e-9                       # same data both regimes
    assert row["n_maxn"] == 20 and row["n_throt"] == 20
