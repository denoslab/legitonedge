"""Unit tests for the paired-Δ analysis script using synthetic traces.jsonl files."""
import json
import sys
from pathlib import Path

# scripts/ is not on the package path; add it.
SCRIPTS_ANALYSIS = Path(__file__).parent.parent / "scripts" / "analysis"
sys.path.insert(0, str(SCRIPTS_ANALYSIS.resolve()))


def _write_traces(path: Path, scores: list[float], lats: list[float] | None = None):
    if lats is None:
        lats = [1.0 + i * 0.01 for i in range(len(scores))]
    rows = [
        {"instance_index": i, "score": float(s), "confidence": None,
         "latency_s": float(l), "joules_interval": 5.0,
         "input": f"q{i}", "target": "x", "output": "x", "score_metadata": {},
         "output_tokens": 1, "input_tokens": 1, "logprobs": None,
         "power_W_mean_interval": 5.0, "timestamp": "x", "temperature_c": None}
        for i, (s, l) in enumerate(zip(scores, lats))
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_paired_delta_capability_basic(tmp_path):
    from output_stability_paired import paired_delta_cell
    maxn_dir = tmp_path / "maxn"; maxn_dir.mkdir()
    throt_dir = tmp_path / "throt"; throt_dir.mkdir()
    _write_traces(maxn_dir / "phi_3_5_mini__jetson__math.traces.jsonl",
                  [1.0]*8 + [0.0]*2)
    _write_traces(throt_dir / "phi_3_5_mini__jetson__math.traces.jsonl",
                  [1.0]*7 + [0.0]*3)
    d = paired_delta_cell(
        maxn_dir / "phi_3_5_mini__jetson__math.traces.jsonl",
        throt_dir / "phi_3_5_mini__jetson__math.traces.jsonl",
    )
    # cap_M = 0.8, cap_T = 0.7, delta_cap = -0.1
    assert d["n"] == 10
    assert abs(d["cap_M"] - 0.8) < 1e-9
    assert abs(d["cap_T"] - 0.7) < 1e-9
    assert abs(d["delta_cap"] - (-0.1)) < 1e-9


def test_paired_delta_latency_p50_p95(tmp_path):
    from output_stability_paired import paired_delta_cell
    maxn_dir = tmp_path / "maxn"; maxn_dir.mkdir()
    throt_dir = tmp_path / "throt"; throt_dir.mkdir()
    # 10 latencies; throttled is uniformly +1.0s
    _write_traces(maxn_dir / "x__a__math.traces.jsonl",
                  [1.0]*10, lats=[float(i) for i in range(10)])
    _write_traces(throt_dir / "x__a__math.traces.jsonl",
                  [1.0]*10, lats=[float(i) + 1.0 for i in range(10)])
    d = paired_delta_cell(
        maxn_dir / "x__a__math.traces.jsonl",
        throt_dir / "x__a__math.traces.jsonl",
    )
    assert abs(d["delta_p50"] - 1.0) < 1e-9
    assert abs(d["delta_p95"] - 1.0) < 1e-9


def test_paired_bootstrap_ci_contains_truth(tmp_path):
    """Paired bootstrap on a constant +0.1 Δcap should produce a CI that
    contains the truth most of the time. With B=2000 and n=50 the CI should
    bracket +0.1 with no slop."""
    from output_stability_paired import paired_bootstrap_ci
    import numpy as np
    rng = np.random.default_rng(42)
    n = 50
    # maxn = 0/1 bernoulli p=0.5; throttled = same but with 10% boost
    a = rng.binomial(1, 0.5, n).astype(float)
    b = np.clip(a + rng.binomial(1, 0.1, n), 0, 1).astype(float)
    lo, hi = paired_bootstrap_ci(a, b, lambda x, y: float(y.mean() - x.mean()),
                                  B=2000, seed=42)
    # mean(b) - mean(a) is something around +0.05 to +0.10 typically; CI must bracket it
    point = float(b.mean() - a.mean())
    assert lo <= point <= hi, f"point {point} not in CI [{lo}, {hi}]"


def test_holm_bonferroni_basic():
    """3 p-values; Holm step-down at alpha=0.05.

    Sorted ascending p: p1=0.01, p2=0.04, p3=0.06.
    Holm thresholds at alpha=0.05: a/3=0.01667, a/2=0.025, a/1=0.05.
    p1 (0.01) < 0.01667 → reject
    p2 (0.04) > 0.025 → accept (and stop)
    p3 stays accepted.
    """
    from output_stability_paired import holm_bonferroni
    decisions = holm_bonferroni([0.01, 0.04, 0.06], alpha=0.05)
    # original-order decisions (matches input order)
    assert decisions == [True, False, False]


def test_main_writes_markdown_report(tmp_path):
    """End-to-end: build a 2-cell synthetic input, run main, verify the .md
    output has a table row per cell + an aggregate section."""
    from output_stability_paired import main
    maxn_dir = tmp_path / "maxn"; maxn_dir.mkdir()
    throt_dir = tmp_path / "throt"; throt_dir.mkdir()
    # 2 cells: math + tooluse (tooluse should be in per-cell table but dropped
    # from the aggregate).
    _write_traces(maxn_dir / "phi_3_5_mini__jetson__math.traces.jsonl",
                  [1.0]*8 + [0.0]*2)
    _write_traces(throt_dir / "phi_3_5_mini__jetson__math.traces.jsonl",
                  [1.0]*7 + [0.0]*3)
    _write_traces(maxn_dir / "phi_3_5_mini__jetson__tooluse.traces.jsonl",
                  [0.5]*16)
    _write_traces(throt_dir / "phi_3_5_mini__jetson__tooluse.traces.jsonl",
                  [0.5]*16)
    out = tmp_path / "report.md"
    rc = main([str(maxn_dir), str(throt_dir), "--out", str(out), "--bootstrap-b", "100"])
    assert rc == 0
    md = out.read_text(encoding="utf-8")
    assert "math" in md
    assert "tooluse" in md
    assert "Non-tool-use aggregate" in md
    # Aggregate row count should refer to 1 non-tool-use cell (math only)
    assert "n_cells = 1" in md or "n_cells=1" in md
