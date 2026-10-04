"""Unit tests for v2_thermal_energy_diag pure helpers using synthetic rows.

Validates the aggregation logic the thermal/energy diagnostics report (v2_thermal_energy_diag.py) relies on:
capability, energy ratios, None-temperature exclusion, and the byte-identical /
score-disagreement counters used for the determinism findings.
"""
import math
import sys
from pathlib import Path

# scripts/ is not on the package path; add it (mirrors test_output_stability_paired).
SCRIPTS_ANALYSIS = Path(__file__).parent.parent / "scripts" / "analysis"
sys.path.insert(0, str(SCRIPTS_ANALYSIS.resolve()))


def _row(idx, score, tokens, joules, temp, output):
    return {"instance_index": idx, "score": score, "output_tokens": tokens,
            "joules_interval": joules, "temperature_c": temp,
            "latency_s": 1.0, "output": output}


def test_cell_summary_basic():
    from v2_thermal_energy_diag import cell_summary
    rows = [
        _row(0, 1.0, 100, 10.0, None, "a"),   # temp None -> excluded from temp stats
        _row(1, 0.0, 200, 30.0, 50.0, "b"),
    ]
    s = cell_summary(rows)
    assert s["n"] == 2
    assert abs(s["cap"] - 0.5) < 1e-9
    assert s["n_correct"] == 1.0
    assert abs(s["mean_tokens"] - 150.0) < 1e-9
    assert abs(s["total_j"] - 40.0) < 1e-9
    assert abs(s["j_per_attempt"] - 20.0) < 1e-9
    assert abs(s["j_per_correct"] - 40.0) < 1e-9   # 40 J / 1 correct
    assert s["n_temp"] == 1                          # row0 None excluded
    assert abs(s["temp_max"] - 50.0) < 1e-9


def test_cell_summary_zero_correct_is_inf():
    from v2_thermal_energy_diag import cell_summary
    rows = [_row(0, 0.0, 10, 5.0, 40.0, "x")]
    s = cell_summary(rows)
    assert s["j_per_correct"] == math.inf


def test_identical_fraction_and_disagree():
    from v2_thermal_energy_diag import identical_fraction
    a = [_row(0, 1.0, 1, 1.0, 40.0, "same"), _row(1, 1.0, 1, 1.0, 40.0, "A")]
    b = [_row(0, 1.0, 1, 1.0, 40.0, "same"), _row(1, 0.0, 1, 1.0, 40.0, "B")]
    frac, disagree, n = identical_fraction(a, b)
    assert n == 2
    assert abs(frac - 0.5) < 1e-9    # 1 of 2 outputs byte-identical
    assert disagree == 1             # 1 of 2 scores differ
