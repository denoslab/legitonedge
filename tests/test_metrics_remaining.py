from legit_edge.metrics import (
    ThroughputStabilityMetric,
    OutputStabilityMetric,
    EnergyMetric,
    CalibrationMetric,
    LatencyMetric,
)
import math


def test_throughput_stability_slope():
    # Constant throughput -> slope ~ 0
    m = ThroughputStabilityMetric()
    times_s = list(range(0, 1800, 10))   # every 10s for 30 min
    tps = [50.0] * len(times_s)
    r = m.compute(samples=list(zip(times_s, tps)))
    assert abs(r.value["slope_tps_per_min"]) < 0.5


def test_output_stability_paired_delta():
    m = OutputStabilityMetric()
    accs_maxn = [0.8] * 50
    accs_throt = [0.7] * 50
    r = m.compute(samples=list(zip(accs_maxn, accs_throt)))
    assert math.isclose(r.value["delta"], 0.1, abs_tol=1e-6)


def test_energy_per_correct():
    m = EnergyMetric()
    # 5 instances; energies in J, correctness 0/1
    samples = [(10.0, 1), (12.0, 1), (8.0, 0), (15.0, 1), (9.0, 0)]
    r = m.compute(samples=samples)
    # j_per_correct = total_J / n_correct = (10+12+8+15+9)/3 = 18.0
    assert math.isclose(r.value["j_per_correct"], 18.0, abs_tol=0.01)


def test_calibration_ece():
    m = CalibrationMetric(n_bins=10)
    # Perfectly calibrated
    confs = [0.0, 0.0, 1.0, 1.0]
    corrects = [0, 0, 1, 1]
    r = m.compute(samples=list(zip(confs, corrects)))
    assert r.value["ece"] < 0.05


def test_bootstrap_b_default_is_10000():
    """Pre-registered B=10,000 per code/PRE_REGISTRATION.md."""
    assert LatencyMetric().bootstrap_n == 10_000
    assert OutputStabilityMetric().bootstrap_n == 10_000
    assert ThroughputStabilityMetric().bootstrap_n == 10_000
