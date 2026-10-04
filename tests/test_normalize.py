from legit_edge.normalize import latency_score, energy_score, throughput_score


def test_latency_score_monotone_and_clipped():
    assert latency_score(0.0) == 1.0
    assert latency_score(120.0) == 0.0          # >= 2x the 60s ref
    assert 0.0 < latency_score(60.0) < 1.0
    assert latency_score(30.0) > latency_score(90.0)


def test_energy_score_handles_inf_and_bounds():
    assert energy_score(float("inf")) == 0.0
    assert energy_score(0.0) == 1.0
    assert energy_score(4000.0) == 0.0          # >= 2x the 2000J ref


def test_throughput_score_penalizes_only_decay():
    assert throughput_score(0.01) == 1.0        # improving/stable -> 1.0
    assert throughput_score(-0.05) == 0.0       # at -ref -> 0.0
    assert 0.0 < throughput_score(-0.025) < 1.0


def test_reliability_composite_geomean_and_weights():
    from legit_edge.scoring import reliability_composite
    full = dict(latency_score=1.0, throughput_score=1.0, energy_score=1.0,
                output_stability=1.0, calibration_r=1.0)
    assert abs(reliability_composite(**full) - 1.0) < 1e-9
    w = {"latency": 1, "throughput": 1, "energy": 1, "output_stability": 1, "calibration": 0}
    bad_cal = dict(latency_score=1.0, throughput_score=1.0, energy_score=1.0,
                   output_stability=1.0, calibration_r=0.0)
    assert abs(reliability_composite(**bad_cal, weights=w) - 1.0) < 1e-9
    import inspect
    assert "capability" not in inspect.signature(reliability_composite).parameters
