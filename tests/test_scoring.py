from legit_edge.scoring import legit_edge_score


def test_perfect_inputs_yield_100():
    s = legit_edge_score(
        capability=1.0,
        latency_score=1.0,
        throughput_score=1.0,
        energy_score=1.0,
        output_stability=1.0,
        calibration_r=1.0,
    )
    assert 99.0 <= s <= 100.001


def test_drift_penalty_reduces_score():
    s_high = legit_edge_score(
        capability=1.0,
        latency_score=1.0,
        throughput_score=1.0,
        energy_score=1.0,
        output_stability=1.0,
        calibration_r=1.0,
    )
    s_low = legit_edge_score(
        capability=1.0,
        latency_score=1.0,
        throughput_score=1.0,
        energy_score=1.0,
        output_stability=0.5,
        calibration_r=1.0,
    )
    assert s_low < s_high
