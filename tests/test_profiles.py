def test_load_profiles_returns_named_weight_vectors():
    from legit_edge.profiles import load_profiles
    profs = load_profiles()
    assert set(profs) >= {"balanced", "latency_critical", "energy_constrained"}
    # the composite weights only the three systems dimensions
    assert set(profs["balanced"]) == {"latency", "throughput", "energy"}
    # latency-critical leans on latency over energy
    assert profs["latency_critical"]["latency"] > profs["latency_critical"]["energy"]


def test_profile_reweights_composite():
    from legit_edge.profiles import load_profiles
    from legit_edge.scoring import reliability_composite
    profs = load_profiles()
    # output_stability / calibration carry no scenario weight, so they are skipped
    dims = dict(latency_score=0.9, throughput_score=0.5, energy_score=0.2,
                output_stability=0.8, calibration_r=0.8)
    lat = reliability_composite(**dims, weights=profs["latency_critical"])
    eng = reliability_composite(**dims, weights=profs["energy_constrained"])
    assert lat > eng    # latency-heavy profile rewards the high latency_score cell
