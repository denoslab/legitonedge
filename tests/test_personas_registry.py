from legit_edge.personas import Persona, PERSONAS, get_persona


def test_four_personas_shipped():
    assert set(PERSONAS.keys()) == {"fast-strong", "slow-strong", "fast-weak", "thermal-drift"}


def test_persona_dataclass_fields():
    p = PERSONAS["fast-strong"]
    assert isinstance(p, Persona)
    assert 0.0 <= p.capability_p <= 1.0
    assert p.latency_mean_s > 0
    assert p.latency_p99_factor >= 1.0
    assert 0.0 <= p.thermal_capability_drop <= 1.0
    assert p.energy_watts > 0
    assert 0 <= p.confidence_bias <= 1.0
    assert p.output_tokens_mean > 0


def test_get_persona_by_name():
    assert get_persona("fast-strong").name == "fast-strong"


def test_get_persona_unknown_raises():
    import pytest
    with pytest.raises(KeyError, match="unknown persona 'nope'"):
        get_persona("nope")


def test_thermal_drift_is_the_killer_case():
    p = PERSONAS["thermal-drift"]
    assert p.thermal_capability_drop >= 0.20
    assert p.latency_p99_factor >= 3.0
    assert p.tps_drift_per_min < 0
