from legit_edge.adapter import MockAdapter
from legit_edge.personas import PERSONAS


def test_mock_adapter_default_persona_still_returns_canned_for_backward_compat():
    a = MockAdapter(canned="42")
    r = a.generate(system="", user="2+2?", expected=None)
    assert r.text == "42"


def test_mock_adapter_with_persona_emits_target_when_capability_succeeds():
    a = MockAdapter.from_persona(PERSONAS["fast-strong"])
    # capability_p=0.85; over 200 trials with seed=42 we expect the mode to be "correct"
    hits = 0
    for i in range(200):
        r = a.generate(system="", user=f"question {i}", expected="42")
        if "42" in r.text:
            hits += 1
    # 0.85 +- a generous band for finite-sample variance
    assert 0.75 <= hits / 200 <= 0.95


def test_mock_adapter_persona_latency_distribution_has_correct_p99():
    a = MockAdapter.from_persona(PERSONAS["slow-strong"])
    lats = []
    for i in range(500):
        r = a.generate(system="", user=f"q{i}", expected="X")
        lats.append(r.latency_s)
    lats.sort()
    p50 = lats[250]
    p99 = lats[int(0.99 * 500)]
    # latency_mean_s=5.0, p99_factor=3.0 -> p99 in roughly [11, 19] for lognormal
    assert 3.0 <= p50 <= 8.0
    assert 9.0 <= p99 <= 25.0


def test_mock_adapter_thermal_drop_lowers_capability_when_throttled():
    a = MockAdapter.from_persona(PERSONAS["thermal-drift"])
    hits_maxn = sum(
        "X" in a.generate(system="", user="q", expected="X", thermal="MAXN").text
        for _ in range(300)
    )
    a2 = MockAdapter.from_persona(PERSONAS["thermal-drift"])
    hits_throt = sum(
        "X" in a2.generate(system="", user="q", expected="X", thermal="throttled").text
        for _ in range(300)
    )
    # capability_p=0.80, thermal_drop=0.25 -> throttled effective p=0.55
    assert hits_maxn / 300 > hits_throt / 300 + 0.10
