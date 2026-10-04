from legit_edge.metrics import LatencyMetric, MetricResult


def test_latency_p50_p95_p99():
    m = LatencyMetric()
    samples = list(range(1, 101))   # 1..100 sec
    r = m.compute(samples)
    assert isinstance(r, MetricResult)
    assert r.value["p50"] == 50.5 or 50 <= r.value["p50"] <= 51
    assert 95 <= r.value["p95"] <= 96
    assert 99 <= r.value["p99"] <= 100


def test_latency_block_bootstrap_returns_ci():
    m = LatencyMetric(bootstrap_n=200, block_size=5)
    samples = [1.0] * 50 + [2.0] * 50
    r = m.compute(samples)
    lo, hi = r.ci_95["p99"]
    assert lo <= r.value["p99"] <= hi
