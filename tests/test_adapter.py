from legit_edge.adapter import AdapterResponse, ModelAdapter, MockAdapter


def test_mock_adapter_returns_canned_response():
    a = MockAdapter(canned="42")
    resp = a.generate(system="", user="What is 6*7?")
    assert isinstance(resp, AdapterResponse)
    assert resp.text == "42"
    assert resp.latency_s > 0
    assert resp.input_tokens > 0
    assert resp.output_tokens > 0


def test_adapter_metadata_includes_tier():
    a = MockAdapter(canned="x", tier="mock-tier")
    assert a.metadata().tier == "mock-tier"
