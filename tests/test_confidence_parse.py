def test_parse_confidence():
    from legit_edge.workload import parse_confidence
    assert parse_confidence("...\nConfidence: 80%") == 0.80
    assert parse_confidence("Confidence: 100%") == 1.0
    assert parse_confidence("no confidence here") is None
    assert parse_confidence("Confidence: 5") == 0.05
    assert parse_confidence("Confidence: 250%") == 1.0
