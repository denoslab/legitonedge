from legit_edge.runner import run_cell, CellResult
from legit_edge.adapter import MockAdapter
from legit_edge.workload import MathWorkload


def test_run_cell_returns_metrics(tmp_path):
    p = tmp_path / "math.jsonl"
    p.write_text('{"input": "x", "target": "42"}\n')
    adapter = MockAdapter(canned="The answer is 42.", tier="mock")
    workload = MathWorkload(jsonl_path=p)
    res = run_cell(
        adapter=adapter,
        workload=workload,
        n_instances=1,
        mode="megaquick",
        thermal="MAXN",
    )
    assert isinstance(res, CellResult)
    assert res.capability == 1.0
    assert res.latency.value["p50"] >= 0


def test_runner_per_instance_has_15_field_schema(tmp_path):
    """Traces schema: 15 fields per row. logprobs is always None (Ollama exposes
    no token logprobs on the targeted builds) and confidence is None unless
    --confidence is used; both fields stay in the schema for stability."""
    from legit_edge.runner import run_cell
    from legit_edge.adapter import MockAdapter
    from legit_edge.personas import get_persona
    from legit_edge.workload import MathWorkload
    p = tmp_path / "math.jsonl"
    p.write_text('{"input": "2+2?", "target": "4"}\n')
    wl = MathWorkload(jsonl_path=p)
    persona = get_persona("fast-strong")
    adapter = MockAdapter.from_persona(persona, tier="mock", name="phi_test")
    cell = run_cell(adapter=adapter, workload=wl, n_instances=1, mode="megaquick", thermal="MAXN")
    expected_keys = {
        "instance_index", "input", "target", "output", "score", "score_metadata",
        "latency_s", "output_tokens", "input_tokens",
        "logprobs", "confidence",
        "joules_interval", "power_W_mean_interval",
        "timestamp", "temperature_c",
    }
    assert set(cell.per_instance[0].keys()) == expected_keys
    # No logprobs, and no confidence without --confidence: both None on Mock
    assert cell.per_instance[0]["logprobs"] is None
    assert cell.per_instance[0]["confidence"] is None
    # joules_interval should be >= 0 on Mock (mock constant_watts × elapsed)
    assert cell.per_instance[0]["joules_interval"] >= 0.0


def test_run_cell_captures_gpu_domain_energy(tmp_path):
    """CellResult carries a cell-level GPU-domain energy figure when the
    monitor exposes snap_secondary() (Mock does). Per-instance schema is untouched."""
    from legit_edge.runner import run_cell
    from legit_edge.adapter import MockAdapter
    from legit_edge.workload import MathWorkload
    p = tmp_path / "math.jsonl"
    p.write_text('{"input": "2+2?", "target": "4"}\n')
    wl = MathWorkload(jsonl_path=p)
    adapter = MockAdapter(canned="The answer is 4.", tier="mock")
    cell = run_cell(adapter=adapter, workload=wl, n_instances=1, mode="megaquick", thermal="MAXN")
    assert cell.energy_gpu_domain_j is not None
    assert cell.energy_gpu_domain_j > 0.0
    # per-instance schema unchanged (no per-instance GPU field)
    assert "energy_gpu_domain_j" not in cell.per_instance[0]
