from legit_edge.report import to_jsonld, to_markdown
from legit_edge.runner import CellResult
from legit_edge.metrics import MetricResult
import json


def _stub_cell():
    return CellResult(
        cell_id="m1|jetson|math|standard|MAXN",
        mode="standard",
        thermal="MAXN",
        capability=0.75,
        latency=MetricResult("latency", {"p50": 1.0, "p95": 2.0, "p99": 3.0}, {}, 100),
        throughput=MetricResult(
            "throughput_stability", {"slope_tps_per_min": -0.1, "mean_tps": 50.0}, {}, 100
        ),
        energy=MetricResult(
            "energy_per_correct", {"j_per_correct": 12.0, "j_per_attempt": 9.0}, {}, 100
        ),
    )


def test_jsonld_round_trip():
    out = to_jsonld(_stub_cell())
    assert out["@context"] == "https://legit-edge.example/v1"
    assert out["cell_id"] == "m1|jetson|math|standard|MAXN"
    assert out["metrics"]["latency"]["p50"] == 1.0


def test_jsonld_includes_gpu_domain_energy():
    """The GPU-domain (overhead-subtracted) energy figure round-trips.

    Defaults to None when the monitor didn't expose a secondary channel (stub cell)."""
    out = to_jsonld(_stub_cell())
    assert "energy_gpu_domain_j" in out
    assert out["energy_gpu_domain_j"] is None
    c = _stub_cell()
    c.energy_gpu_domain_j = 123.4
    assert to_jsonld(c)["energy_gpu_domain_j"] == 123.4


def test_markdown_contains_headline():
    md = to_markdown(_stub_cell())
    assert "## Cell" in md
    assert "Capability" in md


def test_to_traces_jsonl_writes_one_line_per_instance(tmp_path):
    from legit_edge.report import to_traces_jsonl
    per_inst = [
        {"instance_index": 0, "input": "a", "target": "1", "output": "1",
         "score": 1.0, "score_metadata": {}, "latency_s": 0.1,
         "output_tokens": 1, "input_tokens": 1, "logprobs": None,
         "confidence": None, "joules_interval": 0.0,
         "power_W_mean_interval": 0.0, "timestamp": "2026-05-27T00:00:00+00:00",
         "temperature_c": None},
        {"instance_index": 1, "input": "b", "target": "2", "output": "2",
         "score": 1.0, "score_metadata": {}, "latency_s": 0.2,
         "output_tokens": 1, "input_tokens": 1, "logprobs": None,
         "confidence": None, "joules_interval": 1.5,
         "power_W_mean_interval": 7.5, "timestamp": "2026-05-27T00:00:01+00:00",
         "temperature_c": 45.0},
    ]
    path = tmp_path / "x.traces.jsonl"
    to_traces_jsonl(per_inst, path)
    lines = path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["instance_index"] == 0
    assert json.loads(lines[1])["joules_interval"] == 1.5
    assert json.loads(lines[1])["temperature_c"] == 45.0
