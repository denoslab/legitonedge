import json
from pathlib import Path
from typer.testing import CliRunner
from legit_edge.cli.app import app


def _seed_results(root: Path) -> None:
    run_dir = root / "demo-20260512-100000"
    run_dir.mkdir(parents=True)
    cells = [
        {"@context": "x", "cell_id": "llama|mock|math|megaquick|MAXN",
         "mode": "megaquick", "thermal": "MAXN", "capability": 0.82,
         "metrics": {"latency": {"p50": 0.5, "p95": 0.7, "p99": 0.9},
                     "throughput_stability": {"slope_tps_per_min": 0.0},
                     "energy_per_correct": {"j_per_correct": 5.0}},
         "ci_95": {}, "n": {}},
        {"@context": "x", "cell_id": "qwen|mock|math|megaquick|MAXN",
         "mode": "megaquick", "thermal": "MAXN", "capability": 0.78,
         "metrics": {"latency": {"p50": 0.6, "p95": 0.8, "p99": 1.0},
                     "throughput_stability": {"slope_tps_per_min": 0.0},
                     "energy_per_correct": {"j_per_correct": 6.0}},
         "ci_95": {}, "n": {}},
    ]
    for i, c in enumerate(cells):
        (run_dir / f"cell_{i}.json").write_text(json.dumps(c))


def test_report_md_emits_table(tmp_path):
    _seed_results(tmp_path)
    r = CliRunner().invoke(app, ["report", "--out", str(tmp_path)])
    assert r.exit_code == 0, r.stdout
    assert "| Model" in r.stdout or "Model " in r.stdout
    assert "llama" in r.stdout
    assert "qwen" in r.stdout


def test_report_json_format(tmp_path):
    _seed_results(tmp_path)
    r = CliRunner().invoke(app, ["report", "--out", str(tmp_path), "--format", "json"])
    assert r.exit_code == 0
    data = json.loads(r.stdout)
    assert isinstance(data, list)
    assert len(data) == 2
