import pytest
import json
from typer.testing import CliRunner
from legit_edge.cli.app import app


@pytest.mark.requires_datasets
def test_run_mock_megaquick_writes_twenty_json_files(tmp_path):
    out = tmp_path / "results"
    r = CliRunner().invoke(app, [
        "run", "mock", "--mode", "megaquick",
        "--persona", "fast-strong",
        "--out", str(out),
    ])
    assert r.exit_code == 0, r.stdout
    json_files = list(out.rglob("*.json"))
    # 5 models * 4 workloads (math, reasoning, tooluse, tooluse_mt) = 20 cells
    assert len(json_files) == 20
    sample = json.loads(json_files[0].read_text())
    assert "@context" in sample
    assert "capability" in sample


def test_run_mock_rejects_invalid_persona():
    r = CliRunner().invoke(app, ["run", "mock", "--persona", "no-such"])
    assert r.exit_code != 0


@pytest.mark.requires_datasets
def test_run_mock_workloads_filter(tmp_path):
    out = tmp_path / "r"
    r = CliRunner().invoke(app, [
        "run", "mock", "--persona", "fast-strong",
        "--workloads", "math",
        "--out", str(out),
    ])
    assert r.exit_code == 0
    files = list(out.rglob("*.json"))
    assert len(files) == 5   # math x 5 models
    assert all("math" in f.name for f in files)


@pytest.mark.requires_datasets
def test_run_mock_tooluse_mt_routes_and_writes(tmp_path):
    # The multi-step workload must route through run_multiturn_cell and still produce one
    # result file per model in the standard {model}__{tier}__{wl}.json layout.
    out = tmp_path / "mt"
    r = CliRunner().invoke(app, [
        "run", "mock", "--mode", "megaquick", "--persona", "fast-strong",
        "--workloads", "tooluse_mt", "--out", str(out),
    ])
    assert r.exit_code == 0, r.stdout
    files = list(out.rglob("*__mock__tooluse_mt.json"))
    assert len(files) == 5   # tooluse_mt x 5 models
    d = json.loads(files[0].read_text())
    assert 0.0 <= d["capability"] <= 1.0


def test_run_jetson_preflight_fails_when_ollama_unreachable(monkeypatch, tmp_path):
    from legit_edge.config import TierConfig
    monkeypatch.setattr(
        "legit_edge.cli.run.load_tiers",
        lambda: {
            "jetson": TierConfig(name="jetson", host="127.0.0.1", port=1, quantization="Q4_K_M"),
        },
    )
    r = CliRunner().invoke(app, ["run", "jetson", "--out", str(tmp_path)])
    assert r.exit_code == 1
    out = r.stdout.lower()
    assert "preflight failed" in out or "unreachable" in out
