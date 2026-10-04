import pytest
from typer.testing import CliRunner
from legit_edge.cli.app import app


@pytest.mark.requires_datasets
def test_demo_runs_to_completion_and_produces_results(tmp_path, monkeypatch):
    monkeypatch.setenv("LEGIT_EDGE_DEMO_OUT", str(tmp_path))
    r = CliRunner().invoke(app, ["demo"])
    assert r.exit_code == 0, r.stdout
    assert "LegitOnEdge" in r.stdout
    files = list(tmp_path.rglob("*.json"))
    assert len(files) == 15
