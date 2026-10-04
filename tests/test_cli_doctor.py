from typer.testing import CliRunner
from legit_edge.cli.app import app


def test_doctor_runs_and_emits_lines_for_each_check():
    r = CliRunner().invoke(app, ["doctor"])
    assert r.exit_code == 0   # warnings allowed; failures would exit 1
    for label in ("Python", "uv", "Package", "Configs", "Datasets", "Ollama", "INA260"):
        assert label in r.stdout


def test_doctor_summary_includes_mock_jetson_spark_lines():
    r = CliRunner().invoke(app, ["doctor"])
    assert "Mock target" in r.stdout
    assert "Jetson" in r.stdout
    assert "Spark" in r.stdout
