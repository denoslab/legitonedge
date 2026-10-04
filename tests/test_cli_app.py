from typer.testing import CliRunner
from legit_edge.cli.app import app


def test_no_args_prints_help_and_hint():
    r = CliRunner().invoke(app, [])
    assert r.exit_code == 0
    assert "legit-edge demo" in r.stdout
    assert "Usage" in r.stdout or "Commands" in r.stdout


def test_help_flag_works():
    r = CliRunner().invoke(app, ["--help"])
    assert r.exit_code == 0
    assert "legit-edge" in r.stdout.lower()
