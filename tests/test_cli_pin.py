from typer.testing import CliRunner
from legit_edge.cli.app import app


def test_pin_datasets_invokes_loader(monkeypatch):
    called = {}

    def fake_run_dataset_pin():
        called["yes"] = True

    monkeypatch.setattr("legit_edge.cli.pin.run_dataset_pin", fake_run_dataset_pin)
    r = CliRunner().invoke(app, ["pin", "datasets"])
    assert r.exit_code == 0, r.stdout
    assert called["yes"] is True


def test_pin_models_calls_pull_for_each_model(monkeypatch):
    pulls = []

    def fake_pull_model(base_url, tag):
        pulls.append((base_url, tag))
        yield {"status": "success"}

    monkeypatch.setattr("legit_edge.cli.pin.pull_model", fake_pull_model)
    monkeypatch.setenv("LEGIT_EDGE_JETSON_HOST", "jet.test")
    r = CliRunner().invoke(app, ["pin", "models", "jetson"])
    assert r.exit_code == 0, r.stdout
    assert len(pulls) == 5  # llama, qwen, phi, hermes3, groq-tool-use
    assert all(base == "http://jet.test:11434" for base, _ in pulls)


def test_pin_models_rejects_unknown_tier():
    r = CliRunner().invoke(app, ["pin", "models", "noexist"])
    assert r.exit_code != 0
