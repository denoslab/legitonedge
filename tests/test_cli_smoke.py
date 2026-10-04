from typer.testing import CliRunner
from legit_edge.cli.app import app
from legit_edge.adapter import AdapterResponse


def test_smoke_jetson_calls_ollama_for_each_model(monkeypatch):
    calls = []

    def fake_generate(self, system, user, **kw):
        calls.append((self.model_tag, user))
        return AdapterResponse(text="OK", latency_s=0.2, input_tokens=5, output_tokens=2)

    monkeypatch.setattr("legit_edge.adapter.OllamaAdapter.generate", fake_generate)
    monkeypatch.setenv("LEGIT_EDGE_JETSON_HOST", "jet.test")
    r = CliRunner().invoke(app, ["smoke", "jetson"])
    assert r.exit_code == 0
    assert len(calls) == 5  # 5 models
    for _, prompt in calls:
        assert "2+2" in prompt
