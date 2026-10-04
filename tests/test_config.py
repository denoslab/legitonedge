import pytest
from legit_edge.config import expand_env, load_tiers, load_models, TierConfig, ModelConfig


def test_expand_env_with_default(monkeypatch):
    monkeypatch.delenv("FOO_HOST", raising=False)
    assert expand_env("${FOO_HOST:-localhost}") == "localhost"


def test_expand_env_uses_override(monkeypatch):
    monkeypatch.setenv("FOO_HOST", "1.2.3.4")
    assert expand_env("${FOO_HOST:-localhost}") == "1.2.3.4"


def test_expand_env_passes_through_non_template_strings():
    assert expand_env("plain") == "plain"


def test_load_tiers_returns_two_tiers(monkeypatch):
    monkeypatch.setenv("LEGIT_EDGE_JETSON_HOST", "jet.test")
    monkeypatch.setenv("LEGIT_EDGE_SPARK_HOST", "spark.test")
    tiers = load_tiers()
    assert set(tiers.keys()) == {"jetson", "spark"}
    assert tiers["jetson"].host == "jet.test"
    assert tiers["jetson"].port == 11434
    assert tiers["jetson"].base_url == "http://jet.test:11434"


def test_load_models_returns_five_models():
    models = load_models()
    assert set(models.keys()) == {
        "llama_3_1_8b", "qwen_2_5_7b", "phi_3_5_mini",
        "hermes3_8b", "llama3_groq_tooluse_8b",
    }
    assert models["llama_3_1_8b"].ollama_tag["jetson"].startswith("llama3.1:")
    assert models["llama3_groq_tooluse_8b"].ollama_tag["spark"].startswith("llama3-groq-tool-use:")


def test_load_tiers_ssh_user_from_env(monkeypatch):
    monkeypatch.setenv("LEGIT_EDGE_JETSON_HOST", "jet.test")
    monkeypatch.setenv("LEGIT_EDGE_JETSON_SSH_USER", "user")
    monkeypatch.delenv("LEGIT_EDGE_SPARK_SSH_USER", raising=False)
    monkeypatch.setenv("LEGIT_EDGE_SPARK_HOST", "spark.test")
    tiers = load_tiers()
    assert tiers["jetson"].ssh_target == "user@jet.test"
    # no user configured -> bare host (ssh falls back to local user / ~/.ssh/config)
    assert tiers["spark"].ssh_target == "spark.test"
