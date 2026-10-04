import os
import pytest
import requests
from legit_edge.adapter import OllamaAdapter
from legit_edge.config import load_tiers


def _server_up(url: str) -> bool:
    try:
        requests.get(url, timeout=1).status_code
        return True
    except requests.exceptions.RequestException:
        return False


_tier = os.environ.get("LEGIT_EDGE_LIVE_TIER", "jetson")
_tiers = load_tiers()
_base = _tiers[_tier].base_url if _tier in _tiers else "http://localhost:11434"


@pytest.mark.skipif(
    not _server_up(_base + "/"),
    reason=f"Ollama at {_base} not running (set LEGIT_EDGE_LIVE_TIER to switch tiers)",
)
def test_ollama_adapter_smoke():
    a = OllamaAdapter(model_tag="llama3.1:8b-instruct-q4_K_M", base_url=_base)
    resp = a.generate(system="", user="Reply with just OK.", max_tokens=8)
    assert len(resp.text) > 0
    assert resp.latency_s > 0
