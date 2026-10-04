"""OllamaAdapter.metadata() must reflect the configured tier and model_key.

Previously the adapter hardcoded `tier="jetson"` and reported `name=model_tag`
(the long Ollama tag like `qwen2.5:7b-instruct-q4_K_M`) regardless of which
tier the run targeted. That corrupted `cell_id` everywhere — every spark run
produced `cell_id="qwen2.5:7b-instruct-q4_K_M|jetson|math|..."` instead of
`cell_id="qwen_2_5_7b|spark|math|..."`. report_cli and downstream analysis
both read `cell_id` parts for (model, tier, workload, ...), so the bug made
the tier column lie and the model column inconsistent with the result-file
naming convention (`<model_key>__<target>__<workload>.json`).

This test is paired with the fix that adds `tier` and `model_key` to the
OllamaAdapter constructor and routes them through metadata().
"""
from __future__ import annotations
from legit_edge.adapter import OllamaAdapter


def test_ollama_adapter_metadata_uses_configured_tier_and_model_key():
    a = OllamaAdapter(
        model_tag="qwen2.5:7b-instruct-q4_K_M",
        tier="spark",
        model_key="qwen_2_5_7b",
        base_url="http://192.0.2.20:11434",
    )
    md = a.metadata()
    assert md.tier == "spark", f"expected tier='spark', got {md.tier!r}"
    assert md.name == "qwen_2_5_7b", (
        f"expected name='qwen_2_5_7b' (model_key, short form), got {md.name!r}"
    )
    assert md.runtime == "ollama"


def test_ollama_adapter_metadata_defaults_keep_existing_callers_working():
    """The existing live-Ollama smoke test in test_adapter_clients.py constructs
    OllamaAdapter with only model_tag + base_url. After the fix it should still
    construct without error; metadata() falls back to model_tag for name and a
    sentinel tier value, so old call sites compile but cell_id is clearly
    `unknown` rather than silently misattributed to jetson."""
    a = OllamaAdapter(model_tag="llama3.1:8b-instruct-q4_K_M")
    md = a.metadata()
    # Required: no longer claim 'jetson' by default; surface that nothing was passed.
    assert md.tier != "jetson"
    # name falls back to model_tag when model_key not provided
    assert md.name == "llama3.1:8b-instruct-q4_K_M"
