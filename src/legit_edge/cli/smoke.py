"""`legit-edge smoke <tier>` - one-prompt round-trip per model."""
from __future__ import annotations
import typer
from rich.console import Console

from ..adapter import OllamaAdapter
from ..config import load_tiers, load_models

console = Console()
PROMPT = "What is 2+2? Answer with just the number."


def smoke(tier: str) -> None:
    """One-prompt round-trip against the tier's Ollama for every model in models.yaml."""
    tiers = load_tiers()
    if tier not in tiers:
        console.print(f"[red]Unknown tier '{tier}'. Available: {list(tiers)}[/red]")
        raise typer.Exit(code=2)
    base_url = tiers[tier].base_url
    models = load_models()
    console.print(f"Smoke testing {len(models)} models on [bold]{tier}[/] ({base_url}):")
    for name, m in models.items():
        tag = m.ollama_tag[tier]
        adapter = OllamaAdapter(model_tag=tag, base_url=base_url)
        try:
            resp = adapter.generate(system="", user=PROMPT, max_tokens=8)
            console.print(
                f"  [green]✓[/] {name:<14} {tag:<35} {resp.latency_s:>5.1f}s  -> {resp.text!r}"
            )
        except Exception as e:
            console.print(f"  [red]✗[/] {name:<14} {tag:<35} FAILED  {e}")
