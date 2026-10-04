"""`legit-edge doctor` - readiness report with per-tier checks."""
from __future__ import annotations
import typer
from rich.console import Console

from ..config import load_tiers
from .checks import (
    Check, check_python, check_uv, check_package, check_configs,
    check_datasets, check_ollama, check_ina260,
)

console = Console()

_ICON = {"ok": "[green]✓[/]", "warn": "[yellow]⚠[/]", "fail": "[red]✗[/]"}


def _render(c: Check) -> None:
    icon = _ICON[c.status]
    console.print(f"{icon} [bold]{c.name:<12}[/] {c.detail}")
    if c.status != "ok" and c.fix_hint:
        console.print(f"                fix: {c.fix_hint}")


def doctor() -> None:
    """Per-tier readiness report. Exit 0 if mock path is ready (warnings ok)."""
    console.rule("LegitOnEdge readiness report")

    required: list[Check] = [
        check_python(), check_uv(), check_package(),
        check_configs(), check_datasets(),
    ]
    for c in required:
        _render(c)

    tiers = load_tiers()
    tier_checks: dict[str, Check] = {
        name: check_ollama(t.base_url, tier=name) for name, t in tiers.items()
    }
    for c in tier_checks.values():
        _render(c)

    optional: list[Check] = [check_ina260()]
    for c in optional:
        _render(c)

    console.rule()
    mock_ready = all(c.status != "fail" for c in required)
    console.print(
        f"Mock target: [bold]{'READY' if mock_ready else 'NOT READY'}[/]"
        f"{' → try [cyan]legit-edge demo[/]' if mock_ready else ''}"
    )
    for tname, tc in tier_checks.items():
        state = "READY" if tc.status == "ok" else f"needs Ollama @ {tname}"
        console.print(f"{tname.capitalize():<12} {state}")

    if any(c.status == "fail" for c in required):
        raise typer.Exit(code=1)
