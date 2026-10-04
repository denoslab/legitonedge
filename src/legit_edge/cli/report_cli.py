"""`legit-edge report` - aggregate results/<run>/*.json into MD or JSON."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console

from ..paths import RESULTS_DIR

console = Console()


def _collect(root: Path) -> list[dict]:
    out: list[dict] = []
    for f in sorted(root.rglob("*.json")):
        try:
            d = json.loads(f.read_text())
            if "cell_id" in d and "capability" in d:
                out.append(d)
        except (json.JSONDecodeError, OSError):
            continue
    return out


def report(
    out: Optional[Path] = typer.Option(None, "--out", help="results dir to scan (default: results/)"),
    since: Optional[str] = typer.Option(None, "--since", help="ISO date filter on run subdir name"),
    format: str = typer.Option("md", "--format", help="md | json"),
) -> None:
    """Scan results/ and aggregate into Markdown or JSON."""
    root = out or RESULTS_DIR
    if not root.exists():
        console.print(f"[yellow]No results directory at {root}[/]")
        raise typer.Exit(code=0)

    cells = _collect(root)
    if since:
        cells = [c for c in cells if since in c["cell_id"]]

    if format == "json":
        # Use plain print (not Rich) so the output is parseable
        print(json.dumps(cells, indent=2))
        return

    # Markdown table
    print("| Model | Tier | Workload | Mode | Thermal | Cap | p50(s) | p95(s) | p99(s) | J/correct |")
    print("|---|---|---|---|---|---:|---:|---:|---:|---:|")
    for c in cells:
        parts = c["cell_id"].split("|")
        model, tier, wl, mode, therm = (parts + [""] * 5)[:5]
        lat = c["metrics"].get("latency", {})
        en = c["metrics"].get("energy_per_correct", {})
        cap = f"{c['capability']:.3f}"
        p50 = f"{lat.get('p50', float('nan')):.3f}"
        p95 = f"{lat.get('p95', float('nan')):.3f}"
        p99 = f"{lat.get('p99', float('nan')):.3f}"
        jpc = f"{en.get('j_per_correct', float('nan')):.2f}"
        print(f"| {model} | {tier} | {wl} | {mode} | {therm} | {cap} | {p50} | {p95} | {p99} | {jpc} |")
