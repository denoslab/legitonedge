"""`legit-edge demo` - 30-second mock teaser run + summary table."""
from __future__ import annotations
import json
import os
import time
from pathlib import Path
from rich.console import Console
from rich.table import Table

from ..paths import RESULTS_DIR
from .run import run as run_cmd

console = Console()


def demo() -> None:
    """30-second teaser: mock x megaquick x persona=fast-strong."""
    console.rule("[bold cyan]LegitOnEdge demo[/]")
    console.print("mock target * megaquick * persona=fast-strong\n")

    if os.environ.get("LEGIT_EDGE_DEMO_OUT"):
        out = Path(os.environ["LEGIT_EDGE_DEMO_OUT"])
    else:
        out = RESULTS_DIR / f"demo-{time.strftime('%Y%m%d-%H%M%S')}"

    run_cmd(
        target="mock", mode="megaquick", thermal="MAXN",
        persona="fast-strong",
        workloads="math,reasoning,tooluse",
        models_arg="",
        out=out,
    )

    table = Table(title="Per-cell summary", show_header=True, header_style="bold")
    table.add_column("Model")
    table.add_column("Workload")
    table.add_column("Cap", justify="right")
    table.add_column("p50 (s)", justify="right")
    table.add_column("p99 (s)", justify="right")
    for f in sorted(out.rglob("*.json")):
        d = json.loads(f.read_text())
        cell = d["cell_id"].split("|")
        cap = f"{d['capability']:.2f}"
        lat = d["metrics"]["latency"]
        table.add_row(cell[0], cell[2], cap, f"{lat['p50']:.2f}", f"{lat['p99']:.2f}")
    console.print(table)
    console.print(f"\nResults written to [cyan]{out}[/]")
    console.print(
        "\n[dim]Try [bold]legit-edge run mock --persona=thermal-drift --mode=standard "
        "--thermal=throttled[/bold] to see what the framework catches in a worst-case profile.[/dim]"
    )
