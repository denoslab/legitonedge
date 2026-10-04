"""Top-level `legit-edge` Typer app."""
from __future__ import annotations
import typer
from rich.console import Console

app = typer.Typer(
    name="legit-edge",
    help="LegitOnEdge: reliability benchmarking for agentic SLMs on edge hardware.",
    no_args_is_help=False,
)
console = Console()


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())
        console.print(
            "\n[dim]Try [bold]legit-edge demo[/bold] to see a 30-second mock run end-to-end.[/dim]"
        )


from .doctor import doctor as _doctor
app.command("doctor")(_doctor)

from .pin import pin_app
app.add_typer(pin_app, name="pin")

from .smoke import smoke as _smoke
app.command("smoke")(_smoke)

from .run import run as _run
app.command("run")(_run)

from .demo import demo as _demo
app.command("demo")(_demo)

from .report_cli import report as _report
app.command("report")(_report)
