"""`legit-edge run <target>` - mock | jetson | spark."""
from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console

from ..adapter import MockAdapter, OllamaAdapter
from ..config import load_tiers, load_models
from ..paths import RESULTS_DIR, DATA_DIR
from ..personas import get_persona
from ..report import to_jsonld, to_traces_jsonl
from ..runner import run_cell, run_multiturn_cell
from ..workload import (
    MathWorkload, ReasoningWorkload, ToolUseWorkload, MultiTurnToolWorkload,
)
from .checks import check_datasets, check_ollama

console = Console()

ALL_WORKLOADS = {
    "math": MathWorkload,
    "reasoning": ReasoningWorkload,
    "tooluse": ToolUseWorkload,
    "tooluse_mt": MultiTurnToolWorkload,
}


def _make_workloads(names: list[str], mode: str) -> dict[str, object]:
    suffix = "megaquick" if mode == "megaquick" else "standard"
    out = {}
    for name in names:
        cls = ALL_WORKLOADS[name]
        out[name] = cls(jsonl_path=DATA_DIR / f"{name}_{suffix}.jsonl")
    return out


def _make_energy_monitor(target: str):
    from ..telemetry import MockEnergyMonitor, TegraStatsEnergyMonitor, NvidiaSmiEnergyMonitor
    if target == "mock":
        return MockEnergyMonitor()
    tiers = load_tiers()
    if target == "jetson":
        return TegraStatsEnergyMonitor(ssh_target=tiers["jetson"].ssh_target, poll_ms=100)
    if target == "spark":
        return NvidiaSmiEnergyMonitor(ssh_target=tiers["spark"].ssh_target, poll_ms=100)
    return MockEnergyMonitor()


def _make_adapter(target: str, model_key: str, model_cfg, persona_name: Optional[str]):
    if target == "mock":
        p = get_persona(persona_name or "fast-strong")
        return MockAdapter.from_persona(p, tier="mock", name=model_key)
    tiers = load_tiers()
    if target not in tiers:
        raise typer.BadParameter(f"unknown target '{target}'")
    base = tiers[target].base_url
    return OllamaAdapter(
        model_tag=model_cfg.ollama_tag[target],
        tier=target,
        model_key=model_key,
        base_url=base,
    )


def run(
    target: str = typer.Argument(..., help="mock | jetson | spark"),
    mode: str = typer.Option("megaquick", "--mode", help="megaquick | standard"),
    thermal: str = typer.Option("MAXN", "--thermal", help="MAXN | throttled (standard only)"),
    persona: Optional[str] = typer.Option(None, "--persona", help="mock-only persona name"),
    workloads: str = typer.Option("math,reasoning,tooluse,tooluse_mt", "--workloads"),
    models_arg: str = typer.Option("", "--models", help="comma list; empty = all"),
    out: Optional[Path] = typer.Option(None, "--out", help="results dir override"),
    confidence: bool = typer.Option(False, "--confidence", help="append a 'Confidence: N%' elicitation; parse it into per-instance confidence (for ECE)"),
    max_tokens: int = typer.Option(512, "--max-tokens", help="num_predict per generation (raise for thermal-stress runs)"),
    repeat: int = typer.Option(1, "--repeat", help="repeat each cell k times for run-to-run determinism; writes __run{j} files when k>1"),
) -> None:
    """Execute the framework against mock | jetson | spark."""
    # coerce OptionInfo → int when called programmatically (e.g. demo.py)
    repeat = repeat if isinstance(repeat, int) else int(repeat.default)
    if mode not in {"megaquick", "standard"}:
        raise typer.BadParameter("--mode must be megaquick or standard")
    if thermal not in {"MAXN", "throttled"}:
        raise typer.BadParameter("--thermal must be MAXN or throttled")
    if persona is not None:
        get_persona(persona)  # raises if unknown

    # Preflight: for jetson/spark, require Ollama reachable before any cell starts.
    if target != "mock":
        tiers = load_tiers()
        if target not in tiers:
            raise typer.BadParameter(f"unknown target '{target}'")
        c = check_ollama(tiers[target].base_url, tier=target)
        if c.status != "ok":
            console.print(f"[red]Preflight failed:[/] {c.detail}")
            if c.fix_hint:
                console.print(f"  fix: {c.fix_hint}")
            raise typer.Exit(code=1)

    # Preflight: workload JSONLs must exist for any target (mock or hardware).
    # Without this, missing data/workloads/*.jsonl crashes mid-loop with a bare
    # FileNotFoundError. Reuse the same check `doctor` uses so the hint is consistent.
    ds_check = check_datasets()
    if ds_check.status != "ok":
        console.print(f"[red]Preflight failed:[/] {ds_check.detail}")
        if ds_check.fix_hint:
            console.print(f"  fix: {ds_check.fix_hint}")
        raise typer.Exit(code=1)

    wl_names = [w.strip() for w in workloads.split(",") if w.strip()]
    unknown = set(wl_names) - set(ALL_WORKLOADS)
    if unknown:
        raise typer.BadParameter(f"unknown workloads: {unknown}")

    models = load_models()
    if models_arg:
        keep = {m.strip() for m in models_arg.split(",") if m.strip()}
        unknown_m = keep - set(models)
        if unknown_m:
            raise typer.BadParameter(f"unknown models: {unknown_m}")
        models = {k: v for k, v in models.items() if k in keep}

    wls = _make_workloads(wl_names, mode)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = out or (RESULTS_DIR / f"{target}-{mode}-{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    console.print(
        f"Running [bold]{target}[/] / mode={mode} / thermal={thermal} "
        f"/ {len(models)} models * {len(wls)} workloads = {len(models) * len(wls)} cells"
    )
    for model_key, mcfg in models.items():
        adapter = _make_adapter(target, model_key, mcfg, persona)
        for wname, wl in wls.items():
            n = len(wl.instances())
            for j in range(repeat):
                energy = _make_energy_monitor(target)   # fresh monitor per repeat
                suffix = f"__run{j}" if repeat > 1 else ""
                console.print(f"  Cell {model_key} | {target} | {wname}{suffix} | n={n} ...")
                if wname == "tooluse_mt":
                    # The multi-step agentic workload drives a per-turn ReAct loop and is
                    # scored by the state-diff oracle; confidence/ECE does not apply to it.
                    cell = run_multiturn_cell(
                        adapter=adapter, workload=wl,
                        n_instances=n, mode=mode, thermal=thermal,
                        gen_tokens=max_tokens, energy_monitor=energy,
                    )
                else:
                    cell = run_cell(
                        adapter=adapter, workload=wl,
                        n_instances=n, mode=mode, thermal=thermal,
                        confidence=confidence, gen_tokens=max_tokens,
                        energy_monitor=energy,
                    )
                out_file = out_dir / f"{model_key}__{target}__{wname}{suffix}.json"
                traces_file = out_dir / f"{model_key}__{target}__{wname}{suffix}.traces.jsonl"
                out_file.write_text(json.dumps(to_jsonld(cell), indent=2))
                to_traces_jsonl(cell.per_instance, traces_file)
    console.print(f"Results written to [cyan]{out_dir}[/]")
