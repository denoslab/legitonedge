"""Reporter: CellResult -> JSON-LD + Markdown."""
from __future__ import annotations
import json
from pathlib import Path
from .runner import CellResult


def to_jsonld(c: CellResult) -> dict:
    return {
        "@context": "https://legit-edge.example/v1",
        "cell_id": c.cell_id,
        "mode": c.mode,
        "thermal": c.thermal,
        "capability": c.capability,
        # Cell-level GPU-domain (overhead-subtracted) dynamic energy, J.
        # None on Spark (nvidia-smi is already GPU-domain) / older captures without the compute rail.
        "energy_gpu_domain_j": c.energy_gpu_domain_j,
        "metrics": {
            c.latency.name: c.latency.value,
            c.throughput.name: c.throughput.value,
            c.energy.name: c.energy.value,
        },
        "ci_95": {
            c.latency.name: c.latency.ci_95,
            c.throughput.name: c.throughput.ci_95,
            c.energy.name: c.energy.ci_95,
        },
        "n": {
            c.latency.name: c.latency.n,
            c.throughput.name: c.throughput.n,
            c.energy.name: c.energy.n,
        },
    }


def to_markdown(c: CellResult) -> str:
    lines = [
        f"## Cell {c.cell_id}",
        f"- Mode: {c.mode}; Thermal: {c.thermal}",
        f"- Capability: {c.capability:.3f}",
        (
            f"- Latency p50/p95/p99 (s): "
            f"{c.latency.value['p50']:.3f} / {c.latency.value['p95']:.3f} / {c.latency.value['p99']:.3f}"
        ),
        (
            f"- Throughput slope (tps/min): "
            f"{c.throughput.value.get('slope_tps_per_min', float('nan')):.3f}"
        ),
        f"- Energy J/correct: {c.energy.value['j_per_correct']:.2f}",
    ]
    return "\n".join(lines)


def to_traces_jsonl(per_instance: list[dict], path) -> None:
    """Write per-instance traces, one JSON object per line, in instance_index order.

    Captures per-instance data on disk for post-hoc paired-Δ analysis, bootstrap
    CI re-computation and verbalized-confidence ECE. JSON-Lines so a partial run
    leaves a valid streamed file.
    """
    Path(path).write_text(
        "\n".join(json.dumps(row, default=str) for row in per_instance) + "\n",
        encoding="utf-8",
    )
