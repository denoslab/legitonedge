"""Shared readiness checks used by `doctor` and per-target preflight."""
from __future__ import annotations
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import requests

from ..paths import CONFIGS_DIR, DATA_DIR


@dataclass(frozen=True)
class Check:
    name: str
    status: Literal["ok", "warn", "fail"]
    detail: str
    fix_hint: str | None = None


def check_python() -> Check:
    v = sys.version_info
    detail = f"{v.major}.{v.minor}.{v.micro}"
    if (v.major, v.minor) >= (3, 11):
        return Check("Python", "ok", f"{detail} (>=3.11 required)")
    return Check(
        "Python", "fail", f"{detail} (>=3.11 required)",
        fix_hint="Install Python 3.11+ and re-create the env: uv sync --extra dev",
    )


def check_uv() -> Check:
    if shutil.which("uv") is None:
        return Check(
            "uv", "fail", "uv not on PATH",
            fix_hint="install uv: see https://docs.astral.sh/uv/getting-started/installation/",
        )
    try:
        out = subprocess.check_output(["uv", "--version"], text=True).strip()
        return Check("uv", "ok", out)
    except subprocess.CalledProcessError as e:
        return Check("uv", "fail", str(e), fix_hint="reinstall uv")


def check_package() -> Check:
    try:
        import legit_edge
        p = Path(legit_edge.__file__)
        return Check(
            "Package", "ok",
            f"legit_edge {legit_edge.__version__} (editable, {p.parent})",
        )
    except ImportError as e:
        return Check(
            "Package", "fail", str(e),
            fix_hint="uv sync --extra dev",
        )


def check_configs() -> Check:
    missing = [
        f for f in ("models.yaml", "workloads.yaml", "tiers.yaml")
        if not (CONFIGS_DIR / f).exists()
    ]
    if missing:
        return Check(
            "Configs", "fail", f"missing: {missing}",
            fix_hint="git pull / restore the configs/ directory",
        )
    return Check("Configs", "ok", "models.yaml, workloads.yaml, tiers.yaml")


def check_datasets() -> Check:
    expected = [
        "math_megaquick.jsonl", "math_standard.jsonl",
        "reasoning_megaquick.jsonl", "reasoning_standard.jsonl",
        "tooluse_megaquick.jsonl", "tooluse_standard.jsonl",
        "tooluse_mt_megaquick.jsonl", "tooluse_mt_standard.jsonl",
    ]
    missing = [f for f in expected if not (DATA_DIR / f).exists()]
    if missing:
        return Check(
            "Datasets", "warn", f"{len(missing)} of {len(expected)} missing",
            fix_hint="legit-edge pin datasets",
        )
    counts = {f: sum(1 for _ in (DATA_DIR / f).open(encoding="utf-8")) for f in expected}
    summary = ", ".join(f"{f.replace('.jsonl','')} ({n})" for f, n in counts.items())
    return Check("Datasets", "ok", f"{len(expected)} pinned JSONLs: {summary}")


def check_ollama(base_url: str, *, tier: str | None = None) -> Check:
    label = f"Ollama @ {tier}" if tier else "Ollama"
    try:
        r = requests.get(base_url + "/", timeout=2)
        if r.status_code < 500:
            return Check(label, "ok", f"{base_url} reachable")
    except requests.exceptions.RequestException:
        pass
    fix = (
        f"on the {tier or 'target'} device: OLLAMA_HOST=0.0.0.0 ollama serve &"
        if tier else "start ollama serve"
    )
    return Check(label, "warn", f"{base_url} unreachable", fix_hint=fix)


def check_ina260() -> Check:
    try:
        import adafruit_ina260  # noqa: F401
        return Check("INA260", "ok", "adafruit_circuitpython_ina260 importable")
    except ImportError:
        return Check(
            "INA260", "warn", "adafruit_circuitpython_ina260 not importable",
            fix_hint="uv sync --extra energy  (Jetson only, in-prod)",
        )
