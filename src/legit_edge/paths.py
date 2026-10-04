"""Project-root autodetect. The single source of truth for filesystem locations."""
from __future__ import annotations
import os
from pathlib import Path


def _is_project_root(p: Path) -> bool:
    return (p / "pyproject.toml").exists() and (p / "src" / "legit_edge").exists()


def find_project_root() -> Path:
    """Resolve the legit-edge project root.

    Order:
      1. $LEGIT_EDGE_ROOT env var (explicit override)
      2. Walk up from __file__ looking for pyproject.toml + src/legit_edge/
      3. Walk up from cwd looking for the same pattern
      4. Raise RuntimeError with remediation
    """
    if env := os.environ.get("LEGIT_EDGE_ROOT"):
        p = Path(env).resolve()
        if _is_project_root(p):
            return p
        raise RuntimeError(
            f"LEGIT_EDGE_ROOT={env} does not point to a legit-edge project "
            "(missing pyproject.toml or src/legit_edge/)"
        )

    for start in (Path(__file__).resolve(), Path.cwd().resolve()):
        for p in [start, *start.parents]:
            if _is_project_root(p):
                return p

    raise RuntimeError(
        "Could not find legit-edge project root. "
        "Either run from a checkout of the repo, install the package editable "
        "(uv sync --extra dev), or set LEGIT_EDGE_ROOT=/path/to/repo"
    )


PROJECT_ROOT = find_project_root()
DATA_DIR = PROJECT_ROOT / "data" / "workloads"
RESULTS_DIR = PROJECT_ROOT / "results"
CONFIGS_DIR = PROJECT_ROOT / "configs"
LOGS_DIR = PROJECT_ROOT / "logs"
