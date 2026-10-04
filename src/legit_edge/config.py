"""Load tiers.yaml and models.yaml with env-var substitution."""
from __future__ import annotations
import os
import re
from dataclasses import dataclass
from pathlib import Path
import yaml

from .paths import CONFIGS_DIR


_ENV_RE = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)(?::-([^}]*))?\}")


def expand_env(value: str) -> str:
    """Replace ${VAR:-default} with os.environ[VAR] or default."""
    if not isinstance(value, str):
        return value

    def sub(m: re.Match[str]) -> str:
        return os.environ.get(m.group(1), m.group(2) or "")

    return _ENV_RE.sub(sub, value)


def _walk_expand(obj):
    if isinstance(obj, dict):
        return {k: _walk_expand(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk_expand(x) for x in obj]
    if isinstance(obj, str):
        return expand_env(obj)
    return obj


@dataclass(frozen=True)
class TierConfig:
    name: str
    host: str
    port: int
    quantization: str
    notes: str = ""
    ssh_user: str = ""

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def ssh_target(self) -> str:
        """SSH destination for energy telemetry: ``user@host``, or bare ``host``
        when no user is configured (ssh then falls back to the local username
        or a matching ``~/.ssh/config`` entry)."""
        return f"{self.ssh_user}@{self.host}" if self.ssh_user else self.host


@dataclass(frozen=True)
class ModelConfig:
    name: str
    ollama_tag: dict[str, str]
    context_len: int


def load_tiers(path: Path | None = None) -> dict[str, TierConfig]:
    p = path or (CONFIGS_DIR / "tiers.yaml")
    data = _walk_expand(yaml.safe_load(p.read_text()))
    return {
        name: TierConfig(name=name, **{**cfg, "port": int(cfg["port"])})
        for name, cfg in data["tiers"].items()
    }


def load_models(path: Path | None = None) -> dict[str, ModelConfig]:
    p = path or (CONFIGS_DIR / "models.yaml")
    data = yaml.safe_load(p.read_text())
    return {
        name: ModelConfig(name=name, **cfg)
        for name, cfg in data["models"].items()
    }
