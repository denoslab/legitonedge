"""Load named deployment profiles (configs/profiles.yaml) -> dimension weights."""
from __future__ import annotations
from pathlib import Path
import yaml

_PROFILES = Path(__file__).resolve().parent.parent.parent / "configs" / "profiles.yaml"


def load_profiles(path: Path | None = None) -> dict[str, dict[str, float]]:
    p = path or _PROFILES
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    return {name: {k: float(v) for k, v in w.items()} for name, w in data["profiles"].items()}
