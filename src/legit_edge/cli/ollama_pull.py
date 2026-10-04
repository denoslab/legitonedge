"""Streaming wrapper over Ollama's /api/pull. Yields parsed JSON events."""
from __future__ import annotations
import json
from typing import Iterator
import requests


def pull_model(base_url: str, model_tag: str, *, timeout: float = 1800.0) -> Iterator[dict]:
    """POST /api/pull with stream=True. Yields one dict per server event."""
    with requests.post(
        f"{base_url}/api/pull",
        json={"name": model_tag, "stream": True},
        stream=True,
        timeout=timeout,
    ) as r:
        r.raise_for_status()
        for raw in r.iter_lines():
            if not raw:
                continue
            try:
                yield json.loads(raw)
            except json.JSONDecodeError:
                continue
