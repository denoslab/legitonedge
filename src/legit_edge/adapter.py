"""Model adapter abstraction. Same interface across Ollama / vLLM / Mock."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Protocol, TYPE_CHECKING
import math
import os
import time

import numpy as np

if TYPE_CHECKING:
    from .personas import Persona


@dataclass
class AdapterResponse:
    text: str
    latency_s: float
    input_tokens: int
    output_tokens: int
    raw: dict = field(default_factory=dict)


@dataclass
class ModelMetadata:
    name: str
    tier: str             # "jetson" | "spark" | "mock"
    quantization: str     # "Q4_K_M" | "FP16" | etc.
    runtime: str          # "ollama" | "vllm" | "mock"


class ModelAdapter(Protocol):
    def generate(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        expected: str | None = None,
    ) -> AdapterResponse: ...
    def metadata(self) -> ModelMetadata: ...


@dataclass
class MockAdapter:
    canned: str = "ok"
    tier: str = "mock"
    persona: Optional["Persona"] = None
    name: Optional[str] = None  # overrides metadata().name when set (used by mock cells)
    _rng: Optional[np.random.Generator] = None

    @classmethod
    def from_persona(
        cls, persona: "Persona", *, tier: str = "mock", seed: int = 42,
        name: Optional[str] = None,
    ) -> "MockAdapter":
        a = cls(canned="", tier=tier, persona=persona, name=name)
        a._rng = np.random.default_rng(seed)
        return a

    def generate(
        self, system: str, user: str, *,
        max_tokens: int = 512, temperature: float = 0.0,
        expected: str | None = None, thermal: str = "MAXN",
    ) -> AdapterResponse:
        if self.persona is None:
            t0 = time.perf_counter()
            time.sleep(0.001)
            return AdapterResponse(
                text=self.canned,
                latency_s=time.perf_counter() - t0,
                input_tokens=max(1, len(user.split())),
                output_tokens=max(1, len(self.canned.split())),
            )

        rng = self._rng if self._rng is not None else np.random.default_rng(42)
        # lognormal: median = latency_mean_s, sigma chosen so p99 ≈ median * p99_factor
        # ln(p99) - ln(median) = z_0.99 * sigma  =>  sigma = ln(p99_factor) / 2.326
        if self.persona.latency_p99_factor > 1:
            sigma = math.log(self.persona.latency_p99_factor) / 2.326
        else:
            sigma = 0.05
        mu = math.log(self.persona.latency_mean_s)
        latency = float(rng.lognormal(mean=mu, sigma=sigma))

        drop = self.persona.thermal_capability_drop if thermal == "throttled" else 0.0
        eff_p = max(0.0, min(1.0, self.persona.capability_p - drop))
        correct = rng.random() < eff_p

        if correct and expected is not None:
            text = f"The answer is {expected}."
        elif not correct and expected is not None:
            try:
                v = float(expected)
                text = f"The answer is {v + 1}."
            except ValueError:
                text = "The answer is unknown."
        else:
            text = "The answer is unknown."

        output_tokens = max(1, int(rng.poisson(self.persona.output_tokens_mean)))

        return AdapterResponse(
            text=text,
            latency_s=latency,
            input_tokens=max(1, len(user.split())),
            output_tokens=output_tokens,
        )

    def metadata(self) -> ModelMetadata:
        if self.name is not None:
            name = self.name
        elif self.persona is not None:
            name = self.persona.name
        else:
            name = "mock"
        return ModelMetadata(name=name, tier=self.tier, quantization="none", runtime="mock")


import requests as _req


@dataclass
class OllamaAdapter:
    model_tag: str
    tier: str = "unknown"
    model_key: str = ""
    base_url: str = "http://localhost:11434"
    # Wall-clock per /api/generate call. 300s default gives ~2x margin for
    # cold-load of a 4.9 GB model under DVFS-governed clocks on the Jetson;
    # Spark + MAXN runs never come close. Override per-run via
    # LEGIT_EDGE_OLLAMA_TIMEOUT=<seconds>. (A measured ~74 s Jetson cold-load plus
    # generation could exceed an earlier 120 s ceiling.)
    timeout: float = field(
        default_factory=lambda: float(os.environ.get("LEGIT_EDGE_OLLAMA_TIMEOUT", "300.0"))
    )

    def generate(
        self, system: str, user: str, *,
        max_tokens: int = 512, temperature: float = 0.0,
        expected: str | None = None,
        **_: object,
    ) -> AdapterResponse:
        body = {
            "model": self.model_tag,
            "prompt": (system + "\n\n" + user) if system else user,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens, "seed": 42},
        }
        t0 = time.perf_counter()
        r = _req.post(f"{self.base_url}/api/generate", json=body, timeout=self.timeout)
        elapsed = time.perf_counter() - t0
        r.raise_for_status()
        d = r.json()
        return AdapterResponse(
            text=d["response"],
            latency_s=elapsed,
            input_tokens=d.get("prompt_eval_count", 0),
            output_tokens=d.get("eval_count", 0),
            raw=d,
        )

    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            name=self.model_key or self.model_tag,
            tier=self.tier,
            quantization="Q4_K_M",
            runtime="ollama",
        )
