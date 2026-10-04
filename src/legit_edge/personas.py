"""Mock-adapter personas. Each knob maps to one reliability dimension."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class Persona:
    name: str
    description: str
    capability_p: float              # P(correct answer)
    latency_mean_s: float            # median wall-clock seconds
    latency_p99_factor: float        # p99 = median * this
    tps_drift_per_min: float         # tokens/sec/min slope (negative = slowdown)
    thermal_capability_drop: float   # extra capability drop when thermal=throttled
    energy_watts: float              # constant power draw for MockEnergyMonitor
    confidence_bias: float           # 0 = well-calibrated; positive = overconfident
    output_tokens_mean: int          # Poisson mean for output token count


PERSONAS: dict[str, Persona] = {
    "fast-strong": Persona(
        name="fast-strong",
        description="High capability, low latency, stable. The dream model.",
        capability_p=0.85,
        latency_mean_s=0.5,
        latency_p99_factor=1.5,
        tps_drift_per_min=0.0,
        thermal_capability_drop=0.05,
        energy_watts=5.0,
        confidence_bias=0.05,
        output_tokens_mean=64,
    ),
    "slow-strong": Persona(
        name="slow-strong",
        description="High capability but slow; penalized by R's latency factor.",
        capability_p=0.85,
        latency_mean_s=5.0,
        latency_p99_factor=3.0,
        tps_drift_per_min=0.0,
        thermal_capability_drop=0.05,
        energy_watts=15.0,
        confidence_bias=0.05,
        output_tokens_mean=64,
    ),
    "fast-weak": Persona(
        name="fast-weak",
        description="Quick hallucinator: low capability, bad ECE, fast.",
        capability_p=0.45,
        latency_mean_s=0.4,
        latency_p99_factor=1.4,
        tps_drift_per_min=0.0,
        thermal_capability_drop=0.05,
        energy_watts=4.0,
        confidence_bias=0.30,
        output_tokens_mean=48,
    ),
    "thermal-drift": Persona(
        name="thermal-drift",
        description="Looks fine on benchmark, fails in field. This is the H1 case.",
        capability_p=0.80,
        latency_mean_s=1.5,
        latency_p99_factor=5.0,
        tps_drift_per_min=-0.3,
        thermal_capability_drop=0.25,
        energy_watts=12.0,
        confidence_bias=0.10,
        output_tokens_mean=64,
    ),
}


def get_persona(name: str) -> Persona:
    try:
        return PERSONAS[name]
    except KeyError as e:
        raise KeyError(f"unknown persona '{name}'; available: {sorted(PERSONAS)}") from e
