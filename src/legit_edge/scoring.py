"""LegitOnEdge composite score = R * r * (1 - d) * capability * 100."""
from __future__ import annotations
import math


def legit_edge_score(
    *,
    capability: float,           # in [0,1]
    latency_score: float,        # in [0,1]
    throughput_score: float,     # in [0,1]
    energy_score: float,         # in [0,1]
    output_stability: float,     # in [0,1] (1 = no drift between MAXN and throttled)
    calibration_r: float,        # in [0.85, 1.0] per islegit.ai r convention
) -> float:
    """LegitOnEdge score = R * r * (1 - d) * capability, on a 0-100 scale.

    Where:
      - capability = sum(w_i * c_i) (workload accuracy, weights uniform here)
      - r = calibration_r (robustness factor)
      - d = 1 - output_stability (drift penalty)
      - R = geometric mean of edge-deployment-reliability sub-factors
            = (latency * throughput * energy)^(1/3)
    """
    R = (
        max(0.0, latency_score)
        * max(0.0, throughput_score)
        * max(0.0, energy_score)
    ) ** (1.0 / 3.0)
    r = max(0.0, min(1.0, calibration_r))
    d = max(0.0, min(1.0, 1.0 - output_stability))
    cap = max(0.0, min(1.0, capability))
    return float(R * r * (1.0 - d) * cap * 100.0)


def reliability_composite(
    *,
    latency_score: float,
    throughput_score: float,
    energy_score: float,
    output_stability: float,    # 1 = no MAXN->throttled drift
    calibration_r: float,       # 1 - ECE (1.0 neutral when dim-5 omitted)
    weights: dict | None = None,
) -> float:
    """Capability-EXCLUDED reliability composite = weighted geometric mean of the
    five normalized reliability dim scores, in [0,1]. This is the H3 scatter
    Y-axis; capability is the X-axis, so it must not enter here (else the
    correlation is trivially inflated). Profile weights re-rank cells."""
    dims = {
        "latency": latency_score, "throughput": throughput_score,
        "energy": energy_score, "output_stability": output_stability,
        "calibration": calibration_r,
    }
    if weights is None:
        weights = {k: 1.0 for k in dims}
    eps = 1e-6
    num = den = 0.0
    for k, v in dims.items():
        w = float(weights.get(k, 0.0))
        if w <= 0:
            continue
        num += w * math.log(max(eps, min(1.0, v)))
        den += w
    return float(math.exp(num / den)) if den > 0 else 0.0
