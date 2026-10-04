"""Raw-metric -> [0,1] reliability sub-scores for the LegitOnEdge composite.

Bounds pre-declared in PRE_REGISTRATION.md amendment 2026-05-30 so the composite
is not a post-hoc artifact. Lower-is-better metrics (latency, energy) map so that
meeting the reference -> 1.0 and 2x the reference -> 0.0.
"""
from __future__ import annotations

P99_LATENCY_SLA_S = 60.0
J_PER_CORRECT_SLA = 2000.0
THROUGHPUT_SLOPE_REF = 0.05


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def latency_score(p99_s: float, *, ref_s: float = P99_LATENCY_SLA_S) -> float:
    if ref_s <= 0:
        return 0.0
    return _clip01(1.0 - p99_s / (2.0 * ref_s))


def energy_score(j_per_correct: float, *, ref_j: float = J_PER_CORRECT_SLA) -> float:
    if j_per_correct == float("inf") or ref_j <= 0:
        return 0.0
    return _clip01(1.0 - j_per_correct / (2.0 * ref_j))


def throughput_score(slope_tps_per_min: float, *, ref: float = THROUGHPUT_SLOPE_REF) -> float:
    if slope_tps_per_min >= 0:
        return 1.0
    return _clip01(1.0 - abs(slope_tps_per_min) / ref)
