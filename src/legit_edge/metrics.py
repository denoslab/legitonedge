"""Reliability metrics. Each metric returns a MetricResult with point + bootstrap CI."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Protocol
import numpy as np


@dataclass
class MetricResult:
    name: str
    value: dict[str, float]
    ci_95: dict[str, tuple[float, float]] = field(default_factory=dict)
    n: int = 0


class Metric(Protocol):
    name: str

    def compute(self, samples: list[float], **ctx) -> MetricResult: ...


def _block_bootstrap(
    samples: np.ndarray,
    stat_fn,
    n_boot: int = 1000,
    block_size: int = 1,
    rng=None,
) -> tuple[float, float]:
    rng = rng or np.random.default_rng(42)
    n = len(samples)
    if n == 0:
        return (float("nan"), float("nan"))
    n_blocks = max(1, n // block_size)
    boots = []
    for _ in range(n_boot):
        starts = rng.integers(0, max(1, n - block_size + 1), size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block_size) for s in starts])
        idx = idx[idx < n][:n]
        boots.append(stat_fn(samples[idx]))
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


@dataclass
class LatencyMetric:
    name: str = "latency"
    bootstrap_n: int = 10_000
    block_size: int = 10

    def compute(self, samples: list[float], **ctx) -> MetricResult:
        arr = np.asarray(samples, dtype=float)
        if len(arr) == 0:
            return MetricResult(
                name=self.name,
                value={"p50": float("nan"), "p95": float("nan"), "p99": float("nan")},
                n=0,
            )
        value = {
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
        }
        ci = {
            k: _block_bootstrap(
                arr,
                lambda x, p=int(k[1:]): np.percentile(x, p),
                self.bootstrap_n,
                self.block_size,
            )
            for k in value
        }
        return MetricResult(name=self.name, value=value, ci_95=ci, n=len(arr))


@dataclass
class ThroughputStabilityMetric:
    """Slope of tokens-per-sec vs time over a sustained run."""

    name: str = "throughput_stability"
    bootstrap_n: int = 10_000

    def compute(self, samples, **ctx) -> MetricResult:
        # samples: list[(time_s, tokens_per_sec)]
        if not samples:
            return MetricResult(name=self.name, value={"slope_tps_per_min": float("nan")}, n=0)
        ts = np.array([t for t, _ in samples], dtype=float) / 60.0   # min
        tps = np.array([y for _, y in samples], dtype=float)
        slope, _ = np.polyfit(ts, tps, 1)
        rng = np.random.default_rng(42)
        boots = []
        for _ in range(self.bootstrap_n):
            idx = rng.integers(0, len(ts), size=len(ts))
            try:
                s, _ = np.polyfit(ts[idx], tps[idx], 1)
                boots.append(s)
            except Exception:
                pass
        ci = (
            (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
            if boots
            else (float("nan"), float("nan"))
        )
        return MetricResult(
            name=self.name,
            value={"slope_tps_per_min": float(slope)},
            ci_95={"slope_tps_per_min": ci},
            n=len(samples),
        )


@dataclass
class OutputStabilityMetric:
    """Paired delta between MAXN and throttled accuracy on the same instances."""

    name: str = "output_stability"
    bootstrap_n: int = 10_000

    def compute(self, samples, **ctx) -> MetricResult:
        # samples: list[(acc_maxn_per_instance, acc_throttled_per_instance)] - both 0/1
        if not samples:
            return MetricResult(name=self.name, value={"delta": float("nan")}, n=0)
        a = np.array([x for x, _ in samples], dtype=float)
        b = np.array([y for _, y in samples], dtype=float)
        delta = float(a.mean() - b.mean())
        rng = np.random.default_rng(42)
        diffs = a - b
        boots = [
            float(rng.choice(diffs, size=len(diffs), replace=True).mean())
            for _ in range(self.bootstrap_n)
        ]
        ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
        return MetricResult(
            name=self.name, value={"delta": delta}, ci_95={"delta": ci}, n=len(samples)
        )


@dataclass
class EnergyMetric:
    """Joules per correct output."""

    name: str = "energy_per_correct"

    def compute(self, samples, **ctx) -> MetricResult:
        # samples: list[(joules, correct_0_or_1)]
        if not samples:
            return MetricResult(name=self.name, value={"j_per_correct": float("nan")}, n=0)
        total_j = sum(j for j, _ in samples)
        n_correct = sum(c for _, c in samples)
        j_per_correct = total_j / n_correct if n_correct else float("inf")
        return MetricResult(
            name=self.name,
            value={
                "j_per_correct": float(j_per_correct),
                "j_per_attempt": total_j / len(samples),
            },
            n=len(samples),
        )


@dataclass
class CalibrationMetric:
    """Expected Calibration Error with equal-width bins."""

    name: str = "calibration"
    n_bins: int = 10

    def compute(self, samples, **ctx) -> MetricResult:
        if not samples:
            return MetricResult(name=self.name, value={"ece": float("nan")}, n=0)
        confs = np.array([c for c, _ in samples], dtype=float)
        corrects = np.array([y for _, y in samples], dtype=float)
        bins = np.linspace(0, 1, self.n_bins + 1)
        ece = 0.0
        for i in range(self.n_bins):
            lo, hi = bins[i], bins[i + 1]
            if i < self.n_bins - 1:
                mask = (confs >= lo) & (confs < hi)
            else:
                mask = (confs >= lo) & (confs <= hi)
            if mask.any():
                bin_acc = corrects[mask].mean()
                bin_conf = confs[mask].mean()
                ece += (mask.sum() / len(confs)) * abs(bin_acc - bin_conf)
        return MetricResult(name=self.name, value={"ece": float(ece)}, n=len(samples))
