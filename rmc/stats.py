"""Bootstrap statistics for benchmark reports (WikiSkill-style significance)."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class BootstrapCI:
    mean: float
    low: float
    high: float
    n: int
    iterations: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "mean": self.mean,
            "low": self.low,
            "high": self.high,
            "n": self.n,
            "iterations": self.iterations,
        }


@dataclass(frozen=True)
class PairedBootstrapTest:
    mean_a: float
    mean_b: float
    delta: float
    p_value: float
    significant: bool
    iterations: int

    def as_dict(self) -> dict[str, float | int | bool]:
        return {
            "mean_a": self.mean_a,
            "mean_b": self.mean_b,
            "delta": self.delta,
            "p_value": self.p_value,
            "significant": self.significant,
            "iterations": self.iterations,
        }


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def bootstrap_ci(
    values: Sequence[float],
    *,
    iterations: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> BootstrapCI:
    """Percentile bootstrap CI for the mean of ``values``."""
    if not values:
        return BootstrapCI(0.0, 0.0, 0.0, 0, iterations)
    rng = random.Random(seed)
    n = len(values)
    means: list[float] = []
    for _ in range(iterations):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(mean(sample))
    means.sort()
    lo_idx = int((alpha / 2) * iterations)
    hi_idx = int((1 - alpha / 2) * iterations) - 1
    lo_idx = max(0, min(lo_idx, len(means) - 1))
    hi_idx = max(0, min(hi_idx, len(means) - 1))
    return BootstrapCI(
        mean=mean(values),
        low=means[lo_idx],
        high=means[hi_idx],
        n=n,
        iterations=iterations,
    )


def paired_bootstrap_test(
    a: Sequence[float],
    b: Sequence[float],
    *,
    iterations: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> PairedBootstrapTest:
    """Paired bootstrap test: H0 delta = mean(a) - mean(b) <= 0 (one-sided for a > b).

    WikiSkill uses paired bootstrap at p<0.05 for Table 1 bolding.
    """
    if len(a) != len(b) or not a:
        return PairedBootstrapTest(0.0, 0.0, 0.0, 1.0, False, iterations)
    rng = random.Random(seed)
    deltas = [x - y for x, y in zip(a, b)]
    observed = mean(deltas)
    n = len(deltas)
    count = 0
    for _ in range(iterations):
        resampled = [deltas[rng.randrange(n)] for _ in range(n)]
        if mean(resampled) <= 0:
            count += 1
    p_value = count / iterations
    return PairedBootstrapTest(
        mean_a=mean(a),
        mean_b=mean(b),
        delta=observed,
        p_value=p_value,
        significant=observed > 0 and p_value < alpha,
        iterations=iterations,
    )


def accuracy_from_bits(bits: Iterable[bool]) -> float:
    rows = list(bits)
    return sum(1 for b in rows if b) / len(rows) if rows else 0.0
