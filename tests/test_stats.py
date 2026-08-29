"""Tests for bootstrap statistics."""

from __future__ import annotations

from rose.stats import bootstrap_ci, paired_bootstrap_test


def test_bootstrap_ci_contains_mean() -> None:
    values = [1.0, 0.0, 1.0, 1.0, 0.0]
    ci = bootstrap_ci(values, iterations=500, seed=1)
    assert ci.low <= ci.mean <= ci.high
    assert ci.n == 5


def test_paired_bootstrap_detects_improvement() -> None:
    a = [1.0, 1.0, 1.0, 1.0, 0.0]
    b = [0.0, 0.0, 0.0, 0.0, 0.0]
    test = paired_bootstrap_test(a, b, iterations=500, seed=2)
    assert test.delta > 0
    assert test.p_value < 0.05
