"""Tests for the three-tier validation gate (Tier 1 / 2 / 3)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from packages.backtests.validation import (
    BLOCK_SIZE_DAYS,
    STRESS_WINDOWS,
    _block_bootstrap,
    tier2_stress,
    tier3_synthetic,
)
from packages.strategies import TrendFollowing


def _synthetic_prices(n: int = 2600, seed: int = 11, drift: float = 0.0005) -> pd.DataFrame:
    """Build a 10y-ish daily price panel with mild positive drift."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2014-01-02", periods=n)
    cols = ["SPY", "QQQ"]
    rets = rng.normal(drift, 0.01, size=(n, len(cols)))
    prices = 100 * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(prices, index=idx, columns=cols)


def test_block_bootstrap_preserves_length_and_distribution():
    rng = np.random.default_rng(0)
    rets = rng.normal(0.0005, 0.01, size=2000)
    sample = _block_bootstrap(rets, horizon=750, block_size=20, rng=rng)
    assert sample.shape == (750,)
    # Mean of bootstrap sample should be within a few stderrs of the source mean.
    assert abs(float(np.mean(sample)) - float(np.mean(rets))) < 0.005


def test_block_bootstrap_short_history_falls_back_to_iid():
    rng = np.random.default_rng(0)
    rets = rng.normal(0.0, 0.01, size=5)
    sample = _block_bootstrap(rets, horizon=100, block_size=20, rng=rng)
    assert sample.shape == (100,)
    # Every element must come from the source set.
    assert set(np.unique(sample)).issubset(set(np.unique(rets)))


def test_tier3_synthetic_runs_and_reports_metrics():
    prices = _synthetic_prices()
    report = tier3_synthetic(
        TrendFollowing(fast=20, slow=50),
        prices,
        paths=200,  # small for test speed
        block_size=BLOCK_SIZE_DAYS,
    )
    assert report.strategy == "trend-following"
    assert 0.0 <= report.metrics["synthetic_positive_ratio"] <= 1.0
    assert report.metrics["synthetic_paths"] == 200
    assert report.metrics["block_size"] == BLOCK_SIZE_DAYS
    assert "synthetic_median_return" in report.metrics
    assert "synthetic_p05_return" in report.metrics


def test_tier3_synthetic_rejects_too_little_history():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2024-01-02", periods=10)
    prices = pd.DataFrame(
        100 + np.cumsum(rng.normal(0, 1, size=(10, 2)), axis=0),
        index=idx,
        columns=["SPY", "QQQ"],
    )
    report = tier3_synthetic(TrendFollowing(fast=2, slow=5), prices, paths=10)
    assert not report.passed
    assert any("too few" in r for r in report.reasons)


def test_tier2_stress_windows_are_skipped_when_history_absent():
    # Prices that don't overlap any stress window — should not crash.
    idx = pd.bdate_range("2024-01-02", periods=400)
    prices = pd.DataFrame(
        100 + np.cumsum(np.random.default_rng(0).normal(0, 1, size=(400, 2)), axis=0),
        index=idx,
        columns=["SPY", "QQQ"],
    )
    report = tier2_stress(TrendFollowing(fast=20, slow=50), prices)
    # No reasons because no windows applied.
    assert isinstance(report.metrics["stress_drawdowns"], dict)


def test_stress_windows_constant_has_five_entries():
    assert set(STRESS_WINDOWS.keys()) == {"2008", "2015", "2018", "2020", "2022"}
