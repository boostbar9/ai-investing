"""Tests for the production upgrades to TrendFollowing."""
from __future__ import annotations

import numpy as np
import pandas as pd

from packages.strategies import TrendFollowing


def _uptrend_then_crash(n: int = 500, crash_at: int = 350) -> pd.DataFrame:
    """Build a single-name price series: smooth uptrend, then sharp crash."""
    idx = pd.bdate_range("2022-01-01", periods=n)
    base = np.linspace(100.0, 200.0, n)
    if crash_at < n:
        base[crash_at:] = np.linspace(200.0, 120.0, n - crash_at)  # ~40% drop
    return pd.DataFrame({"SPY": base, "QQQ": base * 1.01}, index=idx)


def test_stop_loss_triggers_during_crash():
    prices = _uptrend_then_crash(n=500, crash_at=350)
    strat = TrendFollowing(fast=20, slow=50, stop_loss=0.10, max_gross=1.0)
    w = strat.generate_signals(prices)
    # After enough drawdown post-crash, weights should be flat (stopped out).
    crash_tail = w.iloc[400:]
    assert (crash_tail.sum(axis=1) == 0).any()


def test_vol_targeting_reduces_size_for_volatile_name():
    rng = np.random.default_rng(0)
    n = 500
    idx = pd.bdate_range("2022-01-01", periods=n)
    quiet = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.005, size=n)))
    loud = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.030, size=n)))
    prices = pd.DataFrame({"QUIET": quiet, "LOUD": loud}, index=idx)
    w = TrendFollowing(fast=20, slow=50, vol_target=0.10).generate_signals(prices)
    # Where both legs are active, QUIET should generally carry more weight.
    both_on = w[(w["QUIET"] > 0) & (w["LOUD"] > 0)]
    if len(both_on) > 5:
        assert both_on["QUIET"].mean() > both_on["LOUD"].mean()


def test_gross_exposure_cap_respected():
    rng = np.random.default_rng(1)
    n = 400
    idx = pd.bdate_range("2022-01-01", periods=n)
    cols = ["A", "B", "C", "D"]
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0.001, 0.01, size=(n, len(cols))), axis=0)),
        index=idx,
        columns=cols,
    )
    w = TrendFollowing(fast=10, slow=30, max_gross=0.8).generate_signals(prices)
    assert (w.sum(axis=1) <= 0.8 + 1e-9).all()


def test_trend_filter_blocks_downtrend_entries():
    # Falling slow SMA: no entries should fire even if fast > slow momentarily.
    n = 400
    idx = pd.bdate_range("2022-01-01", periods=n)
    falling = np.linspace(200.0, 100.0, n)
    prices = pd.DataFrame({"SPY": falling, "QQQ": falling}, index=idx)
    w = TrendFollowing(fast=20, slow=50).generate_signals(prices)
    # No long entries on a monotone decline.
    assert w.sum().sum() == 0
