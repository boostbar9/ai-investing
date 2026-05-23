"""Tests for the weekly walk-forward retune."""
from __future__ import annotations

import numpy as np
import pandas as pd

from packages.backtests.walk_forward import (
    DEFAULT_GRID,
    ParamSet,
    _best_params_in_sample,
    equity_from_signal_strategy,
    load_champion,
    run_walk_forward,
    save_champion,
)


def _trending_prices(n: int = 700, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    # Strong trend + noise — should let MA-crossover earn positive Sharpe
    drift = np.linspace(0, 0.4, n)
    noise = rng.normal(0, 0.01, n)
    returns = drift / n + noise
    prices = 100 * np.cumprod(1 + returns)
    return pd.Series(prices, index=pd.date_range("2022-01-01", periods=n, freq="B"))


def test_equity_curve_starts_near_one():
    prices = _trending_prices(400)
    eq = equity_from_signal_strategy(prices, ParamSet())
    # First non-NaN equity value should be 1.0 (cumprod of zeros = 1)
    assert abs(eq.dropna().iloc[0] - 1.0) < 1e-6


def test_equity_curve_short_series_returns_flat():
    prices = pd.Series([100.0, 101.0])
    eq = equity_from_signal_strategy(prices, ParamSet())
    assert len(eq) == 1


def test_best_params_picks_from_grid():
    prices = _trending_prices(600)
    best, sharpe = _best_params_in_sample(prices, DEFAULT_GRID)
    assert best in DEFAULT_GRID
    assert isinstance(sharpe, float)


def test_run_walk_forward_insufficient_history():
    prices = _trending_prices(100)
    result = run_walk_forward(prices, champion=ParamSet())
    assert result.promoted is False
    assert any("insufficient history" in r for r in result.reasons)


def test_run_walk_forward_returns_verdict():
    prices = _trending_prices(700)
    result = run_walk_forward(
        prices, champion=ParamSet(fast_window=10, slow_window=200, zscore_threshold=1.5)
    )
    # Don't assert promotion (depends on randomness) — just verify shape
    assert isinstance(result.promoted, bool)
    assert isinstance(result.challenger, ParamSet)
    assert isinstance(result.metrics, dict)


def test_save_and_load_champion_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_PARAMS_ROOT", str(tmp_path))
    p = ParamSet(fast_window=15, slow_window=75, zscore_threshold=0.75)
    save_champion(p, source="test")
    loaded = load_champion()
    assert loaded == p


def test_load_champion_missing_file_returns_default(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_PARAMS_ROOT", str(tmp_path))
    assert load_champion() == ParamSet()


def test_load_champion_corrupt_file_returns_default(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_PARAMS_ROOT", str(tmp_path))
    (tmp_path / "champion.json").write_text("not json {{{")
    assert load_champion() == ParamSet()
