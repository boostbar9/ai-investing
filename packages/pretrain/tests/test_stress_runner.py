"""Tests for the stress-window replay."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from packages.backtests.walk_forward import ParamSet
from packages.pretrain.stress_runner import (
    DEFAULT_WINDOWS,
    StressMetrics,
    run_stress_windows,
)


@dataclass(frozen=True)
class _W:
    name: str
    start: str
    end: str
    description: str


def _synthetic_prices(start: str = "2007-01-01", n: int = 252 * 4) -> pd.Series:
    rng = np.random.default_rng(42)
    idx = pd.bdate_range(start=start, periods=n)
    rets = rng.normal(0.0005, 0.012, size=n)
    return pd.Series(100.0 * np.exp(np.cumsum(rets)), index=idx)


def test_default_windows_includes_canonical() -> None:
    names = {w.name for w in DEFAULT_WINDOWS}
    # Must include the three canonical crash windows.
    assert "2008-gfc" in names
    assert "2020-covid" in names
    assert "2022-rates" in names


def test_run_stress_windows_returns_one_per_window() -> None:
    prices = _synthetic_prices()
    windows = (
        _W("a", "2007-06-01", "2008-06-01", "win A"),
        _W("b", "2009-01-01", "2009-12-31", "win B"),
    )
    out = run_stress_windows(prices, ParamSet(), windows=windows)
    assert len(out) == 2
    assert {m.window for m in out} == {"a", "b"}
    for m in out:
        assert isinstance(m, StressMetrics)
        assert m.n_days > 0


def test_empty_window_returns_zero_metrics() -> None:
    prices = _synthetic_prices(n=100)
    # Window outside the data.
    far_future = (_W("future", "2099-01-01", "2099-12-31", "future"),)
    out = run_stress_windows(prices, ParamSet(), windows=far_future)
    assert len(out) == 1
    assert out[0].n_days == 0
    assert out[0].sharpe == 0.0
    assert out[0].max_dd == 0.0


def test_partial_window_within_data() -> None:
    prices = _synthetic_prices(start="2010-01-01", n=252 * 2)
    overlap = (_W("partial", "2010-06-01", "2099-01-01", "partial overlap"),)
    out = run_stress_windows(prices, ParamSet(), windows=overlap)
    assert out[0].n_days > 0
    assert out[0].n_days <= len(prices)


@pytest.mark.parametrize("params", [ParamSet(), ParamSet(fast_window=10, slow_window=50)])
def test_metrics_are_finite(params: ParamSet) -> None:
    prices = _synthetic_prices()
    windows = (_W("a", "2007-06-01", "2008-06-01", "w"),)
    out = run_stress_windows(prices, params, windows=windows)
    m = out[0]
    assert np.isfinite(m.sharpe)
    assert np.isfinite(m.max_dd)
    assert np.isfinite(m.cagr)
    assert m.max_dd >= 0.0  # convention: positive number
