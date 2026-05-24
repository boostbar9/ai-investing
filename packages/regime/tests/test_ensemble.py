"""Tests for the regime-gated ensemble combiner."""

from __future__ import annotations

import numpy as np
import pandas as pd

from packages.regime.ensemble import (
    DEFAULT_REGIME_WEIGHTS,
    RegimeGatedEnsemble,
    RegimeWeights,
    backtest_ensemble,
    detect_regime_series,
    regime_series_to_daily,
)


def _toy_prices(n: int = 300, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-02", periods=n)
    spy = 100 * np.exp(np.cumsum(rng.normal(0.0006, 0.012, n)))
    qqq = 100 * np.exp(np.cumsum(rng.normal(0.0007, 0.015, n)))
    iwm = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.014, n)))
    return pd.DataFrame({"SPY": spy, "QQQ": qqq, "IWM": iwm}, index=idx)


def _toy_regime_series(idx: pd.DatetimeIndex) -> pd.Series:
    # Half bull, half chop -- exercises both branches.
    half = len(idx) // 2
    labels = ["bull"] * half + ["chop"] * (len(idx) - half)
    return pd.Series(labels, index=idx, name="regime")


class _DummyStrategy:
    """A predictable strategy: always-on weight=1/N across symbols."""

    def __init__(self, scale: float = 1.0):
        self.scale = scale

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        n = prices.shape[1]
        return pd.DataFrame(self.scale / n, index=prices.index, columns=prices.columns)


def test_regime_weights_lookup_defaults():
    rw = RegimeWeights()
    assert rw.get("trend-following", "bull") == 1.0
    assert rw.get("trend-following", "crisis") == 0.0
    assert rw.get("mean-reversion", "chop") == 1.0
    # Unknown strategy -> defensive: full only in bull.
    assert rw.get("unknown", "bull") == 1.0
    assert rw.get("unknown", "chop") == 0.0


def test_default_weights_have_crisis_zero():
    """No strategy is allowed to run during crisis (§13)."""
    for name, regime_map in DEFAULT_REGIME_WEIGHTS.items():
        assert regime_map["crisis"] == 0.0, f"{name} must be 0 in crisis"


def test_default_weights_in_unit_interval():
    """Multipliers are damping factors only, never amplifiers."""
    for name, regime_map in DEFAULT_REGIME_WEIGHTS.items():
        for regime, m in regime_map.items():
            assert 0.0 <= m <= 1.0, f"{name}/{regime} = {m} not in [0,1]"


def test_regime_series_to_daily_ffill():
    sparse_idx = pd.to_datetime(["2024-01-02", "2024-01-04", "2024-01-08"])
    regimes = pd.Series(["bull", "chop", "bear"], index=sparse_idx)
    target = pd.bdate_range("2024-01-02", "2024-01-09")
    out = regime_series_to_daily(regimes, target)
    # 2024-01-03 should fill from 2024-01-02 ("bull").
    assert out.loc["2024-01-03"] == "bull"
    # 2024-01-05 should fill from 2024-01-04 ("chop").
    assert out.loc["2024-01-05"] == "chop"


def test_ensemble_zero_in_crisis():
    prices = _toy_prices()
    regimes = pd.Series("crisis", index=prices.index)
    ensemble = RegimeGatedEnsemble(
        strategies={"trend-following": _DummyStrategy(), "mean-reversion": _DummyStrategy()}
    )
    w = ensemble.generate_signals(prices, regimes)
    assert (w == 0.0).all().all(), "all weights must be 0 in crisis"


def test_ensemble_caps_gross_at_one():
    """Three full-on strategies summed must scale down to ≤1.0 gross."""
    prices = _toy_prices()
    regimes = pd.Series("bull", index=prices.index)
    ensemble = RegimeGatedEnsemble(
        strategies={
            "trend-following": _DummyStrategy(scale=1.0),  # bull mult=1.0
            "sector-rotation": _DummyStrategy(scale=1.0),  # bull mult=0.8
            "mean-reversion": _DummyStrategy(scale=1.0),    # bull mult=0.5
        },
        max_gross=1.0,
    )
    w = ensemble.generate_signals(prices, regimes)
    gross = w.abs().sum(axis=1)
    assert gross.max() <= 1.0 + 1e-9


def test_ensemble_uses_regime_specific_gates():
    """Verify that a chop bar damps trend-following more than bull does."""
    prices = _toy_prices(n=20)
    # First half bull, second half chop
    half = 10
    regimes = pd.Series(
        ["bull"] * half + ["chop"] * (len(prices) - half),
        index=prices.index,
    )
    ensemble = RegimeGatedEnsemble(
        strategies={"trend-following": _DummyStrategy(scale=1.0)},
    )
    w = ensemble.generate_signals(prices, regimes)
    bull_gross = w.iloc[:half].abs().sum(axis=1).mean()
    chop_gross = w.iloc[half:].abs().sum(axis=1).mean()
    # Trend's bull mult=1.0, chop mult=0.3 -> chop weights should be smaller.
    assert chop_gross < bull_gross


def test_explain_returns_one_row_per_active_strategy_symbol():
    prices = _toy_prices(n=50)
    regimes = pd.Series("bull", index=prices.index)
    ensemble = RegimeGatedEnsemble(
        strategies={
            "trend-following": _DummyStrategy(),
            "mean-reversion": _DummyStrategy(),
        }
    )
    explanation = ensemble.explain(prices, regimes)
    # 2 strategies x 3 symbols (all non-zero) = 6 rows.
    assert len(explanation) == 6
    assert set(explanation["strategy"]) == {"trend-following", "mean-reversion"}
    assert set(explanation["symbol"]) == {"SPY", "QQQ", "IWM"}
    assert (explanation["regime"] == "bull").all()


def test_detect_regime_series_produces_per_bar_labels():
    n = 200
    idx = pd.bdate_range("2024-01-02", periods=n)
    rng = np.random.default_rng(42)
    spy = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, n))), index=idx)
    vix = pd.Series(rng.uniform(12, 22, n), index=idx)
    breadth = pd.Series(rng.uniform(0.4, 0.6, n), index=idx)
    out = detect_regime_series(spy, vix, breadth)
    # 20-day SMA warmup -> at least 150 labels emitted.
    assert len(out) >= 150
    assert set(out.unique()).issubset({"bull", "bear", "chop", "crisis"})


def test_backtest_ensemble_returns_metrics():
    prices = _toy_prices(n=200)
    regimes = _toy_regime_series(prices.index)
    out = backtest_ensemble(
        prices,
        regimes,
        strategies=[
            ("trend-following", _DummyStrategy()),
            ("mean-reversion", _DummyStrategy()),
        ],
    )
    assert "sharpe" in out
    assert "max_dd" in out
    assert out["n_days"] == len(prices)
