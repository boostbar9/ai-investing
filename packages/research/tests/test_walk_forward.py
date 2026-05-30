"""Unit tests for the Phase 14 walk-forward backtest harness.

We test:

1. ``WalkForwardConfig`` -- input validation rejects pathological window sizes.
2. ``_annualised_sharpe`` / ``_max_drawdown`` -- numerical helpers against
   hand-computed values on tiny return vectors.
3. ``run_walk_forward`` -- on synthetic monotonic data the equal-weight
   signal should produce a positive cum-return and a finite Sharpe; on a
   zig-zag panel the harness should respect transaction costs by producing
   a smaller cum-return than the no-cost path.
4. Helper signal functions -- ``equal_weight_signal`` returns 1/N weights;
   ``momentum_signal`` selects top-N by trailing return.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from packages.research.walk_forward import (
    TRADING_DAYS_PER_YEAR,
    WalkForwardConfig,
    _annualised_sharpe,
    _max_drawdown,
    equal_weight_signal,
    momentum_signal,
    run_walk_forward,
)

# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_config_defaults_are_sane() -> None:
    cfg = WalkForwardConfig()
    assert cfg.train_size == 252
    assert cfg.test_size == 21
    assert cfg.step_size == 21
    assert cfg.transaction_cost_bps == 5.0


def test_config_rejects_tiny_train_window() -> None:
    """A train window of 10 days has no signal; refuse rather than mislead."""
    with pytest.raises(ValueError, match="train_size"):
        WalkForwardConfig(train_size=10)


def test_config_rejects_zero_test_window() -> None:
    with pytest.raises(ValueError, match="test_size"):
        WalkForwardConfig(test_size=0)


def test_config_rejects_zero_step() -> None:
    with pytest.raises(ValueError, match="step_size"):
        WalkForwardConfig(step_size=0)


def test_config_rejects_negative_cost() -> None:
    with pytest.raises(ValueError, match="transaction_cost_bps"):
        WalkForwardConfig(transaction_cost_bps=-1.0)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def test_annualised_sharpe_empty_returns_zero() -> None:
    assert _annualised_sharpe(np.array([])) == 0.0
    assert _annualised_sharpe(np.array([0.01])) == 0.0  # need at least 2 points


def test_annualised_sharpe_zero_std_returns_zero() -> None:
    """Constant returns have zero std; Sharpe is undefined -> 0."""
    assert _annualised_sharpe(np.array([0.01, 0.01, 0.01, 0.01])) == 0.0


def test_annualised_sharpe_matches_formula() -> None:
    """Hand-computed: mean/std * sqrt(252)."""
    rets = np.array([0.01, -0.005, 0.02, 0.0, 0.015])
    expected = float(rets.mean() / rets.std(ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR))
    assert _annualised_sharpe(rets) == pytest.approx(expected)


def test_max_drawdown_empty_returns_zero() -> None:
    assert _max_drawdown(np.array([])) == 0.0


def test_max_drawdown_monotone_up_is_zero() -> None:
    """A monotonically increasing equity curve has no drawdown."""
    assert _max_drawdown(np.array([0.01, 0.02, 0.005, 0.03])) == 0.0


def test_max_drawdown_simple_path() -> None:
    """Equity 1 -> 1.1 -> 0.99 is a 10% drawdown from the 1.1 peak.

    Returns vector: +0.10 to reach 1.1, then -0.10 to get 1.1 * 0.9 = 0.99.
    Drawdown at the bottom: (0.99 - 1.1) / 1.1 = -0.10.
    """
    rets = np.array([0.10, -0.10])
    assert _max_drawdown(rets) == pytest.approx(-0.10, abs=1e-9)


# ---------------------------------------------------------------------------
# Signal helpers
# ---------------------------------------------------------------------------


def _make_panel(n_days: int = 400, n_symbols: int = 5, seed: int = 0) -> pd.DataFrame:
    """Synthetic price panel with mild positive drift + noise. SPY at column 0."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")
    symbols = ["SPY"] + [f"SYM{i}" for i in range(1, n_symbols)]
    # Geometric brownian motion-ish path so prices stay positive.
    daily_rets = rng.normal(loc=0.0005, scale=0.015, size=(n_days, n_symbols))
    prices = 100.0 * np.cumprod(1.0 + daily_rets, axis=0)
    return pd.DataFrame(prices, index=dates, columns=symbols)


def test_equal_weight_signal_uniform() -> None:
    panel = _make_panel(n_days=100, n_symbols=4)
    idx = panel.index[-20:]
    weights = equal_weight_signal(panel, idx, list(panel.columns))
    assert weights.shape == (20, 4)
    assert np.allclose(weights.values, 0.25)


def test_equal_weight_signal_empty_universe() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    weights = equal_weight_signal(pd.DataFrame(), idx, [])
    assert weights.empty


def test_momentum_signal_selects_top_n() -> None:
    """Hand-crafted: SYM4 has the highest trailing return -> picked."""
    n = 80
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    # SYM4 grows fastest; SYM0 declines fastest.
    rows = np.column_stack([
        np.linspace(100, 80, n),    # SYM0  declines 20%
        np.linspace(100, 95, n),    # SYM1  declines  5%
        np.linspace(100, 100, n),   # SYM2  flat
        np.linspace(100, 105, n),   # SYM3  +5%
        np.linspace(100, 130, n),   # SYM4  +30%
    ])
    panel = pd.DataFrame(rows, index=dates, columns=[f"SYM{i}" for i in range(5)])
    test_idx = dates[-5:]
    weights = momentum_signal(panel, test_idx, list(panel.columns), top_n=2, lookback=63)
    held = [c for c in weights.columns if weights[c].iloc[0] > 0]
    # The two strongest names get equal weight, others zero.
    assert set(held) == {"SYM3", "SYM4"}
    for sym in held:
        assert weights[sym].iloc[0] == pytest.approx(0.5)


def test_momentum_signal_handles_short_train_window() -> None:
    """If train is shorter than lookback, fall back to all-zeros."""
    dates = pd.date_range("2024-01-01", periods=30, freq="B")
    panel = pd.DataFrame(100.0, index=dates, columns=["A", "B"])
    idx = pd.date_range("2024-03-01", periods=5, freq="B")
    weights = momentum_signal(panel, idx, ["A", "B"], lookback=63)
    assert (weights.values == 0.0).all()


# ---------------------------------------------------------------------------
# Core harness
# ---------------------------------------------------------------------------


def test_run_walk_forward_basic_equal_weight() -> None:
    """Run the harness on synthetic data and verify the shapes / metrics."""
    panel = _make_panel(n_days=400, n_symbols=4, seed=42)
    cfg = WalkForwardConfig(train_size=252, test_size=21, step_size=21, benchmark_symbol="SPY")
    result = run_walk_forward(panel, equal_weight_signal, cfg)

    # 400 - 252 = 148 days remaining; 148 // 21 = 7 full windows.
    assert len(result.windows) == 7
    # Aggregate OOS series has windows * test_size rows.
    assert len(result.oos_returns) == 7 * 21
    # OOS equity is just (1 + r).cumprod -- check the math.
    expected_equity = (1.0 + result.oos_returns).cumprod()
    assert np.allclose(result.oos_equity.values, expected_equity.values)
    # Benchmark series populated when SPY is in panel.
    assert result.benchmark_returns is not None
    assert result.benchmark_sharpe is not None
    # Information ratio exists.
    assert result.information_ratio is not None


def test_run_walk_forward_raises_when_panel_too_short() -> None:
    panel = _make_panel(n_days=100, n_symbols=3)
    cfg = WalkForwardConfig(train_size=252, test_size=21)
    with pytest.raises(ValueError, match="need >="):
        run_walk_forward(panel, equal_weight_signal, cfg)


def test_run_walk_forward_summary_has_expected_keys() -> None:
    panel = _make_panel(n_days=350, n_symbols=3)
    cfg = WalkForwardConfig(train_size=252, test_size=21)
    result = run_walk_forward(panel, equal_weight_signal, cfg)
    s = result.summary()
    for key in (
        "n_windows",
        "n_oos_days",
        "total_return",
        "annualised_return",
        "annualised_vol",
        "sharpe",
        "max_drawdown",
        "hit_rate",
    ):
        assert key in s


def test_run_walk_forward_handles_signal_fn_exceptions() -> None:
    """If signal_fn explodes on one window, that window is skipped but the
    harness still produces other windows."""
    panel = _make_panel(n_days=400, n_symbols=3, seed=7)
    call_count = {"n": 0}

    def flaky(train, idx, universe):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("synthetic crash on second window")
        return equal_weight_signal(train, idx, universe)

    cfg = WalkForwardConfig(train_size=252, test_size=21, benchmark_symbol="SPY")
    result = run_walk_forward(panel, flaky, cfg)
    # 7 attempts, one skipped, so 6 successful windows.
    assert len(result.windows) == 6


def test_run_walk_forward_transaction_costs_drag_returns() -> None:
    """Same signal with higher costs MUST produce lower or equal cum returns."""
    panel = _make_panel(n_days=400, n_symbols=4, seed=1)
    cfg_lo = WalkForwardConfig(train_size=252, test_size=21, transaction_cost_bps=0.0)
    cfg_hi = WalkForwardConfig(train_size=252, test_size=21, transaction_cost_bps=50.0)
    result_lo = run_walk_forward(panel, equal_weight_signal, cfg_lo)
    result_hi = run_walk_forward(panel, equal_weight_signal, cfg_hi)
    assert result_hi.total_return <= result_lo.total_return


def test_run_walk_forward_step_smaller_than_test_gives_more_windows() -> None:
    """Stepping by 10 days through a 21-day test window overlaps the windows."""
    panel = _make_panel(n_days=400, n_symbols=3, seed=2)
    cfg_step21 = WalkForwardConfig(train_size=252, test_size=21, step_size=21)
    cfg_step10 = WalkForwardConfig(train_size=252, test_size=21, step_size=10)
    r1 = run_walk_forward(panel, equal_weight_signal, cfg_step21)
    r2 = run_walk_forward(panel, equal_weight_signal, cfg_step10)
    assert len(r2.windows) > len(r1.windows)


def test_run_walk_forward_no_benchmark_when_missing_symbol() -> None:
    """If the requested benchmark isn't in the panel, leave bench fields None."""
    panel = _make_panel(n_days=400, n_symbols=3, seed=0)
    panel = panel.rename(columns={"SPY": "MARKET"})  # drop SPY name
    cfg = WalkForwardConfig(train_size=252, test_size=21, benchmark_symbol="SPY")
    result = run_walk_forward(panel, equal_weight_signal, cfg)
    assert result.benchmark_returns is None
    assert result.benchmark_sharpe is None
    assert result.information_ratio is None


def test_window_result_to_dict_rounds_floats() -> None:
    """WindowResult.to_dict should produce a compact JSON-ready payload."""
    from packages.research.walk_forward import WindowResult

    w = WindowResult(
        train_start="2024-01-01",
        train_end="2024-12-31",
        test_start="2025-01-02",
        test_end="2025-01-31",
        n_test_days=21,
        cum_return=0.1234567,
        sharpe=1.234567,
        max_drawdown=-0.0567,
        hit_rate=0.6234,
        turnover=0.1234567,
    )
    d = w.to_dict()
    assert d["cum_return"] == 0.1235
    assert d["sharpe"] == 1.235
    assert d["max_drawdown"] == -0.0567
    assert d["hit_rate"] == 0.623


def test_run_walk_forward_oos_period_is_contiguous() -> None:
    """OOS dates should be strictly increasing -- no overlapping window mess
    when step == test_size."""
    panel = _make_panel(n_days=400, n_symbols=3)
    cfg = WalkForwardConfig(train_size=252, test_size=21, step_size=21)
    result = run_walk_forward(panel, equal_weight_signal, cfg)
    dates = result.oos_returns.index.to_list()
    assert all(dates[i] < dates[i + 1] for i in range(len(dates) - 1))
