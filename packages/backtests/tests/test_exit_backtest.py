"""Tests for the exit-threshold backtest harness.

The harness replays the real ``exit_rules.evaluate_position`` over PnL paths, so
these tests pin the realized-return accounting against hand-computed paths and
confirm the aggregate metrics + the sweep plumbing. No IO, no market data.
"""
from __future__ import annotations

import math

import pytest

from packages.backtests import exit_backtest as eb
from packages.cockpit.web.exit_rules import ExitThresholds


def _th(
    *,
    take_profit: float = 0.035,
    arm: float = 0.02,
    giveback: float = 0.015,
    hard_stop: float = 0.025,
    max_hold: float = 0.0,
) -> ExitThresholds:
    return ExitThresholds(
        take_profit_pct=take_profit,
        trail_arm_pct=arm,
        trail_giveback_pct=giveback,
        hard_stop_pct=hard_stop,
        preset="test",
        max_hold_hours=max_hold,
    )


# ---- simulate_trade: each exit rule realizes at the right bar ---------------


def test_take_profit_realizes_at_crossing_bar() -> None:
    # Climbs straight through the 3.5% take-profit.
    path = [(0.01, 1.0), (0.02, 2.0), (0.04, 3.0), (0.06, 4.0)]
    # Disable scale-out by setting arm above the path so only TP can fire.
    th = _th(arm=0.10)
    assert eb.simulate_trade(path, th) == pytest.approx(0.04)


def test_hard_stop_realizes_at_breach_bar() -> None:
    path = [(-0.01, 1.0), (-0.03, 2.0), (-0.10, 3.0)]
    th = _th(hard_stop=0.025)
    # Fires at the first bar <= -2.5% (the -0.03 bar), not the -0.10 bar.
    assert eb.simulate_trade(path, th) == pytest.approx(-0.03)


def test_no_exit_marks_to_market_at_last_bar() -> None:
    # Never hits any threshold; closed at the final bar's PnL.
    path = [(0.001, 1.0), (0.004, 2.0), (0.002, 3.0)]
    th = _th(take_profit=0.10, arm=0.10, hard_stop=0.10)
    assert eb.simulate_trade(path, th) == pytest.approx(0.002)


def test_max_hold_releases_position() -> None:
    path = [(0.001, 10.0), (0.002, 20.0), (0.003, 30.0)]
    th = _th(take_profit=0.10, arm=0.10, hard_stop=0.10, max_hold=24.0)
    # First bar past 24h (the 30.0h bar) releases at its PnL.
    assert eb.simulate_trade(path, th) == pytest.approx(0.003)


def test_scale_out_realizes_fraction_then_rides_remainder() -> None:
    # Peak crosses arm (2%) with TP above it -> scale_out sells half, then the
    # remainder trails out on the giveback.
    th = _th(take_profit=0.10, arm=0.02, giveback=0.015)
    path = [(0.022, 1.0), (0.030, 2.0), (0.012, 3.0)]
    # Bar1: scale_out 0.5 @ 0.022. Bar2: peak=0.030, giveback 0 -> hold.
    # Bar3: pnl 0.012, peak 0.030, giveback 0.018 >= 0.015 -> trailing stop
    # on the remaining 0.5 @ 0.012.
    expected = 0.5 * 0.022 + 0.5 * 0.012
    assert eb.simulate_trade(path, th) == pytest.approx(expected)


def test_empty_path_is_zero() -> None:
    assert eb.simulate_trade([], _th()) == 0.0


# ---- backtest_exits: aggregate metrics --------------------------------------


def test_backtest_metrics_on_known_paths() -> None:
    # Two winners (+2%, +1%) and one loser (-2.5% hard stop).
    paths = [
        [(0.02, 1.0)],
        [(0.01, 1.0)],
        [(-0.03, 1.0)],
    ]
    th = _th(take_profit=0.005, arm=0.10, hard_stop=0.025)
    res = eb.backtest_exits(paths, th)
    assert res.n_trades == 3
    # Winners realize at TP-cross bar value; loser at hard stop bar.
    assert res.win_rate == pytest.approx(2 / 3)
    assert res.gross_profit == pytest.approx(0.03)
    assert res.gross_loss == pytest.approx(0.03)
    assert res.profit_factor == pytest.approx(1.0)
    assert res.expectancy == pytest.approx((0.02 + 0.01 - 0.03) / 3)


def test_profit_factor_infinite_when_no_losses() -> None:
    paths = [[(0.02, 1.0)], [(0.01, 1.0)]]
    th = _th(take_profit=0.005, arm=0.10)
    res = eb.backtest_exits(paths, th)
    assert math.isinf(res.profit_factor)


def test_empty_ensemble_is_safe() -> None:
    res = eb.backtest_exits([], _th())
    assert res.n_trades == 0
    assert res.profit_factor == 0.0
    assert res.max_drawdown == 0.0


# ---- synthetic_paths + sweep ------------------------------------------------


def test_synthetic_paths_are_deterministic() -> None:
    a = eb.synthetic_paths(n_trades=50, seed=3)
    b = eb.synthetic_paths(n_trades=50, seed=3)
    assert a == b
    assert len(a) == 50
    assert len(a[0]) == 96


def test_tighter_hard_stop_cuts_drawdown_on_calibrated_ensemble() -> None:
    """The core validated claim: tightening the hard stop reduces max drawdown
    and lifts profit factor on the calibrated ensemble (the biggest lever)."""
    paths = eb.synthetic_paths(n_trades=1500, seed=11)
    loose = eb.backtest_exits(paths, _th(hard_stop=0.05, max_hold=24.0))
    tight = eb.backtest_exits(paths, _th(hard_stop=0.025, max_hold=24.0))
    assert tight.max_drawdown < loose.max_drawdown
    assert tight.profit_factor > loose.profit_factor
    assert tight.avg_loss > loose.avg_loss  # smaller magnitude loss


def test_sweep_grid_returns_row_per_candidate() -> None:
    paths = eb.synthetic_paths(n_trades=200, seed=5)
    candidates = [
        {"take_profit_pct": 0.035, "trail_giveback_pct": 0.015, "hard_stop_pct": 0.025},
        {"take_profit_pct": 0.03, "trail_giveback_pct": 0.012, "hard_stop_pct": 0.05},
    ]
    rows = eb.sweep_grid(paths, candidates)
    assert len(rows) == 2
    assert rows[0].hard_stop_pct == 0.025
    assert rows[0].max_hold_hours == 0.0  # default when omitted
    assert isinstance(rows[0].result, eb.ExitBacktestResult)
