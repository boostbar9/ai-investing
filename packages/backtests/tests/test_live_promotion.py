"""Tests for Phase 5 live promotion + canary capital."""
from __future__ import annotations

import numpy as np
import pandas as pd

from packages.backtests.live_promotion import (
    DEFAULT_CANARY_SCHEDULE,
    PAPER_MIN_DAYS,
    canary_fraction,
    decide_live_capital,
    live_readiness_gate,
)


def _equity_from_returns(rets: list[float], start: float = 100.0) -> pd.Series:
    """Build an equity curve from a list of daily returns."""
    eq = [start]
    for r in rets:
        eq.append(eq[-1] * (1 + r))
    return pd.Series(eq[1:])


def _steady_equity(days: int, daily_return: float, start: float = 100.0) -> pd.Series:
    return _equity_from_returns([daily_return] * days, start=start)


# ---------------------------------------------------------------------------
# live_readiness_gate
# ---------------------------------------------------------------------------


def test_readiness_blocks_when_flag_unset() -> None:
    # Otherwise-passing curve, but flag is off.
    paper = _steady_equity(PAPER_MIN_DAYS, 0.001)
    v = live_readiness_gate(paper, enable_flag="")
    assert v.ready is False
    assert any("ENABLE_LIVE_TRADING" in r for r in v.reasons)


def test_readiness_blocks_when_too_few_days() -> None:
    paper = _steady_equity(PAPER_MIN_DAYS - 5, 0.001)
    v = live_readiness_gate(paper, enable_flag="true")
    assert v.ready is False
    assert any("trading days" in r for r in v.reasons)


def test_readiness_blocks_when_drawdown_too_large() -> None:
    # Big single-day drop blows max DD past 8%.
    rets = [0.001] * (PAPER_MIN_DAYS - 1) + [-0.10]
    paper = _equity_from_returns(rets)
    v = live_readiness_gate(paper, enable_flag="true")
    assert v.ready is False
    assert any("max drawdown" in r for r in v.reasons)


def test_readiness_blocks_when_sharpe_too_low() -> None:
    # Alternating +/- returns → near-zero mean, non-zero std → Sharpe ~ 0.
    rets = [0.001, -0.001] * (PAPER_MIN_DAYS // 2)
    paper = _equity_from_returns(rets)
    v = live_readiness_gate(paper, enable_flag="true")
    assert v.ready is False
    assert any("Sharpe" in r for r in v.reasons)


def test_readiness_passes_with_strong_curve_and_flag() -> None:
    rng = np.random.default_rng(42)
    rets = list(rng.normal(0.002, 0.005, PAPER_MIN_DAYS))  # strong positive drift
    paper = _equity_from_returns(rets)
    v = live_readiness_gate(paper, enable_flag="true")
    assert v.ready is True, v.reasons
    assert v.metrics["paper_days"] >= PAPER_MIN_DAYS


# ---------------------------------------------------------------------------
# canary_fraction
# ---------------------------------------------------------------------------


def test_canary_starts_at_tier_zero_for_empty_curve() -> None:
    state = canary_fraction(pd.Series(dtype=float))
    assert state.tier_index == 0
    assert state.fraction == DEFAULT_CANARY_SCHEDULE[0][0]
    assert state.next_fraction == DEFAULT_CANARY_SCHEDULE[1][0]


def test_canary_stays_at_tier_zero_when_dwell_not_yet_met() -> None:
    # Only 10 live days; dwell = 30.
    rng = np.random.default_rng(7)
    rets = list(rng.normal(0.002, 0.005, 10))
    state = canary_fraction(_equity_from_returns(rets))
    assert state.tier_index == 0
    assert state.fraction == 0.05
    assert state.days_in_tier == 10
    assert any("dwell" in r for r in state.reasons)


def test_canary_advances_through_tiers_with_strong_curve() -> None:
    # 90 strong live days → should clear tiers 0, 1, 2 → 100%.
    rng = np.random.default_rng(123)
    rets = list(rng.normal(0.003, 0.004, 100))
    state = canary_fraction(_equity_from_returns(rets))
    assert state.tier_index == 3
    assert state.fraction == 1.00
    assert state.next_fraction is None


def test_canary_halts_advance_on_drawdown_breach() -> None:
    # 30 strong days (tier 0 → 1) then a big drop inside tier 1's window.
    rng = np.random.default_rng(9)
    good = list(rng.normal(0.003, 0.003, 30))
    bad = [0.001] * 20 + [-0.10] + [0.001] * 9  # DD > 8% inside tier 1
    state = canary_fraction(_equity_from_returns(good + bad))
    assert state.tier_index == 1  # advanced past tier 0 but stalled at 1
    assert state.fraction == 0.10
    assert any("max-DD" in r for r in state.reasons)


# ---------------------------------------------------------------------------
# decide_live_capital
# ---------------------------------------------------------------------------


def test_decide_returns_zero_when_not_ready() -> None:
    paper = _steady_equity(10, 0.001)  # too few days
    decision = decide_live_capital(paper, enable_flag="true")
    assert decision.live_enabled is False
    assert decision.capital_fraction == 0.0
    assert decision.canary is None


def test_decide_returns_canary_when_ready() -> None:
    rng = np.random.default_rng(11)
    paper = _equity_from_returns(list(rng.normal(0.002, 0.005, PAPER_MIN_DAYS)))
    decision = decide_live_capital(paper, enable_flag="true")
    assert decision.live_enabled is True
    # No live history yet → tier 0 → 5%.
    assert decision.capital_fraction == 0.05
    assert decision.canary is not None
    assert decision.canary.tier_index == 0
