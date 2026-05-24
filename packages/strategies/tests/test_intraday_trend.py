"""Tests for the intraday trend-following strategy."""
from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from packages.strategies.intraday_trend import IntradayTrendFollowing

ET = ZoneInfo("America/New_York")


def _session_bars(
    *,
    date: str = "2026-03-02",
    n_bars: int = 78,
    pattern: list[float] | None = None,
) -> pd.DataFrame:
    """Build a single-session 5-min OHLCV frame with a given close path."""
    start = datetime.fromisoformat(f"{date}T09:30:00").replace(tzinfo=ET)
    idx = pd.DatetimeIndex(
        [start + timedelta(minutes=5 * i) for i in range(n_bars)]
    )
    closes = np.asarray(pattern if pattern is not None else np.full(n_bars, 100.0))
    if len(closes) != n_bars:
        raise ValueError("pattern length must match n_bars")
    opens = np.r_[[closes[0]], closes[:-1]]
    highs = np.maximum(opens, closes) + 0.05
    lows = np.minimum(opens, closes) - 0.05
    vols = np.full(n_bars, 1000.0)
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": vols},
        index=idx,
    )


# ---------------------------------------------------------------------------
# Contract / configuration
# ---------------------------------------------------------------------------


def test_init_rejects_bad_stop_loss():
    with pytest.raises(ValueError):
        IntradayTrendFollowing(stop_loss=0.0)
    with pytest.raises(ValueError):
        IntradayTrendFollowing(stop_loss=0.6)


def test_init_rejects_bad_opening_range():
    with pytest.raises(ValueError):
        IntradayTrendFollowing(opening_range_minutes=0)


def test_generate_signals_raises_on_daily_prices():
    s = IntradayTrendFollowing()
    daily = pd.DataFrame({"SPY": [100, 101, 102]}, index=pd.date_range("2026-01-01", periods=3))
    with pytest.raises(NotImplementedError):
        s.generate_signals(daily)


# ---------------------------------------------------------------------------
# Behavioural tests
# ---------------------------------------------------------------------------


def test_no_entry_during_opening_range():
    """Bars within opening_range_minutes must produce zero participation."""
    n = 78
    # Strict ramp up so the breakout *would* trigger if entry window were open.
    closes = np.linspace(100.0, 110.0, n)
    bars = _session_bars(pattern=closes.tolist())
    s = IntradayTrendFollowing(opening_range_minutes=30, entry_block_minutes=15)
    w = s._symbol_weights(bars)
    # First 9 bars cover 0..40 min => entry window opens at OR(30) + 15 = 45.
    assert w.iloc[:9].sum() == 0.0


def test_breakout_then_vwap_exit():
    """Price ramps above OR high then collapses below VWAP -> long, then flat."""
    n = 78
    closes = []
    for i in range(n):
        if i < 6:           # opening range bars (30 min)
            closes.append(100.0 + 0.05 * i)
        elif i < 12:        # post-OR but pre-entry-window (next 30 min): hold steady
            closes.append(100.3)
        elif i < 30:        # ramp up sharply -> triggers entry
            closes.append(100.3 + 0.5 * (i - 12))
        else:               # crash back below VWAP
            closes.append(90.0)
    bars = _session_bars(pattern=closes)
    s = IntradayTrendFollowing(opening_range_minutes=30, entry_block_minutes=15)
    w = s._symbol_weights(bars)
    # Some long bars exist
    assert w.sum() > 0
    # Crash bars are flat
    assert w.iloc[-10:].sum() == 0.0


def test_session_close_forces_flat():
    """Even on a roaring breakout, weight must be 0 in the final 15 min."""
    n = 78
    closes = np.linspace(100.0, 130.0, n)
    bars = _session_bars(pattern=closes.tolist())
    s = IntradayTrendFollowing(opening_range_minutes=30, exit_block_minutes=15)
    w = s._symbol_weights(bars)
    # Last 3 bars are within the final 15 min of the session (each bar is 5 min).
    assert w.iloc[-3:].sum() == 0.0


def test_panel_normalisation_caps_gross():
    """When two symbols are simultaneously long, gross must <= max_gross."""
    n = 78
    closes = []
    for i in range(n):
        if i < 6:
            closes.append(100.0 + 0.05 * i)
        elif i < 12:
            closes.append(100.3)
        elif i < 50:
            closes.append(100.3 + 0.5 * (i - 12))
        else:
            closes.append(100.3 + 0.5 * 38)
    bars_a = _session_bars(pattern=closes)
    bars_b = _session_bars(pattern=closes)
    s = IntradayTrendFollowing(max_gross=1.0)
    weights = s.generate_weights_for_panel({"AAA": bars_a, "BBB": bars_b})
    row_sums = weights.sum(axis=1)
    assert (row_sums <= 1.0 + 1e-9).all()
    # Should have at least some active bars where both are long: each gets 0.5.
    both_long = weights[(weights["AAA"] > 0) & (weights["BBB"] > 0)]
    if not both_long.empty:
        assert np.allclose(both_long["AAA"], 0.5)
        assert np.allclose(both_long["BBB"], 0.5)


def test_empty_panel_returns_empty():
    s = IntradayTrendFollowing()
    out = s.generate_weights_for_panel({})
    assert out.empty


def test_stop_loss_fires_before_vwap_exit():
    """A tight stop trips while close is still above VWAP."""
    n = 78
    closes = []
    for i in range(n):
        if i < 6:
            closes.append(100.0)
        elif i < 12:
            closes.append(100.3)
        elif i < 18:
            # ramp up sharply to enter (close goes 100.8, 101.3, ... 102.8)
            closes.append(100.3 + 0.5 * (i - 12))
        elif i < 22:
            # ~2% pullback from peak 102.8 -> 100.8 (still above VWAP ~100.5)
            closes.append(100.8)
        else:
            # hover above VWAP (~100.6-101.2) so VWAP exit does not catch us
            closes.append(101.2)
    bars = _session_bars(pattern=closes)
    # A very tight 0.5% stop fires on the pullback; a loose 5% stop does not.
    s_tight = IntradayTrendFollowing(stop_loss=0.005)
    s_loose = IntradayTrendFollowing(stop_loss=0.05)
    w_tight = s_tight._symbol_weights(bars)
    w_loose = s_loose._symbol_weights(bars)
    assert w_tight.sum() < w_loose.sum(), (
        f"tight={w_tight.sum()} loose={w_loose.sum()}"
    )
