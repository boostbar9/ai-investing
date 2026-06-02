"""Phase 28-R step 5: smoke tests for the 'intraday-trend' wiring in paper_trade.

These tests don't exercise the live yfinance HTTP path. They verify three
contract pieces that downstream Phase 28-R work depends on:

  1. ``"intraday-trend"`` is registered in STRATEGIES, STRATEGY_UNIVERSE,
     and STRATEGY_CHOICES so the argparser accepts it.
  2. ``compute_target_weights("intraday-trend")`` dispatches to the
     dedicated ``compute_intraday_trend_weights`` branch (not the
     daily-panel fall-through that would crash).
  3. ``compute_intraday_trend_weights`` fetches intraday bars via the
     yfinance adapter, builds an OHLCV panel, calls
     ``IntradayTrendFollowing.generate_weights_for_panel``, and returns
     last-row weights with zeros stripped.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
import pytest

from packages.data.adapters.base import Bar
from tools import paper_trade as pt


def test_intraday_trend_is_a_registered_strategy() -> None:
    """Step 5 wires 'intraday-trend' alongside trend/sector/mean-reversion."""
    assert "intraday-trend" in pt.STRATEGIES
    assert "intraday-trend" in pt.STRATEGY_UNIVERSE
    assert "intraday-trend" in pt.STRATEGY_CHOICES
    assert len(pt.STRATEGY_UNIVERSE["intraday-trend"]) > 0


def test_compute_target_weights_routes_intraday_to_dedicated_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """compute_target_weights('intraday-trend') must dispatch to
    compute_intraday_trend_weights, not fall through to the daily-panel
    branch that would crash on IntradayTrendFollowing.generate_signals."""
    sentinel = {"SPY": 0.5, "QQQ": 0.5}
    called: dict[str, Any] = {"intraday": False}

    def fake_intraday() -> dict[str, float]:
        called["intraday"] = True
        return sentinel

    monkeypatch.setattr(pt, "compute_intraday_trend_weights", fake_intraday)
    out = pt.compute_target_weights("intraday-trend", equity=10_000.0)

    assert called["intraday"] is True
    assert out == sentinel


def test_sentiment_overlay_short_circuits_for_intraday(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The sentiment overlay path also routes intraday-trend to the
    dedicated branch — daily SentimentOverlay can't operate on 5m bars."""
    sentinel = {"AAPL": 0.33}
    called: dict[str, Any] = {"intraday": False}

    def fake_intraday() -> dict[str, float]:
        called["intraday"] = True
        return sentinel

    monkeypatch.setattr(pt, "compute_intraday_trend_weights", fake_intraday)
    out = pt.compute_target_weights_with_sentiment(
        "intraday-trend", {"AAPL": 0.8}
    )

    assert called["intraday"] is True
    assert out == sentinel


def _make_bars(symbol: str, n: int = 60, *, breakout: bool = False) -> list[Bar]:
    """Build n synthetic 5-minute bars for symbol.

    If ``breakout`` is True, prices climb after the 30-min opening range so
    the strategy should issue a long signal in the entry window.
    """
    base = datetime(2026, 1, 5, 14, 30, tzinfo=timezone.utc)  # 09:30 ET
    bars: list[Bar] = []
    for i in range(n):
        ts = base + timedelta(minutes=5 * i)
        if breakout and i >= 9:
            # After opening range (first 6 bars = 30 min), drive price up.
            price = 100.0 + 0.5 * (i - 5)
        else:
            price = 100.0 + 0.01 * i
        bars.append(
            Bar(
                symbol=symbol,
                ts=ts,
                open=price,
                high=price + 0.2,
                low=price - 0.2,
                close=price,
                volume=10_000.0,
            )
        )
    return bars


def test_compute_intraday_trend_weights_uses_intraday_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """compute_intraday_trend_weights must call YFinanceAdapter.get_intraday_bars
    for each symbol in the universe and return last-row weights."""
    calls: list[tuple[str, str, str]] = []

    class FakeAdapter:
        async def get_intraday_bars(
            self, symbol: str, *, interval: str = "5m", range_: str = "60d"
        ) -> list[Bar]:
            calls.append((symbol, interval, range_))
            # Make SPY break out; others flat -> no signal.
            return _make_bars(symbol, n=60, breakout=(symbol == "SPY"))

        async def aclose(self) -> None:
            pass

    monkeypatch.setattr(
        "packages.data.adapters.yfinance.YFinanceAdapter", FakeAdapter
    )

    out = pt.compute_intraday_trend_weights()

    # The adapter was called once per symbol with 5m/5d.
    universe = pt.STRATEGY_UNIVERSE["intraday-trend"]
    assert {c[0] for c in calls} == set(universe)
    assert all(c[1] == "5m" and c[2] == "5d" for c in calls)

    # Output is a dict of {symbol: weight>0}. Zeros are stripped.
    assert isinstance(out, dict)
    for w in out.values():
        assert w > 0.0


def test_compute_intraday_trend_weights_skips_bad_tickers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A single ticker that 500s must not abort the whole cycle."""

    class FlakeyAdapter:
        async def get_intraday_bars(
            self, symbol: str, *, interval: str = "5m", range_: str = "60d"
        ) -> list[Bar]:
            if symbol == "DIA":
                raise RuntimeError("yfinance DIA: 500")
            return _make_bars(symbol, n=60, breakout=(symbol == "SPY"))

        async def aclose(self) -> None:
            pass

    monkeypatch.setattr(
        "packages.data.adapters.yfinance.YFinanceAdapter", FlakeyAdapter
    )

    out = pt.compute_intraday_trend_weights()
    # No crash, DIA simply absent from the panel/weights.
    assert "DIA" not in out


def test_compute_intraday_trend_weights_raises_when_no_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If every ticker fails / returns no bars, surface a clear error
    instead of silently submitting an empty target."""

    class DeadAdapter:
        async def get_intraday_bars(
            self, symbol: str, *, interval: str = "5m", range_: str = "60d"
        ) -> list[Bar]:
            return []

        async def aclose(self) -> None:
            pass

    monkeypatch.setattr(
        "packages.data.adapters.yfinance.YFinanceAdapter", DeadAdapter
    )

    with pytest.raises(RuntimeError, match="no intraday bars"):
        pt.compute_intraday_trend_weights()
