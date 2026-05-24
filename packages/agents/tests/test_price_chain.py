"""Tests for the multi-provider price chain used by attribution.

We focus on the chain semantics (fall-through, caching, stats, error
isolation). The provider builders themselves (alpaca/polygon/yfinance)
are thin wrappers around already-tested adapters and would require
network/env to exercise end-to-end; we instead inject pure-Python fakes.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from packages.agents import price_chain as pc


def _ts(year: int = 2026, month: int = 5, day: int = 20) -> datetime:
    return datetime(year, month, day, 20, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Fall-through ordering
# ---------------------------------------------------------------------------


def test_chain_returns_first_non_none_in_order() -> None:
    chain = pc.PriceChain()
    chain.add("first", lambda s, t: 100.0)
    chain.add("second", lambda s, t: 200.0)
    assert chain.get_close("SPY", _ts()) == 100.0
    assert chain.stats["first"] == 1
    assert chain.stats.get("second", 0) == 0


def test_chain_falls_through_on_none() -> None:
    chain = pc.PriceChain()
    chain.add("primary", lambda s, t: None)
    chain.add("fallback", lambda s, t: 99.5)
    assert chain.get_close("AAPL", _ts()) == 99.5
    assert chain.stats["fallback"] == 1


def test_chain_records_miss_when_all_none() -> None:
    chain = pc.PriceChain()
    chain.add("a", lambda s, t: None)
    chain.add("b", lambda s, t: None)
    assert chain.get_close("XYZ", _ts()) is None
    assert chain.stats["miss"] == 1


def test_chain_empty_providers_misses() -> None:
    chain = pc.PriceChain()
    assert chain.get_close("SPY", _ts()) is None
    assert chain.stats["miss"] == 1


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


def test_chain_caches_same_symbol_same_day() -> None:
    calls: list[str] = []

    def first(symbol: str, ts: datetime) -> float | None:
        calls.append(symbol)
        return 123.0

    chain = pc.PriceChain()
    chain.add("first", first)
    # Two different intra-day timestamps, same symbol & calendar day.
    t1 = datetime(2026, 5, 20, 14, 30, tzinfo=UTC)
    t2 = datetime(2026, 5, 20, 20, 0, tzinfo=UTC)
    assert chain.get_close("SPY", t1) == 123.0
    assert chain.get_close("SPY", t2) == 123.0
    assert len(calls) == 1
    assert chain.stats["cache"] == 1
    assert chain.stats["first"] == 1


def test_chain_does_not_cache_across_days() -> None:
    calls: list[str] = []

    def first(symbol: str, ts: datetime) -> float | None:
        calls.append(ts.date().isoformat())
        return 50.0

    chain = pc.PriceChain()
    chain.add("first", first)
    chain.get_close("SPY", _ts(day=20))
    chain.get_close("SPY", _ts(day=21))
    assert len(calls) == 2


def test_chain_caches_misses_too() -> None:
    calls = []

    def fn(s: str, t: datetime) -> float | None:
        calls.append((s, t))
        return None

    chain = pc.PriceChain()
    chain.add("only", fn)
    chain.get_close("XYZ", _ts())
    chain.get_close("XYZ", _ts())
    # Second call must hit cache, not re-call provider.
    assert len(calls) == 1
    assert chain.stats["miss"] == 1
    assert chain.stats["cache"] == 1


# ---------------------------------------------------------------------------
# Naive datetimes & error isolation
# ---------------------------------------------------------------------------


def test_chain_handles_naive_datetime() -> None:
    """Caller may pass tz-naive ts; chain must not crash on the cache key."""
    chain = pc.PriceChain()
    chain.add("p", lambda s, t: 1.0)
    naive = datetime(2026, 5, 20, 20, 0)
    assert chain.get_close("SPY", naive) == 1.0


def test_chain_isolates_provider_errors() -> None:
    """A raising provider must not kill the batch \u2014 chain falls through."""

    def boom(s: str, t: datetime) -> float | None:
        raise RuntimeError("simulated provider crash")

    chain = pc.PriceChain()
    chain.add("crashy", boom)
    chain.add("good", lambda s, t: 7.5)
    assert chain.get_close("SPY", _ts()) == 7.5
    assert chain.stats["crashy_error"] == 1
    assert chain.stats["good"] == 1


# ---------------------------------------------------------------------------
# _at_or_after helper
# ---------------------------------------------------------------------------


class _FakeBar:
    def __init__(self, ts: datetime, close: float) -> None:
        self.ts = ts
        self.close = close


def test_at_or_after_picks_first_matching_bar() -> None:
    bars = [
        _FakeBar(datetime(2026, 5, 19, 20, 0, tzinfo=UTC), 100.0),
        _FakeBar(datetime(2026, 5, 20, 20, 0, tzinfo=UTC), 101.0),
        _FakeBar(datetime(2026, 5, 21, 20, 0, tzinfo=UTC), 102.0),
    ]
    assert pc._at_or_after(bars, datetime(2026, 5, 20, 12, 0, tzinfo=UTC)) == 101.0


def test_at_or_after_falls_back_to_last_when_no_match() -> None:
    bars = [
        _FakeBar(datetime(2026, 5, 19, 20, 0, tzinfo=UTC), 100.0),
        _FakeBar(datetime(2026, 5, 20, 20, 0, tzinfo=UTC), 101.0),
    ]
    out = pc._at_or_after(bars, datetime(2027, 1, 1, tzinfo=UTC))
    # No bar at-or-after future date; we return the latest known close.
    assert out == 101.0


def test_at_or_after_empty_bars_returns_none() -> None:
    assert pc._at_or_after([], _ts()) is None


# ---------------------------------------------------------------------------
# Provider builders are env-gated
# ---------------------------------------------------------------------------


def test_alpaca_provider_returns_none_without_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ALPACA_PAPER_KEY_ID", raising=False)
    monkeypatch.delenv("ALPACA_PAPER_SECRET", raising=False)
    assert pc.alpaca_provider() is None


def test_polygon_provider_returns_none_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    assert pc.polygon_provider() is None


def test_yfinance_provider_is_always_available() -> None:
    """yfinance has no API-key requirement \u2014 builder must always return a fn."""
    fn = pc.yfinance_provider()
    assert callable(fn)


def test_build_default_chain_always_includes_yfinance(monkeypatch: pytest.MonkeyPatch) -> None:
    """A fresh box with no env vars must still have a working chain."""
    monkeypatch.delenv("ALPACA_PAPER_KEY_ID", raising=False)
    monkeypatch.delenv("ALPACA_PAPER_SECRET", raising=False)
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    chain = pc.build_default_chain()
    names = [n for n, _ in chain.providers]
    assert names == ["yfinance"]


def test_provider_summary_shape() -> None:
    chain = pc.PriceChain()
    chain.add("fake", lambda s, t: 1.0)
    chain.get_close("SPY", _ts())
    s = pc.provider_summary(chain)
    assert s["providers"] == ["fake"]
    assert s["stats"]["fake"] == 1
    assert s["cache_size"] == 1
