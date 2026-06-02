"""Phase 25.3 — LiveQuoteCache contract tests.

Covers:

* Cache hit / miss + TTL expiry
* Fallback chain (Finnhub → yfinance) on primary failure
* WebSocket ingest seeds the cache and bypasses REST
* ``peek()`` is sync and returns the freshest cached value
* ``status()`` surfaces primary provider + per-symbol freshness
* ``finnhub_price_lookup_factory`` returns a sync lookup that hits
  the cache without an event loop
"""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from packages.cockpit.web.live_quotes import (
    LiveQuoteCache,
    finnhub_price_lookup_factory,
    make_finnhub_fetcher,
    refresh_symbols,
)


class _FakeClock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


# ---------------------------------------------------------------------------
# Cache hit / miss / TTL
# ---------------------------------------------------------------------------


def test_peek_empty_returns_none() -> None:
    cache = LiveQuoteCache()
    assert cache.peek("AAPL") is None


def test_lookup_calls_fetcher_and_caches() -> None:
    calls = []

    async def fetcher(symbol: str) -> float:
        calls.append(symbol)
        return 123.45

    cache = LiveQuoteCache(quote_fetcher=fetcher, ttl_seconds=30.0)
    price = asyncio.run(cache.lookup("AAPL"))
    assert price == 123.45
    assert calls == ["AAPL"]
    # Second call within TTL must hit cache, not re-fetch.
    price2 = asyncio.run(cache.lookup("AAPL"))
    assert price2 == 123.45
    assert calls == ["AAPL"]


def test_ttl_expiry_triggers_refetch() -> None:
    clock = _FakeClock()
    counter = {"n": 0}

    async def fetcher(symbol: str) -> float:
        counter["n"] += 1
        return 100.0 + counter["n"]

    cache = LiveQuoteCache(
        quote_fetcher=fetcher, ttl_seconds=10.0, clock=clock
    )
    p1 = asyncio.run(cache.lookup("SPY"))
    assert p1 == 101.0
    clock.advance(5.0)
    p2 = asyncio.run(cache.lookup("SPY"))  # still fresh
    assert p2 == 101.0
    clock.advance(10.0)  # now stale
    p3 = asyncio.run(cache.lookup("SPY"))
    assert p3 == 102.0
    assert counter["n"] == 2


# ---------------------------------------------------------------------------
# Fallback chain
# ---------------------------------------------------------------------------


def test_fallback_to_yfinance_when_finnhub_raises() -> None:
    async def bad_fetcher(symbol: str) -> float:
        raise RuntimeError("rate limited")

    def yf_fallback(symbol: str) -> list[float] | None:
        return [200.0, 201.0, 202.5]

    cache = LiveQuoteCache(quote_fetcher=bad_fetcher, fallback=yf_fallback)
    price = asyncio.run(cache.lookup("AAPL"))
    assert price == 202.5
    s = cache.status()["stats"]
    assert s["finnhub_err"] == 1
    assert s["fallback_ok"] == 1
    assert s["last_finnhub_err"] is not None


def test_fallback_returns_none_when_all_providers_fail() -> None:
    async def bad_fetcher(symbol: str) -> float:
        raise RuntimeError("boom")

    def bad_fallback(symbol: str) -> list[float] | None:
        return None

    cache = LiveQuoteCache(
        quote_fetcher=bad_fetcher, fallback=bad_fallback
    )
    price = asyncio.run(cache.lookup("NVDA"))
    assert price is None


def test_stale_value_returned_when_refresh_fails() -> None:
    clock = _FakeClock()
    fetch_calls = {"n": 0}

    async def fetcher(symbol: str) -> float:
        fetch_calls["n"] += 1
        if fetch_calls["n"] == 1:
            return 50.0
        raise RuntimeError("flaky")

    cache = LiveQuoteCache(
        quote_fetcher=fetcher, fallback=None, ttl_seconds=5.0, clock=clock
    )
    assert asyncio.run(cache.lookup("AAPL")) == 50.0
    clock.advance(10.0)  # stale
    # Fetch fails; we should still get the stale value, not None.
    assert asyncio.run(cache.lookup("AAPL")) == 50.0


# ---------------------------------------------------------------------------
# WS ingest
# ---------------------------------------------------------------------------


def test_ingest_ws_tick_populates_cache_without_fetcher() -> None:
    cache = LiveQuoteCache(quote_fetcher=None, fallback=None)
    cache.ingest_ws_tick("AAPL", 199.99)
    assert cache.peek("AAPL") == 199.99
    status = cache.status()
    assert status["cache_size"] == 1
    assert status["stats"]["ws_ingested"] == 1
    sym = status["symbols"][0]
    assert sym["source"] == "finnhub_ws"
    assert sym["fresh"] is True


def test_ws_ingest_bypasses_fetcher_on_next_lookup() -> None:
    calls: list[str] = []

    async def fetcher(symbol: str) -> float:
        calls.append(symbol)
        return 1.0

    cache = LiveQuoteCache(quote_fetcher=fetcher, ttl_seconds=30.0)
    cache.ingest_ws_tick("AAPL", 999.0)
    assert asyncio.run(cache.lookup("AAPL")) == 999.0
    assert calls == []


# ---------------------------------------------------------------------------
# status() shape
# ---------------------------------------------------------------------------


def test_status_no_key_shows_yfinance_primary() -> None:
    cache = LiveQuoteCache(quote_fetcher=None)
    s = cache.status()
    assert s["primary_provider"] == "yfinance"
    assert s["has_finnhub_key"] is False
    assert s["cache_size"] == 0
    assert s["symbols"] == []


def test_status_with_fetcher_and_key_shows_finnhub_primary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FINNHUB_API_KEY", "TEST")

    async def fetcher(symbol: str) -> float:
        return 42.0

    cache = LiveQuoteCache(quote_fetcher=fetcher)
    asyncio.run(cache.lookup("SPY"))
    s = cache.status()
    assert s["primary_provider"] == "finnhub"
    assert s["has_finnhub_key"] is True
    assert s["cache_size"] == 1
    assert s["symbols"][0]["symbol"] == "SPY"
    assert s["symbols"][0]["source"] == "finnhub_rest"


# ---------------------------------------------------------------------------
# sync lookup factory + refresh_symbols
# ---------------------------------------------------------------------------


def test_sync_lookup_factory_returns_peeked_price() -> None:
    cache = LiveQuoteCache()
    cache.ingest_ws_tick("AAPL", 150.0)
    lookup = finnhub_price_lookup_factory(cache)
    assert lookup("AAPL") == 150.0
    assert lookup("UNKNOWN") is None


def test_refresh_symbols_warms_multiple_symbols() -> None:
    quotes = {"SPY": 500.0, "AAPL": 150.0, "MSFT": 400.0}

    async def fetcher(symbol: str) -> float:
        return quotes[symbol]

    cache = LiveQuoteCache(quote_fetcher=fetcher)
    out = asyncio.run(refresh_symbols(cache, ["SPY", "AAPL", "MSFT"]))
    assert out == {"SPY": 500.0, "AAPL": 150.0, "MSFT": 400.0}
    assert cache.peek("SPY") == 500.0
    assert cache.peek("MSFT") == 400.0


# ---------------------------------------------------------------------------
# make_finnhub_fetcher adapter glue
# ---------------------------------------------------------------------------


def test_make_finnhub_fetcher_wraps_adapter() -> None:
    class _StubQuote:
        price = 305.55

    class _StubAdapter:
        async def get_quote(self, symbol: str) -> Any:
            return _StubQuote()

    fetch = make_finnhub_fetcher(_StubAdapter())
    price = asyncio.run(fetch("AAPL"))
    assert price == pytest.approx(305.55)


# ---------------------------------------------------------------------------
# Concurrency: single-flight per symbol
# ---------------------------------------------------------------------------


def test_concurrent_lookups_single_flight() -> None:
    """Two concurrent misses on the same symbol must only fetch once."""
    counter = {"n": 0}

    async def fetcher(symbol: str) -> float:
        counter["n"] += 1
        # Force the second caller to wait at the lock.
        await asyncio.sleep(0.01)
        return 77.0

    cache = LiveQuoteCache(quote_fetcher=fetcher, ttl_seconds=60.0)

    async def _go() -> tuple[float | None, float | None]:
        return await asyncio.gather(cache.lookup("AAPL"), cache.lookup("AAPL"))

    a, b = asyncio.run(_go())
    assert a == 77.0
    assert b == 77.0
    # Single-flight: only one fetch should have hit the wire.
    assert counter["n"] == 1
