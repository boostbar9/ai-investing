"""Live-quote cache for the cockpit (Phase 25.3).

A thin TTL cache that fronts the Finnhub ``/quote`` REST endpoint and
falls back to a yfinance-backed daily-close provider when Finnhub is
unavailable (no key, network error, rate-limited). The cache is the
single seam every Phase 25 hook uses to ask "what's the last price?"
so we can later swap in the Finnhub WebSocket stream without touching
the dip-watch / exit-rules / regime callers.

Design contract
---------------

``LiveQuoteCache.lookup(symbol)``:

* Returns the freshest known price as a ``float`` (or ``None`` when
  every provider in the chain failed).
* Honors a TTL — re-uses cached values for ``ttl_seconds`` so a
  bursty dip-watch tick polling 5 symbols doesn't burn the 60/min
  Finnhub budget.
* Records per-symbol last-quote times + per-provider counters which
  ``status()`` surfaces verbatim for the ``/api/data-feed`` endpoint.
* Never raises — all errors are caught, counted, and logged.

WebSocket readiness
-------------------

The cache exposes ``ingest_ws_tick(symbol, price, ts)`` so a future
Finnhub WS subscriber can shove ticks straight into the cache. The
REST polling path simply calls the same primitive, which means the
status endpoint shows a unified view regardless of source.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

log = logging.getLogger(__name__)

# Default cache TTL — 30s is the sweet spot between "fresh enough for
# a 60s fast loop" and "comfortably under the 60/min REST budget even
# with 5+ symbols". Override via LIVE_QUOTE_TTL_S.
DEFAULT_TTL_S = 30.0

# Fallback provider type: a sync callable that returns a list of daily
# closes (yfinance shape). The cache lifts the last element to a price.
FallbackProvider = Callable[[str], list[float] | None]

# Async fetch type for the primary (Finnhub) path.
QuoteFetcher = Callable[[str], Awaitable[float]]


@dataclass
class CacheEntry:
    price: float
    ts: float  # monotonic time of last successful fetch
    wall_ts: datetime  # wall-clock for the status endpoint
    source: str  # "finnhub_rest" | "finnhub_ws" | "yfinance" | "ws"


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    finnhub_ok: int = 0
    finnhub_err: int = 0
    fallback_ok: int = 0
    fallback_err: int = 0
    ws_ingested: int = 0
    last_finnhub_err: str | None = None
    last_fallback_err: str | None = None


class LiveQuoteCache:
    """In-memory TTL cache for live prices.

    Parameters
    ----------
    quote_fetcher
        Async callable that returns the current price for a symbol.
        Defaults to a closure over :class:`FinnhubAdapter` constructed
        lazily when the cache is first used. Tests inject their own.
    fallback
        Sync callable returning daily closes (yfinance shape). When
        the primary fetcher fails we lift the last element as the
        price. Defaults to ``regime.default_price_provider``.
    ttl_seconds
        How long a cached price is considered fresh. Defaults to 30s.
    """

    def __init__(
        self,
        quote_fetcher: QuoteFetcher | None = None,
        fallback: FallbackProvider | None = None,
        *,
        ttl_seconds: float | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._fetcher = quote_fetcher
        self._fallback = fallback
        self._ttl = ttl_seconds if ttl_seconds is not None else float(
            os.getenv("LIVE_QUOTE_TTL_S", DEFAULT_TTL_S)
        )
        self._clock = clock or time.monotonic
        self._entries: dict[str, CacheEntry] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._stats = CacheStats()
        self._has_finnhub_key = bool(os.getenv("FINNHUB_API_KEY"))

    # --------------------------------------------------------------
    # Configuration / wiring
    # --------------------------------------------------------------

    @property
    def ttl_seconds(self) -> float:
        return self._ttl

    @property
    def has_finnhub_key(self) -> bool:
        return self._has_finnhub_key

    def set_fetcher(self, fetcher: QuoteFetcher | None) -> None:
        self._fetcher = fetcher

    def set_fallback(self, fallback: FallbackProvider | None) -> None:
        self._fallback = fallback

    # --------------------------------------------------------------
    # Read path
    # --------------------------------------------------------------

    def _fresh(self, entry: CacheEntry) -> bool:
        return (self._clock() - entry.ts) < self._ttl

    def peek(self, symbol: str) -> float | None:
        """Synchronous, non-refreshing read of the cached price.

        Returns the cached price regardless of freshness, or ``None``
        when the symbol has never been observed. Used by the status
        endpoint and by sync callers (e.g. dip-watch's price_lookup,
        which is a sync function called from inside an async tick).
        """
        sym = symbol.upper()
        entry = self._entries.get(sym)
        return None if entry is None else entry.price

    async def lookup(self, symbol: str) -> float | None:
        """Return the freshest price for ``symbol``.

        Cache hit when an entry exists and is younger than ``ttl_seconds``.
        Otherwise we try the primary fetcher (Finnhub), then the
        fallback (yfinance daily closes). On total failure returns
        ``None`` and the symbol's last known price stays in the cache
        for next-hop callers.
        """
        sym = symbol.upper()
        entry = self._entries.get(sym)
        if entry is not None and self._fresh(entry):
            self._stats.hits += 1
            return entry.price
        self._stats.misses += 1

        # Per-symbol lock so concurrent callers don't double-fetch.
        lock = self._locks.setdefault(sym, asyncio.Lock())
        async with lock:
            # Re-check after acquiring the lock — another task may
            # have refreshed the entry while we were waiting.
            entry = self._entries.get(sym)
            if entry is not None and self._fresh(entry):
                return entry.price

            price = await self._fetch_with_fallback(sym)
            if price is not None:
                return price
            # Stale value beats no value for dip-watch / exit-rules.
            return entry.price if entry is not None else None

    async def _fetch_with_fallback(self, symbol: str) -> float | None:
        # Try Finnhub first.
        if self._fetcher is not None:
            try:
                price = await self._fetcher(symbol)
                if price and price > 0:
                    self._record(symbol, float(price), source="finnhub_rest")
                    self._stats.finnhub_ok += 1
                    return float(price)
            except Exception as exc:  # pragma: no cover — logged
                self._stats.finnhub_err += 1
                self._stats.last_finnhub_err = f"{type(exc).__name__}: {exc}"
                log.debug("live_quotes: finnhub fetch failed %s: %s", symbol, exc)

        # Fall back to yfinance daily closes.
        if self._fallback is not None:
            try:
                series = await asyncio.to_thread(self._fallback, symbol)
                if series:
                    price = float(series[-1])
                    if price > 0:
                        self._record(symbol, price, source="yfinance")
                        self._stats.fallback_ok += 1
                        return price
            except Exception as exc:  # pragma: no cover — logged
                self._stats.fallback_err += 1
                self._stats.last_fallback_err = f"{type(exc).__name__}: {exc}"
                log.debug("live_quotes: fallback failed %s: %s", symbol, exc)

        return None

    def _record(self, symbol: str, price: float, *, source: str) -> None:
        self._entries[symbol] = CacheEntry(
            price=price,
            ts=self._clock(),
            wall_ts=datetime.now(UTC),
            source=source,
        )

    # --------------------------------------------------------------
    # WebSocket-ready ingest
    # --------------------------------------------------------------

    def ingest_ws_tick(
        self,
        symbol: str,
        price: float,
        ts: datetime | None = None,
    ) -> None:
        """Push a tick from the Finnhub WS subscriber into the cache.

        Future Phase 25.4 will spin up a background coroutine that
        subscribes to up to 50 symbols and pumps ticks here. Today
        it's a no-op surface that tests can call to validate the
        contract.
        """
        sym = symbol.upper()
        self._entries[sym] = CacheEntry(
            price=float(price),
            ts=self._clock(),
            wall_ts=ts or datetime.now(UTC),
            source="finnhub_ws",
        )
        self._stats.ws_ingested += 1

    # --------------------------------------------------------------
    # Telemetry for /api/data-feed
    # --------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        symbols: list[dict[str, Any]] = []
        now = self._clock()
        for sym, entry in sorted(self._entries.items()):
            symbols.append(
                {
                    "symbol": sym,
                    "price": entry.price,
                    "source": entry.source,
                    "age_seconds": round(now - entry.ts, 2),
                    "wall_ts": entry.wall_ts.isoformat(timespec="seconds"),
                    "fresh": (now - entry.ts) < self._ttl,
                }
            )
        primary = "finnhub" if (self._fetcher is not None and self._has_finnhub_key) else "yfinance"
        return {
            "primary_provider": primary,
            "has_finnhub_key": self._has_finnhub_key,
            "ttl_seconds": self._ttl,
            "cache_size": len(self._entries),
            "symbols": symbols,
            "stats": {
                "hits": self._stats.hits,
                "misses": self._stats.misses,
                "finnhub_ok": self._stats.finnhub_ok,
                "finnhub_err": self._stats.finnhub_err,
                "fallback_ok": self._stats.fallback_ok,
                "fallback_err": self._stats.fallback_err,
                "ws_ingested": self._stats.ws_ingested,
                "last_finnhub_err": self._stats.last_finnhub_err,
                "last_fallback_err": self._stats.last_fallback_err,
            },
        }


# ---------------------------------------------------------------------------
# Module-level singleton used by server.py — tests construct their own.
# ---------------------------------------------------------------------------

_default_cache: LiveQuoteCache | None = None


def get_default_cache() -> LiveQuoteCache:
    """Lazy-construct the process-wide cache.

    Server startup calls this once and wires the Finnhub fetcher +
    yfinance fallback into the returned instance. The function is
    idempotent.
    """
    global _default_cache
    if _default_cache is None:
        _default_cache = LiveQuoteCache()
    return _default_cache


def reset_default_cache_for_tests() -> None:
    """Test-only seam — wipes the module-level singleton."""
    global _default_cache
    _default_cache = None


# ---------------------------------------------------------------------------
# High-level helpers consumed by server.py
# ---------------------------------------------------------------------------


def make_finnhub_fetcher(adapter: Any) -> QuoteFetcher:
    """Wrap a :class:`FinnhubAdapter` as a ``QuoteFetcher`` closure."""

    async def _fetch(symbol: str) -> float:
        quote = await adapter.get_quote(symbol)
        return float(quote.price)

    return _fetch


def finnhub_price_lookup_factory(
    cache: LiveQuoteCache,
) -> Callable[[str], float | None]:
    """Build a sync ``price_lookup(symbol) -> float | None``.

    The cockpit's Phase 25 hooks (dip-watch, exit-rules, regime)
    consume a sync price lookup. We can't ``await`` inside a sync
    function, so this helper returns whatever the cache currently
    holds. The cache is kept warm by the fast loop, which calls
    ``refresh_symbols`` asynchronously every 60s.
    """

    def _lookup(symbol: str) -> float | None:
        return cache.peek(symbol)

    return _lookup


async def refresh_symbols(
    cache: LiveQuoteCache,
    symbols: list[str],
) -> dict[str, float | None]:
    """Refresh ``cache`` for a list of symbols, returning the new prices.

    Called from the fast loop (every 60s) so the sync ``peek()`` path
    used by dip-watch / exit-rules always returns a value younger than
    the cache TTL.
    """
    out: dict[str, float | None] = {}
    for sym in symbols:
        out[sym] = await cache.lookup(sym)
    return out
