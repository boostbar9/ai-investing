"""Multi-provider EOD price chain used by the attribution batch job.

The attribution job (``packages.agents.attribution.run_attribution``) needs
to ask, "what was the close of ``symbol`` at or after ``ts``?" once per
(decision, horizon) pair. Historically the cockpit had this hard-coded to
Alpaca only, which meant an operator without paper keys could never grow
the agent scorecard — and therefore could never start the 60-day paper
clock the §16 acceptance criteria require.

This module wraps a small chain of free / paid market-data adapters and
exposes a single :class:`PriceChain` whose ``get_close(symbol, ts)`` is
type-compatible with ``attribution.PriceFetcher``. Key properties:

* **Fall-through.** Tries Alpaca → Polygon → Yahoo Finance in order,
  skipping any that aren't configured. yfinance has no key requirement
  so it's always available as a baseline.
* **Cached.** Same ``(symbol, day)`` is fetched once per chain instance.
  Attribution asks for the same date across several horizons (1d/5d/20d
  windows of nearby runs collide), so the cache typically cuts requests
  by 3-4×.
* **Reports.** Tracks which provider served each hit and how many calls
  missed. ``chain.stats`` returns a counter the cockpit endpoint can
  include in its response so the operator can see attribution coverage.
* **Sync.** Attribution is a batch loop, not the inner-loop trading path.
  We wrap async adapter calls with ``asyncio.run`` per request rather
  than threading an event loop through every layer.

Out of scope here: backfilling missing days, adjusting for splits beyond
what the adapters do, intraday execution prices. We use daily closes.
"""

from __future__ import annotations

import asyncio
import logging
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

log = logging.getLogger(__name__)


# A provider returns a close price for ``symbol`` at-or-after ``ts``, or
# ``None`` if it has no bar for that range / isn't configured.
ProviderFn = Callable[[str, datetime], "float | None"]


@dataclass
class PriceChain:
    """Compose multiple provider functions into a single cached fetcher.

    Providers are tried in order; first non-None wins. ``stats`` counts
    hits per provider plus a ``miss`` bucket when no provider produced a
    bar — useful for operator-facing diagnostics.
    """

    providers: list[tuple[str, ProviderFn]] = field(default_factory=list)
    stats: Counter[str] = field(default_factory=Counter)
    # Cache key is (symbol, calendar-day UTC). Attribution always asks for
    # at-or-after a timestamp, so collapsing to the day is safe: every
    # caller for a given day gets the same close.
    _cache: dict[tuple[str, str], float | None] = field(default_factory=dict)

    def add(self, name: str, fn: ProviderFn) -> PriceChain:
        self.providers.append((name, fn))
        return self

    def get_close(self, symbol: str, ts: datetime) -> float | None:
        """The :class:`attribution.PriceFetcher`-compatible call.

        Iterates configured providers and returns the first non-None
        response. Updates ``stats`` and caches the result for the
        ``(symbol, day)`` pair so subsequent calls within the same chain
        lifetime don't re-hit the network.
        """
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        day_key = ts.date().isoformat()
        cache_key = (symbol, day_key)
        if cache_key in self._cache:
            self.stats["cache"] += 1
            return self._cache[cache_key]

        for name, fn in self.providers:
            try:
                price = fn(symbol, ts)
            except Exception as exc:  # provider failure must not kill the batch
                log.warning("price provider %s failed for %s: %s", name, symbol, exc)
                self.stats[f"{name}_error"] += 1
                continue
            if price is not None:
                self.stats[name] += 1
                self._cache[cache_key] = price
                return price
        self.stats["miss"] += 1
        self._cache[cache_key] = None
        return None

    def stats_dict(self) -> dict[str, int]:
        """JSON-safe snapshot of ``stats`` for API responses."""
        return dict(self.stats)


# ---------------------------------------------------------------------------
# Built-in providers — thin sync wrappers over the async adapters.
#
# Each helper is best-effort: returns ``None`` rather than raising, so the
# chain can fall through cleanly to the next provider. Adapter import is
# lazy so the cockpit boots even if a transitive dep is missing.
# ---------------------------------------------------------------------------


def _at_or_after(bars: list[Any], ts: datetime) -> float | None:
    """Pick the first bar whose ``ts >= ts``; fall back to the last bar.

    Adapters return bars in chronological order. Attribution's question is
    "the close at or after this moment" — usually the next session's
    close. If we asked for a window centered on ``ts`` but ``ts`` lands
    on a weekend, the next available bar (Monday) is the right answer.
    """
    if not bars:
        return None
    for b in bars:
        b_ts = getattr(b, "ts", None)
        if b_ts is None:
            continue
        if b_ts.tzinfo is None:
            b_ts = b_ts.replace(tzinfo=UTC)
        if b_ts >= ts:
            return float(b.close)
    # No bar at-or-after; fall back to the last available bar so we don't
    # silently miss when the operator runs attribution late in the day
    # and ts lands inside today's open candle.
    return float(bars[-1].close)


def alpaca_provider() -> ProviderFn | None:
    """Build an Alpaca-backed provider, or return None if keys are absent."""
    try:
        from packages.data.adapters.alpaca_data import AlpacaDataAdapter
    except ImportError:
        return None
    adapter = AlpacaDataAdapter()
    if not adapter.is_configured():
        return None

    def _fetch(symbol: str, ts: datetime) -> float | None:
        # Pull a small window centered on ts so we get the at-or-after bar
        # even if ts lands on a holiday/weekend.
        start = (ts - timedelta(days=2)).isoformat()
        end = (ts + timedelta(days=4)).isoformat()
        try:
            bars = asyncio.run(adapter.get_bars(symbol, start, end))
        except Exception as exc:
            log.debug("alpaca_provider %s failed: %s", symbol, exc)
            return None
        return _at_or_after(bars, ts)

    return _fetch


def polygon_provider() -> ProviderFn | None:
    """Build a Polygon-backed provider, or return None if no key is set.

    Polygon's free tier is 5 req/min — fine for batch attribution since
    the cache collapses (symbol, day) duplicates.
    """
    import os

    if not os.getenv("POLYGON_API_KEY"):
        return None
    try:
        from packages.data.adapters.polygon import PolygonAdapter
    except ImportError:
        return None
    adapter = PolygonAdapter()

    def _fetch(symbol: str, ts: datetime) -> float | None:
        start = (ts - timedelta(days=2)).date().isoformat()
        end = (ts + timedelta(days=4)).date().isoformat()
        try:
            bars = asyncio.run(adapter.get_daily_bars(symbol, start, end))
        except Exception as exc:
            log.debug("polygon_provider %s failed: %s", symbol, exc)
            return None
        return _at_or_after(bars, ts)

    return _fetch


def yfinance_provider() -> ProviderFn | None:
    """Build a Yahoo-backed provider. Always available (no API key).

    yfinance only takes a relative ``range_`` ("5d", "1mo"...), so we pull
    a generous window and filter client-side. For an attribution job that
    asks about points within the last few months, "6mo" is plenty.
    """
    try:
        from packages.data.adapters.yfinance import YFinanceAdapter
    except ImportError:
        return None
    adapter = YFinanceAdapter()

    def _fetch(symbol: str, ts: datetime) -> float | None:
        # If ts is older than 6 months we need a wider range. Pick the
        # narrowest range that comfortably covers the ask to be polite.
        age_days = (datetime.now(UTC) - ts).days
        if age_days <= 30:
            range_ = "3mo"
        elif age_days <= 180:
            range_ = "1y"
        elif age_days <= 365:
            range_ = "2y"
        else:
            range_ = "5y"
        try:
            bars = asyncio.run(adapter.get_daily_bars(symbol, range_=range_))
        except Exception as exc:
            log.debug("yfinance_provider %s failed: %s", symbol, exc)
            return None
        return _at_or_after(bars, ts)

    return _fetch


def build_default_chain() -> PriceChain:
    """Stand up the default Alpaca → Polygon → yfinance chain.

    Providers that aren't configured (no env vars, no module) are skipped
    silently. yfinance is always added so the chain is never empty.
    """
    chain = PriceChain()
    for name, builder in (
        ("alpaca", alpaca_provider),
        ("polygon", polygon_provider),
        ("yfinance", yfinance_provider),
    ):
        fn = builder()
        if fn is not None:
            chain.add(name, fn)
    return chain


def provider_summary(chain: PriceChain) -> dict[str, Any]:
    """Operator-friendly description of which providers are wired."""
    return {
        "providers": [name for name, _ in chain.providers],
        "stats": chain.stats_dict(),
        "cache_size": len(chain._cache),
    }
