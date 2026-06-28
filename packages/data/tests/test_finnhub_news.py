"""Phase 26 tests for the per-ticker Finnhub news sentiment client.

Locks in:
  * Recency weighting (older headlines count less).
  * Confidence grows with article count + source diversity.
  * Label thresholds (bullish > 0.15, bearish < -0.15, low conf → neutral).
  * Caching reduces network calls; expiry forces a refetch.
  * LRU eviction respects ``cache_max``.
  * No API key → neutral payload, no exception, no cache write.
  * Adapter raise → returns neutral, no exception, no cache write.
  * Empty news → returns neutral but does NOT cache (so transient
    emptiness doesn't lock in a bad value for 15 minutes).
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import httpx
import pytest

from packages.data.adapters.base import DataAdapterError, NewsItem
from packages.data.adapters.finnhub import FinnhubAdapter
from packages.data.finnhub_news import (
    FinnhubNewsClient,
    aggregate_news_sentiment,
    reset_news_client_for_tests,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_item(
    *,
    headline: str,
    summary: str | None = None,
    source: str = "reuters",
    ts: datetime | None = None,
) -> NewsItem:
    return NewsItem(
        symbol="AAPL",
        ts=ts or datetime.now(UTC),
        headline=headline,
        summary=summary,
        url="https://example.com",
        source=source,
    )


class _FakeAdapter(FinnhubAdapter):
    """Fake Finnhub adapter — bypasses network entirely."""

    def __init__(self, items: list[NewsItem] | None = None, raise_on_call: bool = False):
        # Avoid calling FinnhubAdapter.__init__ which creates an httpx client.
        self.api_key = "fake-key"
        self._client = httpx.AsyncClient()  # never used; closed in aclose
        self._items = items or []
        self._calls = 0
        self._raise = raise_on_call

    async def get_company_news(self, symbol, frm, to):  # type: ignore[override]
        self._calls += 1
        if self._raise:
            raise DataAdapterError("simulated finnhub failure")
        return self._items


class _NoKeyAdapter(FinnhubAdapter):
    """Like _FakeAdapter but reports has_key=False."""

    def __init__(self):
        self.api_key = ""
        self._client = httpx.AsyncClient()
        self._calls = 0

    async def get_company_news(self, symbol, frm, to):  # type: ignore[override]
        # Should never be called when has_key is False.
        self._calls += 1
        raise AssertionError("network must not be hit without an API key")


class _FakeClock:
    def __init__(self, start: float = 1_000.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture(autouse=True)
def _reset_singleton():
    reset_news_client_for_tests()
    yield
    reset_news_client_for_tests()


# ---------------------------------------------------------------------------
# Pure-function aggregator
# ---------------------------------------------------------------------------


def test_aggregate_empty_is_neutral():
    out = aggregate_news_sentiment("AAPL", items=[])
    assert out.symbol == "AAPL"
    assert out.score == 0.0
    assert out.confidence == 0.0
    assert out.label == "neutral"
    assert out.article_count == 0
    assert out.source_count == 0


def test_aggregate_bullish_headlines():
    now = datetime(2026, 6, 1, tzinfo=UTC)
    items = [
        _make_item(headline="Apple beats earnings, surges to record high", source="reuters", ts=now),
        _make_item(headline="AAPL upgrade triggers rally, breakout looks strong", source="bloomberg", ts=now),
        _make_item(headline="Analysts bullish on Apple's iPhone outlook", source="wsj", ts=now),
    ]
    out = aggregate_news_sentiment("AAPL", items, now=now)
    assert out.score > 0.3, f"expected clearly positive, got {out.score}"
    assert out.label == "bullish"
    assert out.article_count == 3
    assert out.source_count == 3
    assert out.confidence > 0.0


def test_aggregate_bearish_headlines():
    now = datetime(2026, 6, 1, tzinfo=UTC)
    items = [
        _make_item(headline="Apple plunge after earnings miss; bagholders dump shares", source="reuters", ts=now),
        _make_item(headline="AAPL downgrade triggers crash, downtrend confirmed", source="bloomberg", ts=now),
        _make_item(headline="Analysts bearish on Apple's weak iPhone sales", source="wsj", ts=now),
    ]
    out = aggregate_news_sentiment("AAPL", items, now=now)
    assert out.score < -0.3
    assert out.label == "bearish"


def test_low_confidence_forces_neutral_label():
    """A single strong headline from one source must not flip the brain."""
    now = datetime(2026, 6, 1, tzinfo=UTC)
    items = [
        _make_item(headline="Apple surges to record high on bullish news", source="prnewswire", ts=now),
    ]
    out = aggregate_news_sentiment("AAPL", items, now=now)
    # Score itself can be very positive, but the label must be neutral
    # because confidence (1 article, 1 source) is below the 0.2 floor.
    assert out.article_count == 1
    assert out.source_count == 1
    assert out.confidence < 0.2
    assert out.label == "neutral"


def test_recency_weighting_prefers_today_over_last_week():
    """Old headlines should contribute LESS than fresh ones."""
    now = datetime(2026, 6, 1, tzinfo=UTC)
    fresh = [
        _make_item(headline="bullish breakout rally", source="x", ts=now),
        _make_item(headline="strong upgrade buy", source="y", ts=now),
    ]
    stale = [
        _make_item(headline="bearish crash plunge", source="x", ts=now - timedelta(days=6)),
        _make_item(headline="weak downgrade sell", source="y", ts=now - timedelta(days=6)),
    ]
    out = aggregate_news_sentiment("AAPL", fresh + stale, now=now)
    # Fresh bullish should dominate stale bearish.
    assert out.score > 0.2, f"recency weighting failed, score={out.score}"


def test_old_headlines_outside_window_excluded():
    now = datetime(2026, 6, 1, tzinfo=UTC)
    items = [
        _make_item(headline="ancient news, bullish", source="x", ts=now - timedelta(days=30)),
    ]
    out = aggregate_news_sentiment("AAPL", items, now=now)
    assert out.article_count == 0
    assert out.score == 0.0


# ---------------------------------------------------------------------------
# Network-aware client + cache
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_score_symbol_returns_neutral_without_api_key():
    client = FinnhubNewsClient(
        adapter=_NoKeyAdapter(),
        clock=_FakeClock(),
    )
    out = await client.score_symbol("AAPL")
    assert out.label == "neutral"
    assert out.confidence == 0.0
    assert client.stats()["misses"] == 1
    assert client.stats()["hits"] == 0
    # Nothing cached — the moment a key appears we want a fresh call.
    assert client.stats()["cached_symbols"] == 0


@pytest.mark.asyncio
async def test_score_symbol_caches_repeat_calls():
    now = datetime(2026, 6, 1, tzinfo=UTC)
    items = [
        _make_item(headline="Apple beats earnings, surges to record high", source="reuters", ts=now),
        _make_item(headline="AAPL upgrade triggers rally", source="bloomberg", ts=now),
        _make_item(headline="Bullish analysts: strong outlook", source="wsj", ts=now),
    ]
    fake = _FakeAdapter(items=items)
    client = FinnhubNewsClient(adapter=fake, clock=_FakeClock())

    first = await client.score_symbol("AAPL", now=now)
    second = await client.score_symbol("AAPL", now=now)
    assert first.score == second.score
    assert fake._calls == 1  # second served from cache
    stats = client.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1


@pytest.mark.asyncio
async def test_cache_expiry_forces_refetch():
    now = datetime(2026, 6, 1, tzinfo=UTC)
    items = [
        _make_item(headline="bullish breakout strong", source="r", ts=now),
        _make_item(headline="bullish upgrade rally", source="b", ts=now),
        _make_item(headline="bullish surge", source="w", ts=now),
    ]
    fake = _FakeAdapter(items=items)
    clock = _FakeClock()
    client = FinnhubNewsClient(adapter=fake, cache_ttl_s=60, clock=clock)

    await client.score_symbol("AAPL", now=now)
    clock.advance(61)  # past TTL
    await client.score_symbol("AAPL", now=now)
    assert fake._calls == 2


@pytest.mark.asyncio
async def test_lru_eviction_caps_cache_size():
    now = datetime(2026, 6, 1, tzinfo=UTC)
    items = [
        _make_item(headline="bullish breakout strong", source="r", ts=now),
        _make_item(headline="bullish upgrade rally", source="b", ts=now),
        _make_item(headline="bullish surge", source="w", ts=now),
    ]
    fake = _FakeAdapter(items=items)
    client = FinnhubNewsClient(adapter=fake, cache_max=2, clock=_FakeClock())

    await client.score_symbol("AAPL", now=now)
    await client.score_symbol("MSFT", now=now)
    await client.score_symbol("NVDA", now=now)
    assert client.stats()["cached_symbols"] == 2
    # AAPL was first-in, should have been evicted.
    # MSFT and NVDA remain.
    # Calling AAPL again should be a miss (refetch).
    miss_count_before = client.stats()["misses"]
    await client.score_symbol("AAPL", now=now)
    assert client.stats()["misses"] == miss_count_before + 1


@pytest.mark.asyncio
async def test_adapter_failure_returns_neutral_no_cache():
    fake = _FakeAdapter(raise_on_call=True)
    client = FinnhubNewsClient(adapter=fake, clock=_FakeClock())

    out = await client.score_symbol("AAPL")
    assert out.label == "neutral"
    assert out.confidence == 0.0
    assert client.stats()["errors"] == 1
    assert client.stats()["cached_symbols"] == 0
    # Next call should retry (no poison-cache).
    out2 = await client.score_symbol("AAPL")
    assert fake._calls == 2
    assert out2.label == "neutral"


@pytest.mark.asyncio
async def test_empty_news_is_not_cached():
    """A zero-article response is treated as transient; we refetch
    instead of serving stale emptiness for the full TTL."""
    fake = _FakeAdapter(items=[])
    client = FinnhubNewsClient(adapter=fake, clock=_FakeClock())
    await client.score_symbol("AAPL")
    await client.score_symbol("AAPL")
    assert fake._calls == 2
    assert client.stats()["cached_symbols"] == 0


@pytest.mark.asyncio
async def test_invalidate_drops_specific_symbol():
    now = datetime(2026, 6, 1, tzinfo=UTC)
    items = [
        _make_item(headline="bullish surge breakout strong", source="r", ts=now),
        _make_item(headline="bullish upgrade rally", source="b", ts=now),
        _make_item(headline="bullish strong outlook", source="w", ts=now),
    ]
    fake = _FakeAdapter(items=items)
    client = FinnhubNewsClient(adapter=fake, clock=_FakeClock())
    await client.score_symbol("AAPL", now=now)
    await client.score_symbol("MSFT", now=now)
    assert client.stats()["cached_symbols"] == 2

    client.invalidate("AAPL")
    assert client.stats()["cached_symbols"] == 1
    # Refetch AAPL.
    await client.score_symbol("AAPL", now=now)
    assert fake._calls == 3


@pytest.mark.asyncio
async def test_to_dict_shape():
    now = datetime(2026, 6, 1, tzinfo=UTC)
    items = [
        _make_item(headline="bullish breakout strong", source="r", ts=now),
        _make_item(headline="bullish upgrade rally", source="b", ts=now),
        _make_item(headline="bullish surge", source="w", ts=now),
    ]
    out = aggregate_news_sentiment("AAPL", items, now=now)
    d = out.to_dict()
    assert d["symbol"] == "AAPL"
    assert d["label"] == "bullish"
    assert isinstance(d["score"], float)
    assert isinstance(d["confidence"], float)
    assert isinstance(d["article_count"], int)
    assert isinstance(d["sample_headlines"], list)
    assert d["sample_headlines"]  # non-empty


@pytest.mark.asyncio
async def test_naive_datetime_normalized():
    """A naive ``now`` arg shouldn't crash the recency comparison."""
    naive = datetime(2026, 6, 1)  # no tzinfo
    items = [
        _make_item(headline="bullish breakout", source="r", ts=datetime(2026, 6, 1, tzinfo=UTC)),
    ]
    out = aggregate_news_sentiment("AAPL", items, now=naive)
    assert out.article_count == 1
