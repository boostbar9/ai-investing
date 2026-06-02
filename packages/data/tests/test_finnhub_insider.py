"""Phase 27 tests — Finnhub insider-transactions cluster signal.

Locks in:
  * Empty txns → neutral with confidence 0 and label "neutral".
  * Cluster detection: 3 directors fire, a lone VP sell does not.
  * Buy/sell asymmetry: buys outweigh sells of equal notional in score.
  * Seniority weighting: CEO > Director > VP in cluster_score.
  * Single-buyer confidence cap at 0.15.
  * Cluster_buy forces confidence floor of 0.35.
  * A single CEO alone does NOT fire cluster_buy (cluster needs ≥2 names).
  * Outside the cluster window, buys still count toward score but not
    toward cluster_score.
  * Cache hit/miss accounting + LRU eviction.
  * No API key → neutral, no network call.
  * Adapter failure → neutral, error counter increments, no cache write.
  * Empty result is NOT cached (treat as transient).
"""
from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import httpx
import pytest

from packages.data.adapters.base import DataAdapterError
from packages.data.adapters.finnhub import FinnhubAdapter
from packages.data.finnhub_insider import (
    CLUSTER_THRESHOLD,
    FinnhubInsiderClient,
    InsiderTransaction,
    aggregate_insider_signal,
    reset_insider_client_for_tests,
    seniority_weight,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _txn(
    *,
    name: str,
    title: str,
    code: str = "P",
    shares: float = 1000.0,
    price: float = 200.0,
    when: date | None = None,
    symbol: str = "AAPL",
) -> InsiderTransaction:
    when = when or date(2026, 6, 1)
    return InsiderTransaction(
        symbol=symbol,
        name=name,
        title=title,
        transaction_date=when,
        transaction_code=code,
        shares=shares,
        price=price,
    )


class _FakeAdapter(FinnhubAdapter):
    """Looks like a Finnhub adapter but never talks to the network."""

    def __init__(self):
        self.api_key = "fake-key"
        self._client = httpx.AsyncClient()


class _NoKeyAdapter(FinnhubAdapter):
    def __init__(self):
        self.api_key = ""
        self._client = httpx.AsyncClient()


class _FakeFetcher:
    """Callable used as the ``fetcher`` injection point on the client."""

    def __init__(self, items: list[InsiderTransaction], *, raise_err: bool = False):
        self._items = items
        self._raise = raise_err
        self.calls = 0

    async def __call__(self, adapter, symbol, *, lookback_days=30):
        self.calls += 1
        if self._raise:
            raise DataAdapterError("simulated finnhub failure")
        return [t for t in self._items if t.symbol == symbol.upper()] or list(self._items)


class _FakeClock:
    def __init__(self, start: float = 1_000.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture(autouse=True)
def _reset_singleton():
    reset_insider_client_for_tests()
    yield
    reset_insider_client_for_tests()


# ---------------------------------------------------------------------------
# Pure-function aggregator
# ---------------------------------------------------------------------------


def test_seniority_weight_picks_highest_match():
    assert seniority_weight("Chief Executive Officer") == 2.0
    assert seniority_weight("President & CEO") == 2.0  # CEO dominates
    assert seniority_weight("CFO") == 1.8
    assert seniority_weight("Director") == 1.2
    assert seniority_weight("VP, Engineering") == 1.0
    assert seniority_weight("Janitor") == 1.0  # baseline
    assert seniority_weight(None) == 1.0
    assert seniority_weight("") == 1.0


def test_empty_transactions_returns_neutral():
    now = datetime(2026, 6, 1, tzinfo=UTC)
    s = aggregate_insider_signal("AAPL", [], now=now)
    assert s.symbol == "AAPL"
    assert s.score == 0.0
    assert s.confidence == 0.0
    assert s.label == "neutral"
    assert s.cluster_buy is False
    assert s.cluster_score == 0.0
    assert s.buy_count == 0
    assert s.sell_count == 0
    assert s.unique_buyers == 0
    assert s.top_buyers == ()


def test_cluster_three_directors_fires_buy_vp_sell_ignored():
    """3 directors all buying in <14 days → cluster_buy.

    A lone VP selling in the same window does NOT cancel the cluster
    (cluster_score is buy-side only). The VP shows up in sell_count.
    """
    today = date(2026, 6, 1)
    now = datetime(2026, 6, 1, tzinfo=UTC)
    txns = [
        _txn(name="Alice", title="Director", when=today, code="P"),
        _txn(name="Bob", title="Director", when=today - timedelta(days=2), code="P"),
        _txn(name="Carol", title="Director", when=today - timedelta(days=5), code="P"),
        # The VP sells — must NOT contribute to cluster_score.
        _txn(name="Vince", title="Vice President", when=today, code="S", shares=-500),
    ]
    s = aggregate_insider_signal("AAPL", txns, now=now)
    assert s.cluster_buy is True
    assert s.label == "cluster_buy"
    # 3 directors × 1.2 weight = 3.6 (above CLUSTER_THRESHOLD).
    assert s.cluster_score >= CLUSTER_THRESHOLD
    assert s.cluster_score == pytest.approx(3.6, rel=1e-9)
    assert s.unique_buyers == 3
    assert s.buy_count == 3
    assert s.sell_count == 1
    # Cluster floor on confidence.
    assert s.confidence >= 0.35
    # Top buyers by seniority (all directors tied at 1.2) — we just
    # check we got 3 names back.
    assert len(s.top_buyers) == 3


def test_single_ceo_alone_is_not_a_cluster():
    """One CEO buying alone hits the seniority-weight threshold but
    a cluster requires ≥2 distinct insiders. Should land as 'neutral'
    because single-buyer confidence is capped at 0.15."""
    today = date(2026, 6, 1)
    now = datetime(2026, 6, 1, tzinfo=UTC)
    txns = [_txn(name="Tim Cook", title="Chief Executive Officer", when=today, code="P")]
    s = aggregate_insider_signal("AAPL", txns, now=now)
    assert s.cluster_buy is False
    assert s.unique_buyers == 1
    assert s.confidence == 0.15  # capped
    assert s.label == "neutral"


def test_buy_sell_asymmetry_in_score():
    """Equal dollar buys and sells should net positive (buys 2x-weighted)."""
    now = datetime(2026, 6, 1, tzinfo=UTC)
    today = date(2026, 6, 1)
    txns = [
        # $200k buy
        _txn(name="A", title="Director", when=today, code="P", shares=1000, price=200),
        # $200k sell
        _txn(name="B", title="Director", when=today, code="S", shares=-1000, price=200),
    ]
    s = aggregate_insider_signal("AAPL", txns, now=now)
    # raw_buy = 200k * 2 = 400k; raw_sell = 200k; score = (400-200)/600 = 1/3
    assert s.score == pytest.approx(1.0 / 3.0, rel=1e-6)
    assert s.buy_count == 1
    assert s.sell_count == 1


def test_seniority_weighting_ceo_outranks_director():
    """A buy by a CEO contributes more to cluster_score than a buy by a Director."""
    today = date(2026, 6, 1)
    now = datetime(2026, 6, 1, tzinfo=UTC)
    txns_ceo_dir = [
        _txn(name="A", title="Chief Executive Officer", when=today, code="P"),
        _txn(name="B", title="Director", when=today, code="P"),
    ]
    txns_two_dirs = [
        _txn(name="A", title="Director", when=today, code="P"),
        _txn(name="B", title="Director", when=today, code="P"),
    ]
    s1 = aggregate_insider_signal("AAPL", txns_ceo_dir, now=now)
    s2 = aggregate_insider_signal("AAPL", txns_two_dirs, now=now)
    # CEO(2.0) + Director(1.2) = 3.2  vs  Director(1.2) + Director(1.2) = 2.4
    assert s1.cluster_score == pytest.approx(3.2, rel=1e-9)
    assert s2.cluster_score == pytest.approx(2.4, rel=1e-9)
    assert s1.cluster_score > s2.cluster_score
    # CEO should be first in top_buyers.
    assert s1.top_buyers[0] == "A"


def test_single_buyer_confidence_cap():
    """Even a $50M single-CEO buy stays ≤ 0.15 confidence."""
    today = date(2026, 6, 1)
    now = datetime(2026, 6, 1, tzinfo=UTC)
    # 250k shares × $200 = $50M
    txns = [_txn(name="CEO", title="Chief Executive Officer", when=today,
                 code="P", shares=250_000, price=200)]
    s = aggregate_insider_signal("AAPL", txns, now=now)
    assert s.unique_buyers == 1
    assert s.confidence <= 0.15
    assert s.cluster_buy is False  # single name


def test_cluster_buy_forces_confidence_floor():
    """A tiny-notional cluster buy still gets confidence ≥ 0.35
    so the brain actually sees it."""
    today = date(2026, 6, 1)
    now = datetime(2026, 6, 1, tzinfo=UTC)
    # 2 directors, but only $500 each — total $1k notional.
    txns = [
        _txn(name="A", title="Director", when=today, code="P", shares=100, price=5),
        _txn(name="B", title="Director", when=today, code="P", shares=100, price=5),
    ]
    s = aggregate_insider_signal("AAPL", txns, now=now)
    assert s.cluster_buy is True
    assert s.confidence >= 0.35
    assert s.label == "cluster_buy"


def test_buys_outside_cluster_window_excluded_from_cluster_score():
    """Buys older than CLUSTER_WINDOW_DAYS still count toward score
    and buy_count, but NOT toward cluster_score."""
    today = date(2026, 6, 1)
    now = datetime(2026, 6, 1, tzinfo=UTC)
    txns = [
        _txn(name="A", title="Director", when=today - timedelta(days=20), code="P"),
        _txn(name="B", title="Director", when=today - timedelta(days=22), code="P"),
    ]
    s = aggregate_insider_signal("AAPL", txns, now=now)
    assert s.buy_count == 2
    assert s.cluster_score == 0.0
    assert s.cluster_buy is False


def test_heavy_selling_label():
    """5 distinct EVPs each dumping $2M of stock → heavy_selling."""
    today = date(2026, 6, 1)
    now = datetime(2026, 6, 1, tzinfo=UTC)
    txns = [
        _txn(name=f"P{i}", title="EVP", when=today, code="S",
             shares=-10_000, price=200)
        for i in range(5)
    ]
    s = aggregate_insider_signal("AAPL", txns, now=now)
    assert s.sell_count == 5
    assert s.score == -1.0  # buy_notional==0
    assert s.confidence > 0.5  # 5 unique sellers, $10M notional
    assert s.label == "heavy_selling"


def test_to_dict_shape():
    today = date(2026, 6, 1)
    now = datetime(2026, 6, 1, tzinfo=UTC)
    txns = [
        _txn(name="A", title="Director", when=today, code="P"),
        _txn(name="B", title="Director", when=today, code="P"),
    ]
    d = aggregate_insider_signal("AAPL", txns, now=now).to_dict()
    assert d["symbol"] == "AAPL"
    assert d["label"] == "cluster_buy"
    assert isinstance(d["score"], float)
    assert isinstance(d["confidence"], float)
    assert isinstance(d["cluster_score"], float)
    assert isinstance(d["cluster_buy"], bool)
    assert isinstance(d["top_buyers"], list)
    assert "fresh_at" in d


def test_naive_datetime_does_not_crash():
    naive = datetime(2026, 6, 1)
    s = aggregate_insider_signal(
        "AAPL",
        [_txn(name="A", title="Director", when=date(2026, 6, 1), code="P")],
        now=naive,
    )
    assert s.buy_count == 1


# ---------------------------------------------------------------------------
# Network-aware client + cache
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_score_symbol_returns_neutral_without_api_key():
    fetcher = _FakeFetcher(items=[])  # would explode if called
    client = FinnhubInsiderClient(
        adapter=_NoKeyAdapter(),
        clock=_FakeClock(),
        fetcher=fetcher,
    )
    out = await client.score_symbol("AAPL")
    assert out.label == "neutral"
    assert out.confidence == 0.0
    assert fetcher.calls == 0
    stats = client.stats()
    assert stats["cached_symbols"] == 0
    assert stats["misses"] == 1


@pytest.mark.asyncio
async def test_score_symbol_caches_repeat_calls():
    today = date(2026, 6, 1)
    now = datetime(2026, 6, 1, tzinfo=UTC)
    items = [
        _txn(name="A", title="Director", when=today, code="P"),
        _txn(name="B", title="Director", when=today, code="P"),
    ]
    fetcher = _FakeFetcher(items=items)
    client = FinnhubInsiderClient(
        adapter=_FakeAdapter(),
        clock=_FakeClock(),
        fetcher=fetcher,
    )

    first = await client.score_symbol("AAPL", now=now)
    second = await client.score_symbol("AAPL", now=now)
    assert fetcher.calls == 1  # second served from cache
    assert first.score == second.score
    assert first.cluster_buy is True
    stats = client.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1


@pytest.mark.asyncio
async def test_cache_expiry_forces_refetch():
    today = date(2026, 6, 1)
    now = datetime(2026, 6, 1, tzinfo=UTC)
    items = [
        _txn(name="A", title="Director", when=today, code="P"),
        _txn(name="B", title="Director", when=today, code="P"),
    ]
    fetcher = _FakeFetcher(items=items)
    clock = _FakeClock()
    client = FinnhubInsiderClient(
        adapter=_FakeAdapter(),
        cache_ttl_s=60,
        clock=clock,
        fetcher=fetcher,
    )
    await client.score_symbol("AAPL", now=now)
    clock.advance(61)
    await client.score_symbol("AAPL", now=now)
    assert fetcher.calls == 2


@pytest.mark.asyncio
async def test_lru_eviction_caps_cache_size():
    today = date(2026, 6, 1)
    now = datetime(2026, 6, 1, tzinfo=UTC)
    items = [
        _txn(name="A", title="Director", when=today, code="P"),
        _txn(name="B", title="Director", when=today, code="P"),
    ]
    fetcher = _FakeFetcher(items=items)
    client = FinnhubInsiderClient(
        adapter=_FakeAdapter(),
        cache_max=2,
        clock=_FakeClock(),
        fetcher=fetcher,
    )
    await client.score_symbol("AAPL", now=now)
    await client.score_symbol("MSFT", now=now)
    await client.score_symbol("NVDA", now=now)
    stats = client.stats()
    assert stats["cached_symbols"] == 2
    # AAPL was first-in → evicted. Refetch becomes a miss.
    misses_before = stats["misses"]
    await client.score_symbol("AAPL", now=now)
    assert client.stats()["misses"] == misses_before + 1


@pytest.mark.asyncio
async def test_adapter_failure_returns_neutral_no_cache():
    fetcher = _FakeFetcher(items=[], raise_err=True)
    client = FinnhubInsiderClient(
        adapter=_FakeAdapter(),
        clock=_FakeClock(),
        fetcher=fetcher,
    )
    out = await client.score_symbol("AAPL")
    assert out.label == "neutral"
    assert out.confidence == 0.0
    stats = client.stats()
    assert stats["errors"] == 1
    assert stats["cached_symbols"] == 0
    # Next call retries (no poison-cache).
    await client.score_symbol("AAPL")
    assert fetcher.calls == 2


@pytest.mark.asyncio
async def test_empty_result_is_not_cached():
    fetcher = _FakeFetcher(items=[])
    client = FinnhubInsiderClient(
        adapter=_FakeAdapter(),
        clock=_FakeClock(),
        fetcher=fetcher,
    )
    await client.score_symbol("AAPL")
    await client.score_symbol("AAPL")
    assert fetcher.calls == 2  # transient emptiness → always refetch
    assert client.stats()["cached_symbols"] == 0


@pytest.mark.asyncio
async def test_invalidate_drops_specific_symbol():
    today = date(2026, 6, 1)
    now = datetime(2026, 6, 1, tzinfo=UTC)
    items = [
        _txn(name="A", title="Director", when=today, code="P"),
        _txn(name="B", title="Director", when=today, code="P"),
    ]
    fetcher = _FakeFetcher(items=items)
    client = FinnhubInsiderClient(
        adapter=_FakeAdapter(),
        clock=_FakeClock(),
        fetcher=fetcher,
    )
    await client.score_symbol("AAPL", now=now)
    await client.score_symbol("MSFT", now=now)
    assert client.stats()["cached_symbols"] == 2

    client.invalidate("AAPL")
    assert client.stats()["cached_symbols"] == 1
    await client.score_symbol("AAPL", now=now)
    assert fetcher.calls == 3
