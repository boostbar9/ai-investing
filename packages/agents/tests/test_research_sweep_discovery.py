"""Tests for the Phase 2 discovery overhaul (the 0-candidates fix).

These cover the two ADDITIVE, fail-safe candidate sources that keep the
sweep alive when Reddit/StockTwits are 403-blocked:

  * ``candidates_from_news`` -- ticker extraction from the RSS/news feed
    that already works (extraction, mention-count confidence, top-N,
    empty input = no-op).
  * ``merge_news_candidates`` / ``merge_movers_candidates`` -- ADDITIVE
    union semantics (empty = no-op, never displaces/down-ranks existing
    sentiment/portfolio candidates).
  * ``_gather_finnhub_movers`` -- fail-safe behaviour with the network
    mocked (no key => no call; adapter error => []; happy path counts
    ticker frequency across general-news headlines).

All network is mocked -- no live calls.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from packages.agents.research_sweep import (
    Candidate,
    _gather_finnhub_movers,
    _symbols_from_news,
    candidates_from_news,
    merge_movers_candidates,
    merge_news_candidates,
    run_sweep,
)
from packages.data.adapters.base import NewsItem


def _news(headline: str, *, symbol: str | None = None, summary: str | None = None,
          when: datetime | None = None) -> NewsItem:
    return NewsItem(
        symbol=symbol,
        ts=when or datetime.now(UTC),
        headline=headline,
        summary=summary,
        url="https://example.com",
        source="rss",
    )


def _cand(symbol: str, *, confidence: float, kind: str = "sentiment") -> Candidate:
    return Candidate(
        symbol=symbol,
        signal_kind=kind,  # type: ignore[arg-type]
        thesis="x",
        confidence=confidence,
        sentiment_score=0.5,
        mentions=5,
        sources=["reddit", "rss"],
        sample_headlines=["seed"],
        created_at="2026-01-01T00:00:00+00:00",
    )


# ---------------------------------------------------------------------------
# _symbols_from_news
# ---------------------------------------------------------------------------


def test_symbols_from_news_extracts_cashtag_and_known_ticker() -> None:
    item = _news("$NVDA breakout as Apple rallies into earnings")
    syms = _symbols_from_news(item)
    assert "NVDA" in syms
    assert "AAPL" in syms  # "Apple" -> AAPL via the name map


def test_symbols_from_news_folds_in_pretagged_symbol() -> None:
    # A Finnhub `related` / RSS-tagged symbol with no mention in the headline
    # text should still be surfaced via the cashtag path.
    item = _news("Quarterly results beat expectations", symbol="MSFT")
    assert "MSFT" in _symbols_from_news(item)


def test_symbols_from_news_drops_blacklisted_junk() -> None:
    # "$CEO" would pass the raw cashtag regex, but the shared denylist must
    # strip it so we never emit junk tickers.
    item = _news("The $CEO said the $USA economy is strong")
    assert _symbols_from_news(item) == []


# ---------------------------------------------------------------------------
# candidates_from_news
# ---------------------------------------------------------------------------


def test_candidates_from_news_empty_is_noop() -> None:
    assert candidates_from_news([]) == []


def test_candidates_from_news_ticker_less_input_is_noop() -> None:
    items = [_news("markets mixed amid macro uncertainty") for _ in range(5)]
    assert candidates_from_news(items) == []


def test_candidates_from_news_counts_mentions_and_builds_news_kind() -> None:
    items = [_news(f"$NVDA rallies hard #{i}") for i in range(4)]
    items.append(_news("Apple ships new product"))  # single AAPL mention
    out = candidates_from_news(items)
    by_sym = {c.symbol: c for c in out}
    assert by_sym["NVDA"].mentions == 4
    assert by_sym["NVDA"].signal_kind == "news"
    assert by_sym["NVDA"].sources == ["rss_news"]
    assert by_sym["AAPL"].mentions == 1


def test_candidates_from_news_confidence_rises_with_mentions() -> None:
    # Same neutral headline text => same per-headline score; more mentions
    # must yield >= confidence (mention factor only grows).
    few = candidates_from_news([_news("$NVDA update") for _ in range(2)])
    many = candidates_from_news([_news("$NVDA update") for _ in range(15)])
    assert few and many
    assert many[0].confidence >= few[0].confidence


def test_candidates_from_news_respects_max_candidates() -> None:
    # Five distinct known tickers, capped at 2.
    items = [
        _news("$NVDA up"), _news("$AAPL up"), _news("$MSFT up"),
        _news("$TSLA up"), _news("$AMZN up"),
    ]
    out = candidates_from_news(items, max_candidates=2)
    assert len(out) == 2


def test_candidates_from_news_respects_min_mentions() -> None:
    items = [_news("$NVDA up"), _news("$AAPL up"), _news("$AAPL again")]
    out = candidates_from_news(items, min_mentions=2)
    assert {c.symbol for c in out} == {"AAPL"}


# ---------------------------------------------------------------------------
# merge_news_candidates
# ---------------------------------------------------------------------------


def test_merge_news_empty_is_noop_same_object() -> None:
    base = [_cand("NVDA", confidence=0.8)]
    assert merge_news_candidates(base, []) is base


def test_merge_news_appends_only_new_symbols() -> None:
    base = [_cand("NVDA", confidence=0.8)]
    news = [
        Candidate(symbol="SOFI", signal_kind="news", thesis="t",
                  confidence=0.3, sources=["rss_news"]),
    ]
    out = merge_news_candidates(base, news)
    assert {c.symbol for c in out} == {"NVDA", "SOFI"}


def test_merge_news_never_displaces_or_downranks_existing() -> None:
    base = [_cand("NVDA", confidence=0.8)]
    # A news candidate for an already-present, higher-confidence symbol must
    # NOT change its kind or confidence -- only union the source provenance.
    news = [Candidate(symbol="NVDA", signal_kind="news", thesis="t",
                      confidence=0.99, sources=["rss_news"])]
    out = merge_news_candidates(base, news)
    nvda = next(c for c in out if c.symbol == "NVDA")
    assert nvda.signal_kind == "sentiment"
    assert nvda.confidence == 0.8
    assert "rss_news" in nvda.sources  # provenance unioned in


# ---------------------------------------------------------------------------
# merge_movers_candidates
# ---------------------------------------------------------------------------


def test_merge_movers_empty_is_noop_same_object() -> None:
    base = [_cand("NVDA", confidence=0.8)]
    assert merge_movers_candidates(base, []) is base


def test_merge_movers_adds_low_confidence_scan_candidates() -> None:
    base = [_cand("NVDA", confidence=0.8)]
    out = merge_movers_candidates(base, ["MSFT", "", "  "])
    msft = next(c for c in out if c.symbol == "MSFT")
    assert msft.signal_kind == "scan"
    assert msft.confidence == 0.3
    assert msft.sources == ["finnhub_movers"]
    # blank/whitespace entries are ignored
    assert {c.symbol for c in out} == {"NVDA", "MSFT"}


def test_merge_movers_never_downranks_existing() -> None:
    base = [_cand("NVDA", confidence=0.8)]
    out = merge_movers_candidates(base, ["NVDA"])
    nvda = next(c for c in out if c.symbol == "NVDA")
    assert nvda.confidence == 0.8
    assert nvda.signal_kind == "sentiment"  # not displaced by a scan kind


# ---------------------------------------------------------------------------
# _gather_finnhub_movers (network mocked)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gather_finnhub_movers_no_key_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
    assert await _gather_finnhub_movers() == []


@pytest.mark.asyncio
async def test_gather_finnhub_movers_counts_ticker_frequency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FINNHUB_API_KEY", "test-key")

    now = datetime.now(UTC)
    fake_news = [
        _news("$NVDA jumps on AI demand", when=now),
        _news("More $NVDA momentum", when=now - timedelta(hours=1)),
        _news("Apple unveils device", symbol="AAPL", when=now),
    ]

    class _FakeAdapter:
        has_key = True

        def __init__(self, *a, **k) -> None:
            pass

        async def get_market_news(self, category: str = "general"):
            assert category == "general"
            return fake_news

        async def aclose(self) -> None:
            return None

    monkeypatch.setattr(
        "packages.data.adapters.finnhub.FinnhubAdapter", _FakeAdapter
    )
    syms = await _gather_finnhub_movers()
    # NVDA mentioned twice => ranked ahead of AAPL.
    assert syms[0] == "NVDA"
    assert "AAPL" in syms


@pytest.mark.asyncio
async def test_gather_finnhub_movers_failsafe_on_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FINNHUB_API_KEY", "test-key")

    class _BoomAdapter:
        has_key = True

        def __init__(self, *a, **k) -> None:
            pass

        async def get_market_news(self, category: str = "general"):
            raise RuntimeError("finnhub is down")

        async def aclose(self) -> None:
            return None

    monkeypatch.setattr(
        "packages.data.adapters.finnhub.FinnhubAdapter", _BoomAdapter
    )
    # Must NOT raise -- a dead feed degrades to an empty universe.
    assert await _gather_finnhub_movers() == []


# ---------------------------------------------------------------------------
# run_sweep wiring: news candidates survive a fully-dead sentiment path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_sweep_surfaces_news_candidates_without_sentiment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The headline regression this task fixes: even with the sentiment
    aggregation yielding ~nothing (Reddit/StockTwits blocked), RSS headlines
    must still produce candidates."""
    monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
    from unittest.mock import AsyncMock

    now = datetime.now(UTC)
    # NewsItems with symbol=None (the case aggregate_sentiment drops) but
    # whose headlines clearly name tradable tickers.
    items = [
        _news("Apple rallies into product launch", when=now),
        _news("$NVDA breaks out on AI demand", when=now),
        _news("Tesla deliveries beat estimates", when=now),
    ]
    fake_adapter = AsyncMock()
    fake_adapter.fetch_all.return_value = items
    fake_adapter.aclose.return_value = None

    result = await run_sweep(
        adapter=fake_adapter,
        portfolio_symbols=[],
        enable_trust_gate=False,
    )
    assert result.status == "done"
    symbols = {c.symbol for c in result.candidates}
    assert {"AAPL", "NVDA", "TSLA"} <= symbols
    assert "news_candidates" in result.sources_meta
