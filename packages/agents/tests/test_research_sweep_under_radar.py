"""Tests for the under-the-radar catalyst discovery lane in research_sweep.

Covers the ADDITIVE, fail-safe source that surfaces under-followed
small/micro-cap names WITH a real catalyst:

  * :func:`candidates_from_under_radar` — catalyst-gated, liquidity-gated,
    under-radar-only; tags ``lane="under_radar"`` + catalyst type/score.
  * :func:`merge_under_radar_candidates` — additive union, never displaces.

All inputs are plain objects — no network.
"""

from __future__ import annotations

from datetime import UTC, datetime

from packages.agents.research_sweep import (
    Candidate,
    candidates_from_under_radar,
    merge_under_radar_candidates,
)
from packages.data.adapters.base import NewsItem


def _news(headline: str, *, symbol: str, summary: str | None = None) -> NewsItem:
    return NewsItem(
        symbol=symbol,
        ts=datetime(2024, 1, 1, tzinfo=UTC),
        headline=headline,
        summary=summary,
        url="https://example.com",
        source="rss",
    )


def _liq(price, dvol, spread, *, market_cap=None):
    d = {"price": price, "avg_dollar_volume": dvol, "spread_pct": spread}
    if market_cap is not None:
        d["market_cap"] = market_cap
    return d


def test_surfaces_under_radar_catalyst_name():
    news = [
        _news("$BCRX receives FDA approval for rare disease drug", symbol="BCRX"),
        _news("$BCRX FDA approval confirmed", symbol="BCRX"),
    ]
    liq = {"BCRX": _liq(1.50, 5_000_000, 0.01, market_cap=3e8)}
    cands = candidates_from_under_radar(news, liquidity_by_symbol=liq, min_mentions=1)
    assert [c.symbol for c in cands] == ["BCRX"]
    c = cands[0]
    assert c.lane == "under_radar"
    assert c.catalyst_type == "fda"
    assert c.catalyst_score > 0.0
    assert c.confidence == c.catalyst_score
    assert "under_radar" in c.sources


def test_excludes_mainstream_megacap():
    news = [_news("$AAPL receives FDA-style approval blah", symbol="AAPL")]
    liq = {"AAPL": _liq(190.0, 1e10, 0.001, market_cap=3e12)}
    cands = candidates_from_under_radar(news, liquidity_by_symbol=liq, min_mentions=1)
    assert cands == []


def test_excludes_no_catalyst_name():
    news = [_news("$TINY trades sideways on no news", symbol="TINY")]
    liq = {"TINY": _liq(1.50, 5_000_000, 0.01, market_cap=3e8)}
    cands = candidates_from_under_radar(news, liquidity_by_symbol=liq, min_mentions=1)
    assert cands == []


def test_excludes_low_volume_name():
    news = [_news("$LOWV gets FDA nod for new therapy", symbol="LOWV")]
    liq = {"LOWV": _liq(2.0, 100_000, 0.01, market_cap=1e8)}
    cands = candidates_from_under_radar(news, liquidity_by_symbol=liq, min_mentions=1)
    assert cands == []


def test_excludes_missing_liquidity_data_failsafe():
    news = [_news("$NODATA FDA approval granted", symbol="NODATA")]
    # No entry in liquidity map => fail-safe exclude.
    cands = candidates_from_under_radar(news, liquidity_by_symbol={}, min_mentions=1)
    assert cands == []


def test_excludes_wide_spread_name():
    news = [_news("$WIDE wins major government contract award", symbol="WIDE")]
    liq = {"WIDE": _liq(2.0, 5_000_000, 0.25, market_cap=1e8)}
    cands = candidates_from_under_radar(news, liquidity_by_symbol=liq, min_mentions=1)
    assert cands == []


def test_empty_news_is_noop():
    assert candidates_from_under_radar([], liquidity_by_symbol={}) == []


def test_merge_is_additive_appends_new_symbol():
    base = [
        Candidate(
            symbol="AAPL",
            signal_kind="sentiment",  # type: ignore[arg-type]
            thesis="x",
            confidence=0.5,
            sentiment_score=0.5,
            mentions=5,
            sources=["reddit"],
            sample_headlines=["seed"],
            created_at="2026-01-01T00:00:00+00:00",
        )
    ]
    news = [_news("$BCRX FDA approval confirmed", symbol="BCRX")]
    liq = {"BCRX": _liq(1.50, 5_000_000, 0.01, market_cap=3e8)}
    ur = candidates_from_under_radar(news, liquidity_by_symbol=liq, min_mentions=1)
    merged = merge_under_radar_candidates(base, ur)
    syms = {c.symbol for c in merged}
    assert syms == {"AAPL", "BCRX"}
    # base candidate preserved unchanged
    aapl = next(c for c in merged if c.symbol == "AAPL")
    assert aapl.confidence == 0.5


def test_merge_empty_under_radar_is_noop():
    base = [
        Candidate(
            symbol="AAPL",
            signal_kind="sentiment",  # type: ignore[arg-type]
            thesis="x",
            confidence=0.5,
            sentiment_score=0.5,
            mentions=5,
            sources=["reddit"],
            sample_headlines=["seed"],
            created_at="2026-01-01T00:00:00+00:00",
        )
    ]
    merged = merge_under_radar_candidates(base, [])
    assert [c.symbol for c in merged] == ["AAPL"]
