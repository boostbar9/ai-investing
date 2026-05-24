"""Tests for sentiment scoring, RSS parsing, and the sentiment adapter."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import httpx
import pytest

from packages.data.adapters.base import NewsItem
from packages.data.adapters.sentiment import (
    SentimentAdapter,
    _parse_rss,
    aggregate_sentiment,
    extract_tickers,
    score_headline,
)

# ---------------------------------------------------------------------------
# Lexicon scoring
# ---------------------------------------------------------------------------


def test_score_empty_returns_zero():
    assert score_headline("") == 0.0
    assert score_headline("   ") == 0.0
    assert score_headline("nothing polarized here at all today") == 0.0


def test_score_strongly_positive():
    s = score_headline("Bullish rally — strong beat, big gains for calls")
    assert s > 0.5


def test_score_strongly_negative():
    s = score_headline("Crash and bloodbath: stocks plunge, downgrade, weak miss")
    assert s < -0.5


def test_score_negation_flip():
    # "not bullish" should NOT count as positive
    s_plain = score_headline("bullish")
    s_neg = score_headline("not bullish")
    assert s_plain > 0
    assert s_neg < 0


def test_score_in_unit_range():
    assert -1.0 <= score_headline("rally crash dump moon") <= 1.0


# ---------------------------------------------------------------------------
# Ticker extraction
# ---------------------------------------------------------------------------


def test_extract_tickers_basic():
    assert extract_tickers("Loaded up on $SPY and $QQQ today") == ["SPY", "QQQ"]


def test_extract_tickers_dedup():
    assert extract_tickers("$AAPL is amazing, $AAPL to the moon") == ["AAPL"]


def test_extract_tickers_no_tickers():
    assert extract_tickers("no dollar signs here") == []
    assert extract_tickers("") == []


def test_extract_tickers_company_names():
    # Mainstream news mentions company names, not $TICKER.
    assert "TSLA" in extract_tickers("Tesla shares rally on robotaxi news")
    assert "AAPL" in extract_tickers("Apple unveils new AI chip")
    assert "SPY" in extract_tickers("S&P 500 hits new high")
    assert "QQQ" in extract_tickers("Nasdaq jumps 2% on tech earnings")


def test_extract_tickers_bare_uppercase_allowlist():
    # Bare tickers only match known symbols.
    assert "AAPL" in extract_tickers("AAPL beats earnings")
    # Common acronyms must NOT be mistaken for tickers.
    assert extract_tickers("The CEO of the company met with the SEC") == []
    assert extract_tickers("Fed signals dovish stance, USA growth strong") == []


def test_extract_tickers_mixed():
    text = "Tesla and $NVDA surge as the Fed holds rates"
    out = extract_tickers(text)
    assert "TSLA" in out
    assert "NVDA" in out
    assert "FED" not in out  # blacklist
    # No duplicates
    assert len(out) == len(set(out))


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def test_aggregate_sentiment_groups_by_symbol():
    now = datetime.now(UTC)
    items = [
        NewsItem(symbol="SPY", ts=now, headline="bullish rally", url="x", source="s"),
        NewsItem(symbol="SPY", ts=now, headline="strong gains", url="x", source="s"),
        NewsItem(symbol="QQQ", ts=now, headline="crash and dump", url="x", source="s"),
    ]
    agg = aggregate_sentiment(items, window_hours=24)
    assert agg["SPY"]["n"] == 2
    assert agg["SPY"]["score"] > 0
    assert agg["QQQ"]["n"] == 1
    assert agg["QQQ"]["score"] < 0


def test_aggregate_sentiment_drops_old_items():
    now = datetime.now(UTC)
    old = now - timedelta(hours=48)
    items = [
        NewsItem(symbol="SPY", ts=old, headline="bullish", url="x", source="s"),
        NewsItem(symbol="SPY", ts=now, headline="rally", url="x", source="s"),
    ]
    agg = aggregate_sentiment(items, window_hours=24)
    assert agg["SPY"]["n"] == 1


def test_aggregate_sentiment_skips_no_symbol():
    now = datetime.now(UTC)
    items = [NewsItem(symbol=None, ts=now, headline="bullish", url="x", source="s")]
    agg = aggregate_sentiment(items)
    assert agg == {}


# ---------------------------------------------------------------------------
# RSS parser
# ---------------------------------------------------------------------------


def test_parse_rss_basic():
    xml = """<?xml version="1.0"?>
    <rss>
      <channel>
        <item>
          <title>$AAPL beats earnings, rally expected</title>
          <link>https://example.com/a</link>
          <pubDate>Wed, 02 Jan 2024 15:00:00 +0000</pubDate>
        </item>
        <item>
          <title><![CDATA[Crash fears for $TSLA]]></title>
          <link>https://example.com/b</link>
        </item>
      </channel>
    </rss>"""
    items = _parse_rss(xml, "https://example.com/feed")
    titles = [i.headline for i in items]
    assert any("AAPL" in t for t in titles)
    assert any("TSLA" in t for t in titles)
    # Each item with a ticker yields one NewsItem per ticker
    assert len(items) == 2
    assert items[0].symbol == "AAPL"
    assert items[1].symbol == "TSLA"


def test_parse_rss_no_tickers_emits_unkeyed_item():
    xml = """<rss><channel>
      <item><title>Market closes flat</title><link>https://x.com/a</link></item>
    </channel></rss>"""
    items = _parse_rss(xml, "https://x.com/feed")
    assert len(items) == 1
    assert items[0].symbol is None


# ---------------------------------------------------------------------------
# Adapter (mocked HTTP)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_reddit_parses_posts():
    class _T(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            assert "/r/stocks/hot.json" in request.url.path
            return httpx.Response(
                200,
                json={
                    "data": {
                        "children": [
                            {
                                "data": {
                                    "title": "$SPY going to moon tomorrow",
                                    "selftext": "calls calls calls",
                                    "created_utc": 1700000000,
                                    "permalink": "/r/stocks/abc",
                                }
                            },
                            {
                                "data": {
                                    "title": "Market crash incoming",
                                    "selftext": "",
                                    "created_utc": 1700000100,
                                    "permalink": "/r/stocks/def",
                                }
                            },
                        ]
                    }
                },
            )

    client = httpx.AsyncClient(transport=_T(), base_url="https://www.reddit.com")
    adapter = SentimentAdapter(client=client)
    items = await adapter.fetch_reddit("stocks", limit=10)
    assert len(items) == 2
    assert items[0].symbol == "SPY"
    assert items[1].symbol is None  # no ticker mention
    await adapter.aclose()


@pytest.mark.asyncio
async def test_fetch_all_tolerates_failures():
    class _T(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            # Every call fails — fetch_all should NOT raise
            return httpx.Response(500, text="server error")

    client = httpx.AsyncClient(transport=_T(), base_url="https://www.reddit.com")
    adapter = SentimentAdapter(
        subreddits=("stocks",),
        rss_feeds=("https://example.com/feed",),
        client=client,
    )
    items = await adapter.fetch_all()
    assert items == []
    await adapter.aclose()
