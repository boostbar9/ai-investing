"""Tests for the Phase 10 Yahoo Finance per-ticker adapter."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from packages.data.adapters.yahoo_news import YahooNewsAdapter


def _client(handler) -> httpx.AsyncClient:
    """Build an AsyncClient that routes through an in-process MockTransport."""
    transport = httpx.MockTransport(handler)
    return httpx.AsyncClient(
        transport=transport,
        headers={"User-Agent": "test", "Accept": "application/json"},
    )


# ---------------------------------------------------------------------------
# Per-ticker news
# ---------------------------------------------------------------------------


def test_fetch_ticker_news_happy_path():
    def handler(req: httpx.Request) -> httpx.Response:
        assert "search" in req.url.path
        return httpx.Response(
            200,
            json={
                "news": [
                    {
                        "title": "NVDA pops on earnings",
                        "link": "https://example.com/a",
                        "publisher": "Reuters",
                        "providerPublishTime": 1_700_000_000,
                    },
                    {
                        "title": "Analyst upgrades NVDA",
                        "link": "https://example.com/b",
                        "publisher": "Bloomberg",
                        "providerPublishTime": 1_700_001_000,
                    },
                ]
            },
        )

    async def go():
        adapter = YahooNewsAdapter(client=_client(handler))
        try:
            items = await adapter.fetch_ticker_news("NVDA", limit=5)
        finally:
            await adapter.aclose()
        return items

    items = asyncio.run(go())
    assert len(items) == 2
    assert items[0].symbol == "NVDA"
    assert items[0].source == "yahoo/reuters"
    assert items[1].source == "yahoo/bloomberg"
    assert items[0].headline.startswith("NVDA pops")


def test_fetch_ticker_news_skips_items_missing_link_or_title():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "news": [
                    {"title": "no link", "publisher": "x"},
                    {"link": "https://x.com/", "publisher": "x"},
                    {
                        "title": "good",
                        "link": "https://example.com/",
                        "publisher": "WSJ",
                        "providerPublishTime": 1_700_000_000,
                    },
                ]
            },
        )

    async def go():
        adapter = YahooNewsAdapter(client=_client(handler))
        try:
            return await adapter.fetch_ticker_news("AAPL")
        finally:
            await adapter.aclose()

    items = asyncio.run(go())
    assert len(items) == 1
    assert items[0].source == "yahoo/wsj"


def test_fetch_ticker_news_returns_empty_on_5xx():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="upstream wedged")

    async def go():
        adapter = YahooNewsAdapter(client=_client(handler))
        try:
            return await adapter.fetch_ticker_news("AAPL")
        finally:
            await adapter.aclose()

    assert asyncio.run(go()) == []


def test_fetch_ticker_news_returns_empty_on_bad_json():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json")

    async def go():
        adapter = YahooNewsAdapter(client=_client(handler))
        try:
            return await adapter.fetch_ticker_news("AAPL")
        finally:
            await adapter.aclose()

    assert asyncio.run(go()) == []


def test_fetch_ticker_news_sanitizes_publisher():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "news": [
                    {
                        "title": "x",
                        "link": "https://x.com/",
                        "publisher": "Some! Weird/Publisher Name",
                        "providerPublishTime": 1_700_000_000,
                    }
                ]
            },
        )

    async def go():
        adapter = YahooNewsAdapter(client=_client(handler))
        try:
            return await adapter.fetch_ticker_news("X")
        finally:
            await adapter.aclose()

    items = asyncio.run(go())
    assert items[0].source.startswith("yahoo/")
    # Underscores instead of punctuation, lowercased.
    assert "/" not in items[0].source.split("yahoo/")[1]
    assert " " not in items[0].source


# ---------------------------------------------------------------------------
# Analyst signal
# ---------------------------------------------------------------------------


def test_fetch_analyst_signal_extracts_upgrade():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "quoteSummary": {
                    "result": [
                        {
                            "financialData": {
                                "recommendationMean": {"raw": 1.8},
                                "numberOfAnalystOpinions": {"raw": 42},
                                "targetMeanPrice": {"raw": 175.0},
                                "targetHighPrice": {"raw": 220.0},
                                "targetLowPrice": {"raw": 130.0},
                            },
                            "upgradeDowngradeHistory": {
                                "history": [
                                    {
                                        "action": "up",
                                        "firm": "Morgan Stanley",
                                    },
                                    {"action": "main", "firm": "Goldman"},
                                ]
                            },
                        }
                    ]
                }
            },
        )

    async def go():
        adapter = YahooNewsAdapter(client=_client(handler))
        try:
            return await adapter.fetch_analyst_signal("NVDA")
        finally:
            await adapter.aclose()

    sig = asyncio.run(go())
    assert sig["mean_rating"] == pytest.approx(1.8)
    assert sig["num_analysts"] == 42
    assert sig["target_mean"] == 175.0
    assert sig["recent_upgrade"] is True
    assert sig["recent_downgrade"] is False
    assert sig["recent_action"] == "upgrade"
    assert sig["recent_firm"] == "Morgan Stanley"


def test_fetch_analyst_signal_handles_downgrade():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "quoteSummary": {
                    "result": [
                        {
                            "financialData": {},
                            "upgradeDowngradeHistory": {
                                "history": [
                                    {"action": "down", "firm": "Citi"}
                                ]
                            },
                        }
                    ]
                }
            },
        )

    async def go():
        adapter = YahooNewsAdapter(client=_client(handler))
        try:
            return await adapter.fetch_analyst_signal("X")
        finally:
            await adapter.aclose()

    sig = asyncio.run(go())
    assert sig["recent_downgrade"] is True
    assert sig["recent_action"] == "downgrade"


def test_fetch_analyst_signal_empty_on_no_result():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"quoteSummary": {"result": []}})

    async def go():
        adapter = YahooNewsAdapter(client=_client(handler))
        try:
            return await adapter.fetch_analyst_signal("X")
        finally:
            await adapter.aclose()

    assert asyncio.run(go()) == {}


# ---------------------------------------------------------------------------
# Insider summary
# ---------------------------------------------------------------------------


def test_fetch_insider_summary_extracts_net_buy():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "quoteSummary": {
                    "result": [
                        {
                            "netSharePurchaseActivity": {
                                "netInfoShares": {"raw": 50_000},
                                "netPercentInsiderShares": {"raw": 0.012},
                                "buyInfoCount": {"raw": 5},
                                "sellInfoCount": {"raw": 1},
                            }
                        }
                    ]
                }
            },
        )

    async def go():
        adapter = YahooNewsAdapter(client=_client(handler))
        try:
            return await adapter.fetch_insider_summary("X")
        finally:
            await adapter.aclose()

    s = asyncio.run(go())
    assert s["net_shares"] == 50_000
    assert s["buy_count"] == 5
    assert s["sell_count"] == 1


def test_fetch_insider_summary_handles_403():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(403)

    async def go():
        adapter = YahooNewsAdapter(client=_client(handler))
        try:
            return await adapter.fetch_insider_summary("X")
        finally:
            await adapter.aclose()

    assert asyncio.run(go()) == {}


def test_health_check_round_trip():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"news": []})

    async def go():
        adapter = YahooNewsAdapter(client=_client(handler))
        try:
            return await adapter.health()
        finally:
            await adapter.aclose()

    h = asyncio.run(go())
    assert h["ok"] is True
    assert h["latency_ms"] >= 0.0
