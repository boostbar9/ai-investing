"""Tests for the Phase 10 StockTwits trending adapter."""

from __future__ import annotations

import asyncio

import httpx

from packages.data.adapters.stocktwits import StockTwitsAdapter


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        headers={"User-Agent": "test", "Accept": "application/json"},
    )


def test_fetch_trending_happy_path():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "symbols": [
                    {
                        "symbol": "NVDA",
                        "title": "NVIDIA",
                        "watchlist_count": 1_200_000,
                    },
                    {
                        "symbol": "TSLA",
                        "title": "Tesla",
                        "watchlist_count": 900_000,
                    },
                ]
            },
        )

    async def go():
        adapter = StockTwitsAdapter(client=_client(handler))
        try:
            return await adapter.fetch_trending(limit=5)
        finally:
            await adapter.aclose()

    out = asyncio.run(go())
    assert len(out) == 2
    assert out[0]["symbol"] == "NVDA"
    assert out[0]["watchlist_count"] == 1_200_000


def test_fetch_trending_respects_limit():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "symbols": [
                    {"symbol": f"T{i}", "title": "x"} for i in range(50)
                ]
            },
        )

    async def go():
        adapter = StockTwitsAdapter(client=_client(handler))
        try:
            return await adapter.fetch_trending(limit=10)
        finally:
            await adapter.aclose()

    assert len(asyncio.run(go())) == 10


def test_fetch_trending_skips_empty_symbols():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "symbols": [
                    {"symbol": "", "title": "x"},
                    {"symbol": "GOOD", "title": "y"},
                ]
            },
        )

    async def go():
        adapter = StockTwitsAdapter(client=_client(handler))
        try:
            return await adapter.fetch_trending()
        finally:
            await adapter.aclose()

    out = asyncio.run(go())
    assert len(out) == 1
    assert out[0]["symbol"] == "GOOD"


def test_fetch_trending_handles_429():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="rate limited")

    async def go():
        adapter = StockTwitsAdapter(client=_client(handler))
        try:
            return await adapter.fetch_trending()
        finally:
            await adapter.aclose()

    assert asyncio.run(go()) == []


def test_health_check_ok():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"symbols": []})

    async def go():
        adapter = StockTwitsAdapter(client=_client(handler))
        try:
            return await adapter.health()
        finally:
            await adapter.aclose()

    h = asyncio.run(go())
    assert h["ok"] is True
