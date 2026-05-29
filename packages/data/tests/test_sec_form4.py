"""Tests for the Phase 10 SEC EDGAR Form 4 insider-count helper."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import httpx

import packages.data.adapters.sec_edgar as edgar
from packages.data.adapters.sec_edgar import SecEdgarAdapter


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        headers={"User-Agent": "test"},
    )


def setup_function():
    # Reset the module-level ticker cache between tests so each test
    # starts deterministic.
    edgar._TICKER_CIK_CACHE.clear()


def test_lookup_cik_returns_padded_value():
    def handler(req: httpx.Request) -> httpx.Response:
        if "company_tickers" in req.url.path:
            return httpx.Response(
                200,
                json={
                    "0": {"cik_str": 320193, "ticker": "AAPL"},
                    "1": {"cik_str": 789019, "ticker": "MSFT"},
                },
            )
        return httpx.Response(404)

    async def go():
        a = SecEdgarAdapter(client=_client(handler))
        try:
            return await a.lookup_cik("AAPL"), await a.lookup_cik("MSFT")
        finally:
            await a.aclose()

    aapl, msft = asyncio.run(go())
    assert aapl == "0000320193"
    assert msft == "0000789019"


def test_lookup_cik_unknown_ticker_returns_none():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"0": {"cik_str": 1, "ticker": "X"}},
        )

    async def go():
        a = SecEdgarAdapter(client=_client(handler))
        try:
            return await a.lookup_cik("ZZZZZ")
        finally:
            await a.aclose()

    assert asyncio.run(go()) is None


def test_lookup_cik_caches_after_first_hit():
    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(
            200,
            json={"0": {"cik_str": 1, "ticker": "X"}},
        )

    async def go():
        a = SecEdgarAdapter(client=_client(handler))
        try:
            await a.lookup_cik("X")
            await a.lookup_cik("X")
            await a.lookup_cik("X")
        finally:
            await a.aclose()

    asyncio.run(go())
    assert calls["n"] == 1


def test_get_recent_form4_count_inside_window():
    today = datetime.now(UTC).date()
    yesterday = (today - timedelta(days=1)).isoformat()
    last_week = (today - timedelta(days=7)).isoformat()
    too_old = (today - timedelta(days=90)).isoformat()

    def handler(req: httpx.Request) -> httpx.Response:
        if "company_tickers" in req.url.path:
            return httpx.Response(
                200, json={"0": {"cik_str": 1, "ticker": "X"}}
            )
        return httpx.Response(
            200,
            json={
                "filings": {
                    "recent": {
                        "form": ["4", "4", "10-K", "4/A", "4"],
                        "filingDate": [
                            yesterday,
                            last_week,
                            yesterday,
                            yesterday,
                            too_old,
                        ],
                    }
                }
            },
        )

    async def go():
        a = SecEdgarAdapter(client=_client(handler))
        try:
            return await a.get_recent_form4_count("X", window_days=30)
        finally:
            await a.aclose()

    out = asyncio.run(go())
    # 3 inside window (2x "4" + 1x "4/A"); the 10-K and the 90-day-old
    # filing are excluded.
    assert out["count"] == 3
    assert out["latest"] == yesterday
    assert out["cik"] == "0000000001"


def test_get_recent_form4_count_missing_ticker_returns_zero():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"0": {"cik_str": 1, "ticker": "Y"}}
        )

    async def go():
        a = SecEdgarAdapter(client=_client(handler))
        try:
            return await a.get_recent_form4_count("UNKNOWN")
        finally:
            await a.aclose()

    out = asyncio.run(go())
    assert out == {"count": 0, "latest": "", "cik": ""}


def test_get_recent_form4_count_handles_submissions_error():
    def handler(req: httpx.Request) -> httpx.Response:
        if "company_tickers" in req.url.path:
            return httpx.Response(
                200, json={"0": {"cik_str": 1, "ticker": "X"}}
            )
        return httpx.Response(503)

    async def go():
        a = SecEdgarAdapter(client=_client(handler))
        try:
            return await a.get_recent_form4_count("X")
        finally:
            await a.aclose()

    out = asyncio.run(go())
    assert out["count"] == 0
    assert out["cik"] == "0000000001"
