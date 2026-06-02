"""Phase 25.3 — Finnhub /quote endpoint adapter tests.

Uses ``httpx.MockTransport`` to verify the adapter's request shape and
response parsing without touching the network. Validates:

* URL + params include token and symbol
* Happy path returns a normalized :class:`Quote`
* Missing API key raises immediately (no HTTP call)
* Empty payload (Finnhub's silent "unknown symbol" shape) raises
* HTTP error status raises with a useful message
* ``t == 0`` is replaced with a sensible UTC timestamp
"""
from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest

from packages.data.adapters.base import DataAdapterError
from packages.data.adapters.finnhub import FinnhubAdapter, Quote


def _make_adapter(handler) -> FinnhubAdapter:
    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport, timeout=5)
    return FinnhubAdapter(api_key="TEST_KEY", client=client)


@pytest.mark.asyncio
async def test_get_quote_happy_path() -> None:
    seen: dict[str, httpx.Request] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["req"] = req
        return httpx.Response(
            200,
            json={
                "c": 306.31,
                "d": 1.2,
                "dp": 0.4,
                "h": 308.0,
                "l": 304.5,
                "o": 305.0,
                "pc": 305.11,
                "t": 1_700_000_000,
            },
        )

    adapter = _make_adapter(handler)
    try:
        q = await adapter.get_quote("AAPL")
    finally:
        await adapter.aclose()

    assert isinstance(q, Quote)
    assert q.symbol == "AAPL"
    assert q.price == pytest.approx(306.31)
    assert q.prev_close == pytest.approx(305.11)
    assert q.high == pytest.approx(308.0)
    assert q.ts == datetime.fromtimestamp(1_700_000_000, tz=UTC)

    req = seen["req"]
    assert req.url.path == "/api/v1/quote"
    assert req.url.params["symbol"] == "AAPL"
    assert req.url.params["token"] == "TEST_KEY"


@pytest.mark.asyncio
async def test_get_quote_no_key_raises() -> None:
    adapter = FinnhubAdapter(api_key="")
    try:
        with pytest.raises(DataAdapterError, match="FINNHUB_API_KEY"):
            await adapter.get_quote("AAPL")
    finally:
        await adapter.aclose()


@pytest.mark.asyncio
async def test_get_quote_empty_payload_raises() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        # Finnhub returns all zeros for unknown symbols rather than 4xx.
        return httpx.Response(
            200,
            json={"c": 0, "h": 0, "l": 0, "o": 0, "pc": 0, "t": 0},
        )

    adapter = _make_adapter(handler)
    try:
        with pytest.raises(DataAdapterError, match="empty payload"):
            await adapter.get_quote("ZZZZ")
    finally:
        await adapter.aclose()


@pytest.mark.asyncio
async def test_get_quote_http_error_raises() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="rate limited")

    adapter = _make_adapter(handler)
    try:
        with pytest.raises(DataAdapterError, match="HTTP 429"):
            await adapter.get_quote("AAPL")
    finally:
        await adapter.aclose()


@pytest.mark.asyncio
async def test_get_quote_zero_timestamp_substitutes_now() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "c": 100.0,
                "h": 101.0,
                "l": 99.0,
                "o": 100.5,
                "pc": 99.5,
                "t": 0,  # zero -> we substitute now()
            },
        )

    before = datetime.now(UTC)
    adapter = _make_adapter(handler)
    try:
        q = await adapter.get_quote("XYZ")
    finally:
        await adapter.aclose()
    after = datetime.now(UTC)

    assert before <= q.ts <= after


def test_has_key_flag() -> None:
    assert FinnhubAdapter(api_key="abc").has_key is True
    assert FinnhubAdapter(api_key="").has_key is False
