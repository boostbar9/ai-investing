"""Tests for the Alpaca market-data adapter."""
from __future__ import annotations

import httpx
import pytest

from packages.data.adapters.alpaca_data import AlpacaDataAdapter
from packages.data.adapters.base import DataAdapterError


@pytest.mark.asyncio
async def test_health_missing_keys():
    adapter = AlpacaDataAdapter(key_id="", secret="")
    h = await adapter.health()
    assert h["ok"] is False
    assert "ALPACA_PAPER_KEY_ID" in h["error"]
    await adapter.aclose()


@pytest.mark.asyncio
async def test_get_bars_parses_response():
    class _T(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/v2/stocks/SPY/bars"
            assert request.url.params["timeframe"] == "1Day"
            assert request.url.params["feed"] == "iex"
            return httpx.Response(
                200,
                json={
                    "bars": [
                        {
                            "t": "2024-01-02T00:00:00Z",
                            "o": 470.0,
                            "h": 472.5,
                            "l": 469.0,
                            "c": 471.2,
                            "v": 80_000_000,
                        },
                        {
                            "t": "2024-01-03T00:00:00Z",
                            "o": 471.5,
                            "h": 473.0,
                            "l": 470.5,
                            "c": 472.0,
                            "v": 75_000_000,
                        },
                    ],
                    "symbol": "SPY",
                    "next_page_token": None,
                },
            )

    client = httpx.AsyncClient(transport=_T(), base_url="https://data.alpaca.markets")
    adapter = AlpacaDataAdapter(key_id="k", secret="s", client=client)
    bars = await adapter.get_bars("SPY", "2024-01-01", "2024-01-05")
    assert len(bars) == 2
    assert bars[0].close == 471.2
    assert bars[1].volume == 75_000_000
    await adapter.aclose()


@pytest.mark.asyncio
async def test_get_bars_skips_malformed_rows():
    class _T(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "bars": [
                        {"t": "2024-01-02T00:00:00Z", "o": 1, "h": 2, "l": 1, "c": 1.5, "v": 100},
                        {"oops": "missing fields"},
                        {"t": "2024-01-03T00:00:00Z", "o": 2, "h": 3, "l": 2, "c": 2.5, "v": 200},
                    ]
                },
            )

    client = httpx.AsyncClient(transport=_T(), base_url="https://data.alpaca.markets")
    adapter = AlpacaDataAdapter(key_id="k", secret="s", client=client)
    bars = await adapter.get_bars("SPY", "2024-01-01", "2024-01-05")
    assert len(bars) == 2  # malformed row dropped
    await adapter.aclose()


@pytest.mark.asyncio
async def test_get_bars_non_200_raises():
    class _T(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            return httpx.Response(403, text="forbidden")

    client = httpx.AsyncClient(transport=_T(), base_url="https://data.alpaca.markets")
    adapter = AlpacaDataAdapter(key_id="k", secret="s", client=client)
    with pytest.raises(DataAdapterError):
        await adapter.get_bars("SPY", "2024-01-01", "2024-01-05")
    await adapter.aclose()
