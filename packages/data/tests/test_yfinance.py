"""Tests for the yfinance adapter."""
from __future__ import annotations

import httpx
import pytest

from packages.data.adapters.base import DataAdapterError
from packages.data.adapters.yfinance import YFinanceAdapter, _parse_chart_response


def test_parse_chart_response_happy_path():
    payload = {
        "chart": {
            "result": [
                {
                    "timestamp": [1700000000, 1700086400],
                    "indicators": {
                        "quote": [
                            {
                                "open": [100.0, 101.0],
                                "high": [102.0, 103.0],
                                "low": [99.0, 100.5],
                                "close": [101.5, 102.5],
                                "volume": [1_000_000, 1_200_000],
                            }
                        ]
                    },
                }
            ]
        }
    }
    bars = _parse_chart_response("SPY", payload)
    assert len(bars) == 2
    assert bars[0].symbol == "SPY"
    assert bars[0].open == 100.0
    assert bars[0].close == 101.5
    assert bars[1].volume == 1_200_000


def test_parse_chart_response_skips_null_bars():
    payload = {
        "chart": {
            "result": [
                {
                    "timestamp": [1700000000, 1700086400, 1700172800],
                    "indicators": {
                        "quote": [
                            {
                                "open": [100.0, None, 102.0],
                                "high": [102.0, None, 104.0],
                                "low": [99.0, None, 101.0],
                                "close": [101.5, None, 103.0],
                                "volume": [1_000_000, None, 900_000],
                            }
                        ]
                    },
                }
            ]
        }
    }
    bars = _parse_chart_response("SPY", payload)
    assert len(bars) == 2  # middle null row dropped
    assert bars[1].open == 102.0


def test_parse_chart_response_empty_raises():
    with pytest.raises(DataAdapterError):
        _parse_chart_response("SPY", {"chart": {"result": []}})


@pytest.mark.asyncio
async def test_get_daily_bars_round_trip():
    class _T(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/v8/finance/chart/SPY"
            return httpx.Response(
                200,
                json={
                    "chart": {
                        "result": [
                            {
                                "timestamp": [1700000000],
                                "indicators": {
                                    "quote": [
                                        {
                                            "open": [100.0],
                                            "high": [101.0],
                                            "low": [99.0],
                                            "close": [100.5],
                                            "volume": [500_000],
                                        }
                                    ]
                                },
                            }
                        ]
                    }
                },
            )

    client = httpx.AsyncClient(transport=_T(), base_url="https://query1.finance.yahoo.com")
    adapter = YFinanceAdapter(client=client)
    bars = await adapter.get_daily_bars("SPY", range_="1mo")
    assert len(bars) == 1
    assert bars[0].close == 100.5
    await adapter.aclose()


@pytest.mark.asyncio
async def test_get_daily_bars_non_200_raises():
    class _T(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            return httpx.Response(429, text="rate limited")

    client = httpx.AsyncClient(transport=_T(), base_url="https://query1.finance.yahoo.com")
    adapter = YFinanceAdapter(client=client)
    with pytest.raises(DataAdapterError):
        await adapter.get_daily_bars("SPY")
    await adapter.aclose()
