"""Polygon.io bars adapter (§14)."""
from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Any

import httpx

from packages.shared.otel import span
from packages.shared.rate_limit import BUCKETS

from .base import Bar, DataAdapter, DataAdapterError


class PolygonAdapter(DataAdapter):
    name = "polygon"
    BASE = "https://api.polygon.io"

    def __init__(self, api_key: str | None = None, client: httpx.AsyncClient | None = None) -> None:
        self.api_key = api_key or os.getenv("POLYGON_API_KEY", "")
        self._client = client or httpx.AsyncClient(timeout=30)

    async def health(self) -> dict[str, Any]:
        async with span("data.polygon.health"):
            try:
                r = await self._client.get(f"{self.BASE}/v3/reference/tickers", params={"limit": 1, "apiKey": self.api_key})
                return {"ok": r.status_code == 200, "latency_ms": r.elapsed.total_seconds() * 1000}
            except Exception as e:
                return {"ok": False, "error": str(e)}

    async def get_daily_bars(self, symbol: str, start: str, end: str) -> list[Bar]:
        await BUCKETS["polygon"].acquire()
        url = f"{self.BASE}/v2/aggs/ticker/{symbol}/range/1/day/{start}/{end}"
        with span("data.polygon.daily_bars", {"symbol": symbol, "start": start, "end": end}):
            r = await self._client.get(url, params={"apiKey": self.api_key, "adjusted": "true", "sort": "asc"})
            if r.status_code != 200:
                raise DataAdapterError(f"polygon {symbol}: {r.status_code} {r.text[:200]}")
            data = r.json().get("results") or []
            return [
                Bar(
                    symbol=symbol,
                    ts=datetime.fromtimestamp(row["t"] / 1000, tz=UTC),
                    open=row["o"],
                    high=row["h"],
                    low=row["l"],
                    close=row["c"],
                    volume=row["v"],
                )
                for row in data
            ]

    async def aclose(self) -> None:
        await self._client.aclose()
