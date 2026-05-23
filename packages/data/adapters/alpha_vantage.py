"""Alpha Vantage fundamentals adapter (§14)."""
from __future__ import annotations

import os
from typing import Any

import httpx

from packages.shared.otel import span
from packages.shared.rate_limit import BUCKETS

from .base import DataAdapter, DataAdapterError


class AlphaVantageAdapter(DataAdapter):
    name = "alpha_vantage"
    BASE = "https://www.alphavantage.co/query"

    def __init__(self, api_key: str | None = None, client: httpx.AsyncClient | None = None) -> None:
        self.api_key = api_key or os.getenv("ALPHA_VANTAGE_API_KEY", "")
        self._client = client or httpx.AsyncClient(timeout=30)

    async def health(self) -> dict[str, Any]:
        with span("data.alpha_vantage.health"):
            try:
                r = await self._client.get(self.BASE, params={"function": "GLOBAL_QUOTE", "symbol": "SPY", "apikey": self.api_key})
                return {"ok": r.status_code == 200, "latency_ms": r.elapsed.total_seconds() * 1000}
            except Exception as e:
                return {"ok": False, "error": str(e)}

    async def get_overview(self, symbol: str) -> dict[str, Any]:
        await BUCKETS["alpha_vantage"].acquire()
        with span("data.alpha_vantage.overview", {"symbol": symbol}):
            r = await self._client.get(self.BASE, params={"function": "OVERVIEW", "symbol": symbol, "apikey": self.api_key})
            if r.status_code != 200:
                raise DataAdapterError(f"alpha_vantage {symbol}: {r.status_code}")
            return r.json() or {}

    async def aclose(self) -> None:
        await self._client.aclose()
