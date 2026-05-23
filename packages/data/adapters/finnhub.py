"""Finnhub news adapter (§14)."""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import httpx

from packages.shared.otel import span
from packages.shared.rate_limit import BUCKETS

from .base import DataAdapter, DataAdapterError, NewsItem


class FinnhubAdapter(DataAdapter):
    name = "finnhub"
    BASE = "https://finnhub.io/api/v1"

    def __init__(self, api_key: str | None = None, client: httpx.AsyncClient | None = None) -> None:
        self.api_key = api_key or os.getenv("FINNHUB_API_KEY", "")
        self._client = client or httpx.AsyncClient(timeout=30)

    async def health(self) -> dict[str, Any]:
        with span("data.finnhub.health"):
            try:
                r = await self._client.get(f"{self.BASE}/quote", params={"symbol": "SPY", "token": self.api_key})
                return {"ok": r.status_code == 200, "latency_ms": r.elapsed.total_seconds() * 1000}
            except Exception as e:  # noqa: BLE001
                return {"ok": False, "error": str(e)}

    async def get_company_news(self, symbol: str, frm: str, to: str) -> list[NewsItem]:
        await BUCKETS["finnhub"].acquire()
        with span("data.finnhub.company_news", {"symbol": symbol}):
            r = await self._client.get(
                f"{self.BASE}/company-news",
                params={"symbol": symbol, "from": frm, "to": to, "token": self.api_key},
            )
            if r.status_code != 200:
                raise DataAdapterError(f"finnhub {symbol}: {r.status_code}")
            return [
                NewsItem(
                    symbol=symbol,
                    ts=datetime.fromtimestamp(row.get("datetime", 0), tz=timezone.utc),
                    headline=row.get("headline", ""),
                    summary=row.get("summary"),
                    url=row.get("url", ""),
                    source=row.get("source", "finnhub"),
                )
                for row in r.json() or []
            ]

    async def aclose(self) -> None:
        await self._client.aclose()
