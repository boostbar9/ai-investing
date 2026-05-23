"""FRED macro adapter (§14). Free, unlimited; we still throttle politely."""
from __future__ import annotations

import os
from typing import Any

import httpx

from packages.shared.otel import span
from packages.shared.rate_limit import BUCKETS

from .base import DataAdapter, DataAdapterError


class FredAdapter(DataAdapter):
    name = "fred"
    BASE = "https://api.stlouisfed.org/fred"

    def __init__(self, api_key: str | None = None, client: httpx.AsyncClient | None = None) -> None:
        self.api_key = api_key or os.getenv("FRED_API_KEY", "")
        self._client = client or httpx.AsyncClient(timeout=30)

    async def health(self) -> dict[str, Any]:
        with span("data.fred.health"):
            try:
                r = await self._client.get(
                    f"{self.BASE}/series", params={"series_id": "GDP", "api_key": self.api_key, "file_type": "json"}
                )
                return {"ok": r.status_code == 200, "latency_ms": r.elapsed.total_seconds() * 1000}
            except Exception as e:  # noqa: BLE001
                return {"ok": False, "error": str(e)}

    async def get_series(self, series_id: str) -> list[dict[str, Any]]:
        await BUCKETS["fred"].acquire()
        with span("data.fred.series", {"series_id": series_id}):
            r = await self._client.get(
                f"{self.BASE}/series/observations",
                params={"series_id": series_id, "api_key": self.api_key, "file_type": "json"},
            )
            if r.status_code != 200:
                raise DataAdapterError(f"fred {series_id}: {r.status_code}")
            return r.json().get("observations") or []

    async def aclose(self) -> None:
        await self._client.aclose()
