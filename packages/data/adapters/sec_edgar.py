"""SEC EDGAR filings adapter (§14). Self-throttled; identifies via User-Agent."""
from __future__ import annotations

import os
from typing import Any

import httpx

from packages.shared.otel import span
from packages.shared.rate_limit import BUCKETS

from .base import DataAdapter, DataAdapterError


class SecEdgarAdapter(DataAdapter):
    name = "sec_edgar"
    BASE = "https://data.sec.gov"

    def __init__(self, user_agent: str | None = None, client: httpx.AsyncClient | None = None) -> None:
        ua = user_agent or os.getenv("SEC_EDGAR_USER_AGENT", "ai-investing dev@example.com")
        self._client = client or httpx.AsyncClient(timeout=30, headers={"User-Agent": ua})

    async def health(self) -> dict[str, Any]:
        with span("data.sec_edgar.health"):
            try:
                r = await self._client.get(f"{self.BASE}/submissions/CIK0000320193.json")
                return {"ok": r.status_code == 200, "latency_ms": r.elapsed.total_seconds() * 1000}
            except Exception as e:  # noqa: BLE001
                return {"ok": False, "error": str(e)}

    async def get_submissions(self, cik: str) -> dict[str, Any]:
        """``cik`` zero-padded to 10 digits, e.g. ``0000320193`` for Apple."""
        await BUCKETS["sec_edgar"].acquire()
        with span("data.sec_edgar.submissions", {"cik": cik}):
            r = await self._client.get(f"{self.BASE}/submissions/CIK{cik}.json")
            if r.status_code != 200:
                raise DataAdapterError(f"sec_edgar {cik}: {r.status_code}")
            return r.json()

    async def aclose(self) -> None:
        await self._client.aclose()
