"""SEC EDGAR filings adapter (§14). Self-throttled; identifies via User-Agent.

Phase 10 adds :meth:`get_recent_form4_count` so the research sweep can
treat a burst of insider Form 4 filings as a corroboration signal
independent of Yahoo's own insider summary.
"""
from __future__ import annotations

import logging
import os
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx

from packages.shared.otel import span
from packages.shared.rate_limit import BUCKETS

from .base import DataAdapter, DataAdapterError

logger = logging.getLogger(__name__)

# Ticker -> CIK lookup. Populated lazily on first call; refreshed only
# on process restart since CIK assignments effectively never change.
_TICKER_CIK_CACHE: dict[str, str] = {}
_TICKER_CIK_URL = "https://www.sec.gov/files/company_tickers.json"


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
            except Exception as e:
                return {"ok": False, "error": str(e)}

    async def get_submissions(self, cik: str) -> dict[str, Any]:
        """``cik`` zero-padded to 10 digits, e.g. ``0000320193`` for Apple."""
        await BUCKETS["sec_edgar"].acquire()
        with span("data.sec_edgar.submissions", {"cik": cik}):
            r = await self._client.get(f"{self.BASE}/submissions/CIK{cik}.json")
            if r.status_code != 200:
                raise DataAdapterError(f"sec_edgar {cik}: {r.status_code}")
            return r.json()

    async def lookup_cik(self, symbol: str) -> str | None:
        """Return the 10-digit zero-padded CIK for ``symbol`` or None.

        Uses SEC's public ticker map and caches the whole dict in
        process memory after the first hit. Returns None on any
        network failure — callers must treat insider signal as
        unavailable, not as 'no insider activity'.
        """
        global _TICKER_CIK_CACHE
        if not _TICKER_CIK_CACHE:
            await BUCKETS["sec_edgar"].acquire()
            try:
                r = await self._client.get(_TICKER_CIK_URL)
            except Exception as exc:
                logger.debug("sec ticker map fetch failed: %s", exc)
                return None
            if r.status_code != 200:
                return None
            try:
                data = r.json()
            except ValueError:
                return None
            # Map is keyed by integer-as-string index; values have
            # cik_str (int) and ticker (str).
            built: dict[str, str] = {}
            for entry in data.values():
                t = (entry.get("ticker") or "").strip().upper()
                cik = entry.get("cik_str")
                if t and isinstance(cik, int):
                    built[t] = f"{cik:010d}"
            if built:
                _TICKER_CIK_CACHE = built
        return _TICKER_CIK_CACHE.get(symbol.upper())

    async def get_recent_form4_count(
        self, symbol: str, *, window_days: int = 30
    ) -> dict[str, Any]:
        """Count Form 4 (insider transactions) filings for ``symbol``
        in the last ``window_days`` days.

        Returns ``{"count": int, "latest": iso-str or '', "cik": str}``.
        A ``count`` of 0 means no recent insider activity OR data
        unavailable — caller cannot distinguish; treat as neutral.
        """
        cik = await self.lookup_cik(symbol)
        if not cik:
            return {"count": 0, "latest": "", "cik": ""}
        try:
            subs = await self.get_submissions(cik)
        except DataAdapterError:
            return {"count": 0, "latest": "", "cik": cik}
        recent = (subs.get("filings") or {}).get("recent") or {}
        forms = recent.get("form") or []
        dates = recent.get("filingDate") or []
        cutoff = datetime.now(UTC).date() - timedelta(days=window_days)
        count = 0
        latest = ""
        for form, dstr in zip(forms, dates, strict=False):
            if form not in ("4", "4/A"):
                continue
            try:
                d = datetime.strptime(dstr, "%Y-%m-%d").date()
            except (TypeError, ValueError):
                continue
            if d < cutoff:
                continue
            count += 1
            if not latest or dstr > latest:
                latest = dstr
        return {"count": count, "latest": latest, "cik": cik}

    async def aclose(self) -> None:
        await self._client.aclose()
