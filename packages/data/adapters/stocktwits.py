"""StockTwits trending tickers adapter — Phase 10.

StockTwits' public trending endpoint is unauthenticated and returns
the symbols with the highest message-volume in the last few hours.
This is an early-momentum signal — by the time a ticker shows up in
WSB hot posts it's often already moved 20%. StockTwits trending
catches the *start* of the move.

We only consume the ticker list + watchlist counts; we deliberately do
not pull individual messages (low signal, high noise).
"""

from __future__ import annotations

import contextlib
import logging
import os
from typing import Any

import httpx

from packages.data.adapters.base import DataAdapter
from packages.data.adapters.http import ResilientHTTPClient
from packages.shared.otel import span

logger = logging.getLogger(__name__)

STOCKTWITS_TRENDING_URL = os.getenv(
    "STOCKTWITS_TRENDING_URL",
    "https://api.stocktwits.com/api/2/trending/symbols.json",
)

# StockTwits returns 403 to obvious bot UAs — present as a real browser.
USER_AGENT = os.getenv(
    "STOCKTWITS_UA",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
)

DEFAULT_TIMEOUT_S = 8.0


class StockTwitsAdapter(DataAdapter):
    """Lightweight trending-ticker feed from StockTwits.

    Fetches through the shared :class:`ResilientHTTPClient` (browser UA,
    backoff on 429/5xx, the ``rss`` rate-limit bucket, health recording)
    and degrades to ``[]`` on any failure — it never raises and a blocked
    feed is never a negative signal.
    """

    name = "stocktwits"

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self._http = ResilientHTTPClient(
            "stocktwits",
            bucket="rss",
            client=client,
            user_agent=USER_AGENT,
            timeout_s=DEFAULT_TIMEOUT_S,
        )

    async def health(self) -> dict[str, Any]:
        res = await self._http.get(STOCKTWITS_TRENDING_URL, record_health=False)
        return {"ok": res.ok, "latency_ms": 0.0}

    async def fetch_trending(
        self, *, limit: int = 30
    ) -> list[dict[str, Any]]:
        """Return up to ``limit`` trending symbols, each as a dict with
        ``symbol``, ``title`` (company name), and ``watchlist_count``.
        Returns ``[]`` on any failure — never raises.
        """
        with span("data.stocktwits.trending"):
            res = await self._http.get(STOCKTWITS_TRENDING_URL)
            if not res.ok:
                if res.unavailable and res.error != "disabled":
                    logger.warning(
                        "stocktwits trending unavailable: %s", res.error
                    )
                return []
            payload = res.json()
            if not isinstance(payload, dict):
                return []
            symbols = payload.get("symbols") or []
            out: list[dict[str, Any]] = []
            for s in symbols[:limit]:
                sym = (s.get("symbol") or "").strip().upper()
                if not sym:
                    continue
                out.append(
                    {
                        "symbol": sym,
                        "title": (s.get("title") or "").strip(),
                        "watchlist_count": int(s.get("watchlist_count") or 0),
                    }
                )
            return out

    async def aclose(self) -> None:
        with contextlib.suppress(Exception):
            await self._http.aclose()
