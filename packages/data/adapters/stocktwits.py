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
import time
from typing import Any

import httpx

from packages.data.adapters.base import DataAdapter
from packages.shared.otel import span
from packages.shared.rate_limit import BUCKETS

logger = logging.getLogger(__name__)

STOCKTWITS_TRENDING_URL = os.getenv(
    "STOCKTWITS_TRENDING_URL",
    "https://api.stocktwits.com/api/2/trending/symbols.json",
)

USER_AGENT = os.getenv(
    "STOCKTWITS_UA",
    "ai-investing/0.4 (+https://github.com/boostbar9/ai-investing)",
)

DEFAULT_TIMEOUT_S = 8.0


class StockTwitsAdapter(DataAdapter):
    """Lightweight trending-ticker feed from StockTwits.

    Shares the ``rss`` rate-limit bucket (1 req/s) since we hit it at
    most once per sweep.
    """

    name = "stocktwits"

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self._own_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=DEFAULT_TIMEOUT_S,
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        )

    async def health(self) -> dict[str, Any]:
        t0 = time.monotonic()
        try:
            r = await self._client.get(STOCKTWITS_TRENDING_URL)
            ok = r.status_code == 200
        except Exception:
            ok = False
        return {"ok": ok, "latency_ms": (time.monotonic() - t0) * 1000.0}

    async def fetch_trending(
        self, *, limit: int = 30
    ) -> list[dict[str, Any]]:
        """Return up to ``limit`` trending symbols, each as a dict with
        ``symbol``, ``title`` (company name), and ``watchlist_count``.
        Returns ``[]`` on any failure — never raises.
        """
        await BUCKETS["rss"].acquire()
        with span("data.stocktwits.trending"):
            try:
                r = await self._client.get(STOCKTWITS_TRENDING_URL)
            except Exception as exc:
                logger.warning(
                    "stocktwits trending failed: %s",
                    exc.__class__.__name__,
                )
                return []
            if r.status_code != 200:
                logger.warning(
                    "stocktwits trending: HTTP %s", r.status_code
                )
                return []
            try:
                payload = r.json()
            except ValueError:
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
        if self._own_client:
            with contextlib.suppress(Exception):
                await self._client.aclose()
