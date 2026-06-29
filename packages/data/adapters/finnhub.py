"""Finnhub news + real-time quote adapter (§14, Phase 25.3)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import httpx

from packages.data import health as health_mod
from packages.shared.otel import span
from packages.shared.rate_limit import BUCKETS

from .base import DataAdapter, DataAdapterError, NewsItem


@dataclass(frozen=True)
class Quote:
    """Normalized real-time quote from Finnhub /quote."""

    symbol: str
    price: float           # current price (c)
    open: float            # day open (o)
    high: float            # day high (h)
    low: float             # day low (l)
    prev_close: float      # previous close (pc)
    ts: datetime           # quote timestamp (server-reported)


class FinnhubAdapter(DataAdapter):
    name = "finnhub"
    BASE = "https://finnhub.io/api/v1"

    def __init__(self, api_key: str | None = None, client: httpx.AsyncClient | None = None) -> None:
        self.api_key = api_key or os.getenv("FINNHUB_API_KEY", "")
        self._client = client or httpx.AsyncClient(timeout=30)

    @property
    def has_key(self) -> bool:
        return bool(self.api_key)

    async def health(self) -> dict[str, Any]:
        with span("data.finnhub.health"):
            try:
                r = await self._client.get(f"{self.BASE}/quote", params={"symbol": "SPY", "token": self.api_key})
                return {"ok": r.status_code == 200, "latency_ms": r.elapsed.total_seconds() * 1000}
            except Exception as e:
                return {"ok": False, "error": str(e)}

    async def get_quote(self, symbol: str) -> Quote:
        """Fetch a real-time quote for ``symbol``.

        Raises :class:`DataAdapterError` when the API key is missing,
        the response is non-200, or the response payload signals an
        invalid symbol (``c == 0`` with ``pc == 0``).

        Returns a :class:`Quote` with the current price (``c``) plus
        the day's OHLC and previous close. Timestamps follow the
        Finnhub server clock (``t``); when ``t`` is zero we substitute
        ``datetime.now(UTC)`` so downstream code never has to deal
        with epoch-0 sentinels.
        """
        if not self.api_key:
            raise DataAdapterError("finnhub: FINNHUB_API_KEY not set")
        await BUCKETS["finnhub"].acquire()
        with span("data.finnhub.quote", {"symbol": symbol}):
            r = await self._client.get(
                f"{self.BASE}/quote",
                params={"symbol": symbol, "token": self.api_key},
            )
            if r.status_code != 200:
                raise DataAdapterError(
                    f"finnhub quote {symbol}: HTTP {r.status_code}"
                )
            data = r.json() or {}
            # Finnhub returns {c:0, h:0, l:0, o:0, pc:0, t:0} for
            # unknown symbols rather than a 4xx — treat that as an error.
            if not data.get("c") and not data.get("pc"):
                raise DataAdapterError(
                    f"finnhub quote {symbol}: empty payload (unknown symbol?)"
                )
            raw_ts = int(data.get("t") or 0)
            ts = (
                datetime.fromtimestamp(raw_ts, tz=UTC)
                if raw_ts > 0
                else datetime.now(UTC)
            )
            return Quote(
                symbol=symbol.upper(),
                price=float(data.get("c", 0.0)),
                open=float(data.get("o", 0.0)),
                high=float(data.get("h", 0.0)),
                low=float(data.get("l", 0.0)),
                prev_close=float(data.get("pc", 0.0)),
                ts=ts,
            )

    async def get_company_news(self, symbol: str, frm: str, to: str) -> list[NewsItem]:
        await BUCKETS["finnhub"].acquire()
        reg = health_mod.get_registry()
        reg.record_attempt("finnhub")
        with span("data.finnhub.company_news", {"symbol": symbol}):
            r = await self._client.get(
                f"{self.BASE}/company-news",
                params={"symbol": symbol, "from": frm, "to": to, "token": self.api_key},
            )
            if r.status_code != 200:
                reg.record_failure("finnhub", f"HTTP {r.status_code}")
                raise DataAdapterError(f"finnhub {symbol}: {r.status_code}")
            reg.record_success("finnhub")
            return [
                NewsItem(
                    symbol=symbol,
                    ts=datetime.fromtimestamp(row.get("datetime", 0), tz=UTC),
                    headline=row.get("headline", ""),
                    summary=row.get("summary"),
                    url=row.get("url", ""),
                    source=row.get("source", "finnhub"),
                )
                for row in r.json() or []
            ]

    async def get_market_news(self, category: str = "general") -> list[NewsItem]:
        """Fetch Finnhub market news for ``category`` (free-tier endpoint).

        The free tier has no dedicated gainers/losers/most-active endpoint,
        but ``/news?category=general`` IS reachable with the same key and
        returns a fresh feed of market-moving headlines, each tagged with a
        ``related`` ticker field. Discovery uses these as a movers proxy.

        Mirrors :meth:`get_company_news`: rate-limited, health-tracked, and
        raises :class:`DataAdapterError` on a missing key / non-200 so the
        caller can fail safe. When a row carries a ``related`` ticker we set
        ``NewsItem.symbol`` to the first one; otherwise it stays ``None`` and
        the headline text still flows downstream for extraction.
        """
        if not self.api_key:
            raise DataAdapterError("finnhub: FINNHUB_API_KEY not set")
        await BUCKETS["finnhub"].acquire()
        reg = health_mod.get_registry()
        reg.record_attempt("finnhub")
        with span("data.finnhub.market_news", {"category": category}):
            r = await self._client.get(
                f"{self.BASE}/news",
                params={"category": category, "token": self.api_key},
            )
            if r.status_code != 200:
                reg.record_failure("finnhub", f"HTTP {r.status_code}")
                raise DataAdapterError(f"finnhub news {category}: {r.status_code}")
            reg.record_success("finnhub")
            out: list[NewsItem] = []
            for row in r.json() or []:
                related = str(row.get("related") or "").strip()
                first = related.split(",")[0].strip().upper() if related else ""
                out.append(
                    NewsItem(
                        symbol=first or None,
                        ts=datetime.fromtimestamp(row.get("datetime", 0) or 0, tz=UTC),
                        headline=row.get("headline", ""),
                        summary=row.get("summary"),
                        url=row.get("url", ""),
                        source=row.get("source", "finnhub"),
                    )
                )
            return out

    async def aclose(self) -> None:
        await self._client.aclose()
