"""Yahoo Finance adapter — free, no API key required.

Calls Yahoo's public chart API directly so we don't need the ``yfinance``
package (which is unmaintained-prone and pulls in a heavy dep tree). For
production-grade fundamentals or earnings data we'd swap in a paid source;
yfinance is our zero-cost fallback for daily OHLCV and a baseline data
source for the first-install bootstrap.

Endpoint reference:
  https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range=20y
"""
from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Any

import httpx

from packages.shared.otel import span
from packages.shared.rate_limit import BUCKETS

from .base import Bar, DataAdapter, DataAdapterError

YAHOO_HOST = "https://query1.finance.yahoo.com"
# Yahoo rejects requests without a UA string. Use a generic browser one.
_UA = "Mozilla/5.0 (Windows NT 11.0; Win64; x64) ai-investing/0.1"


class YFinanceAdapter(DataAdapter):
    """Free Yahoo Finance OHLCV via the public chart API."""

    name = "yfinance"

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self._client = client or httpx.AsyncClient(
            timeout=30, headers={"User-Agent": os.getenv("YF_USER_AGENT", _UA)}
        )

    async def health(self) -> dict[str, Any]:
        with span("data.yfinance.health"):
            try:
                r = await self._client.get(
                    f"{YAHOO_HOST}/v8/finance/chart/SPY",
                    params={"interval": "1d", "range": "5d"},
                )
                return {"ok": r.status_code == 200, "latency_ms": r.elapsed.total_seconds() * 1000}
            except Exception as e:
                return {"ok": False, "error": str(e)}

    async def get_daily_bars(self, symbol: str, range_: str = "5y") -> list[Bar]:
        """Fetch ``range_`` of daily bars. Valid ranges: 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, 20y, max."""
        return await self._get_bars(symbol, interval="1d", range_=range_)

    async def get_intraday_bars(
        self,
        symbol: str,
        *,
        interval: str = "5m",
        range_: str = "60d",
    ) -> list[Bar]:
        """Fetch intraday bars from Yahoo (no API key required).

        Yahoo's documented limits per interval:
            1m  -> max range 7d
            2m  -> max range 60d
            5m  -> max range 60d
            15m -> max range 60d
            30m -> max range 60d
            60m -> max range 730d
            90m -> max range 60d
        Caller is responsible for picking a (interval, range) pair that
        respects Yahoo's caps; we just forward the request.
        """
        if interval not in _VALID_INTRADAY_INTERVALS:
            raise DataAdapterError(
                f"yfinance intraday: interval={interval!r} not in {_VALID_INTRADAY_INTERVALS}"
            )
        return await self._get_bars(symbol, interval=interval, range_=range_)

    async def _get_bars(self, symbol: str, *, interval: str, range_: str) -> list[Bar]:
        await BUCKETS["yfinance"].acquire()
        with span("data.yfinance.bars", {"symbol": symbol, "interval": interval, "range": range_}):
            r = await self._client.get(
                f"{YAHOO_HOST}/v8/finance/chart/{symbol}",
                params={"interval": interval, "range": range_, "events": "div,split"},
            )
            if r.status_code != 200:
                raise DataAdapterError(f"yfinance {symbol}: {r.status_code} {r.text[:200]}")
            return _parse_chart_response(symbol, r.json())

    async def aclose(self) -> None:
        await self._client.aclose()


_VALID_INTRADAY_INTERVALS = ("1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h")


def _parse_chart_response(symbol: str, payload: dict[str, Any]) -> list[Bar]:
    """Parse Yahoo's chart JSON into a list of :class:`Bar` records.

    Yahoo returns parallel arrays (timestamps + quote columns). Missing values
    show up as ``None`` — we skip those bars rather than synthesizing fills.
    """
    chart = (payload.get("chart") or {}).get("result") or []
    if not chart:
        raise DataAdapterError(f"yfinance {symbol}: empty result")
    block = chart[0]
    timestamps: list[int] = block.get("timestamp") or []
    indicators = block.get("indicators") or {}
    quote_list = indicators.get("quote") or []
    if not quote_list:
        return []
    q = quote_list[0]
    opens = q.get("open") or []
    highs = q.get("high") or []
    lows = q.get("low") or []
    closes = q.get("close") or []
    volumes = q.get("volume") or []

    out: list[Bar] = []
    for i, ts in enumerate(timestamps):
        if i >= min(len(opens), len(highs), len(lows), len(closes), len(volumes)):
            break
        o, h, lo, c, v = opens[i], highs[i], lows[i], closes[i], volumes[i]
        if None in (o, h, lo, c, v):
            continue
        out.append(
            Bar(
                symbol=symbol,
                ts=datetime.fromtimestamp(ts, tz=UTC),
                open=float(o),
                high=float(h),
                low=float(lo),
                close=float(c),
                volume=float(v),
            )
        )
    return out
