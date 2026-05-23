"""Alpaca market-data adapter — free IEX feed, real-time delayed.

Distinct from the *broker* adapter in :mod:`packages.execution.broker`:
the broker submits orders; this one only reads market data. Uses the same
``ALPACA_PAPER_KEY_ID`` / ``ALPACA_PAPER_SECRET`` env vars by default since
free market data is available on the paper plan.

Endpoint reference:
  https://docs.alpaca.markets/reference/stockbars
"""
from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Any

import httpx

from packages.shared.otel import span
from packages.shared.rate_limit import BUCKETS

from .base import Bar, DataAdapter, DataAdapterError


class AlpacaDataAdapter(DataAdapter):
    """Free IEX bars (daily + intraday) from Alpaca's market-data v2 API."""

    name = "alpaca_data"
    BASE = "https://data.alpaca.markets"

    def __init__(
        self,
        key_id: str | None = None,
        secret: str | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.key_id = key_id or os.getenv("ALPACA_PAPER_KEY_ID", "")
        self.secret = secret or os.getenv("ALPACA_PAPER_SECRET", "")
        self._client = client or httpx.AsyncClient(
            timeout=30,
            headers={"APCA-API-KEY-ID": self.key_id, "APCA-API-SECRET-KEY": self.secret},
        )

    async def health(self) -> dict[str, Any]:
        with span("data.alpaca_data.health"):
            try:
                # /v2/stocks/{symbol}/bars requires a key; if keys are missing we report unhealthy.
                if not self.key_id:
                    return {"ok": False, "error": "ALPACA_PAPER_KEY_ID not set"}
                r = await self._client.get(
                    f"{self.BASE}/v2/stocks/SPY/bars",
                    params={"timeframe": "1Day", "limit": 1, "feed": "iex"},
                )
                return {"ok": r.status_code == 200, "latency_ms": r.elapsed.total_seconds() * 1000}
            except Exception as e:
                return {"ok": False, "error": str(e)}

    async def get_bars(
        self,
        symbol: str,
        start: str,
        end: str,
        timeframe: str = "1Day",
        feed: str = "iex",
    ) -> list[Bar]:
        """Fetch bars between ``start`` and ``end`` (ISO 8601).

        Common timeframes: ``"1Min"``, ``"5Min"``, ``"15Min"``, ``"1Hour"``, ``"1Day"``.
        ``feed="iex"`` is free for paper accounts; ``"sip"`` requires a paid plan.
        """
        await BUCKETS["alpaca_data"].acquire()
        with span(
            "data.alpaca_data.bars",
            {"symbol": symbol, "timeframe": timeframe, "start": start, "end": end},
        ):
            r = await self._client.get(
                f"{self.BASE}/v2/stocks/{symbol}/bars",
                params={
                    "timeframe": timeframe,
                    "start": start,
                    "end": end,
                    "feed": feed,
                    "limit": 10000,
                    "adjustment": "all",
                },
            )
            if r.status_code != 200:
                raise DataAdapterError(f"alpaca_data {symbol}: {r.status_code} {r.text[:200]}")
            data = r.json().get("bars") or []
            out: list[Bar] = []
            for row in data:
                try:
                    out.append(
                        Bar(
                            symbol=symbol,
                            ts=datetime.fromisoformat(row["t"].replace("Z", "+00:00")).astimezone(UTC),
                            open=float(row["o"]),
                            high=float(row["h"]),
                            low=float(row["l"]),
                            close=float(row["c"]),
                            volume=float(row["v"]),
                        )
                    )
                except (KeyError, ValueError, TypeError):
                    continue
            return out

    async def aclose(self) -> None:
        await self._client.aclose()
