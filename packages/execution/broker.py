"""Broker abstraction (§18 mitigation: broker outage → failover).

Today: Alpaca paper. Tomorrow: add Interactive Brokers, Tradier, etc. and the
router falls over to the next healthy broker. Trading keys live ONLY in this
process — see SECURITY.md.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import httpx

from packages.shared.otel import span


class BrokerError(RuntimeError):
    """Raised when an order cannot be placed."""


@dataclass(frozen=True)
class OrderRequest:
    symbol: str
    side: str  # "buy" | "sell"
    qty: float
    type: str = "market"  # "market" | "limit"
    limit_price: float | None = None
    time_in_force: str = "day"


@dataclass(frozen=True)
class OrderAck:
    broker: str
    broker_order_id: str
    status: str
    submitted_at: str


class Broker(ABC):
    name: str

    @abstractmethod
    async def health(self) -> bool: ...

    @abstractmethod
    async def submit(self, req: OrderRequest) -> OrderAck: ...


class AlpacaPaperBroker(Broker):
    name = "alpaca_paper"

    def __init__(
        self,
        key_id: str | None = None,
        secret: str | None = None,
        base_url: str | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.key_id = key_id or os.getenv("ALPACA_PAPER_KEY_ID", "")
        self.secret = secret or os.getenv("ALPACA_PAPER_SECRET", "")
        self.base_url = base_url or os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
        self._client = client or httpx.AsyncClient(
            timeout=15,
            headers={"APCA-API-KEY-ID": self.key_id, "APCA-API-SECRET-KEY": self.secret},
        )

    async def health(self) -> bool:
        with span("broker.alpaca.health"):
            try:
                r = await self._client.get(f"{self.base_url}/v2/account")
                return r.status_code == 200
            except Exception:  # noqa: BLE001
                return False

    async def submit(self, req: OrderRequest) -> OrderAck:
        client_order_id = str(uuid4())
        with span("broker.alpaca.submit", {"symbol": req.symbol, "side": req.side, "qty": req.qty}):
            r = await self._client.post(
                f"{self.base_url}/v2/orders",
                json={
                    "symbol": req.symbol,
                    "qty": str(req.qty),
                    "side": req.side,
                    "type": req.type,
                    "limit_price": str(req.limit_price) if req.limit_price else None,
                    "time_in_force": req.time_in_force,
                    "client_order_id": client_order_id,
                },
            )
            if r.status_code >= 300:
                raise BrokerError(f"alpaca {r.status_code}: {r.text[:200]}")
            data: dict[str, Any] = r.json()
            return OrderAck(
                broker=self.name,
                broker_order_id=data["id"],
                status=data.get("status", "unknown"),
                submitted_at=data.get("submitted_at", ""),
            )

    async def aclose(self) -> None:
        await self._client.aclose()


class BrokerRouter:
    """Healthy-first failover across multiple brokers.

    On total failure sets ``TRADING_PAUSED=true`` semantics by raising
    BrokerError — the Risk Engine treats this as a halt condition (§20).
    """

    def __init__(self, brokers: list[Broker]) -> None:
        if not brokers:
            raise ValueError("BrokerRouter requires at least one broker")
        self.brokers = brokers

    async def submit(self, req: OrderRequest) -> OrderAck:
        last_err: Exception | None = None
        for b in self.brokers:
            try:
                if not await b.health():
                    continue
                return await b.submit(req)
            except Exception as e:  # noqa: BLE001
                last_err = e
                continue
        raise BrokerError(f"all brokers down: {last_err}")
