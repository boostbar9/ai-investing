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


@dataclass(frozen=True)
class BrokerPosition:
    symbol: str
    qty: float
    avg_price: float
    last_price: float | None
    pnl_pct: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "qty": self.qty,
            "avg_price": self.avg_price,
            "last_price": self.last_price,
            "pnl_pct": self.pnl_pct,
        }


class Broker(ABC):
    name: str

    @abstractmethod
    async def health(self) -> bool: ...

    @abstractmethod
    async def submit(self, req: OrderRequest) -> OrderAck: ...

    @abstractmethod
    async def positions(self) -> list[BrokerPosition]: ...


class AlpacaPaperBroker(Broker):
    """Alpaca paper-trading adapter. Free $100k fake-cash account.

    Env vars: ``ALPACA_PAPER_KEY_ID``, ``ALPACA_PAPER_SECRET``.
    Get keys at https://app.alpaca.markets/paper/dashboard/overview.
    """

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
        raw_base = base_url or os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
        # Be tolerant: users frequently paste the docs URL which includes /v2.
        # We always append /v2/<endpoint> ourselves, so strip any trailing /v2
        # or slashes before storing.
        self.base_url = raw_base.rstrip("/")
        if self.base_url.endswith("/v2"):
            self.base_url = self.base_url[: -len("/v2")]
        self._client = client or httpx.AsyncClient(
            timeout=15,
            headers={"APCA-API-KEY-ID": self.key_id, "APCA-API-SECRET-KEY": self.secret},
        )

    async def health(self) -> bool:
        with span("broker.alpaca.health"):
            try:
                r = await self._client.get(f"{self.base_url}/v2/account")
                return r.status_code == 200
            except Exception:
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

    async def positions(self) -> list[BrokerPosition]:
        with span("broker.alpaca.positions"):
            r = await self._client.get(f"{self.base_url}/v2/positions")
            if r.status_code >= 300:
                raise BrokerError(f"alpaca positions {r.status_code}: {r.text[:200]}")
            out: list[BrokerPosition] = []
            for p in r.json():
                try:
                    qty = float(p["qty"])
                    avg = float(p["avg_entry_price"])
                    last = float(p["current_price"]) if p.get("current_price") else None
                    pnl = (last - avg) / avg if (last is not None and avg > 0) else None
                    out.append(
                        BrokerPosition(
                            symbol=p["symbol"],
                            qty=qty,
                            avg_price=avg,
                            last_price=last,
                            pnl_pct=pnl,
                        )
                    )
                except (KeyError, ValueError, TypeError):
                    continue
            return out

    async def aclose(self) -> None:
        await self._client.aclose()

    async def account(self) -> dict[str, Any]:
        """Return raw account data: equity, cash, buying_power, day P&L.

        Used by the cockpit to surface paper-account stats in the training view.
        """
        with span(f"broker.{self.name}.account"):
            r = await self._client.get(f"{self.base_url}/v2/account")
            if r.status_code >= 300:
                raise BrokerError(f"{self.name} account {r.status_code}: {r.text[:200]}")
            return r.json()  # type: ignore[no-any-return]


class AlpacaLiveBroker(AlpacaPaperBroker):
    """Alpaca live-trading adapter (real money).

    Identical wire protocol to paper; only the base URL and keys differ.
    Env vars: ``ALPACA_LIVE_KEY_ID``, ``ALPACA_LIVE_SECRET``.

    SAFETY: Only constructed when ``ENABLE_LIVE_TRADING=true`` AND the
    ``live_promotion`` gate has cleared. Never constructed in test envs.
    """

    name = "alpaca_live"

    def __init__(
        self,
        key_id: str | None = None,
        secret: str | None = None,
        base_url: str | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(
            key_id=key_id or os.getenv("ALPACA_LIVE_KEY_ID", ""),
            secret=secret or os.getenv("ALPACA_LIVE_SECRET", ""),
            base_url=base_url or os.getenv("ALPACA_LIVE_BASE_URL", "https://api.alpaca.markets"),
            client=client,
        )


class IBKRBroker(Broker):
    """Interactive Brokers adapter (stub).

    IBKR is pro-grade but heavy: it requires the IB Gateway or TWS desktop app
    running locally with a logged-in session. This stub implements the
    :class:`Broker` interface so callers can already wire IBKR into the router
    config; calls raise ``NotImplementedError`` until the gateway integration
    lands.

    Planned env vars (do not set yet):
      * ``IBKR_GATEWAY_HOST`` (default: 127.0.0.1)
      * ``IBKR_GATEWAY_PORT`` (default: 7497 paper, 7496 live)
      * ``IBKR_ACCOUNT_ID``

    See ``docs/runbooks/ibkr-setup.md`` (TODO) for the full setup once enabled.
    """

    name = "ibkr"

    def __init__(self, paper: bool = True) -> None:
        self.paper = paper
        self.host = os.getenv("IBKR_GATEWAY_HOST", "127.0.0.1")
        self.port = int(os.getenv("IBKR_GATEWAY_PORT", "7497" if paper else "7496"))
        self.account_id = os.getenv("IBKR_ACCOUNT_ID", "")

    async def health(self) -> bool:
        # Stub: not wired yet, so always unhealthy. The router will skip past it.
        return False

    async def submit(self, req: OrderRequest) -> OrderAck:
        raise NotImplementedError(
            "IBKR adapter is a stub. Install ib_insync and implement against"
            " the local IB Gateway socket. See docs/runbooks/ibkr-setup.md."
        )

    async def positions(self) -> list[BrokerPosition]:
        raise NotImplementedError("IBKR adapter is a stub.")


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
            except Exception as e:
                last_err = e
                continue
        raise BrokerError(f"all brokers down: {last_err}")

    async def positions(self) -> list[BrokerPosition]:
        """Return positions from the first healthy broker (positions are
        broker-local; we never merge across brokers)."""
        last_err: Exception | None = None
        for b in self.brokers:
            try:
                if not await b.health():
                    continue
                return await b.positions()
            except Exception as e:
                last_err = e
                continue
        raise BrokerError(f"all brokers down for positions: {last_err}")
