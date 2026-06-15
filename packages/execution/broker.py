"""Broker abstraction (§18 mitigation: broker outage → failover).

Today: Alpaca paper. Tomorrow: add Interactive Brokers, Tradier, etc. and the
router falls over to the next healthy broker. Trading keys live ONLY in this
process — see SECURITY.md.
"""
from __future__ import annotations

import contextlib
import hashlib
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import httpx

from packages.shared.otel import span

logger = logging.getLogger(__name__)


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
    # Idempotency identity. When the planner can supply a stable decision
    # id (e.g. one decision -> one order) we hash it into the broker's
    # client_order_id so a retried submission dedupes server-side instead
    # of placing a duplicate order. ``bar_ts`` is the bar/decision
    # timestamp (ISO string or epoch) that, combined with symbol/side/qty,
    # makes the identity stable across retries of the SAME logical order.
    decision_id: str | None = None
    bar_ts: str | None = None


def deterministic_client_order_id(
    *,
    symbol: str,
    side: str,
    qty: float,
    decision_id: str | None = None,
    bar_ts: str | None = None,
    prefix: str = "seer",
) -> str:
    """Derive a STABLE client_order_id from an order's identity.

    The same logical order retried must produce the SAME id so the broker
    dedupes it instead of placing a duplicate. We hash the strongest
    identity available:

      * If ``decision_id`` (and optionally ``bar_ts``) is present, the
        hash is fully deterministic and survives process restarts.
      * Otherwise we fall back to a hash of (symbol, side, qty, bar_ts).
        This still dedupes identical retries within the same bar.
      * Only when NO identity at all is available (no decision_id and no
        bar_ts) do we fall back to a fresh uuid4 -- a retry then can't be
        recognized, but we have nothing stable to key on. Callers that
        care about idempotency MUST supply a decision_id or bar_ts.

    Alpaca caps client_order_id at 128 chars; a hex digest plus prefix is
    well under that and uses only URL-safe chars.
    """
    if decision_id is None and bar_ts is None:
        logger.warning(
            "order for %s %s has no decision_id/bar_ts -- falling back to "
            "uuid4 client_order_id; retries will NOT dedupe",
            side,
            symbol,
        )
        return f"{prefix}-{uuid4()}"
    parts = [
        str(decision_id or ""),
        symbol.upper(),
        side.lower(),
        f"{float(qty):.8f}",
        str(bar_ts or ""),
    ]
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return f"{prefix}-{digest[:32]}"


@dataclass(frozen=True)
class OrderAck:
    broker: str
    broker_order_id: str
    status: str
    submitted_at: str


@dataclass(frozen=True)
class FillReconciliation:
    """Outcome of comparing an order's actual fill against intent."""

    broker_order_id: str
    intended_qty: float
    filled_qty: float
    status: str
    matched: bool
    polls: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "broker_order_id": self.broker_order_id,
            "intended_qty": self.intended_qty,
            "filled_qty": self.filled_qty,
            "status": self.status,
            "matched": self.matched,
            "polls": self.polls,
        }


# Order statuses that mean "this order is done; no more fills coming".
_TERMINAL_ORDER_STATES = {
    "filled",
    "canceled",
    "cancelled",
    "rejected",
    "expired",
    "done_for_day",
    "replaced",
}


async def reconcile_fill_via_poll(
    *,
    poll,
    broker_order_id: str,
    intended_qty: float,
    max_polls: int = 5,
    delay_s: float = 1.0,
    sleep=None,
) -> FillReconciliation:
    """Poll an order's status a BOUNDED number of times and report fills.

    ``poll`` is an async callable returning a dict snapshot with (best
    effort) ``filled_qty`` and ``status`` keys -- the per-broker adapter
    supplies it. We stop early once the order reaches a terminal state.
    A shortfall (filled < intended) logs a structured warning so a partial
    fill can NEVER silently pass. This function only READS -- it can never
    place an order.
    """
    import asyncio

    _sleep = sleep or asyncio.sleep
    filled = 0.0
    status = "unknown"
    polls = 0
    for i in range(max(1, max_polls)):
        polls = i + 1
        try:
            snap = await poll()
        except Exception as exc:
            logger.warning(
                "fill reconciliation poll failed for %s: %s",
                broker_order_id,
                exc.__class__.__name__,
            )
            break
        snap = snap or {}
        with contextlib.suppress(TypeError, ValueError):
            filled = float(
                snap.get("filled_qty", snap.get("cumulative_quantity", filled))
                or 0.0
            )
        status = str(snap.get("status", status) or status).lower()
        if status in _TERMINAL_ORDER_STATES or filled >= float(intended_qty):
            break
        if polls < max_polls:
            await _sleep(delay_s)

    matched = filled >= float(intended_qty)
    if not matched:
        logger.warning(
            "fill mismatch: order %s filled %.6f of intended %.6f (status=%s)",
            broker_order_id,
            filled,
            float(intended_qty),
            status,
        )
    return FillReconciliation(
        broker_order_id=broker_order_id,
        intended_qty=float(intended_qty),
        filled_qty=filled,
        status=status,
        matched=matched,
        polls=polls,
    )


@dataclass(frozen=True)
class BracketOrderRequest:
    """Phase 35 — broker-side OCO bracket order.

    Submitted as a single ``order_class=bracket`` parent order so the
    exchange itself triggers the take-profit limit OR the stop-loss
    leg at machine speed — we don't need our 60s fast loop in the
    loop for these exits. The two child legs are mutually exclusive
    (one cancels the other when filled).

    ``take_profit_price`` and ``stop_loss_stop_price`` are absolute
    dollar prices, not percent offsets. The caller is responsible for
    deriving them from the entry price + exit thresholds (see
    ``tools/paper_trade.py``).

    ``stop_loss_limit_price`` is optional — when set, the stop leg
    becomes a stop-limit instead of a stop-market. Useful in fast
    moves to avoid getting filled at a price worse than the limit.
    """

    symbol: str
    side: str  # parent order side — typically "buy"
    qty: float
    take_profit_price: float
    stop_loss_stop_price: float
    stop_loss_limit_price: float | None = None
    type: str = "market"  # parent order type
    limit_price: float | None = None  # required when type == "limit"
    time_in_force: str = "day"
    # Idempotency identity -- see ``OrderRequest`` / ``deterministic_client_order_id``.
    decision_id: str | None = None
    bar_ts: str | None = None


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
        client_order_id = deterministic_client_order_id(
            symbol=req.symbol,
            side=req.side,
            qty=req.qty,
            decision_id=req.decision_id,
            bar_ts=req.bar_ts,
        )
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

    async def reconcile_fill(
        self,
        broker_order_id: str,
        intended_qty: float,
        *,
        max_polls: int = 5,
        delay_s: float = 1.0,
    ) -> FillReconciliation:
        """Poll an Alpaca order's status and report fills vs intent (P0-3).

        Reads ``GET /v2/orders/{id}`` a BOUNDED number of times until the
        order is terminal or fully filled. Read-only -- never places an
        order. Surfaces a structured warning on any shortfall.
        """
        async def _poll() -> dict[str, Any]:
            with span("broker.alpaca.reconcile_poll"):
                r = await self._client.get(
                    f"{self.base_url}/v2/orders/{broker_order_id}"
                )
                if r.status_code >= 300:
                    raise BrokerError(
                        f"alpaca order-status {r.status_code}: {r.text[:200]}"
                    )
                return r.json()

        return await reconcile_fill_via_poll(
            poll=_poll,
            broker_order_id=broker_order_id,
            intended_qty=intended_qty,
            max_polls=max_polls,
            delay_s=delay_s,
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

    async def submit_bracket(self, req: BracketOrderRequest) -> OrderAck:
        """Phase 35 — submit a parent order with attached OCO bracket.

        Alpaca supports ``order_class=bracket`` natively: the parent
        order fills first, then the broker arms both child legs
        (take-profit limit + stop-loss). Whichever fills first cancels
        the other. Exits run at exchange speed independent of our loop.

        We treat all bracket prices as 2-decimal stock-equity prices,
        matching Alpaca's tick-size requirements.
        """
        client_order_id = deterministic_client_order_id(
            symbol=req.symbol,
            side=req.side,
            qty=req.qty,
            decision_id=req.decision_id,
            bar_ts=req.bar_ts,
            prefix="seer-bracket",
        )
        # Alpaca's stock equities require <= 2 decimal places for prices.
        def _px(p: float | None) -> str | None:
            if p is None:
                return None
            return f"{float(p):.2f}"

        take_profit_leg = {"limit_price": _px(req.take_profit_price)}
        stop_loss_leg: dict[str, Any] = {
            "stop_price": _px(req.stop_loss_stop_price),
        }
        if req.stop_loss_limit_price is not None:
            stop_loss_leg["limit_price"] = _px(req.stop_loss_limit_price)

        body: dict[str, Any] = {
            "symbol": req.symbol,
            "qty": str(req.qty),
            "side": req.side,
            "type": req.type,
            "time_in_force": req.time_in_force,
            "order_class": "bracket",
            "take_profit": take_profit_leg,
            "stop_loss": stop_loss_leg,
            "client_order_id": client_order_id,
        }
        if req.type == "limit":
            body["limit_price"] = _px(req.limit_price)

        with span(
            "broker.alpaca.submit_bracket",
            {"symbol": req.symbol, "side": req.side, "qty": req.qty},
        ):
            r = await self._client.post(f"{self.base_url}/v2/orders", json=body)
            if r.status_code >= 300:
                raise BrokerError(
                    f"alpaca bracket {r.status_code}: {r.text[:200]}"
                )
            data: dict[str, Any] = r.json()
            return OrderAck(
                broker=self.name,
                broker_order_id=data["id"],
                status=data.get("status", "unknown"),
                submitted_at=data.get("submitted_at", ""),
            )

    async def liquidate_all(self, cancel_orders: bool = True) -> dict[str, Any]:
        """Cancel open orders and close every open position at market.

        Uses Alpaca's bulk endpoints:
        - ``DELETE /v2/orders`` -- cancel all open orders
        - ``DELETE /v2/positions?cancel_orders=true`` -- close all positions

        Both endpoints are atomic on Alpaca's side. Returns a summary dict
        with counts and the per-symbol responses from Alpaca. Raises
        ``BrokerError`` if either call fails with HTTP >= 300.

        This is the recommended way to free up buying power that's stuck in
        old positions -- one round-trip, no per-symbol retries needed.
        """
        with span("broker.alpaca.liquidate_all"):
            cancelled_orders: list[dict[str, Any]] = []
            if cancel_orders:
                ro = await self._client.delete(f"{self.base_url}/v2/orders")
                # 207 multi-status is normal here; only fail on hard errors.
                if ro.status_code >= 400:
                    raise BrokerError(
                        f"alpaca cancel-orders {ro.status_code}: {ro.text[:200]}"
                    )
                try:
                    cancelled_orders = ro.json() if ro.text else []
                except ValueError:
                    cancelled_orders = []
            rp = await self._client.delete(
                f"{self.base_url}/v2/positions",
                params={"cancel_orders": "true"} if cancel_orders else None,
            )
            # Alpaca returns 207 multi-status with a per-symbol list.
            if rp.status_code >= 400:
                raise BrokerError(
                    f"alpaca liquidate {rp.status_code}: {rp.text[:200]}"
                )
            try:
                closed = rp.json() if rp.text else []
            except ValueError:
                closed = []
            return {
                "cancelled_orders": len(cancelled_orders),
                "closed_positions": len(closed) if isinstance(closed, list) else 0,
                "orders_response": cancelled_orders,
                "positions_response": closed,
            }

    async def open_orders(self) -> list[dict[str, Any]]:
        """Phase 36g — list every open (un-filled) order on the account.

        Returns the raw JSON from ``GET /v2/orders?status=open``. Each row
        contains at minimum ``symbol``, ``side``, ``qty``, and ``filled_qty``;
        we deliberately don't dataclass-wrap it so the planner can compute
        in-flight notional with whatever fields Alpaca actually returns.

        This is what unblocks the buying-power 403 cascade: if the planner
        knows about pending orders, it can avoid re-queueing duplicates and
        subtract their reserved cash from the buying-power budget.
        """
        with span("broker.alpaca.open_orders"):
            r = await self._client.get(
                f"{self.base_url}/v2/orders",
                params={"status": "open", "limit": 500, "nested": "true"},
            )
            if r.status_code >= 300:
                raise BrokerError(
                    f"alpaca open_orders {r.status_code}: {r.text[:200]}"
                )
            data = r.json()
            return data if isinstance(data, list) else []

    async def cancel_all_orders(self) -> dict[str, Any]:
        """Phase 36g — cancel every open order WITHOUT closing positions.

        ``liquidate_all`` is too aggressive when the problem is just stuck
        pending orders eating buying power. This is the surgical version:
        DELETE /v2/orders only, leave positions untouched.

        Returns a summary dict like :meth:`liquidate_all`.
        """
        with span("broker.alpaca.cancel_all_orders"):
            r = await self._client.delete(f"{self.base_url}/v2/orders")
            if r.status_code >= 400:
                raise BrokerError(
                    f"alpaca cancel-orders {r.status_code}: {r.text[:200]}"
                )
            try:
                rows = r.json() if r.text else []
            except ValueError:
                rows = []
            return {
                "cancelled_orders": len(rows) if isinstance(rows, list) else 0,
                "orders_response": rows,
            }

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
