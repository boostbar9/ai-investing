"""Robinhood-realistic paper simulator (``robinhood_paper`` backend).

The Seer learns from a journal of resolved paper trades that feeds
confidence calibration AND agent reweighting. The plain Alpaca paper loop
fills at a single ``last_price`` with no spread or slippage, so the AI was
calibrating against a slightly fictional world. This backend makes the
practice trades behave like REAL Robinhood trades, using ONLY read-only
Robinhood data:

  * **Real prices** -- entries price at the ask, exits at the bid, taken
    from Robinhood's ``get_equity_quotes`` read tool. If a live quote is
    unavailable we fall back to the existing parquet data feed and mark the
    fill ``pricing_source="fallback"``. If BOTH are unavailable the order
    FAILS SAFE: we raise ``BrokerError`` so the caller records an error,
    never a fabricated fill.
  * **Real account constraints** -- a sim cash ledger seeded from the
    user's real Robinhood Agentic-account cash (cached) or a configured
    training balance, plus the existing trading-controls caps (per-trade,
    budget, max trades/day, max open positions) and the float cap.
  * **Realistic fills** -- bid/ask spread, a small conservative slippage,
    fractional-share rounding honoring ``get_equity_tradability``, partial
    fills when the quote shows thin size, and market-hours gating (closed
    market -> no fill).
  * **Optional grounding** -- opt-in, rate-limited read-only
    ``review_equity_order`` to anchor acceptance to Robinhood's own
    response. Never places an order.

HARD SAFETY: this module is READ-ONLY against Robinhood. It calls only
read tools (quotes/tradability/account) and at most read-only
``review_equity_order``. It NEVER calls ``place_equity_order`` /
``cancel_*`` and selecting it does NOT enable live trading. The ledger is
in-memory only -- nothing is written to ``data/``.
"""

from __future__ import annotations

import logging
import os
import time as _time
from collections.abc import Callable
from datetime import UTC, datetime
from datetime import time as dt_time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from packages.execution.broker import (
    Broker,
    BrokerError,
    BrokerPosition,
    OrderAck,
    OrderRequest,
    deterministic_client_order_id,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Conservative, NAMED fill-model constants. These bias the simulator toward
# *pessimism* (slightly worse fills than reality) so the AI never learns
# from rosier-than-real outcomes. Override via env for experiments.
# ---------------------------------------------------------------------------

# Half-spread synthesized when we only know a single price (last trade or
# the fallback parquet close) and have no live bid/ask. 5 bps each side =>
# a 10 bps round-trip, a reasonable liquid-large-cap assumption.
SYNTHETIC_HALF_SPREAD_BPS = float(os.getenv("RH_PAPER_HALF_SPREAD_BPS", "5.0"))
# Adverse slippage applied ON TOP of the spread (buys fill a touch higher,
# sells a touch lower). Conservative default.
SLIPPAGE_BPS = float(os.getenv("RH_PAPER_SLIPPAGE_BPS", "2.0"))

# Robinhood fractional-order minimum is $1; fractional shares round to 6 dp.
MIN_FRACTIONAL_NOTIONAL_USD = 1.0
FRACTIONAL_QTY_DECIMALS = 6

# Real-cash lookups are cached so the GET endpoints stay cheap and we don't
# hammer Robinhood on every tick.
REAL_CASH_CACHE_TTL_S = 60.0

# Order-review grounding is opt-in AND rate-limited: at most one review per
# this interval so it can never slow the trading loop.
REVIEW_MIN_INTERVAL_S = 10.0

# Used when no override is set and real cash can't be fetched -- a bounded,
# meaningful training balance rather than $0 (which would block all buys).
DEFAULT_PAPER_START_BALANCE_USD = 1000.0

# Where the fallback data feed lives (mirrors tools/paper_trade.DATA_ROOT).
_DATA_ROOT = Path("data/parquet/daily")

_ET = ZoneInfo("America/New_York")
_RTH_OPEN = dt_time(9, 30)
_RTH_CLOSE = dt_time(16, 0)
# Extended hours: 4:00am-8:00pm ET (Robinhood's pre/after-market window).
_EXT_OPEN = dt_time(4, 0)
_EXT_CLOSE = dt_time(20, 0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _bps(x: float) -> float:
    return x / 10_000.0


def market_is_open(
    now_utc: datetime | None = None, *, extended_hours: bool = False
) -> bool:
    """True if US equities are open at ``now_utc`` (Mon-Fri ET window).

    Holidays are not modeled (a holiday costs at most one skipped cycle and
    zero correctness -- there are no quotes to fill against anyway). On a
    weekend this returns ``False`` so the simulator correctly does NOTHING
    rather than inventing fills (today is Sunday -> closed)."""
    now = now_utc or datetime.now(UTC)
    et = now.astimezone(_ET)
    if et.weekday() >= 5:  # Sat / Sun
        return False
    lo, hi = (_EXT_OPEN, _EXT_CLOSE) if extended_hours else (_RTH_OPEN, _RTH_CLOSE)
    return lo <= et.time() < hi


def _fallback_price(symbol: str) -> float | None:
    """Latest parquet close for ``symbol`` -- the same data feed the plain
    paper loop already uses. Returns ``None`` when no bar is available so
    the caller fails safe."""
    sym = str(symbol or "").strip().upper()
    if not sym:
        return None
    path = _DATA_ROOT / f"{sym}.parquet"
    if not path.exists():
        return None
    try:
        import pandas as pd

        df = pd.read_parquet(path)
        if df.empty or "close" not in df.columns:
            return None
        return float(df["close"].iloc[-1])
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            "fallback price read failed for %s: %s", sym, exc.__class__.__name__
        )
        return None


# ---------------------------------------------------------------------------
# Real-cash resolution (cached)
# ---------------------------------------------------------------------------
_real_cash_cache: dict[str, Any] = {"value": None, "ts": 0.0, "source": "none"}


async def fetch_real_rh_cash(*, force: bool = False) -> dict[str, Any]:
    """Read the real Robinhood Agentic-account cash (read-only, cached).

    Returns ``{"cash","buying_power","source","cached_at","connected"}``.
    Never raises; on any failure the values are ``None`` and ``source`` is
    a short reason. The cockpit's "use my real Robinhood cash" button calls
    this; the simulator uses it as the default starting balance.
    """
    now = _time.monotonic()
    if (
        not force
        and _real_cash_cache["value"] is not None
        and (now - float(_real_cash_cache["ts"])) < REAL_CASH_CACHE_TTL_S
    ):
        return {
            "cash": _real_cash_cache["value"],
            "buying_power": _real_cash_cache.get("buying_power"),
            "source": _real_cash_cache.get("source", "cache"),
            "cached_at": _real_cash_cache["ts"],
            "connected": True,
        }

    cash: float | None = None
    buying_power: float | None = None
    source = "unavailable"
    connected = False
    try:
        from packages.execution import robinhood as rh

        connected = rh.is_connected()
        if connected:
            snap = await rh.robinhood_account_snapshot()
            cash = snap.get("cash")
            buying_power = snap.get("buying_power")
            if cash is not None:
                source = "rh_account"
            elif buying_power is not None:
                # Some accounts only surface buying power; use it as cash.
                cash = buying_power
                source = "rh_buying_power"
        else:
            source = "not_connected"
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("real-cash fetch failed: %s", exc.__class__.__name__)
        source = "error"

    if cash is not None:
        _real_cash_cache.update(
            value=float(cash),
            buying_power=buying_power,
            ts=now,
            source=source,
        )
    return {
        "cash": cash,
        "buying_power": buying_power,
        "source": source,
        "cached_at": now,
        "connected": connected,
    }


def _load_controls() -> Any:
    """Read trading-controls, failing safe to library defaults."""
    try:
        from packages.cockpit import trading_controls as tc

        return tc.load_controls()
    except Exception:  # pragma: no cover - defensive
        from packages.cockpit.trading_controls import TradingControls

        return TradingControls()


def _resolve_float_cap() -> float:
    try:
        from packages.execution.robinhood import resolve_float_cap

        return float(resolve_float_cap())
    except Exception:  # pragma: no cover - defensive
        return 300.0


async def resolve_paper_start_balance() -> tuple[float, str]:
    """Resolve the simulator's starting cash balance + its source label.

    Priority:
      1. ``trading_controls.paper_start_balance_usd`` when the user set an
         explicit training balance (``paper_use_real_cash`` is False).
      2. The user's REAL Robinhood Agentic-account cash (cached, read-only).
      3. The configured budget (float cap) when real cash is unavailable.
      4. ``DEFAULT_PAPER_START_BALANCE_USD`` as a final floor.
    """
    controls = _load_controls()
    override = getattr(controls, "paper_start_balance_usd", None)
    use_real = getattr(controls, "paper_use_real_cash", True)

    if not use_real and override is not None and float(override) > 0:
        return float(override), "configured"

    real = await fetch_real_rh_cash()
    if real.get("cash") is not None and float(real["cash"]) > 0:
        return float(real["cash"]), "rh_cash"

    # Real cash unavailable. If the user set an override even with
    # use_real=True, honor it before falling back further.
    if override is not None and float(override) > 0:
        return float(override), "configured"

    budget = _resolve_float_cap()
    if budget > 0:
        return float(budget), "budget"
    return DEFAULT_PAPER_START_BALANCE_USD, "default"


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------
class _LedgerPosition:
    __slots__ = ("avg_price", "qty")

    def __init__(self, qty: float = 0.0, avg_price: float = 0.0) -> None:
        self.qty = qty
        self.avg_price = avg_price


# ---------------------------------------------------------------------------
# The simulator broker
# ---------------------------------------------------------------------------
class RobinhoodPaperBroker(Broker):
    """A ``Broker`` that simulates fills from live Robinhood quotes.

    Read-only against Robinhood. Maintains an in-memory cash/positions
    ledger. Every ``submit`` records provenance in :attr:`last_fill_meta`
    (``pricing_source``, spread/slippage applied, partial flag) so the
    learning layer/UI can show how realistic each simulated outcome is.
    """

    name = "robinhood_paper"

    def __init__(
        self,
        *,
        read_broker: Any | None = None,
        fallback_price: Callable[[str], float | None] | None = None,
        start_balance: float | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        # ``read_broker`` supplies equity_quote / equity_tradability /
        # review_order / account_snapshot. Injected in tests; built lazily
        # in production (always SHADOW -- reads are mode-agnostic but we
        # never want this path to be a live broker).
        self._read_broker = read_broker
        self._fallback_price = fallback_price or _fallback_price
        self._clock = clock or (lambda: datetime.now(UTC))
        self._start_balance_override = start_balance

        self._cash: float | None = None
        self._start_balance: float | None = None
        self._start_source: str = "pending"
        self._positions: dict[str, _LedgerPosition] = {}
        self._trades_by_day: dict[str, int] = {}
        self._last_review_at = 0.0
        self.last_fill_meta: dict[str, Any] | None = None

    # ---- lazy setup ----------------------------------------------------
    async def _ensure_read_broker(self) -> Any:
        if self._read_broker is None:
            from packages.execution import robinhood as rh
            from packages.execution.modes import ExecutionMode

            self._read_broker = rh.RobinhoodAgenticBroker(
                mode=ExecutionMode.SHADOW,
                account_number=rh.resolve_agentic_account_number(),
            )
        return self._read_broker

    async def _ensure_ledger(self) -> None:
        if self._cash is not None:
            return
        if self._start_balance_override is not None:
            self._start_balance = float(self._start_balance_override)
            self._start_source = "injected"
        else:
            bal, src = await resolve_paper_start_balance()
            self._start_balance = float(bal)
            self._start_source = src
        self._cash = float(self._start_balance)

    # ---- Broker contract ----------------------------------------------
    async def health(self) -> bool:
        """Always healthy: the simulator works offline (fallback prices)
        even when Robinhood isn't connected."""
        return True

    async def positions(self) -> list[BrokerPosition]:
        """Sim positions, priced from the latest quote/fallback for P&L."""
        await self._ensure_ledger()
        out: list[BrokerPosition] = []
        for sym, pos in self._positions.items():
            if pos.qty <= 0:
                continue
            last = await self._mark_price(sym)
            pnl = (
                (last - pos.avg_price) / pos.avg_price
                if (last is not None and pos.avg_price > 0)
                else None
            )
            out.append(
                BrokerPosition(
                    symbol=sym,
                    qty=round(pos.qty, FRACTIONAL_QTY_DECIMALS),
                    avg_price=pos.avg_price,
                    last_price=last,
                    pnl_pct=pnl,
                )
            )
        return out

    async def _mark_price(self, symbol: str) -> float | None:
        quote = await self._safe_quote(symbol)
        if quote is not None:
            for key in ("mid", "last", "ask", "bid"):
                v = quote.get(key)
                if v is not None and v > 0:
                    return float(v)
        return self._fallback_price(symbol)

    async def _safe_quote(self, symbol: str) -> dict[str, Any] | None:
        try:
            broker = await self._ensure_read_broker()
            return await broker.equity_quote(symbol)
        except Exception as exc:  # pragma: no cover - read path is defensive
            logger.warning(
                "quote lookup failed for %s: %s", symbol, exc.__class__.__name__
            )
            return None

    async def _safe_tradability(self, symbol: str) -> dict[str, Any]:
        try:
            broker = await self._ensure_read_broker()
            return await broker.equity_tradability(symbol)
        except Exception:  # pragma: no cover - defensive
            return {"tradable": True, "fractional": False, "known": False}

    def account_state(self) -> dict[str, Any]:
        """Read-only snapshot of the sim ledger for the cockpit/status."""
        used = 0.0
        positions = []
        for sym, pos in self._positions.items():
            if pos.qty <= 0:
                continue
            used += pos.qty * pos.avg_price
            positions.append(
                {"symbol": sym, "qty": pos.qty, "avg_price": pos.avg_price}
            )
        return {
            "start_balance_usd": self._start_balance,
            "start_balance_source": self._start_source,
            "cash_usd": self._cash,
            "invested_usd": round(used, 2),
            "open_positions": len(positions),
            "positions": positions,
        }

    async def submit(self, req: OrderRequest) -> OrderAck:
        """Simulate one fill against live Robinhood economics.

        Safety order (cheapest gates first); ANY gate that can't be
        satisfied raises ``BrokerError`` so the caller records an ERROR,
        never a fabricated fill:
          1. validate request shape
          2. market-hours gate (closed -> no fill)
          3. tradability (explicit untradable -> skip)
          4. price from live quote, else fallback feed, else FAIL SAFE
          5. apply spread + slippage to get the fill price
          6. enforce trading-controls caps + sim buying power; size down
          7. optional read-only order-review grounding (rate-limited)
          8. update the ledger; record provenance
        """
        await self._ensure_ledger()
        self.last_fill_meta = None

        sym = str(req.symbol or "").strip().upper()
        side = str(req.side or "").strip().lower()
        if not sym or side not in ("buy", "sell") or req.qty <= 0:
            raise BrokerError(f"invalid order: {req.side} {req.qty} {req.symbol}")

        controls = _load_controls()
        extended = bool(getattr(controls, "paper_extended_hours", False))
        if not market_is_open(self._clock(), extended_hours=extended):
            raise BrokerError(
                f"market closed -- no simulated fill for {side} {sym}"
            )

        trade = await self._safe_tradability(sym)
        if trade.get("known") and not trade.get("tradable", True):
            raise BrokerError(f"{sym} is not tradable (halted/restricted)")

        quote = await self._safe_quote(sym)
        pricing_source = "rh_quote"
        if quote is None:
            fb = self._fallback_price(sym)
            if fb is None:
                # FAIL SAFE: no live quote AND no fallback -> never invent.
                raise BrokerError(
                    f"no quote or fallback price for {sym} -- skipping (fail safe)"
                )
            quote = {
                "symbol": sym,
                "bid": None,
                "ask": None,
                "last": fb,
                "mid": fb,
                "bid_size": None,
                "ask_size": None,
            }
            pricing_source = "fallback"

        fill_price, spread_bps = self._fill_price(side, quote)
        if fill_price is None or fill_price <= 0:
            raise BrokerError(f"unpriceable quote for {sym} -- skipping (fail safe)")

        fractional_ok = bool(trade.get("fractional", False))
        filled_qty, partial, reason = self._size_fill(
            side=side,
            sym=sym,
            requested_qty=float(req.qty),
            fill_price=fill_price,
            quote=quote,
            controls=controls,
            fractional_ok=fractional_ok,
        )
        if filled_qty <= 0:
            raise BrokerError(
                f"order for {sym} could not be filled: {reason}"
            )

        # Optional, rate-limited read-only grounding.
        review_anchored = False
        if bool(getattr(controls, "paper_review_grounding", False)):
            review_anchored = await self._maybe_review(
                sym, side, filled_qty, fill_price
            )

        # ---- apply to ledger ----
        notional = filled_qty * fill_price
        pos = self._positions.setdefault(sym, _LedgerPosition())
        if side == "buy":
            new_qty = pos.qty + filled_qty
            pos.avg_price = (
                (pos.avg_price * pos.qty + notional) / new_qty
                if new_qty > 0
                else fill_price
            )
            pos.qty = new_qty
            self._cash = float(self._cash) - notional
        else:  # sell
            pos.qty = max(0.0, pos.qty - filled_qty)
            self._cash = float(self._cash) + notional
            if pos.qty <= 0:
                self._positions.pop(sym, None)

        day = self._clock().astimezone(_ET).date().isoformat()
        self._trades_by_day[day] = self._trades_by_day.get(day, 0) + 1

        oid = deterministic_client_order_id(
            symbol=sym,
            side=side,
            qty=filled_qty,
            decision_id=req.decision_id,
            bar_ts=req.bar_ts,
            prefix="rhpaper",
        )
        status = "partially_filled" if partial else "filled"
        self.last_fill_meta = {
            "pricing_source": pricing_source,
            "bid": quote.get("bid"),
            "ask": quote.get("ask"),
            "mid": quote.get("mid"),
            "fill_price": round(fill_price, 6),
            "spread_bps": round(spread_bps, 3),
            "slippage_bps": SLIPPAGE_BPS,
            "requested_qty": round(float(req.qty), FRACTIONAL_QTY_DECIMALS),
            "filled_qty": round(filled_qty, FRACTIONAL_QTY_DECIMALS),
            "partial": partial,
            "notional_usd": round(notional, 2),
            "review_anchored": review_anchored,
            "fractional": fractional_ok,
        }
        logger.info(
            "rh_paper %s %s %.6f @ %.4f (%s, spread=%.1fbps%s)",
            side,
            sym,
            filled_qty,
            fill_price,
            pricing_source,
            spread_bps,
            ", partial" if partial else "",
        )
        return OrderAck(
            broker=self.name,
            broker_order_id=oid,
            status=status,
            submitted_at=self._clock().isoformat(timespec="seconds"),
        )

    # ---- fill model ----------------------------------------------------
    def _fill_price(
        self, side: str, quote: dict[str, Any]
    ) -> tuple[float | None, float]:
        """Compute the realistic fill price + the effective spread (bps).

        Buys cross the spread to the ask and pay extra slippage; sells hit
        the bid and give up slippage. When only a single price is known we
        synthesize a half-spread around it."""
        bid = quote.get("bid")
        ask = quote.get("ask")
        mid = quote.get("mid")
        last = quote.get("last")
        slip = _bps(SLIPPAGE_BPS)
        hs = _bps(SYNTHETIC_HALF_SPREAD_BPS)

        # Effective spread for provenance.
        if bid and ask and mid and mid > 0:
            spread_bps = (float(ask) - float(bid)) / float(mid) * 10_000.0
        else:
            spread_bps = 2.0 * SYNTHETIC_HALF_SPREAD_BPS

        base_single = mid or last
        if side == "buy":
            if ask and ask > 0:
                price = float(ask) * (1.0 + slip)
            elif base_single and base_single > 0:
                price = float(base_single) * (1.0 + hs + slip)
            else:
                return None, spread_bps
        else:  # sell
            if bid and bid > 0:
                price = float(bid) * (1.0 - slip)
            elif base_single and base_single > 0:
                price = float(base_single) * (1.0 - hs - slip)
            else:
                return None, spread_bps
        return price, spread_bps

    def _size_fill(
        self,
        *,
        side: str,
        sym: str,
        requested_qty: float,
        fill_price: float,
        quote: dict[str, Any],
        controls: Any,
        fractional_ok: bool,
    ) -> tuple[float, bool, str]:
        """Return ``(filled_qty, partial, reason)`` after applying caps,
        buying power, liquidity and fractional rounding."""
        qty = float(requested_qty)
        partial = False

        if side == "sell":
            # Can't sell more than held; never short in the sim.
            held = self._positions.get(sym, _LedgerPosition()).qty
            if held <= 0:
                return 0.0, False, "no position to sell"
            if qty > held:
                qty = held
                partial = True
            # Thin-liquidity partial: can't sell more than the displayed bid
            # size when the quote provides it.
            bid_size = quote.get("bid_size")
            if bid_size is not None and bid_size > 0 and bid_size < qty:
                qty = float(bid_size)
                partial = True
        else:  # buy -- enforce caps + buying power
            per_trade = float(getattr(controls, "max_per_trade_usd", 50.0) or 0.0)
            budget = float(getattr(controls, "total_budget_usd", 0.0) or 0.0)
            cap = _resolve_float_cap()
            invested = sum(
                p.qty * p.avg_price for p in self._positions.values() if p.qty > 0
            )
            remaining_budget = max(0.0, budget - invested)
            buying_power = max(0.0, float(self._cash or 0.0))

            # Max trades/day + max open positions (defense in depth).
            max_trades = int(getattr(controls, "max_trades_per_day", 5) or 0)
            day = self._clock().astimezone(_ET).date().isoformat()
            if max_trades and self._trades_by_day.get(day, 0) >= max_trades:
                return 0.0, False, "hit max trades/day"
            max_open = int(getattr(controls, "max_open_positions", 3) or 0)
            is_new_symbol = self._positions.get(sym, _LedgerPosition()).qty <= 0
            if max_open and is_new_symbol and self._open_count() >= max_open:
                return 0.0, False, "at max open positions"

            notional_cap = min(
                v
                for v in (per_trade, remaining_budget, buying_power, cap)
                if v > 0
            ) if any(
                v > 0 for v in (per_trade, remaining_budget, buying_power, cap)
            ) else 0.0
            max_qty = notional_cap / fill_price if fill_price > 0 else 0.0
            if max_qty < qty:
                qty = max_qty
                partial = True
            # Thin-liquidity partial: can't buy more than the displayed ask
            # size when the quote provides it.
            ask_size = quote.get("ask_size")
            if ask_size is not None and ask_size > 0 and ask_size < qty:
                qty = float(ask_size)
                partial = True

        # ---- fractional rounding (Robinhood rules) ----
        if fractional_ok:
            qty = round(qty, FRACTIONAL_QTY_DECIMALS)
        else:
            import math

            qty = float(math.floor(qty))

        if qty <= 0:
            return 0.0, False, "size rounds to zero after caps/rounding"

        # Fractional minimum notional ($1) for buys.
        if side == "buy" and fractional_ok and (qty * fill_price) < MIN_FRACTIONAL_NOTIONAL_USD:
            return 0.0, False, "below $1 fractional minimum"

        return qty, partial, "ok"

    def _open_count(self) -> int:
        return sum(1 for p in self._positions.values() if p.qty > 0)

    async def _maybe_review(
        self, symbol: str, side: str, qty: float, price: float
    ) -> bool:
        """Best-effort, rate-limited read-only order-review grounding.

        Returns True if a review call was made and the server accepted the
        preview. Never raises, never places an order; failures are silent
        (grounding is advisory)."""
        now = _time.monotonic()
        if (now - self._last_review_at) < REVIEW_MIN_INTERVAL_S:
            return False
        self._last_review_at = now
        try:
            broker = await self._ensure_read_broker()
            review = await broker.review_order(
                symbol=symbol, side=side, qty=qty
            )
        except Exception:  # pragma: no cover - advisory
            return False
        return bool(review)

    async def aclose(self) -> None:
        if self._read_broker is not None:
            close = getattr(self._read_broker, "aclose", None)
            if close is not None:
                import contextlib

                with contextlib.suppress(Exception):
                    await close()


def build_robinhood_paper_broker() -> RobinhoodPaperBroker:
    """Construct a default simulator. Cheap -- no network at build time
    (quotes/cash resolve lazily on first use)."""
    return RobinhoodPaperBroker()


__all__ = [
    "DEFAULT_PAPER_START_BALANCE_USD",
    "MIN_FRACTIONAL_NOTIONAL_USD",
    "REAL_CASH_CACHE_TTL_S",
    "SLIPPAGE_BPS",
    "SYNTHETIC_HALF_SPREAD_BPS",
    "RobinhoodPaperBroker",
    "build_robinhood_paper_broker",
    "fetch_real_rh_cash",
    "market_is_open",
    "resolve_paper_start_balance",
]
