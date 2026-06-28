"""Phase 35 — bracket-order auto-attach helper.

After a successful entry order, attach an Alpaca-side OCO bracket so
the *broker* enforces our take-profit / stop-loss thresholds at
exchange speed. This is the third leg of Phase 35's "faster profit
taking" work (after adaptive fast-loop cadence + scale-out partial
exits): even if our cockpit loop hangs or the WAN drops, the bracket
sitting at Alpaca will still fire when the price hits either side.

Design notes
------------
* We do NOT replace the cockpit's exit-rules loop. Brackets are a
  defense-in-depth layer; the trailing stop + scale-out logic still
  runs locally because Alpaca brackets are static (no peak-tracked
  trailing). The bracket levels are derived from the entry price and
  the active ``ExitThresholds`` so the *worst case* lock-in matches
  the cockpit policy.
* The helper is async and best-effort: a bracket failure must never
  unwind a successful entry. Returns a result dict for logging.
* Sell-side entries (i.e. shorts) are NOT bracketed here. The bot
  is long-only intraday today; revisit when shorts ship.
* Fractional-share entries skip the bracket — Alpaca rejects bracket
  legs on non-integer qty. Whole-share entries get the full OCO.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from packages.execution.broker import (
    AlpacaPaperBroker,
    BracketOrderRequest,
    BrokerError,
    OrderAck,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class BracketLevels:
    """Computed bracket prices for a long entry."""

    take_profit_price: float
    stop_loss_stop_price: float
    stop_loss_limit_price: float | None


def compute_bracket_levels(
    *,
    entry_price: float,
    take_profit_pct: float,
    hard_stop_pct: float,
    stop_limit_slack_pct: float = 0.002,
) -> BracketLevels | None:
    """Convert exit-rules thresholds (fractions) into absolute prices.

    Returns ``None`` when the inputs are unusable (non-positive entry,
    zero thresholds), signalling "do not attach a bracket".

    ``stop_limit_slack_pct`` widens the stop-limit a touch below the
    stop-price so the limit doesn't itself become the binding price
    in a fast move. Default 0.2%.
    """
    try:
        ep = float(entry_price)
        tp = float(take_profit_pct)
        sl = float(hard_stop_pct)
    except (TypeError, ValueError):
        return None
    if ep <= 0 or tp <= 0 or sl <= 0:
        return None
    tp_price = ep * (1.0 + tp)
    sl_stop = ep * (1.0 - sl)
    if sl_stop <= 0 or tp_price <= ep:
        return None
    # Stop-limit must be < stop-price for a sell stop; widen by slack.
    sl_limit_raw = sl_stop * (1.0 - max(0.0, float(stop_limit_slack_pct)))
    sl_limit: float | None = sl_limit_raw if sl_limit_raw > 0 else None
    return BracketLevels(
        take_profit_price=round(tp_price, 4),
        stop_loss_stop_price=round(sl_stop, 4),
        stop_loss_limit_price=round(sl_limit, 4) if sl_limit is not None else None,
    )


async def attach_bracket_after_entry(
    *,
    broker: Any,
    symbol: str,
    qty: float,
    side: str,
    entry_price: float,
    take_profit_pct: float,
    hard_stop_pct: float,
) -> dict[str, Any]:
    """Submit an OCO bracket for a freshly-filled entry.

    Returns a result dict regardless of outcome so callers can append
    it to their per-cycle audit record. Never raises.

    Skip conditions (all logged + returned, never raise):
      * Side != "buy" (Phase 35 brackets are long-only)
      * Fractional qty (Alpaca rejects bracket on fractional shares)
      * Threshold pair invalid / produces unusable prices
      * Broker lacks ``submit_bracket`` (e.g. IBKR stub)
    """
    if str(side).lower() != "buy":
        return {"attached": False, "reason": "side_not_buy"}

    try:
        q = float(qty)
    except (TypeError, ValueError):
        return {"attached": False, "reason": "qty_unparseable"}
    if q <= 0:
        return {"attached": False, "reason": "qty_non_positive"}
    if q != int(q):
        # Alpaca rejects brackets on fractional-share orders.
        return {"attached": False, "reason": "fractional_qty"}

    levels = compute_bracket_levels(
        entry_price=entry_price,
        take_profit_pct=take_profit_pct,
        hard_stop_pct=hard_stop_pct,
    )
    if levels is None:
        return {"attached": False, "reason": "invalid_thresholds"}

    submit_bracket = getattr(broker, "submit_bracket", None)
    if submit_bracket is None or not callable(submit_bracket):
        # Mixed-broker setups (e.g. IBKR stub) won't support brackets.
        return {"attached": False, "reason": "broker_no_bracket"}

    req = BracketOrderRequest(
        symbol=symbol,
        side="buy",
        qty=int(q),
        take_profit_price=levels.take_profit_price,
        stop_loss_stop_price=levels.stop_loss_stop_price,
        stop_loss_limit_price=levels.stop_loss_limit_price,
        type="market",
        time_in_force="day",
    )
    try:
        ack: OrderAck = await submit_bracket(req)
    except BrokerError as exc:
        log.warning("bracket attach failed for %s: %s", symbol, exc)
        return {
            "attached": False,
            "reason": "broker_error",
            "error": str(exc)[:200],
        }
    except Exception as exc:  # pragma: no cover — defensive
        log.warning("bracket attach crashed for %s: %s", symbol, exc)
        return {
            "attached": False,
            "reason": "unexpected_error",
            "error": str(exc)[:200],
        }
    return {
        "attached": True,
        "broker_order_id": ack.broker_order_id,
        "status": ack.status,
        "take_profit_price": levels.take_profit_price,
        "stop_loss_stop_price": levels.stop_loss_stop_price,
        "stop_loss_limit_price": levels.stop_loss_limit_price,
    }


__all__ = [
    "AlpacaPaperBroker",
    "BracketLevels",
    "attach_bracket_after_entry",
    "compute_bracket_levels",
]
