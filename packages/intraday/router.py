"""Intraday router — Phase 28-R step 4.

Bridges the setup finder to the broker. Given a list of
``RankedSetup``s from ``setup_finder.find_morning_setups``, this module:

  * Computes per-symbol share quantity from the notional split and the
    current intraday close (rounded down to whole shares; fractional
    Alpaca orders are allowed but we keep it simple at first).
  * Skips symbols already held (no doubling up on an existing position).
  * Skips symbols where the price feed is missing (degraded safely).
  * Submits market BUY orders via the injected ``submit_order`` callable.
  * Appends an audit row to ``data/paper_log/intraday_router.jsonl``
    for every decision (submit / skip / error).

The router is intentionally **dumb**: it owns no scoring or filtering
logic. All ranking decisions happened in setup_finder. The router's
job is purely "turn ranked setups into orders, audit everything".

Like the setup finder, the router is gated behind ``INTRADAY_MODE=1``;
``route_setups`` is a no-op unless that flag is set, so wiring it into
the cockpit cron is safe even before the user opts in.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from packages.intraday.setup_finder import (
    RankedSetup,
    is_intraday_mode_enabled,
)

log = logging.getLogger("intraday.router")


DEFAULT_LOG_PATH = Path("data/paper_log/intraday_router.jsonl")

# Refuse to ship orders smaller than this. Below ~$5 the order is mostly
# spread + fees on the Alpaca side.
MIN_NOTIONAL_USD = 5.0


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RouteAttempt:
    """One per-setup result."""

    symbol: str
    action: str  # "submitted" | "skip_held" | "skip_no_price" | "skip_too_small" | "error"
    qty: float = 0.0
    notional_usd: float = 0.0
    price: float = 0.0
    reason: str = ""
    broker_order_id: str | None = None


@dataclass
class RouteResult:
    """Aggregate result of one router run."""

    submitted: list[RouteAttempt] = field(default_factory=list)
    skipped: list[RouteAttempt] = field(default_factory=list)
    errors: list[RouteAttempt] = field(default_factory=list)
    ts: str = ""

    def all(self) -> list[RouteAttempt]:
        return self.submitted + self.skipped + self.errors


# Injectable provider signatures.
SubmitOrder = Callable[[str, float], Awaitable[Any]]
"""``async (symbol, qty) -> broker ack`` — qty is whole shares."""

PriceLookup = Callable[[str], float | None]
"""``symbol -> last_price`` (sync)."""

HeldSymbols = Callable[[], set[str]]
"""``() -> {symbol, ...}`` of currently-open positions (sync)."""


# ---------------------------------------------------------------------------
# Log writer
# ---------------------------------------------------------------------------


def _resolve_log_path(override: Path | str | None) -> Path:
    if override is not None:
        path = Path(override)
    else:
        env_path = os.environ.get("INTRADAY_ROUTER_LOG_PATH")
        path = Path(env_path) if env_path else DEFAULT_LOG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _append_log(record: dict[str, Any], *, log_path: Path | str | None) -> None:
    try:
        path = _resolve_log_path(log_path)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")
    except OSError as exc:  # pragma: no cover — disk is rarely the bug
        log.warning("intraday_router audit write failed: %s", exc)


# ---------------------------------------------------------------------------
# Sizing helper
# ---------------------------------------------------------------------------


def shares_for_notional(notional_usd: float, price: float) -> float:
    """Whole-share quantity for a target notional. Always >= 0.

    Returns 0 when price is non-positive or the notional is below the
    cost of a single share — the router will then skip the setup with
    a "too small" reason.
    """
    if price <= 0 or notional_usd <= 0:
        return 0.0
    qty = int(notional_usd // price)
    return float(max(qty, 0))


# ---------------------------------------------------------------------------
# Core route fn
# ---------------------------------------------------------------------------


async def route_setups(
    setups: list[RankedSetup],
    *,
    submit_order: SubmitOrder,
    price_lookup: PriceLookup,
    held_symbols_getter: HeldSymbols | None = None,
    log_path: Path | str | None = None,
    now: datetime | None = None,
    force_enabled: bool = False,
) -> RouteResult:
    """Submit a market BUY for each ranked setup, with safety skips.

    Args:
        setups: Ranked output of ``find_morning_setups``.
        submit_order: Async broker hook ``(symbol, qty) -> ack``.
        price_lookup: Returns last price for sizing.
        held_symbols_getter: Optional ``() -> set[symbol]`` so we never
            double up on an existing position.
        log_path: Override for the audit log path (mostly tests).
        now: Clock override for tests.
        force_enabled: Bypass the INTRADAY_MODE flag (tests only).

    Returns a ``RouteResult`` summarising every decision. Errors are
    captured per-symbol; one bad order never blocks the others.
    """
    ts = (now or datetime.now(UTC)).astimezone(UTC).isoformat(timespec="seconds")
    result = RouteResult(ts=ts)

    if not (force_enabled or is_intraday_mode_enabled()):
        # Mode disabled — log a single audit row and bail.
        _append_log(
            {"ts": ts, "action": "disabled", "setups": len(setups)},
            log_path=log_path,
        )
        return result

    held: set[str] = set()
    if held_symbols_getter is not None:
        try:
            held = {s.upper() for s in (held_symbols_getter() or set())}
        except Exception as exc:  # pragma: no cover — defensive
            log.warning("held_symbols_getter failed: %s", exc)

    for setup in setups:
        symbol = setup.symbol.upper()

        if symbol in held:
            attempt = RouteAttempt(
                symbol=symbol,
                action="skip_held",
                reason="position_already_open",
            )
            result.skipped.append(attempt)
            _append_log(
                {
                    "ts": ts,
                    "symbol": symbol,
                    "action": attempt.action,
                    "reason": attempt.reason,
                    "score": setup.score,
                },
                log_path=log_path,
            )
            continue

        price = None
        try:
            price = price_lookup(symbol)
        except Exception as exc:  # pragma: no cover — defensive
            log.warning("price_lookup failed for %s: %s", symbol, exc)
        if price is None or price <= 0:
            attempt = RouteAttempt(
                symbol=symbol,
                action="skip_no_price",
                reason="no_quote",
                notional_usd=setup.notional_usd,
            )
            result.skipped.append(attempt)
            _append_log(
                {
                    "ts": ts,
                    "symbol": symbol,
                    "action": attempt.action,
                    "reason": attempt.reason,
                },
                log_path=log_path,
            )
            continue

        qty = shares_for_notional(setup.notional_usd, float(price))
        effective_notional = qty * float(price)
        if qty <= 0 or effective_notional < MIN_NOTIONAL_USD:
            attempt = RouteAttempt(
                symbol=symbol,
                action="skip_too_small",
                qty=qty,
                price=float(price),
                notional_usd=effective_notional,
                reason="below_min_notional",
            )
            result.skipped.append(attempt)
            _append_log(
                {
                    "ts": ts,
                    "symbol": symbol,
                    "action": attempt.action,
                    "qty": qty,
                    "price": float(price),
                    "notional_usd": round(effective_notional, 2),
                    "reason": attempt.reason,
                },
                log_path=log_path,
            )
            continue

        try:
            ack = await submit_order(symbol, qty)
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"[:240]
            log.warning("submit_order failed for %s: %s", symbol, msg)
            attempt = RouteAttempt(
                symbol=symbol,
                action="error",
                qty=qty,
                price=float(price),
                notional_usd=effective_notional,
                reason=msg,
            )
            result.errors.append(attempt)
            _append_log(
                {
                    "ts": ts,
                    "symbol": symbol,
                    "action": attempt.action,
                    "qty": qty,
                    "price": float(price),
                    "reason": msg,
                },
                log_path=log_path,
            )
            continue

        broker_order_id = (
            getattr(ack, "broker_order_id", None)
            or (ack.get("broker_order_id") if isinstance(ack, dict) else None)
        )
        attempt = RouteAttempt(
            symbol=symbol,
            action="submitted",
            qty=qty,
            price=float(price),
            notional_usd=effective_notional,
            reason=setup.reason,
            broker_order_id=broker_order_id,
        )
        result.submitted.append(attempt)
        _append_log(
            {
                "ts": ts,
                "symbol": symbol,
                "action": attempt.action,
                "qty": qty,
                "price": float(price),
                "notional_usd": round(effective_notional, 2),
                "score": setup.score,
                "components": setup.components,
                "reason": setup.reason,
                "broker_order_id": broker_order_id,
            },
            log_path=log_path,
        )

    return result
