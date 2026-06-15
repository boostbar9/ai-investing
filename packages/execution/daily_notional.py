"""Per-day cumulative buy-notional ledger for the Robinhood float cap.

The float cap (``resolve_float_cap``) was originally enforced PER ORDER:
a single $300 buy was rejected, but ten $290 buys in one day -- $2,900 of
deployed capital -- all slipped through. That defeats the point of a
"first float" blast-radius limiter.

This ledger fixes that by tracking the *aggregate* buy notional deployed
today. Before a live buy is allowed, the broker sums today's already-
recorded buys + the new order and rejects if the total exceeds the cap.

Persistence mirrors the ``shadow_trades.jsonl`` pattern: one JSON line per
recorded buy under ``data/``. Reads filter to the current calendar day in
the same timezone the rest of the cockpit uses (the account-local tz,
falling back to UTC). Sells are never recorded here -- the cap is a
*deployment* ceiling, and sells reduce exposure.

Shadow-mode buys ARE recorded (for realism / parity with how a live day
would accrue) but are never blocked from logging -- only live buys consult
the aggregate before being allowed.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

# Where the ledger lives. Env-overridable for test isolation, mirroring
# ``ROBINHOOD_SHADOW_TRADES_PATH``.
DAILY_NOTIONAL_PATH = Path(
    os.getenv("ROBINHOOD_DAILY_NOTIONAL_PATH", "data/cockpit/daily_notional.jsonl")
)

# Timezone for the calendar-day boundary. The cockpit uses US/Eastern for
# market-day reasoning elsewhere; we follow that so "today's deployed
# notional" resets at the same midnight the trading day does.
LEDGER_TZ = os.getenv("COCKPIT_TZ", "America/New_York")


def _tz() -> ZoneInfo:
    try:
        return ZoneInfo(LEDGER_TZ)
    except Exception:  # pragma: no cover - bad tz name
        return ZoneInfo("UTC")


def _today_key(now: datetime | None = None) -> str:
    ts = now if now is not None else datetime.now(_tz())
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=_tz())
    return ts.astimezone(_tz()).strftime("%Y-%m-%d")


def _path() -> Path:
    # Resolve at call time so monkeypatching the module attr works.
    return sys.modules[__name__].DAILY_NOTIONAL_PATH


def record_buy(
    *,
    symbol: str,
    notional: float,
    mode: str,
    now: datetime | None = None,
) -> None:
    """Append one buy to the ledger. Side classification is the caller's
    job; only buys must be passed here (sells are never cap-counted)."""
    entry = {
        "day": _today_key(now),
        "ts": (now or datetime.now(_tz())).isoformat(timespec="seconds"),
        "symbol": symbol,
        "notional": float(notional),
        "mode": mode,
    }
    target = _path()
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, separators=(",", ":")) + "\n")


def deployed_today(now: datetime | None = None) -> float:
    """Sum of buy notional recorded for the current calendar day.

    Skips malformed lines and rows from other days. Never raises -- a
    read failure returns 0.0 so a corrupt ledger can't wedge trading
    (the per-order cap still applies as a backstop)."""
    target = _path()
    if not target.exists():
        return 0.0
    day = _today_key(now)
    total = 0.0
    try:
        lines = target.read_text(encoding="utf-8").splitlines()
    except OSError:  # pragma: no cover - fs error
        return 0.0
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        try:
            row: dict[str, Any] = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("day") != day:
            continue
        try:
            total += float(row.get("notional", 0.0))
        except (TypeError, ValueError):
            continue
    return total


def would_exceed_cap(
    new_notional: float,
    cap: float,
    *,
    now: datetime | None = None,
) -> tuple[bool, float]:
    """Return ``(exceeds, projected_total)`` for a prospective buy.

    ``exceeds`` is True when today's deployed buy notional PLUS the new
    order would push the aggregate strictly above ``cap``."""
    projected = deployed_today(now) + float(new_notional)
    return projected > cap, projected
