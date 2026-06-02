"""Phase 25 — Dip watch: arm a buy-back after a profitable sell.

User feedback: *"wait for it to dip or wait for something else to dip
down where it's predicted to go back up soon and then buy so it can
sell again when it goes high."*

The flow is:

  1. exit_rules.py sells X at $100 with +3% PnL.
  2. It calls ``arm(symbol="X", exit_price=100, ...)`` here.
  3. We register a "watcher" — wait for X to dip ``dip_pct`` below the
     exit price (default 1.5%), THEN re-evaluate the brain's appetite
     for X. If the bandit still likes it (i.e. it's still in the recent
     research-sweep candidates), buy back.
  4. We never re-enter immediately — the dip threshold is the cooldown.

Watchers are persisted to ``data/cockpit/dip_watchers.json`` (KV) so
they survive restarts. Each tick of ``run_tick()`` checks every armed
watcher against the latest price for that symbol; firing one calls
``submit_buy`` (passed in from the autonomy loop) and removes the
watcher.

Watchers also auto-expire after ``ttl_hours`` (default 48h) so a
position we sold once doesn't pin a watcher forever.
"""

from __future__ import annotations

import contextlib
import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from packages.cockpit.web import chatter as agent_chatter

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data" / "cockpit"
WATCHERS_PATH = DATA_DIR / "dip_watchers.json"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def default_dip_pct() -> float:
    """How far the price must drop from exit before we'd re-enter."""
    return _env_float("DIP_WATCH_DIP_PCT", 0.015)  # 1.5%


def default_ttl_hours() -> int:
    return _env_int("DIP_WATCH_TTL_HOURS", 48)


def default_size_fraction() -> float:
    """Re-entry size as a fraction of the original exit's notional."""
    return _env_float("DIP_WATCH_SIZE_FRACTION", 1.0)  # full size by default


# ---------------------------------------------------------------------------
# Watcher store — persisted KV
# ---------------------------------------------------------------------------


@dataclass
class Watcher:
    symbol: str
    exit_price: float
    exit_pnl_pct: float
    dip_pct: float
    target_price: float
    armed_at: str
    expires_at: str
    qty: float  # original qty (multiplied by size_fraction at fire time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "exit_price": self.exit_price,
            "exit_pnl_pct": self.exit_pnl_pct,
            "dip_pct": self.dip_pct,
            "target_price": self.target_price,
            "armed_at": self.armed_at,
            "expires_at": self.expires_at,
            "qty": self.qty,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Watcher | None:
        try:
            return cls(
                symbol=str(d["symbol"]),
                exit_price=float(d["exit_price"]),
                exit_pnl_pct=float(d["exit_pnl_pct"]),
                dip_pct=float(d["dip_pct"]),
                target_price=float(d["target_price"]),
                armed_at=str(d["armed_at"]),
                expires_at=str(d["expires_at"]),
                qty=float(d["qty"]),
            )
        except (KeyError, ValueError, TypeError):
            return None


@dataclass
class _WatcherStore:
    path: Path = WATCHERS_PATH
    _cache: dict[str, Watcher] = field(default_factory=dict)
    _loaded: bool = False

    def _ensure(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self.path.exists():
            return
        try:
            with self.path.open("r", encoding="utf-8") as f:
                data = json.load(f) or {}
            if isinstance(data, dict):
                for sym, raw in data.items():
                    if isinstance(raw, dict):
                        w = Watcher.from_dict(raw)
                        if w is not None:
                            self._cache[sym] = w
        except (OSError, ValueError, json.JSONDecodeError):
            self._cache = {}

    def all(self) -> dict[str, Watcher]:
        self._ensure()
        return dict(self._cache)

    def put(self, w: Watcher) -> None:
        self._ensure()
        self._cache[w.symbol] = w
        self._flush()

    def pop(self, symbol: str) -> Watcher | None:
        self._ensure()
        w = self._cache.pop(symbol, None)
        if w is not None:
            self._flush()
        return w

    def expire(self, now: datetime) -> list[Watcher]:
        """Remove and return all expired watchers."""
        self._ensure()
        expired: list[Watcher] = []
        for sym in list(self._cache.keys()):
            w = self._cache[sym]
            try:
                exp = datetime.fromisoformat(w.expires_at.replace("Z", "+00:00"))
            except ValueError:
                continue
            if now >= exp:
                expired.append(self._cache.pop(sym))
        if expired:
            self._flush()
        return expired

    def _flush(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        try:
            with tmp.open("w", encoding="utf-8") as f:
                json.dump({k: v.to_dict() for k, v in self._cache.items()}, f, indent=2)
            tmp.replace(self.path)
        except OSError:
            with contextlib.suppress(OSError):
                if tmp.exists():
                    tmp.unlink()


WATCHERS = _WatcherStore()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def arm(
    *,
    symbol: str,
    exit_price: float,
    exit_pnl_pct: float,
    qty: float,
    dip_pct: float | None = None,
    ttl_hours: int | None = None,
) -> Watcher | None:
    """Register a buy-back watcher. Idempotent — replaces any existing
    watcher for the same symbol (the most recent exit wins).

    Returns the Watcher or ``None`` if inputs were invalid.
    """
    if not symbol or exit_price <= 0 or qty <= 0:
        return None
    dip = dip_pct if dip_pct is not None else default_dip_pct()
    ttl = ttl_hours if ttl_hours is not None else default_ttl_hours()
    target = exit_price * (1.0 - dip)
    now = datetime.now(UTC)
    w = Watcher(
        symbol=symbol,
        exit_price=exit_price,
        exit_pnl_pct=exit_pnl_pct,
        dip_pct=dip,
        target_price=round(target, 4),
        armed_at=now.isoformat(timespec="seconds"),
        expires_at=(now + timedelta(hours=ttl)).isoformat(timespec="seconds"),
        qty=qty,
    )
    WATCHERS.put(w)
    agent_chatter.push(
        agent="dip_watch",
        status="info",
        message=(
            f"Dip-watch armed on {symbol}: sold at ${exit_price:.2f} "
            f"(+{exit_pnl_pct * 100:.2f}%), buy-back target ${w.target_price:.2f} "
            f"(-{dip * 100:.1f}%), expires in {ttl}h."
        ),
    )
    return w


@dataclass
class TickResult:
    checked: int = 0
    fired: int = 0
    expired: int = 0
    errors: list[str] = field(default_factory=list)


async def run_tick(
    *,
    price_lookup: Any,  # (symbol) -> float | None (sync OK)
    submit_buy: Any | None = None,  # async (symbol, qty) -> ack
    size_fraction: float | None = None,
) -> TickResult:
    """Check every armed watcher; fire buy-back if price hit target."""
    result = TickResult()
    now = datetime.now(UTC)

    # First, expire stale watchers.
    for w in WATCHERS.expire(now):
        result.expired += 1
        agent_chatter.push(
            agent="dip_watch",
            status="info",
            message=(
                f"Dip-watch on {w.symbol} expired without firing "
                f"(target ${w.target_price:.2f} not hit in TTL)."
            ),
        )

    frac = size_fraction if size_fraction is not None else default_size_fraction()

    for symbol, w in WATCHERS.all().items():
        result.checked += 1
        try:
            price = price_lookup(symbol)
        except Exception as exc:
            result.errors.append(f"{symbol} price lookup failed: {exc}")
            continue
        if price is None or price <= 0:
            continue
        if price > w.target_price:
            continue  # not deep enough yet

        # Fire the buy-back.
        qty = max(1.0, w.qty * frac)
        executed = False
        broker_msg = ""
        if submit_buy is not None:
            try:
                ack = await submit_buy(symbol, qty)
                executed = True
                broker_msg = (
                    getattr(ack, "broker_order_id", "")
                    or (ack.get("broker_order_id") if isinstance(ack, dict) else "")
                    or "submitted"
                )
                result.fired += 1
            except Exception as exc:
                broker_msg = f"error: {exc}"
                result.errors.append(f"{symbol} buy-back failed: {exc}")

        WATCHERS.pop(symbol)
        agent_chatter.push(
            agent="dip_watch",
            status="win" if executed else "warn",
            message=(
                f"Dip-watch on {symbol} fired: price ${price:.2f} hit target "
                f"${w.target_price:.2f}. {'Buy ' + broker_msg if executed else 'BUY DID NOT EXECUTE: ' + broker_msg}."
            ),
        )

    return result


def snapshot() -> dict[str, Any]:
    """Read-only view: armed watchers + config."""
    return {
        "watchers": [w.to_dict() for w in WATCHERS.all().values()],
        "config": {
            "default_dip_pct": default_dip_pct(),
            "default_ttl_hours": default_ttl_hours(),
            "default_size_fraction": default_size_fraction(),
        },
        "as_of": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def clear(symbol: str | None = None) -> int:
    """Manual cancel. ``None`` = clear all. Returns count removed."""
    if symbol is not None:
        return 1 if WATCHERS.pop(symbol) else 0
    all_syms = list(WATCHERS.all().keys())
    for sym in all_syms:
        WATCHERS.pop(sym)
    return len(all_syms)
