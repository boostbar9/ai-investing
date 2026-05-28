"""Round-trip pairing.

The shadow audit log is a stream of one-side orders -- buys and sells
appended in time order. To compute PnL we need *round-trips*: each
buy paired with the next matching sell of the same symbol.

We use a simple FIFO queue per symbol:

* on BUY: enqueue the open lot
* on SELL: pop the oldest open lot, emit a ``PairedTrade``

Quantities are matched lot-for-lot. If a sell quantity exceeds the
open lot we split the sell across multiple open lots; if the open lot
exceeds the sell qty we split the lot. This is the same logic any
broker uses for FIFO cost-basis accounting; the only difference is we
do it on shadow records.

Trades without a usable price (``limit_price`` is the only field we
have for shadow fills) are skipped -- there's nothing to compute.
"""
from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PairedTrade:
    """One round-trip: bought at ``buy_px``, sold at ``sell_px``."""

    symbol: str
    buy_ts: str
    sell_ts: str
    qty: float
    buy_px: float
    sell_px: float

    @property
    def pnl(self) -> float:
        return (self.sell_px - self.buy_px) * self.qty


@dataclass
class _OpenLot:
    ts: str
    qty: float
    px: float


def _coerce_qty(raw: Any) -> float | None:
    try:
        q = float(raw)
    except (TypeError, ValueError):
        return None
    return q if q > 0 else None


def _coerce_price(raw: Any) -> float | None:
    if raw is None:
        return None
    try:
        p = float(raw)
    except (TypeError, ValueError):
        return None
    return p if p > 0 else None


def pair_round_trips(trades: Iterable[dict[str, Any]]) -> list[PairedTrade]:
    """Walk shadow trades in time order and emit round-trip pairs."""
    # Sort by ts so we don't depend on file order.
    rows = sorted(
        [r for r in trades if r.get("symbol") and r.get("ts")],
        key=lambda r: r["ts"],
    )

    open_lots: dict[str, deque[_OpenLot]] = defaultdict(deque)
    paired: list[PairedTrade] = []
    for r in rows:
        side = str(r.get("side", "")).lower()
        symbol = str(r["symbol"]).upper()
        qty = _coerce_qty(r.get("qty"))
        price = _coerce_price(r.get("limit_price"))
        ts = str(r["ts"])
        if qty is None or price is None:
            continue
        if side == "buy":
            open_lots[symbol].append(_OpenLot(ts=ts, qty=qty, px=price))
            continue
        if side != "sell":
            continue
        # Sell -- match against FIFO queue.
        remaining = qty
        queue = open_lots[symbol]
        while remaining > 0 and queue:
            lot = queue[0]
            take = min(lot.qty, remaining)
            paired.append(
                PairedTrade(
                    symbol=symbol,
                    buy_ts=lot.ts,
                    sell_ts=ts,
                    qty=take,
                    buy_px=lot.px,
                    sell_px=price,
                )
            )
            lot.qty -= take
            remaining -= take
            if lot.qty <= 0:
                queue.popleft()
        # Any unmatched sell (no matching open lot) is silently dropped --
        # this is shadow data, not a real position.
    return paired
