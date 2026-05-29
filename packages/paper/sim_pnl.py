"""Simulated round-trip stream derived from the paper-trade run log.

Phase 11: the cockpit's ``/shadow`` page is supposed to show realised
PnL so the user can decide whether the strategy is performing. Until
now it only knew about the Robinhood ``shadow_trades.jsonl`` audit log
(which is empty -- the user hasn't enabled RH shadow mode). Meanwhile,
``tools/paper_trade.py`` has been writing rich planned-order records
to ``data/paper_log/runs.jsonl`` every cycle, including target weights
and last-known prices. That data is enough to *simulate* what the
strategy would have done, by treating each planned order as if it had
filled at the printed last_price.

This module's job: read ``runs.jsonl``, emit synthetic ``{ts, symbol,
side, qty, limit_price}`` dicts that the existing
``packages.shadow.pairing.pair_round_trips`` can pair into
``PairedTrade`` rows. The output flows through the same PnL/greenlight
pipeline as real shadow trades, so the dashboard renders identically.

This is explicitly a *simulation*, not a backtest: we use the prices the
loop already observed at decision time, and we assume the orders filled
at those prices with no slippage. That's optimistic; the user gets to
see "if our planned orders had filled, this is what we'd be sitting on."
Once real shadow or live trades start flowing, the dashboard merges
both streams (real takes priority).
"""
from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Reuse the same log the paper loop writes. Override for tests.
DEFAULT_RUNS_PATH = Path(os.getenv("PAPER_RUNS_PATH", "data/paper_log/runs.jsonl"))


def _iter_runs(path: Path) -> Iterator[dict[str, Any]]:
    """Yield run records oldest-first. Skips malformed lines."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def synth_trades_from_runs(
    runs_path: Path | None = None,
    *,
    include_dry_run: bool = True,
) -> list[dict[str, Any]]:
    """Convert ``runs.jsonl`` records into synthetic shadow-trade dicts.

    Each planned order becomes one ``buy`` or ``sell`` entry shaped like
    the Robinhood shadow log so ``pair_round_trips`` consumes them
    uniformly. Halt cycles and zero-order cycles are silently skipped.

    ``include_dry_run`` defaults True: dry-run cycles are valuable for
    simulation since the user *wanted* to see what the strategy would
    have done. The /shadow page exposes a toggle later if needed.
    """
    import sys

    target = (
        runs_path
        if runs_path is not None
        else sys.modules[__name__].DEFAULT_RUNS_PATH
    )
    synth: list[dict[str, Any]] = []
    for run in _iter_runs(target):
        if run.get("halted"):
            continue
        if not include_dry_run and run.get("dry_run"):
            continue
        ts = run.get("ts")
        if not ts:
            continue
        # Prefer the rich planned-orders array (each entry has symbol,
        # side, qty, last_price). Skip entries without a price -- we
        # can't pair those.
        for po in run.get("orders_planned") or []:
            symbol = po.get("symbol")
            side = str(po.get("side", "")).lower()
            qty = po.get("qty")
            price = po.get("last_price")
            if not symbol or side not in ("buy", "sell"):
                continue
            try:
                qty_f = float(qty)
                price_f = float(price)
            except (TypeError, ValueError):
                continue
            if qty_f <= 0 or price_f <= 0:
                continue
            synth.append(
                {
                    "ts": str(ts),
                    "symbol": str(symbol).upper(),
                    "side": side,
                    "qty": qty_f,
                    # pair_round_trips reads ``limit_price`` for the fill
                    # px -- the schema is shared with Robinhood shadow.
                    "limit_price": price_f,
                    "synthetic": True,
                    "strategy": str(run.get("strategy") or ""),
                }
            )
    return synth


def merge_real_and_synth(
    real_trades: list[dict[str, Any]],
    synth_trades: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Union of real + synthetic shadow trades for the dashboard.

    Real trades (from the Robinhood shadow log, once enabled) take
    priority: if a real trade exists for a given ``(symbol, ts)`` we drop
    the synthetic one. This way the dashboard's PnL number is always at
    least as conservative as reality.
    """
    seen: set[tuple[str, str, str]] = set()
    out: list[dict[str, Any]] = []
    for r in real_trades or []:
        key = (str(r.get("symbol", "")).upper(), str(r.get("ts", "")), str(r.get("side", "")))
        seen.add(key)
        out.append(r)
    for s in synth_trades or []:
        key = (str(s.get("symbol", "")).upper(), str(s.get("ts", "")), str(s.get("side", "")))
        if key in seen:
            continue
        out.append(s)
    return out


def daily_equity_curve(
    paired,  # type: ignore[no-untyped-def]
    starting_equity: float = 100_000.0,
) -> list[dict[str, Any]]:
    """Cumulative equity curve for the simulated round-trips.

    Used by the /shadow page to plot a single line. Each entry is
    ``{day: ISO date, equity: float}``. Starting equity defaults to
    $100k, matching the Alpaca paper account.
    """
    from packages.shadow.pnl import aggregate_daily

    daily = aggregate_daily(paired)
    if not daily:
        return [{"day": None, "equity": starting_equity}]

    out: list[dict[str, Any]] = []
    eq = float(starting_equity)
    for row in daily:
        eq += float(row.pnl)
        out.append({"day": row.day.isoformat(), "equity": round(eq, 2)})
    return out


__all__ = [
    "DEFAULT_RUNS_PATH",
    "daily_equity_curve",
    "merge_real_and_synth",
    "synth_trades_from_runs",
]
