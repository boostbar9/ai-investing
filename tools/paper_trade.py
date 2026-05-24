"""Nightly paper-trading runner.

Reads the latest daily Parquet data, runs the champion strategy, diffs the
target weights against current Alpaca paper positions, and submits the
minimum set of orders to close the gap. Logs everything to
``data/paper_log/runs.jsonl`` (append-only).

Kill switches (any one of these halts before any order is sent):

1. Equity drawdown from session peak > ``MAX_DD_PCT`` (default 8%)
2. Margin utilization > ``MARGIN_HALT_PCT`` (default 95%)
3. ``ENABLE_PAPER_TRADING`` env var not set to ``true``
4. Alpaca account status not ACTIVE

Designed to run from cron once per trading day after the close. Idempotent:
running it twice in the same session is a no-op (orders are only created
when the target weight changes by more than ``MIN_REBALANCE_BPS``).

Usage::

    PYTHONPATH=. python3 tools/paper_trade.py --strategy mean-reversion --dry-run
    PYTHONPATH=. python3 tools/paper_trade.py --strategy mean-reversion
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from packages.agents.paper_bridge import advise as agent_advise
from packages.execution.broker import (
    AlpacaPaperBroker,
    BrokerError,
    OrderRequest,
)
from packages.paper.streak import compute_paper_streak
from packages.regime.ensemble import (
    RegimeGatedEnsemble,
    RegimeWeights,
    detect_regime_series,
)
from packages.shared.schemas import Position
from packages.strategies import (
    MeanReversion,
    SectorRotation,
    SentimentOverlay,
    TrendFollowing,
)

log = logging.getLogger("paper_trade")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DATA_ROOT = Path("data/parquet/daily")
PAPER_LOG_DIR = Path("data/paper_log")
PAPER_LOG_FILE = PAPER_LOG_DIR / "runs.jsonl"
EQUITY_PEAK_FILE = PAPER_LOG_DIR / "session_peak.json"

# Defaults; overridable via env.
MAX_DD_PCT = float(os.getenv("MAX_DD_PCT", "0.08"))
MARGIN_HALT_PCT = float(os.getenv("MARGIN_HALT_PCT", "0.95"))
MIN_REBALANCE_BPS = float(os.getenv("MIN_REBALANCE_BPS", "25"))  # 0.25% min weight change


STRATEGIES = {
    "trend-following": lambda: TrendFollowing(fast=50, slow=200),
    "sector-rotation": lambda: SectorRotation(top_n=3),
    # Walk-forward-tuned params (see docs/mean-reversion-tuning.md).
    "mean-reversion": lambda: MeanReversion(rsi_entry=15.0, rsi_exit=60.0, sma=200),
}

STRATEGY_UNIVERSE = {
    "trend-following": ["SPY", "QQQ", "IWM", "DIA", "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "TSLA"],
    "sector-rotation": ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU"],
    "mean-reversion": ["SPY", "QQQ", "IWM"],
    # Ensemble = union of trend + sector + mean-reversion, gated by HMM regime.
    "ensemble": [
        "SPY", "QQQ", "IWM", "DIA", "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "TSLA",
        "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU",
    ],
}

STRATEGY_CHOICES = [*STRATEGIES, "ensemble"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_panel(symbols: list[str]) -> pd.DataFrame:
    frames: list[pd.Series] = []
    for sym in symbols:
        p = DATA_ROOT / f"{sym}.parquet"
        if not p.exists():
            log.warning("missing parquet for %s; skipping", sym)
            continue
        df = pd.read_parquet(p)
        df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(None)
        df = df.set_index("ts").sort_index()
        frames.append(df["close"].rename(sym))
    panel = pd.concat(frames, axis=1).ffill().dropna(how="any")
    return panel


# ---------------------------------------------------------------------------
# Kill switches
# ---------------------------------------------------------------------------


@dataclass
class KillSwitchResult:
    halt: bool
    reasons: list[str]


def update_session_peak(equity: float) -> float:
    PAPER_LOG_DIR.mkdir(parents=True, exist_ok=True)
    peak = equity
    if EQUITY_PEAK_FILE.exists():
        try:
            data = json.loads(EQUITY_PEAK_FILE.read_text())
            peak = max(float(data.get("peak", 0.0)), equity)
        except (json.JSONDecodeError, ValueError, OSError):
            pass
    EQUITY_PEAK_FILE.write_text(json.dumps({"peak": peak, "updated_at": datetime.now(UTC).isoformat()}))
    return peak


def check_kill_switches(account: dict[str, Any]) -> KillSwitchResult:
    reasons: list[str] = []
    if os.getenv("ENABLE_PAPER_TRADING", "false").lower() != "true":
        reasons.append("ENABLE_PAPER_TRADING != true")
    if account.get("status") != "ACTIVE":
        reasons.append(f"account status={account.get('status')!r}")
    if account.get("trading_blocked"):
        reasons.append("trading_blocked=true")
    if account.get("account_blocked"):
        reasons.append("account_blocked=true")

    equity = float(account.get("equity", 0))
    last_equity = float(account.get("last_equity", equity))
    peak = update_session_peak(equity)
    if peak > 0:
        dd = (peak - equity) / peak
        if dd > MAX_DD_PCT:
            reasons.append(f"DD {dd*100:.2f}% > {MAX_DD_PCT*100:.0f}% (peak ${peak:,.0f}, now ${equity:,.0f})")

    buying_power = float(account.get("buying_power", 0))
    long_market_value = float(account.get("long_market_value", 0))
    if buying_power > 0:
        util = long_market_value / (long_market_value + buying_power)
        if util > MARGIN_HALT_PCT:
            reasons.append(f"margin util {util*100:.1f}% > {MARGIN_HALT_PCT*100:.0f}%")

    # daily P&L info (purely advisory; logged not enforced)
    log.info(
        "account equity=$%s last_equity=$%s peak=$%s dd_today=%.2f%%",
        f"{equity:,.0f}",
        f"{last_equity:,.0f}",
        f"{peak:,.0f}",
        (peak - equity) / peak * 100 if peak > 0 else 0.0,
    )
    return KillSwitchResult(halt=bool(reasons), reasons=reasons)


# ---------------------------------------------------------------------------
# Order planning
# ---------------------------------------------------------------------------


def compute_target_weights(strategy_name: str) -> dict[str, float]:
    """Run the strategy on real data; return last-bar weights as a dict."""
    if strategy_name == "ensemble":
        return compute_ensemble_weights()
    symbols = STRATEGY_UNIVERSE[strategy_name]
    panel = load_panel(symbols)
    if panel.empty:
        raise RuntimeError(f"no price panel for {strategy_name}")
    strategy = STRATEGIES[strategy_name]()
    # Sentiment overlay handled below if requested by caller.
    weights = strategy.generate_signals(panel)
    last_row = weights.iloc[-1].to_dict()
    return {k: float(v) for k, v in last_row.items() if not pd.isna(v)}


def _build_regime_series(panel: pd.DataFrame) -> pd.Series:
    """Daily regime labels using realised-vol VIX proxy + cross-section breadth.

    Mirrors the construction used in ``tools/stress_ensemble.py`` so paper
    behaviour matches the Tier-2 stress results.
    """
    # Fallback to first column as broad-market proxy when SPY is absent.
    spy = panel["SPY"] if "SPY" in panel.columns else panel.iloc[:, 0]
    realised_vol = spy.pct_change().rolling(20).std() * np.sqrt(252) * 100
    vix_proxy = realised_vol.fillna(15.0)
    rets_5d = panel.pct_change(5)
    breadth = (rets_5d > 0).mean(axis=1).fillna(0.5)
    return detect_regime_series(spy, vix_proxy, breadth)


def compute_ensemble_weights() -> dict[str, float]:
    """Run the regime-gated ensemble and return last-bar target weights."""
    symbols = STRATEGY_UNIVERSE["ensemble"]
    panel = load_panel(symbols)
    if panel.empty:
        raise RuntimeError("no price panel for ensemble")
    regimes = _build_regime_series(panel)
    ensemble = RegimeGatedEnsemble(
        strategies={
            "trend-following": TrendFollowing(fast=50, slow=200),
            "mean-reversion": MeanReversion(rsi_entry=15.0, rsi_exit=60.0, sma=200),
            "sector-rotation": SectorRotation(top_n=3),
        },
        regime_weights=RegimeWeights.from_calibrated(),
    )
    weights = ensemble.generate_signals(panel, regimes)
    last_row = weights.iloc[-1].to_dict()
    return {k: float(v) for k, v in last_row.items() if not pd.isna(v) and float(v) > 0}


def compute_target_weights_with_sentiment(
    base_name: str,
    sentiment_scores: dict[str, float] | None,
) -> dict[str, float]:
    """Wrap the base strategy with SentimentOverlay using real scores."""
    if base_name == "ensemble":
        # Ensemble already aggregates per-strategy signals; sentiment overlay
        # is intentionally not stacked on top.
        return compute_ensemble_weights()
    symbols = STRATEGY_UNIVERSE[base_name]
    panel = load_panel(symbols)
    base = STRATEGIES[base_name]()
    # Convert sentiment scores in [-1, 1] to multipliers in [0.5, 1.25].
    # Negative sentiment dampens to 0.5x, neutral=1.0x, max bullish=1.25x.
    mults: dict[str, float] = {}
    for sym in panel.columns:
        s = (sentiment_scores or {}).get(sym, 0.0)
        # linear map -1 -> 0.5, 0 -> 1.0, +1 -> 1.25 (slightly asymmetric)
        if s >= 0:
            mults[sym] = 1.0 + 0.25 * s
        else:
            mults[sym] = 1.0 + 0.5 * s  # -1 -> 0.5
    overlay = SentimentOverlay(base=base, sentiment=mults)
    weights = overlay.generate_signals(panel)
    return {k: float(v) for k, v in weights.iloc[-1].to_dict().items() if not pd.isna(v)}


@dataclass
class PlannedOrder:
    symbol: str
    side: str
    qty: float
    target_weight: float
    current_weight: float
    delta_weight: float
    last_price: float


async def plan_orders(
    target_weights: dict[str, float],
    broker: AlpacaPaperBroker,
    equity: float,
) -> list[PlannedOrder]:
    """Diff target weights against current positions; size in shares."""
    positions = await broker.positions()
    pos_by_sym = {p.symbol: p for p in positions}

    # Current weights = position market value / equity
    current_weights: dict[str, float] = {}
    last_price: dict[str, float] = {}
    for p in positions:
        if p.last_price is None:
            continue
        mv = p.qty * p.last_price
        current_weights[p.symbol] = mv / equity if equity > 0 else 0.0
        last_price[p.symbol] = p.last_price

    # For symbols in target but not currently held, pull a last-price from
    # the most recent parquet bar.
    for sym in target_weights:
        if sym not in last_price:
            p = DATA_ROOT / f"{sym}.parquet"
            if p.exists():
                df = pd.read_parquet(p)
                if not df.empty:
                    last_price[sym] = float(df["close"].iloc[-1])

    all_symbols = set(target_weights) | set(current_weights)
    planned: list[PlannedOrder] = []
    for sym in sorted(all_symbols):
        tw = target_weights.get(sym, 0.0)
        cw = current_weights.get(sym, 0.0)
        delta = tw - cw
        if abs(delta) * 10000 < MIN_REBALANCE_BPS:
            continue
        px = last_price.get(sym)
        if px is None or px <= 0:
            log.warning("no price for %s; cannot size order", sym)
            continue
        delta_dollars = delta * equity
        qty = abs(delta_dollars / px)
        # Round to 4 decimals -- Alpaca supports fractional shares.
        qty = round(qty, 4)
        if qty <= 0:
            continue
        side = "buy" if delta > 0 else "sell"
        # Don't sell more than we hold; cap to current qty.
        if side == "sell":
            current_qty = pos_by_sym[sym].qty if sym in pos_by_sym else 0.0
            qty = min(qty, current_qty)
            if qty <= 0:
                continue
        planned.append(
            PlannedOrder(
                symbol=sym,
                side=side,
                qty=qty,
                target_weight=tw,
                current_weight=cw,
                delta_weight=delta,
                last_price=px,
            )
        )
    return planned


# ---------------------------------------------------------------------------
# Run loop
# ---------------------------------------------------------------------------


def log_run(record: dict[str, Any]) -> None:
    PAPER_LOG_DIR.mkdir(parents=True, exist_ok=True)
    with PAPER_LOG_FILE.open("a") as f:
        f.write(json.dumps(record, default=str) + "\n")


async def run(
    strategy_name: str,
    *,
    dry_run: bool = False,
    use_sentiment: bool = False,
    sentiment_scores: dict[str, float] | None = None,
) -> dict[str, Any]:
    started = datetime.now(UTC)
    broker = AlpacaPaperBroker()
    try:
        if not broker.key_id or not broker.secret:
            return {"halted": True, "reasons": ["Alpaca paper keys not in environment"]}

        account = await broker.account()
        kill = check_kill_switches(account)
        if kill.halt:
            log.warning("HALT: %s", "; ".join(kill.reasons))
            record = {
                "ts": started.isoformat(),
                "strategy": strategy_name,
                "halted": True,
                "reasons": kill.reasons,
                "account_equity": float(account.get("equity", 0)),
                "orders_planned": 0,
                "orders_submitted": 0,
            }
            log_run(record)
            return record

        equity = float(account["equity"])
        if use_sentiment and strategy_name in STRATEGIES:
            target = compute_target_weights_with_sentiment(strategy_name, sentiment_scores)
        else:
            target = compute_target_weights(strategy_name)
        log.info("target weights for %s: %s", strategy_name, target)

        # ------------------------------------------------------------------
        # LangGraph advisory pass (Research -> Strategy -> Risk -> Approval).
        # Runs in advisory mode: never submits orders itself, but can halt
        # the run via the risk agent or veto specific concentration breaches.
        # ------------------------------------------------------------------
        symbols = STRATEGY_UNIVERSE.get(strategy_name, list(target.keys()))
        agent_positions: list[Position] = []
        try:
            for p in await broker.positions():
                agent_positions.append(
                    Position(symbol=p.symbol, qty=float(p.qty), avg_price=float(p.avg_price))
                )
        except Exception as e:
            log.warning("could not fetch positions for agent advisory: %s", e)

        agent_result = await agent_advise(
            symbols=symbols,
            regime="bull",  # placeholder; regime agent lands later
            positions=agent_positions,
            target_weights=target,
            sentiment_scores=sentiment_scores,
        )
        agent_audit = [
            {"actor": a.actor, "event_type": a.event_type, "payload": a.payload}
            for a in agent_result.audit
        ]
        if agent_result.halted:
            reason = agent_result.risk.halt_reason or "risk-agent halted"
            log.warning("AGENT HALT: %s", reason)
            record = {
                "ts": started.isoformat(),
                "strategy": strategy_name,
                "halted": True,
                "reasons": [f"agent_halt: {reason}"],
                "account_equity": equity,
                "agent_audit": agent_audit,
                "agent_decision_id": str(agent_result.decision_id),
                "agent_sentiment": agent_result.research.sentiment,
            }
            log_run(record)
            return record

        # Filter target weights by the symbols the risk agent approved.
        approved_syms = {s.symbol for s in agent_result.risk.approved}
        if approved_syms:
            target = {
                sym: (w if sym in approved_syms else 0.0)
                for sym, w in target.items()
            }
            log.info("agent-approved symbols: %s", sorted(approved_syms))

        planned = await plan_orders(target, broker, equity)
        log.info("planned %d orders", len(planned))

        submitted = []
        errors = []
        if not dry_run:
            for po in planned:
                try:
                    req = OrderRequest(
                        symbol=po.symbol,
                        side=po.side,
                        qty=po.qty,
                        type="market",
                        time_in_force="day",
                    )
                    ack = await broker.submit(req)
                    submitted.append({
                        "symbol": po.symbol,
                        "side": po.side,
                        "qty": po.qty,
                        "broker_order_id": ack.broker_order_id,
                        "status": ack.status,
                    })
                    log.info("submitted %s %s %.4f -> %s", po.side, po.symbol, po.qty, ack.status)
                except BrokerError as e:
                    log.warning("order failed %s %s: %s", po.side, po.symbol, e)
                    errors.append({"symbol": po.symbol, "side": po.side, "error": str(e)})

        record = {
            "ts": started.isoformat(),
            "strategy": strategy_name,
            "dry_run": dry_run,
            "halted": False,
            "account_equity": equity,
            "account_buying_power": float(account.get("buying_power", 0)),
            "target_weights": target,
            "orders_planned": [
                {
                    "symbol": po.symbol, "side": po.side, "qty": po.qty,
                    "target_w": po.target_weight, "current_w": po.current_weight,
                    "delta_w": po.delta_weight, "last_price": po.last_price,
                }
                for po in planned
            ],
            "orders_submitted": submitted,
            "errors": errors,
            "agent_decision_id": str(agent_result.decision_id),
            "agent_sentiment": agent_result.research.sentiment,
            "agent_thesis": agent_result.research.thesis,
            "agent_audit": agent_audit,
            "duration_sec": (datetime.now(UTC) - started).total_seconds(),
        }
        log_run(record)
        # Refresh the §16 streak snapshot AFTER appending this run so the
        # dashboard always sees the latest day. Best-effort: a failure here
        # must not poison the actual run record.
        try:
            streak = compute_paper_streak()
            (PAPER_LOG_DIR / "streak.json").write_text(
                json.dumps(streak.to_dict(), indent=2, default=str)
            )
            log.info(
                "§16 streak: %d/%d clean paper days (longest %d)",
                streak.current_streak,
                streak.gate_target_days,
                streak.longest_streak,
            )
        except Exception as e:
            log.warning("could not refresh paper streak: %s", e)
        return record
    finally:
        await broker.aclose()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", choices=STRATEGY_CHOICES, default="mean-reversion")
    ap.add_argument("--dry-run", action="store_true", help="Plan orders but do not submit.")
    ap.add_argument("--use-sentiment", action="store_true", help="Apply real sentiment overlay.")
    args = ap.parse_args()

    sentiment_scores = None
    if args.use_sentiment:
        # Pull live sentiment (best-effort).
        try:
            from tools.fetch_sentiment import fetch_scores  # late import to avoid hard dep
            sentiment_scores = asyncio.run(fetch_scores(list(set().union(*STRATEGY_UNIVERSE.values()))))
            log.info("loaded %d sentiment scores", len(sentiment_scores))
        except Exception as e:
            log.warning("sentiment fetch failed (%s); falling back to neutral", e)

    result = asyncio.run(
        run(
            args.strategy,
            dry_run=args.dry_run,
            use_sentiment=args.use_sentiment,
            sentiment_scores=sentiment_scores,
        )
    )
    print(json.dumps(result, indent=2, default=str))
    return 0 if not result.get("halted") else 1


if __name__ == "__main__":
    raise SystemExit(main())
