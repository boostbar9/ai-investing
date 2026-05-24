"""FastAPI cockpit - local monitor + control panel for the paper bot.

Endpoints
---------
GET  /                     -> server-rendered HTML dashboard (polls /api/state)
GET  /api/state            -> snapshot JSON of everything below
GET  /api/positions        -> current Alpaca paper positions
GET  /api/account          -> account equity, cash, buying power, day P&L
GET  /api/trades           -> recent trades (last 50 from runs.jsonl)
GET  /api/streak           -> paper-day streak summary
GET  /api/regime           -> latest regime classification + override
GET  /api/equity-curve     -> equity curve points for the chart
POST /api/pause            -> toggle global pause
POST /api/resume           -> resume from pause
POST /api/override-regime  -> set manual regime override (bull/chop/bear/crisis/auto)
POST /api/flatten          -> emergency: cancel orders + sell all positions
POST /api/run-now          -> trigger a paper run immediately (background task)

Launch with ``python tools/cockpit.py`` or
``uvicorn packages.cockpit.web.server:app --port 8765``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from packages.cockpit.state import (
    VALID_OVERRIDES,
    load_state,
    record_action,
    save_state,
)
from packages.execution.broker import AlpacaPaperBroker, BrokerError, OrderRequest
from packages.paper.streak import compute_paper_streak

log = logging.getLogger("cockpit")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

PAPER_LOG = Path("data/paper_log/runs.jsonl")
# packages/cockpit/web/server.py -> repo root is 3 levels up.
REPO_ROOT = Path(__file__).resolve().parents[3]
INDEX_HTML = Path(__file__).parent / "templates" / "index.html"


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------


class OverrideRequest(BaseModel):
    """Manual regime override payload."""

    regime: str


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def read_runs(limit: int | None = None) -> list[dict[str, Any]]:
    """Read paper-log runs, newest first. Tolerant to malformed lines."""
    if not PAPER_LOG.exists():
        return []
    rows: list[dict[str, Any]] = []
    with PAPER_LOG.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    rows.reverse()
    return rows[:limit] if limit else rows


def latest_account_snapshot() -> dict[str, Any]:
    """Pull the most recent equity/buying-power snapshot recorded by the paper runner.

    The runner writes a record per invocation; we surface the latest one. Cash
    and day-PnL aren't currently logged, so they come back as ``None`` until
    the runner is extended.
    """
    runs = read_runs(limit=1)
    if not runs:
        return {
            "equity": None,
            "cash": None,
            "buying_power": None,
            "day_pnl": None,
            "as_of": None,
            "strategy": None,
            "halted": None,
        }
    r = runs[0]
    return {
        "equity": r.get("account_equity"),
        "cash": r.get("account_cash"),
        "buying_power": r.get("account_buying_power"),
        "day_pnl": r.get("day_pnl"),
        "as_of": r.get("ts"),
        "strategy": r.get("strategy"),
        "halted": r.get("halted"),
    }


def latest_positions() -> list[dict[str, Any]]:
    """Derive an approximate positions view from the latest run's planned weights.

    The runner doesn't persist a positions list to JSONL today; it logs the
    target weights and the orders it submitted. The cockpit reconstructs an
    approximate per-symbol view by combining the latest target weights with
    the most recent equity figure.
    """
    runs = read_runs(limit=1)
    if not runs:
        return []
    r = runs[0]
    target = r.get("target_weights") or {}
    equity = float(r.get("account_equity") or 0.0)
    out: list[dict[str, Any]] = []
    for symbol, weight in target.items():
        try:
            w = float(weight)
        except (TypeError, ValueError):
            continue
        out.append(
            {
                "symbol": symbol,
                "target_weight": w,
                "approx_value": round(w * equity, 2) if equity else None,
            }
        )
    out.sort(key=lambda p: abs(p["target_weight"]), reverse=True)
    return out


def latest_trades(limit: int = 50) -> list[dict[str, Any]]:
    """Flatten submitted orders across recent runs."""
    trades: list[dict[str, Any]] = []
    for run in read_runs():
        ts = run.get("ts")
        strategy = run.get("strategy")
        for order in run.get("orders_submitted", []) or []:
            trades.append({**order, "run_ts": ts, "strategy": strategy})
            if len(trades) >= limit:
                return trades
    return trades


def latest_regime() -> dict[str, Any]:
    """Pull the latest regime reading from the most recent run, plus the override."""
    state = load_state()
    runs = read_runs(limit=1)
    auto_regime = None
    confidence = None
    if runs:
        # Try a few likely keys; the ensemble logger may evolve over time.
        auto_regime = runs[0].get("regime") or runs[0].get("current_regime")
        confidence = runs[0].get("regime_confidence")
    effective = state.regime_override if state.regime_override != "auto" else auto_regime
    return {
        "auto": auto_regime,
        "override": state.regime_override,
        "effective": effective,
        "confidence": confidence,
    }


def equity_curve_points(window: int = 90) -> list[dict[str, Any]]:
    """Return chronological equity points for the last N runs."""
    runs = read_runs()
    runs.reverse()  # chronological
    runs = runs[-window:]
    out: list[dict[str, Any]] = []
    for r in runs:
        ts = r.get("ts")
        eq = r.get("account_equity")
        if ts is None or eq is None:
            continue
        try:
            eq_f = float(eq)
        except (TypeError, ValueError):
            continue
        out.append({"t": ts, "equity": eq_f})
    return out


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------

app = FastAPI(title="ai-investing cockpit", version="0.1.0")


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    """Serve the dashboard shell."""
    if not INDEX_HTML.exists():
        return HTMLResponse("<h1>cockpit template missing</h1>", status_code=500)
    return HTMLResponse(INDEX_HTML.read_text(encoding="utf-8"))


@app.get("/api/state")
def api_state() -> JSONResponse:
    """One-shot snapshot used by the dashboard's poll loop."""
    cstate = load_state()
    summary = compute_paper_streak(PAPER_LOG)
    return JSONResponse(
        {
            "now": datetime.now(UTC).isoformat(timespec="seconds"),
            "control": cstate.to_dict(),
            "account": latest_account_snapshot(),
            "regime": latest_regime(),
            "streak": {
                "current": summary.current_streak,
                "longest": summary.longest_streak,
                "target": summary.gate_target_days,
                "days_remaining": summary.days_remaining,
                "peak_equity": summary.peak_equity,
                "current_drawdown": summary.current_drawdown,
                "gate_passed": summary.gate_passed,
                "clean_days": summary.clean_days,
                "total_days": summary.total_days,
            },
            "positions": latest_positions(),
            "trades": latest_trades(limit=20),
            "equity_curve": equity_curve_points(),
        }
    )


@app.get("/api/positions")
def api_positions() -> list[dict[str, Any]]:
    return latest_positions()


@app.get("/api/account")
def api_account() -> dict[str, Any]:
    return latest_account_snapshot()


@app.get("/api/trades")
def api_trades(limit: int = 50) -> list[dict[str, Any]]:
    return latest_trades(limit=limit)


@app.get("/api/streak")
def api_streak() -> dict[str, Any]:
    return compute_paper_streak(PAPER_LOG).to_dict()


@app.get("/api/regime")
def api_regime() -> dict[str, Any]:
    return latest_regime()


@app.get("/api/equity-curve")
def api_equity_curve(window: int = 90) -> list[dict[str, Any]]:
    return equity_curve_points(window=window)


@app.post("/api/pause")
def api_pause() -> dict[str, Any]:
    state = load_state()
    state.paused = True
    state = record_action(state, "Bot paused from cockpit")
    save_state(state)
    return state.to_dict()


@app.post("/api/resume")
def api_resume() -> dict[str, Any]:
    state = load_state()
    state.paused = False
    state = record_action(state, "Bot resumed from cockpit")
    save_state(state)
    return state.to_dict()


@app.post("/api/override-regime")
def api_override_regime(req: OverrideRequest) -> dict[str, Any]:
    if req.regime not in VALID_OVERRIDES:
        raise HTTPException(
            status_code=400,
            detail=f"regime must be one of {VALID_OVERRIDES}; got {req.regime!r}",
        )
    state = load_state()
    state.regime_override = req.regime  # type: ignore[assignment]
    state = record_action(state, f"Regime override -> {req.regime}")
    save_state(state)
    return state.to_dict()


@app.post("/api/flatten")
async def api_flatten(background_tasks: BackgroundTasks) -> dict[str, Any]:
    """Pause + queue a flatten run. The actual broker calls happen in the
    background task so the HTTP request returns immediately."""
    state = load_state()
    state.paused = True
    state = record_action(state, "EMERGENCY FLATTEN requested from cockpit")
    save_state(state)
    background_tasks.add_task(_run_flatten)
    return {**state.to_dict(), "queued": "flatten"}


@app.get("/api/live-positions")
async def api_live_positions() -> list[dict[str, Any]]:
    """Fetch positions directly from Alpaca paper (real-time, not log-derived).

    Falls back to the log-derived view if Alpaca creds aren't configured.
    """
    if not (os.getenv("ALPACA_PAPER_KEY_ID") and os.getenv("ALPACA_PAPER_SECRET")):
        return latest_positions()
    broker = AlpacaPaperBroker()
    try:
        positions = await broker.positions()
        return [p.to_dict() for p in positions]
    except BrokerError as e:
        log.warning("live positions fetch failed: %s", e)
        return latest_positions()
    finally:
        with contextlib.suppress(Exception):
            await broker.aclose()


@app.post("/api/run-now")
def api_run_now(background_tasks: BackgroundTasks) -> dict[str, Any]:
    """Kick off a paper-trade run in the background (uses defaults)."""
    background_tasks.add_task(_run_paper_trade)
    return {"queued": "paper-trade", "ts": datetime.now(UTC).isoformat(timespec="seconds")}


# --------------------------------------------------------------------------
# Background helpers (run subprocesses; never block the request)
# --------------------------------------------------------------------------


def _python_exe() -> str:
    """Use the same Python interpreter that started the server."""
    return sys.executable


def _run_paper_trade() -> None:
    cmd = [_python_exe(), "tools/paper_trade.py", "--strategy", "ensemble"]
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", ".")
    try:
        subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=False, timeout=600)
    except subprocess.TimeoutExpired:
        log.warning("paper-trade run timed out after 600s")


def _run_flatten() -> None:
    """Cancel orders + submit closing market orders for every open position.

    Runs synchronously in a background thread (FastAPI BackgroundTasks). Uses
    ``asyncio.run`` to drive the async broker calls.
    """
    try:
        asyncio.run(_flatten_async())
    except Exception:
        log.exception("flatten failed")


async def _flatten_async() -> None:
    if not (os.getenv("ALPACA_PAPER_KEY_ID") and os.getenv("ALPACA_PAPER_SECRET")):
        log.warning("flatten skipped: Alpaca creds not in environment")
        return
    broker = AlpacaPaperBroker()
    try:
        positions = await broker.positions()
        log.info("flatten: %d positions to close", len(positions))
        for p in positions:
            qty = abs(float(p.qty))
            if qty <= 0:
                continue
            side = "sell" if float(p.qty) > 0 else "buy"
            try:
                ack = await broker.submit(
                    OrderRequest(symbol=p.symbol, side=side, qty=qty, type="market")
                )
                log.info("flatten: closed %s %s qty=%s order=%s", side, p.symbol, qty, ack.broker_order_id)
            except BrokerError as e:
                log.error("flatten: failed to close %s: %s", p.symbol, e)
    finally:
        with contextlib.suppress(Exception):
            await broker.aclose()


# --------------------------------------------------------------------------
# Error handlers
# --------------------------------------------------------------------------


@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
    log.exception("unhandled error on %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={"error": str(exc) or exc.__class__.__name__},
    )


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="info")
