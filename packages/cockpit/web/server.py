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
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from packages.cockpit import errors as err_log
from packages.cockpit import proc as job_mgr
from packages.cockpit import updater
from packages.cockpit.state import (
    VALID_MODES,
    VALID_OVERRIDES,
    load_state,
    record_action,
    save_state,
)
from packages.execution.broker import AlpacaPaperBroker, BrokerError, OrderRequest
from packages.paper.streak import compute_paper_streak
from packages.shared import conn_checks, secrets
from packages.shared.dotenv import load_dotenv

# Auto-load .env so the cockpit and any subprocesses it spawns inherit the
# Alpaca paper keys without the user having to set them manually first.
load_dotenv()
# Hydrate any secrets stored in the OS keystore (Windows Credential Manager).
secrets.hydrate_environment()

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


class ModeRequest(BaseModel):
    """Trading mode toggle payload."""

    mode: str
    confirm_live: bool = False


class SecretsUpdate(BaseModel):
    """Bulk secrets update payload from the Settings page."""

    values: dict[str, str]


class StartTradingRequest(BaseModel):
    """Start a paper-trade loop."""

    strategy: str = "ensemble"
    dry_run: bool = False


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

# Mount static assets (shared CSS/JS for every page).
from fastapi.staticfiles import StaticFiles  # noqa: E402

_STATIC_DIR = Path(__file__).parent / "static"
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    """Serve the dashboard shell."""
    return _render("index.html")


@app.get("/settings", response_class=HTMLResponse)
def settings_page() -> HTMLResponse:
    return _render("settings.html")


@app.get("/updates", response_class=HTMLResponse)
def updates_page() -> HTMLResponse:
    return _render("updates.html")


@app.get("/models", response_class=HTMLResponse)
def models_page() -> HTMLResponse:
    return _render("models.html")


@app.get("/trading", response_class=HTMLResponse)
def trading_page() -> HTMLResponse:
    return _render("trading.html")


@app.get("/errors", response_class=HTMLResponse)
def errors_page() -> HTMLResponse:
    return _render("errors.html")


def _render(name: str) -> HTMLResponse:
    path = Path(__file__).parent / "templates" / name
    if not path.exists():
        return HTMLResponse(f"<h1>template missing: {name}</h1>", status_code=500)
    return HTMLResponse(path.read_text(encoding="utf-8"))


@app.get("/api/health")
def api_health() -> dict[str, Any]:
    """Compact health summary for the global topbar and health panel.

    Includes data freshness, the most recent paper-loop run, error counts,
    git commit so the user can verify what build is currently running, and
    a derived ``status`` (ok / warn / down) that the UI maps to a color.
    """
    cstate = load_state()
    err_counts = err_log.count_unresolved()
    job_states = {k: job_mgr.status(k).to_dict() for k in ("paper_loop", "pretrain")}
    last_run_ts: str | None = None
    last_halted = False
    try:
        if PAPER_LOG.exists():
            with PAPER_LOG.open(encoding="utf-8") as f:
                last_line = ""
                for line in f:
                    if line.strip():
                        last_line = line
            if last_line:
                obj = json.loads(last_line)
                last_run_ts = obj.get("ts")
                last_halted = bool(obj.get("halted", False))
    except (OSError, json.JSONDecodeError, ValueError):
        pass
    # Roll up an overall status
    if err_counts.get("error", 0) > 0 or last_halted:
        status = "warn"
    elif not PAPER_LOG.exists():
        status = "idle"
    else:
        status = "ok"
    try:
        commit = updater.current_commit()
    except Exception:
        commit = {"sha": "", "summary": ""}
    return {
        "status": status,
        "now": datetime.now(UTC).isoformat(timespec="seconds"),
        "mode": cstate.trading_mode,
        "paused": cstate.paused,
        "last_paper_run": last_run_ts,
        "last_paper_halted": last_halted,
        "errors": err_counts,
        "jobs": job_states,
        "commit": commit,
    }


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
# Settings (API keys / secrets)
# --------------------------------------------------------------------------


@app.get("/api/secrets")
def api_secrets_get() -> dict[str, Any]:
    """Return per-provider status + masked values for the Settings UI."""
    return {
        "backend": secrets.backend(),
        "providers": secrets.provider_status(),
    }


@app.post("/api/secrets")
def api_secrets_post(req: SecretsUpdate) -> dict[str, Any]:
    """Persist secrets. Empty string deletes a key."""
    secrets.set_secrets(req.values)
    return {
        "backend": secrets.backend(),
        "providers": secrets.provider_status(),
    }


@app.post("/api/secrets/test/{provider}")
def api_secrets_test(provider: str) -> dict[str, Any]:
    ok, msg = conn_checks.check_provider(provider)
    return {"provider": provider, "ok": ok, "message": msg}


# --------------------------------------------------------------------------
# Updates (git pull + reinstall)
# --------------------------------------------------------------------------


@app.get("/api/updates/check")
def api_updates_check() -> dict[str, Any]:
    return updater.check_updates()


@app.get("/api/updates/current")
def api_updates_current() -> dict[str, Any]:
    return updater.current_commit()


@app.post("/api/updates/apply")
def api_updates_apply() -> dict[str, Any]:
    """Synchronously pull + reinstall. Returns the log + new HEAD.

    Note: the UI typically POSTs and waits up to ~5 min for pip. Page should
    show a spinner. After this returns ok, the user is expected to click the
    'Restart cockpit' button to load the new code.
    """
    return updater.apply_update()


@app.post("/api/updates/restart")
def api_updates_restart(background_tasks: BackgroundTasks) -> dict[str, Any]:
    """Schedule a cockpit restart shortly after this request returns."""
    background_tasks.add_task(_restart_self)
    return {
        "ok": True,
        "message": "Cockpit restarting. This window will reconnect automatically.",
    }


def _restart_self() -> None:
    """Exec the current Python interpreter with the same argv to reload code."""
    import time

    time.sleep(1.0)  # give the HTTP response time to flush
    try:
        os.execv(sys.executable, [sys.executable, *sys.argv])
    except Exception:
        log.exception("restart failed")
        os._exit(1)


# --------------------------------------------------------------------------
# Models (pretrain / retune)
# --------------------------------------------------------------------------


PRETRAIN_KIND = "pretrain"
RETUNE_NIGHTLY_KIND = "retune_nightly"
RETUNE_WEEKLY_KIND = "retune_weekly"
PAPER_LOOP_KIND = "paper_loop"

_MODEL_KINDS = (PRETRAIN_KIND, RETUNE_NIGHTLY_KIND, RETUNE_WEEKLY_KIND)


@app.get("/api/jobs")
def api_jobs() -> list[dict[str, Any]]:
    return job_mgr.all_status()


@app.get("/api/jobs/{kind}")
def api_job(kind: str) -> dict[str, Any]:
    return job_mgr.status(kind).to_dict()


@app.get("/api/jobs/{kind}/log")
def api_job_log(kind: str) -> dict[str, Any]:
    return {"kind": kind, "tail": job_mgr.tail_log(kind)}


@app.get("/api/jobs/{kind}/stream")
async def api_job_stream(kind: str) -> StreamingResponse:
    """Server-Sent Events stream of the job's log file as it grows."""
    return StreamingResponse(_log_stream(kind), media_type="text/event-stream")


async def _log_stream(kind: str):
    """Yield SSE chunks of the log as it appends. Stops 10s after process exits."""
    path = job_mgr.log_path(kind)
    # Wait briefly for the file to appear if the job just started.
    for _ in range(20):
        if path.exists():
            break
        await asyncio.sleep(0.1)
    if not path.exists():
        yield "data: (no log yet)\n\n"
        return

    # Stream existing content first, then tail forever.
    pos = 0
    idle_after_exit = 0.0
    while True:
        try:
            with path.open("r", encoding="utf-8", errors="replace") as f:
                f.seek(pos)
                chunk = f.read()
                pos = f.tell()
        except OSError:
            chunk = ""
        if chunk:
            for raw_line in chunk.splitlines():
                yield f"data: {raw_line}\n\n"
        info = job_mgr.status(kind)
        if not info.is_running():
            idle_after_exit += 0.5
            if idle_after_exit > 10.0:
                yield "event: end\ndata: done\n\n"
                return
        else:
            idle_after_exit = 0.0
        await asyncio.sleep(0.5)


@app.post("/api/models/pretrain")
def api_models_pretrain() -> dict[str, Any]:
    cmd = [_python_exe(), "-m", "packages.data.pretrain"]
    info = job_mgr.start(PRETRAIN_KIND, cmd)
    return info.to_dict()


@app.post("/api/models/retune-nightly")
def api_models_retune_nightly() -> dict[str, Any]:
    # Tolerate the module not yet existing; fall back to a stub command that
    # writes a friendly message so the UI still streams something.
    cmd = [_python_exe(), "-m", "packages.data.retune", "--cadence", "nightly"]
    info = job_mgr.start(RETUNE_NIGHTLY_KIND, cmd)
    return info.to_dict()


@app.post("/api/models/retune-weekly")
def api_models_retune_weekly() -> dict[str, Any]:
    cmd = [_python_exe(), "-m", "packages.data.retune", "--cadence", "weekly"]
    info = job_mgr.start(RETUNE_WEEKLY_KIND, cmd)
    return info.to_dict()


@app.post("/api/models/{kind}/stop")
def api_models_stop(kind: str) -> dict[str, Any]:
    if kind not in _MODEL_KINDS:
        raise HTTPException(status_code=400, detail=f"unknown job kind: {kind}")
    return job_mgr.stop(kind).to_dict()


# --------------------------------------------------------------------------
# Paper-trade loop control
# --------------------------------------------------------------------------


@app.post("/api/trading/start")
def api_trading_start(req: StartTradingRequest) -> dict[str, Any]:
    cmd = [_python_exe(), "tools/paper_trade.py", "--strategy", req.strategy, "--loop"]
    if req.dry_run:
        cmd.append("--dry-run")
    info = job_mgr.start(PAPER_LOOP_KIND, cmd)
    return info.to_dict()


@app.post("/api/trading/stop")
def api_trading_stop() -> dict[str, Any]:
    return job_mgr.stop(PAPER_LOOP_KIND).to_dict()


@app.get("/api/trading/status")
def api_trading_status() -> dict[str, Any]:
    return job_mgr.status(PAPER_LOOP_KIND).to_dict()


# --------------------------------------------------------------------------
# Trading mode (paper vs live)
# --------------------------------------------------------------------------


@app.get("/api/mode")
def api_mode_get() -> dict[str, Any]:
    state = load_state()
    return {
        "mode": state.trading_mode,
        "live_keys_present": bool(
            os.getenv("ALPACA_LIVE_KEY_ID") and os.getenv("ALPACA_LIVE_SECRET")
        ),
        "paper_keys_present": bool(
            os.getenv("ALPACA_PAPER_KEY_ID") and os.getenv("ALPACA_PAPER_SECRET")
        ),
    }


@app.post("/api/mode")
def api_mode_set(req: ModeRequest) -> dict[str, Any]:
    if req.mode not in VALID_MODES:
        raise HTTPException(
            status_code=400, detail=f"mode must be one of {VALID_MODES}; got {req.mode!r}"
        )
    if req.mode == "live":
        if not req.confirm_live:
            raise HTTPException(
                status_code=400,
                detail="Switching to live requires confirm_live=true. Real money will be at risk.",
            )
        if not (os.getenv("ALPACA_LIVE_KEY_ID") and os.getenv("ALPACA_LIVE_SECRET")):
            raise HTTPException(
                status_code=400,
                detail="Cannot switch to live: ALPACA_LIVE_KEY_ID / ALPACA_LIVE_SECRET are not configured. Add them on the Settings page first.",
            )
    state = load_state()
    state.trading_mode = req.mode  # type: ignore[assignment]
    # Switching mode pauses the bot defensively so the user can review.
    state.paused = True
    state = record_action(state, f"Trading mode -> {req.mode} (bot auto-paused)")
    save_state(state)
    return {"mode": state.trading_mode, "paused": state.paused}


# --------------------------------------------------------------------------
# Errors log
# --------------------------------------------------------------------------


@app.get("/api/errors")
def api_errors_list(limit: int = 200, severity: str | None = None) -> dict[str, Any]:
    return {
        "counts": err_log.count_unresolved(),
        "entries": err_log.list_errors(limit=limit, severity=severity),
    }


@app.get("/api/errors/markdown")
def api_errors_markdown(limit: int = 50) -> dict[str, Any]:
    return {"markdown": err_log.to_markdown(limit=limit)}


@app.post("/api/errors/clear")
def api_errors_clear() -> dict[str, Any]:
    n = err_log.clear()
    return {"cleared": n}


# --------------------------------------------------------------------------
# Error handlers
# --------------------------------------------------------------------------


@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
    log.exception("unhandled error on %s", request.url.path)
    with contextlib.suppress(Exception):
        err_log.record_exception(
            source="cockpit.api",
            exc=exc,
            path=str(request.url.path),
            method=request.method,
        )
    return JSONResponse(
        status_code=500,
        content={"error": str(exc) or exc.__class__.__name__},
    )


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="info")
