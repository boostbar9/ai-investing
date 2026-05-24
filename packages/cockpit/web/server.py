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
# Append-only log of every LangGraph agent advisory run. Capped at AGENT_LOG_MAX
# lines via simple tail-trim so it never grows unbounded.
AGENT_LOG = Path("data/agents_log.jsonl")
AGENT_LOG_MAX = 500
# packages/cockpit/web/server.py -> repo root is 3 levels up.
REPO_ROOT = Path(__file__).resolve().parents[3]
INDEX_HTML = Path(__file__).parent / "templates" / "index.html"


def _append_agent_log(payload: dict[str, Any]) -> None:
    """Persist one agent-graph run so the dashboard can show history.

    Writes a compact one-line JSON row and tail-trims the file to
    ``AGENT_LOG_MAX`` lines so it never grows without bound.
    """
    try:
        AGENT_LOG.parent.mkdir(parents=True, exist_ok=True)
        # Persist a compact row, but ALSO include the strategy signals so
        # outcome attribution (packages.agents.attribution) can join each
        # call to forward returns. Keep the row small — fills/audit are
        # already in data/paper_log; we don't duplicate them here.
        per_agent: dict[str, dict[str, Any]] = {}
        for name, agent in (payload.get("agents") or {}).items():
            cell: dict[str, Any] = {
                "status": agent.get("status"),
                "detail": agent.get("detail"),
            }
            if name == "strategy":
                cell["signals"] = [
                    {
                        "symbol": s.get("symbol"),
                        "side": s.get("side"),
                        "strength": s.get("strength"),
                    }
                    for s in (agent.get("signals") or [])
                ]
            per_agent[name] = cell
        row = {
            "ts": payload.get("ran_at"),
            "decision_id": payload.get("decision_id"),
            "halted": payload.get("halted"),
            "halt_reason": payload.get("halt_reason"),
            "used_llm": payload.get("used_llm"),
            "regime": payload.get("regime"),
            "symbols": payload.get("symbols"),
            "agents": per_agent,
        }
        with AGENT_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")
        # Tail-trim if needed.
        try:
            with AGENT_LOG.open(encoding="utf-8") as f:
                lines = f.readlines()
            if len(lines) > AGENT_LOG_MAX:
                AGENT_LOG.write_text("".join(lines[-AGENT_LOG_MAX:]), encoding="utf-8")
        except OSError:
            pass
    except OSError as e:
        log.warning("failed to persist agents log: %s", e)


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


@app.get("/agents", response_class=HTMLResponse)
def agents_page() -> HTMLResponse:
    return _render("agents.html")


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
# Agent graph (§5)
# --------------------------------------------------------------------------


class AgentRunRequest(BaseModel):
    """Trigger a single LangGraph advisory run."""

    symbols: list[str] | None = None
    regime: str | None = None
    use_llm: bool = False


# In-memory cache of the most recent agent graph run. Kept on the module so
# the dashboard /api/agents/last endpoint can show status lights without
# re-running the graph on every poll.
_LAST_AGENT_RUN: dict[str, Any] = {
    "ran_at": None,
    "decision_id": None,
    "halted": False,
    "halt_reason": None,
    "used_llm": False,
    "agents": {
        "research": {"status": "idle", "detail": ""},
        "strategy": {"status": "idle", "detail": ""},
        "risk": {"status": "idle", "detail": ""},
        "execution": {"status": "idle", "detail": ""},
        "discovery": {"status": "idle", "detail": ""},
    },
    "audit": [],
}


# Discovery audit log — the agent runs in advisory mode (NOT in the order
# path) so we persist its proposals separately for offline review and
# eventual promotion into the strategy playbook by hand.
DISCOVERY_LOG = Path("data/discoveries_log.jsonl")
DISCOVERY_LOG_MAX = 500

# Agent self-improvement scorecard: each row attributes one matured run to
# realized forward returns. Written by the nightly attribution job; read by
# the dashboard panel + the prompt self-reflection injection.
SCORECARD_LOG = Path("data/agent_scorecard.jsonl")
SCORECARD_PROMOTION_LOG = Path("data/promotion_candidates.jsonl")


def _append_discovery_log(payload: dict[str, Any]) -> None:
    """Append one discovery row (compact), tail-trim to ``DISCOVERY_LOG_MAX``."""
    if not payload.get("patterns"):
        return  # nothing interesting — skip the write
    try:
        DISCOVERY_LOG.parent.mkdir(parents=True, exist_ok=True)
        row = json.dumps(payload, default=str, separators=(",", ":"))
        with DISCOVERY_LOG.open("a", encoding="utf-8") as f:
            f.write(row + "\n")
        # Tail-trim opportunistically (cheap because rows are small).
        lines = DISCOVERY_LOG.read_text(encoding="utf-8").splitlines()
        if len(lines) > DISCOVERY_LOG_MAX:
            keep = lines[-DISCOVERY_LOG_MAX:]
            DISCOVERY_LOG.write_text("\n".join(keep) + "\n", encoding="utf-8")
    except OSError as e:
        log.warning("failed to append discovery log: %s", e)


def _last_run_payload() -> dict[str, Any]:
    """Return a deep-ish copy of the cached run so callers can't mutate it."""
    return json.loads(json.dumps(_LAST_AGENT_RUN, default=str))


@app.get("/api/agents/last")
def api_agents_last() -> dict[str, Any]:
    """Last advisory graph result (for the dashboard status lights)."""
    return _last_run_payload()


@app.post("/api/agents/run")
async def api_agents_run(req: AgentRunRequest) -> dict[str, Any]:
    """Run one advisory pass of the LangGraph agent graph.

    Defaults to the deterministic stub agents in ``paper_bridge`` so we don't
    require a running Ollama. ``use_llm=true`` swaps in the LLM-backed
    runners; if Ollama isn't reachable, those runners fall back to safe
    defaults (no signals, risk halts), which we surface to the UI.
    """
    from packages.agents import paper_bridge
    from packages.shared.schemas import Position

    symbols = req.symbols or ["SPY", "TLT", "QQQ"]
    # latest_regime() exposes keys auto/effective/override; we prefer the
    # effective (after manual override) and fall back to chop if everything
    # is missing so the chain still runs.
    reg_info = latest_regime()
    fallback_regime = (
        reg_info.get("effective") or reg_info.get("auto") or "chop"
    )
    regime = (req.regime or fallback_regime or "chop").lower()
    if regime not in ("bull", "bear", "chop", "crisis"):
        regime = "chop"

    # Equal-weight stub for the advisory run; real weights come from the
    # paper loop. The graph treats these as candidate signals only.
    target_weights = {s: 1.0 / len(symbols) for s in symbols}

    # Compute the recent self-assessment summary once. Used by every LLM
    # runner below (research, strategy, risk, execution, discovery) so all
    # agents share the same recent-performance frame of reference. Cheap to
    # compute even on every request — reads at most last_n_runs jsonl rows.
    scorecard_summary: dict[str, Any] | None = None
    if req.use_llm:
        try:
            from packages.agents.attribution import summarize_scorecard
            scorecard_summary = summarize_scorecard(SCORECARD_LOG, last_n_runs=20).to_jsonable()
        except Exception as e:
            log.warning("scorecard summary skipped: %s", e)
            scorecard_summary = None

    if req.use_llm:
        # LLM-backed runners (Ollama). Build a fresh graph so we don't have
        # to import inside the bridge module.
        from packages.agents.graph import AgentGraph
        from packages.agents.llm_router import LLMRouter
        from packages.agents.runners import (
            build_execution_runner,
            build_research_runner,
            build_risk_runner,
            build_strategy_runner,
        )

        async def _auto_approve(sigs, _did):  # type: ignore[no-untyped-def]
            return sigs

        router = LLMRouter()
        try:
            graph = AgentGraph(
                research=build_research_runner(router, scorecard_summary=scorecard_summary),
                strategy=build_strategy_runner(router, scorecard_summary=scorecard_summary),
                risk=build_risk_runner(router, scorecard_summary=scorecard_summary),
                execution=build_execution_runner(router, scorecard_summary=scorecard_summary),
                approval=_auto_approve,
            )
            result = await graph.run(
                symbols=symbols,
                regime=regime,
                positions=[Position(symbol=s, qty=0.0, avg_price=0.0) for s in symbols],
                features={},
            )
        finally:
            await router.aclose()
    else:
        result = await paper_bridge.advise(
            symbols=symbols,
            regime=regime,
            positions=[Position(symbol=s, qty=0.0, avg_price=0.0) for s in symbols],
            target_weights=target_weights,
            sentiment_scores=None,
        )

    # Build per-agent status. "ok" / "warn" / "halt" maps cleanly to the
    # cockpit pill classes (green / yellow / red).
    n_signals = len(result.strategy.signals)
    n_approved = len(result.risk.approved)
    n_rejected = len(result.risk.rejected)
    research_status = "ok"
    if abs(result.research.sentiment) > 0.5:
        research_status = "warn"
    strategy_status = "ok" if n_signals else "warn"
    if regime == "crisis":
        strategy_status = "halt"
    if result.risk.halted:
        risk_status = "halt"
    elif n_rejected and not n_approved:
        risk_status = "warn"
    else:
        risk_status = "ok"
    if result.execution is None:
        execution_status = "halt" if result.halted else "idle"
        n_fills = 0
    else:
        execution_status = "ok"
        n_fills = len(result.execution.fills)

    # Discovery agent (advisory only). Failure here MUST NOT halt the chain;
    # an exception is logged and the dashboard simply shows "idle".
    discovery_payload: dict[str, Any] = {"status": "idle", "detail": "not run", "patterns": [], "notes": ""}
    try:
        from packages.shared.universe import DEFAULT_UNIVERSE

        disc_universe = DEFAULT_UNIVERSE.filter(symbols) or list(symbols)
        disc_features = {
            "sentiment": float(result.research.sentiment),
            "regime_bull": 1.0 if regime == "bull" else 0.0,
            "regime_bear": 1.0 if regime == "bear" else 0.0,
            "regime_chop": 1.0 if regime == "chop" else 0.0,
            "regime_crisis": 1.0 if regime == "crisis" else 0.0,
            "n_signals": float(n_signals),
            "n_approved": float(n_approved),
        }
        if req.use_llm:
            from packages.agents.llm_router import LLMRouter
            from packages.agents.runners import build_discovery_runner
            from packages.shared.schemas import DiscoveryInput

            d_router = LLMRouter()
            try:
                d_runner = build_discovery_runner(d_router, scorecard_summary=scorecard_summary)
                d_in = DiscoveryInput(
                    decision_id=result.decision_id,
                    regime=regime,  # type: ignore[arg-type]
                    universe=disc_universe,
                    features=disc_features,
                    recent_thesis=result.research.thesis,
                )
                d_out = await d_runner(d_in)
            finally:
                await d_router.aclose()
        else:
            # Deterministic stub: derive at most one pattern from the existing
            # research/strategy outputs so the cockpit shows something useful
            # without an Ollama dependency.
            from packages.shared.schemas import DiscoveryOutput, PatternCandidate

            patterns: list[PatternCandidate] = []
            if regime != "crisis" and disc_universe and abs(result.research.sentiment) >= 0.3:
                side = "momentum-follow" if result.research.sentiment > 0 else "defensive-rotation"
                patterns.append(
                    PatternCandidate(
                        name=f"{regime}-{side}",
                        hypothesis=(
                            f"In {regime} regime with sentiment {result.research.sentiment:+.2f}, "
                            f"lean toward {('long-momentum' if result.research.sentiment > 0 else 'defensives')} "
                            f"across {', '.join(disc_universe[:3])}."
                        ),
                        symbols=disc_universe[:3],
                        feature_keys=["sentiment", f"regime_{regime}"],
                        confidence=min(0.6, abs(result.research.sentiment)),
                        horizon_days=5,
                    )
                )
            d_out = DiscoveryOutput(
                decision_id=result.decision_id,
                patterns=patterns,
                notes="deterministic stub: derived from research sentiment + regime",
            )
        disc_patterns = [p.model_dump() for p in d_out.patterns]
        discovery_payload = {
            "status": "ok" if disc_patterns else "idle",
            "detail": (
                f"{len(disc_patterns)} pattern(s) proposed (advisory only)"
                if disc_patterns
                else "no novel patterns"
            ),
            "advisory_only": True,
            "patterns": disc_patterns,
            "notes": d_out.notes,
        }
        if disc_patterns:
            _append_discovery_log(
                {
                    "ts": datetime.now(UTC).isoformat(timespec="seconds"),
                    "decision_id": str(result.decision_id),
                    "regime": regime,
                    "used_llm": bool(req.use_llm),
                    "patterns": disc_patterns,
                    "notes": d_out.notes,
                }
            )
    except Exception as e:
        log.warning("discovery agent failed (advisory only, ignored): %s", e)
        discovery_payload = {
            "status": "warn",
            "detail": f"discovery error (ignored): {type(e).__name__}",
            "advisory_only": True,
            "patterns": [],
            "notes": "",
        }

    payload: dict[str, Any] = {
        "ran_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "decision_id": str(result.decision_id),
        "halted": bool(result.halted),
        "halt_reason": result.risk.halt_reason,
        "used_llm": bool(req.use_llm),
        "regime": regime,
        "symbols": symbols,
        "agents": {
            "research": {
                "status": research_status,
                "detail": f"sentiment={result.research.sentiment:+.2f}",
                "thesis": result.research.thesis,
                "citations": list(result.research.citations),
            },
            "strategy": {
                "status": strategy_status,
                "detail": f"{n_signals} signal(s)",
                "signals": [s.model_dump() for s in result.strategy.signals],
            },
            "risk": {
                "status": risk_status,
                "detail": (
                    result.risk.halt_reason
                    or f"{n_approved} approved, {n_rejected} rejected"
                ),
                "approved": [s.model_dump() for s in result.risk.approved],
                "rejected": [s.model_dump() for s in result.risk.rejected],
            },
            "execution": {
                "status": execution_status,
                "detail": (
                    "no orders (risk halt)" if result.execution is None
                    else f"{n_fills} fill(s)"
                ),
                "fills": (
                    [] if result.execution is None
                    else [f.model_dump(mode="json") for f in result.execution.fills]
                ),
            },
            "discovery": discovery_payload,
        },
        "audit": [
            {
                "actor": ev.actor,
                "event_type": ev.event_type,
                "payload": ev.payload,
            }
            for ev in result.audit
        ],
    }
    _LAST_AGENT_RUN.clear()
    _LAST_AGENT_RUN.update(payload)
    _append_agent_log(payload)
    return payload


# --------------------------------------------------------------------------
# Agent auto-scheduler
# --------------------------------------------------------------------------
#
# Single in-process asyncio task that periodically POSTs to the same
# pipeline used by the manual Run button. Defaults are:
#   * disabled until the operator turns it on
#   * 30 minute interval (well above the 15-min paper loop cadence)
#   * stub backend so it never depends on Ollama
#
# The task is fully self-contained: it skips runs while the cockpit is
# paused, retries on transport failure with an exponential backoff capped
# at the interval, and emits the same _LAST_AGENT_RUN / agent log a manual
# run would. Config is in-memory (resets on restart) by design — the user
# explicitly opts in each session.

_AGENT_SCHED: dict[str, Any] = {
    "enabled": False,
    "interval_seconds": 1800,  # 30 minutes
    "use_llm": False,
    "symbols": ["SPY", "QQQ", "TLT"],
    "last_run_at": None,
    "last_run_status": None,
    "last_error": None,
    "_task": None,
}


class AgentScheduleConfig(BaseModel):
    """Toggle and tune the auto-scheduler."""

    enabled: bool | None = None
    interval_seconds: int | None = None
    use_llm: bool | None = None
    symbols: list[str] | None = None


async def _scheduler_tick_once() -> dict[str, Any]:
    """Execute one auto-scheduled pipeline pass.

    Skips when the cockpit is paused so the user's pause button universally
    halts background work. Returns the same payload shape as /api/agents/run.
    """
    cstate = load_state()
    if cstate.paused:
        _AGENT_SCHED["last_run_at"] = datetime.now(UTC).isoformat(timespec="seconds")
        _AGENT_SCHED["last_run_status"] = "skipped_paused"
        return {"skipped": True, "reason": "cockpit paused"}
    req = AgentRunRequest(
        symbols=list(_AGENT_SCHED["symbols"] or []),
        regime=None,
        use_llm=bool(_AGENT_SCHED["use_llm"]),
    )
    result = await api_agents_run(req)
    _AGENT_SCHED["last_run_at"] = result.get("ran_at")
    _AGENT_SCHED["last_run_status"] = "halted" if result.get("halted") else "ok"
    return result


async def _scheduler_loop() -> None:  # pragma: no cover — long-lived task
    """Background loop — sleeps `interval_seconds` between ticks."""
    backoff = 5
    while _AGENT_SCHED.get("enabled"):
        try:
            await _scheduler_tick_once()
            backoff = 5
        except asyncio.CancelledError:
            raise
        except Exception as e:
            _AGENT_SCHED["last_error"] = str(e)[:300]
            log.warning("agent scheduler tick failed: %s", e)
            backoff = min(backoff * 2, int(_AGENT_SCHED["interval_seconds"]) or 60)
            try:
                await asyncio.sleep(backoff)
            except asyncio.CancelledError:
                raise
            continue
        try:
            await asyncio.sleep(int(_AGENT_SCHED["interval_seconds"]))
        except asyncio.CancelledError:
            raise


def _start_scheduler_task() -> None:
    """Spawn the background loop if not already running."""
    task = _AGENT_SCHED.get("_task")
    if task is not None and not task.done():
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return  # no loop yet — will be started by FastAPI startup hook
    _AGENT_SCHED["_task"] = loop.create_task(_scheduler_loop())


def _stop_scheduler_task() -> None:
    task = _AGENT_SCHED.get("_task")
    if task is not None and not task.done():
        task.cancel()
    _AGENT_SCHED["_task"] = None


@app.on_event("startup")
async def _agent_scheduler_startup() -> None:  # pragma: no cover
    if _AGENT_SCHED.get("enabled"):
        _start_scheduler_task()


@app.on_event("shutdown")
async def _agent_scheduler_shutdown() -> None:  # pragma: no cover
    _stop_scheduler_task()


@app.get("/api/agents/schedule")
def api_agents_schedule_get() -> dict[str, Any]:
    """Current auto-scheduler config + status."""
    task = _AGENT_SCHED.get("_task")
    running = bool(task is not None and not task.done())
    return {
        "enabled": bool(_AGENT_SCHED["enabled"]),
        "running": running,
        "interval_seconds": int(_AGENT_SCHED["interval_seconds"]),
        "use_llm": bool(_AGENT_SCHED["use_llm"]),
        "symbols": list(_AGENT_SCHED["symbols"] or []),
        "last_run_at": _AGENT_SCHED["last_run_at"],
        "last_run_status": _AGENT_SCHED["last_run_status"],
        "last_error": _AGENT_SCHED["last_error"],
    }


@app.post("/api/agents/schedule")
async def api_agents_schedule_set(cfg: AgentScheduleConfig) -> dict[str, Any]:
    """Update the auto-scheduler config. Starts/stops the loop as needed."""
    if cfg.interval_seconds is not None:
        if cfg.interval_seconds < 60:
            raise HTTPException(
                status_code=400,
                detail="interval_seconds must be >= 60",
            )
        _AGENT_SCHED["interval_seconds"] = int(cfg.interval_seconds)
    if cfg.use_llm is not None:
        _AGENT_SCHED["use_llm"] = bool(cfg.use_llm)
    if cfg.symbols is not None:
        cleaned = [s.strip().upper() for s in cfg.symbols if s and s.strip()]
        if cleaned:
            _AGENT_SCHED["symbols"] = cleaned
    if cfg.enabled is not None:
        _AGENT_SCHED["enabled"] = bool(cfg.enabled)
        if _AGENT_SCHED["enabled"]:
            _start_scheduler_task()
        else:
            _stop_scheduler_task()
    return api_agents_schedule_get()


@app.post("/api/agents/schedule/tick")
async def api_agents_schedule_tick() -> dict[str, Any]:
    """Manually execute one scheduler tick (for tests and ad-hoc triggers)."""
    return await _scheduler_tick_once()


@app.get("/api/agents/history")
def api_agents_history(limit: int = 50) -> dict[str, Any]:
    """Return up to ``limit`` most-recent persisted agent runs (newest first)."""
    rows: list[dict[str, Any]] = []
    try:
        if AGENT_LOG.exists():
            with AGENT_LOG.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except OSError as e:
        log.warning("failed to read agents log: %s", e)
    rows.reverse()
    if limit > 0:
        rows = rows[:limit]
    return {"runs": rows, "total": len(rows)}


@app.get("/api/agents/discoveries")
def api_agents_discoveries(limit: int = 50) -> dict[str, Any]:
    """Most recent patterns proposed by the advisory Discovery agent."""
    rows: list[dict[str, Any]] = []
    try:
        if DISCOVERY_LOG.exists():
            with DISCOVERY_LOG.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except OSError as e:
        log.warning("failed to read discoveries log: %s", e)
    rows.reverse()
    if limit > 0:
        rows = rows[:limit]
    return {"discoveries": rows, "total": len(rows), "advisory_only": True}


# --------------------------------------------------------------------------
# Self-improvement: outcome attribution + promotion candidates
# --------------------------------------------------------------------------


@app.get("/api/agents/scorecard")
def api_agents_scorecard(limit: int = 50) -> dict[str, Any]:
    """Return the most recent scorecard rows + a rolled-up summary.

    The scorecard is written by the nightly attribution job. It joins each
    matured agent run to the realized close prices N days later so the
    operator can see hit-rate and avg PnL per recent run.
    """
    from packages.agents.attribution import summarize_scorecard

    rows: list[dict[str, Any]] = []
    try:
        if SCORECARD_LOG.exists():
            with SCORECARD_LOG.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except OSError as e:
        log.warning("failed to read scorecard log: %s", e)
    summary = summarize_scorecard(SCORECARD_LOG, last_n_runs=max(limit, 20)).to_jsonable()
    rows.reverse()
    if limit > 0:
        rows = rows[:limit]
    return {"runs": rows, "total": len(rows), "summary": summary}


@app.post("/api/agents/attribute")
def api_agents_attribute(force: bool = False) -> dict[str, Any]:
    """Trigger attribution: walk agents_log.jsonl and append matured runs
    to ``agent_scorecard.jsonl``. Returns the count of new rows written.

    ``force=true`` ignores the look-ahead safety horizon and re-attributes
    using whatever prices the data adapter can produce — mainly for tests
    and operator debugging. Default behavior only attributes runs whose
    shortest horizon has fully matured.
    """
    from packages.agents.attribution import (
        DEFAULT_HORIZONS_DAYS,
        run_attribution,
    )

    # Build a price fetcher. Try Alpaca first; if not configured, return a
    # callable that always returns None so attribution is a no-op (the
    # scorecard simply won't grow until the operator wires credentials).
    def _no_price(_symbol: str, _ts: Any) -> float | None:
        return None

    get_close = _no_price
    try:
        from packages.data.adapters.alpaca_data import AlpacaDataAdapter

        adapter = AlpacaDataAdapter()
        if adapter.is_configured():
            import asyncio
            from datetime import timedelta

            # Sync wrapper around the async bar fetcher — attribution is a
            # batch job, not an inner-loop call.
            def _close_via_alpaca(symbol: str, ts: Any) -> float | None:
                start = (ts - timedelta(days=1)).isoformat()
                end = (ts + timedelta(days=1)).isoformat()
                try:
                    bars = asyncio.run(adapter.get_bars(symbol, start, end))
                except Exception:
                    return None
                if not bars:
                    return None
                # Pick the first bar at or after ts.
                for b in bars:
                    if b.ts >= ts:
                        return float(b.close)
                return float(bars[-1].close)

            get_close = _close_via_alpaca
    except ImportError:
        pass

    n = run_attribution(
        AGENT_LOG,
        SCORECARD_LOG,
        get_close,
        horizons_days=DEFAULT_HORIZONS_DAYS,
        now=None if not force else datetime.now(UTC).replace(year=datetime.now(UTC).year + 10),
    )
    return {"appended": n, "scorecard_path": str(SCORECARD_LOG)}


@app.get("/api/agents/promotion_candidates")
def api_agents_promotion_candidates(limit: int = 50) -> dict[str, Any]:
    """Discovery patterns that have passed the §16 backtest gate — awaiting
    human approval before being promoted into the Strategy playbook.
    """
    rows: list[dict[str, Any]] = []
    try:
        if SCORECARD_PROMOTION_LOG.exists():
            with SCORECARD_PROMOTION_LOG.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except OSError as e:
        log.warning("failed to read promotion candidates log: %s", e)
    rows.reverse()
    if limit > 0:
        rows = rows[:limit]
    return {"candidates": rows, "total": len(rows)}


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
