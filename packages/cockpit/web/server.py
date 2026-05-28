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
POST /api/trading/liquidate -> bulk close all positions via Alpaca DELETE /v2/positions
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
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    Response,
    StreamingResponse,
)
from pydantic import BaseModel

from packages.cockpit import automation, diagnostics, paper_autopilot, updater, watchdog
from packages.cockpit import errors as err_log
from packages.cockpit import proc as job_mgr
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


# --------------------------------------------------------------------------
# Quiet the uvicorn access log
# --------------------------------------------------------------------------
# The cockpit UI polls a handful of status endpoints every 3-5 seconds
# (``/api/state``, ``/api/jobs``, ``/api/errors``, ``/api/health``,
# ``/api/ollama/status``, ``/api/agents/...``). Without this filter, each
# poll fires an INFO access-log line on the terminal, which buries real
# events like POSTs, exit codes, and the once-per-startup OTel warning
# under a wall of identical 200 OKs.
#
# This filter drops the access record only when *all* of the following are
# true: GET method, 200/304 status, and the URL is one of the high-frequency
# polling paths or any SSE ``/stream`` endpoint. Everything else (POSTs,
# page loads, non-2xx, downloads, one-off API hits) still logs normally.
#
# Note: uvicorn emits access logs by writing a single argument tuple
# ``(client_addr, method, path, http_version, status)`` to the
# ``uvicorn.access`` logger. We sniff that tuple shape and pattern-match
# the path; if uvicorn ever changes its access format the filter falls
# through to ``True`` so we never accidentally silence real errors.
_QUIET_PATH_PREFIXES: tuple[str, ...] = (
    "/api/state",
    "/api/jobs",  # also covers /api/jobs/<kind>/stream
    "/api/errors",
    "/api/health",
    "/api/mode",
    "/api/ollama/status",
    "/api/ollama/warmup_status",
    "/api/ollama/gpu_fix",  # GET status; the POST is one-shot and rare
    "/api/ollama/pin",  # POST is rare; included so SSE log polling stays quiet
    "/api/agents/last",
    "/api/agents/in_flight",
    "/api/agents/schedule",
    "/api/agents/history",
    "/api/agents/scorecard",
    "/api/agents/promotion_candidates",
    "/api/autopilot",
    "/api/trading/status",
    "/api/watchdog",
    "/api/equity-summary",
    "/api/promote",
    "/static/",
    "/favicon.ico",
)


class _UvicornAccessQuietFilter(logging.Filter):
    """Suppress 2xx/304 GET access log lines for high-frequency poll paths."""

    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        if not isinstance(args, tuple) or len(args) < 5:
            return True
        method = args[1]
        path = args[2]
        status = args[4]
        if not isinstance(method, str) or not isinstance(path, str):
            return True
        try:
            status_int = int(status)
        except (TypeError, ValueError):
            return True
        if method != "GET":
            return True
        if status_int not in (200, 304):
            return True
        # Strip the query string before prefix-matching.
        bare = path.split("?", 1)[0]
        return not bare.startswith(_QUIET_PATH_PREFIXES)


def _install_uvicorn_access_filter() -> None:
    """Attach the quiet filter to ``uvicorn.access``. Idempotent.

    Safe to call multiple times: we tag the filter instance with a marker
    attribute and skip if an instance with that marker is already attached.
    """
    access_log = logging.getLogger("uvicorn.access")
    for existing in access_log.filters:
        if getattr(existing, "_cockpit_quiet", False):
            return
    flt = _UvicornAccessQuietFilter()
    flt._cockpit_quiet = True  # type: ignore[attr-defined]
    access_log.addFilter(flt)


_install_uvicorn_access_filter()

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


class LiquidateRequest(BaseModel):
    """Body for /api/trading/liquidate.

    Must include ``confirm: true`` -- guards against an accidental click that
    blows away every paper position. The cockpit UI sends this only after a
    typed-confirmation dialog.
    """

    confirm: bool = False
    cancel_orders: bool = True


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

    Fast path: ``data/cockpit/snapshot.json`` is rewritten atomically after
    every cycle, so the dashboard has equity/streak the moment the cockpit
    boots even before any new run fires (§17, task 5/8). Falls back to the
    JSONL run log if the snapshot file is missing or stale.
    """
    # Fast path: the cycle snapshot.
    try:
        from packages.persistence import load_snapshot as _load_snapshot
        snap = _load_snapshot()
    except Exception:  # pragma: no cover - import guard
        snap = None
    if snap:
        return {
            "equity": snap.get("equity"),
            "cash": snap.get("cash"),
            "buying_power": snap.get("buying_power"),
            "day_pnl": snap.get("day_pnl"),
            "as_of": snap.get("ts"),
            "strategy": snap.get("strategy"),
            "halted": snap.get("halted"),
        }
    # Fallback: scan the JSONL log (slower, but works pre-snapshot).
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


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    """Serve the legacy /favicon.ico path.

    Modern browsers pick up the icons declared in the template <head> and
    never hit this route; older browsers (and Windows pinned tabs) still
    request /favicon.ico directly. The real multi-size ICO lives under
    ``static/brand/favicon.ico`` -- this route just bridges the well-known
    path to it. Falls back to a legacy top-level file or a 204 so the
    server log stays clean either way.
    """
    for candidate in (_STATIC_DIR / "brand" / "favicon.ico", _STATIC_DIR / "favicon.ico"):
        if candidate.exists():
            return Response(
                content=candidate.read_bytes(),
                media_type="image/x-icon",
            )
    return Response(status_code=204)


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


@app.get("/health", response_class=HTMLResponse)
def health_page() -> HTMLResponse:
    return _render("health.html")


@app.get("/welcome", response_class=HTMLResponse)
def welcome_page() -> HTMLResponse:
    """First-boot wizard. Served on demand -- the dashboard banner deep-
    links here when onboarding is incomplete. The actual state lives in
    ``data/cockpit/onboarding.json`` and is read/written by the
    ``/api/onboarding/*`` endpoints below."""
    return _render("welcome.html")


_TEMPLATES_DIR = Path(__file__).parent / "templates"
_INCLUDE_PATTERN = re.compile(r"\{%\s*include\s+['\"]([^'\"]+)['\"]\s*%\}")
# Strip Jinja-style comments {# ... #}. DOTALL so multi-line blocks vanish too.
_COMMENT_PATTERN = re.compile(r"\{#.*?#\}", re.DOTALL)


def _render(name: str) -> HTMLResponse:
    """Read a template file and expand any ``{% include 'foo.html' %}`` tags.

    A full Jinja2 dependency would be overkill for the handful of partials
    the cockpit needs (currently just ``_head_icons.html``), so we do a
    one-pass include expansion ourselves, then strip Jinja-style comments
    ``{# ... #}`` so partials can document themselves without that
    documentation leaking out as visible text in the browser. Includes
    resolve relative to the templates directory; missing partials degrade
    gracefully with an HTML comment so a typo doesn't break the whole page.
    """
    path = _TEMPLATES_DIR / name
    if not path.exists():
        return HTMLResponse(f"<h1>template missing: {name}</h1>", status_code=500)
    body = _expand_includes(path.read_text(encoding="utf-8"))
    body = _COMMENT_PATTERN.sub("", body)
    return HTMLResponse(body)


def _expand_includes(body: str, _depth: int = 0) -> str:
    if _depth > 4:  # paranoid recursion guard
        return body

    def repl(m: re.Match[str]) -> str:
        included = _TEMPLATES_DIR / m.group(1)
        if not included.exists():
            return f"<!-- include not found: {m.group(1)} -->"
        return _expand_includes(included.read_text(encoding="utf-8"), _depth + 1)

    return _INCLUDE_PATTERN.sub(repl, body)


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


@app.get("/api/health/full")
def api_health_full() -> dict[str, Any]:
    """Comprehensive diagnostics for the /health UI page.

    Runs every check in :mod:`packages.cockpit.diagnostics` and returns a
    rollup with per-check status, remediation commands, and which checks
    are auto-fixable from the Health page.
    """
    return diagnostics.summary()


@app.post("/api/health/fix/{name}")
def api_health_fix(name: str) -> dict[str, Any]:
    """Run the auto-heal action for a single diagnostic check.

    The Health page exposes a "Fix it" button for any check whose
    ``auto_fixable`` flag is true. This endpoint dispatches that action
    and returns a plain ``{ok, message}`` payload for the toast UI.
    """
    return diagnostics.auto_heal(name)


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
def api_job_log(kind: str, download: int = 0) -> Any:
    """Return the tail of the job log as JSON, or the full file as a download.

    The cockpit UI uses JSON for the on-page snapshot but offers a Download
    button that hits this same endpoint with ``?download=1`` to grab the
    complete file (up to whatever the on-disk rotation has kept).
    """
    if download:
        path = job_mgr.log_path(kind)
        if not path.exists():
            return PlainTextResponse("(no log yet)", media_type="text/plain")
        return FileResponse(
            path,
            media_type="text/plain",
            filename=f"{kind}.log",
        )
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
# Ollama setup (GUI wrapper around tools/check_ollama.py --auto)
# --------------------------------------------------------------------------

OLLAMA_SETUP_KIND = "ollama_setup"

# Track warmup results across the session so the UI can show "models ready"
# vs. "warming". Updated by both the manual /api/ollama/warmup endpoint and
# the startup background task.
_WARMUP_STATE: dict[str, Any] = {
    "started_at": None,
    "finished_at": None,
    "in_progress": False,
    "results": [],  # list of {model, elapsed_s, ok, error?}
}

# Module-level reference for the startup pre-warm task so it isn't GC'd
# mid-run (ruff RUF006).
_WARMUP_BG_TASK: asyncio.Task | None = None


async def _warmup_models() -> dict[str, Any]:
    """Issue a one-token generate at every installed declared model.

    Causes Ollama to mmap the weights and (on GPU builds) move them into
    VRAM. After a successful warmup the first agent Run typically drops
    from 30-90s per agent to 5-15s per agent.

    Returns a result dict; also caches it in ``_WARMUP_STATE`` so the
    Ollama panel can read it without re-running the warmup.
    """
    import httpx as _httpx

    from tools.check_ollama import ensure_daemon, pull_model, status_snapshot

    _WARMUP_STATE["started_at"] = time.time()
    _WARMUP_STATE["finished_at"] = None
    _WARMUP_STATE["in_progress"] = True
    _WARMUP_STATE["results"] = []

    # Auto-start the Ollama daemon if it's not already running. Honors
    # COCKPIT_OLLAMA_AUTO_START (default on). Set to 0 to keep startup
    # idempotent on machines where the daemon is managed by systemd.
    snap = status_snapshot()
    host = snap.get("host") or "http://127.0.0.1:11434"
    if not snap.get("daemon_alive") and os.environ.get(
        "COCKPIT_OLLAMA_AUTO_START", "1"
    ) in ("1", "true", "True"):
        log.info("Ollama daemon not running; attempting auto-start")
        try:
            started = await asyncio.get_event_loop().run_in_executor(
                None, lambda: ensure_daemon(host, verbose=False)
            )
            if started:
                log.info("Ollama daemon auto-started")
                snap = status_snapshot()
            else:
                log.warning("Ollama auto-start failed; warmup will be skipped")
        except Exception as e:
            log.warning("Ollama auto-start error: %s", e)

    if not snap.get("daemon_alive"):
        _WARMUP_STATE["in_progress"] = False
        _WARMUP_STATE["finished_at"] = time.time()
        return {
            "ok": False,
            "error": "Ollama daemon not running (auto-start failed; set COCKPIT_OLLAMA_AUTO_START=0 to suppress this attempt)",
            "results": [],
        }

    # Auto-pull missing models if COCKPIT_OLLAMA_AUTO_PULL is on (default on).
    # We pull in the background so warmup can proceed with whatever's already
    # installed. Pull progress is visible in the server log.
    missing = list(snap.get("missing") or [])
    if missing and os.environ.get("COCKPIT_OLLAMA_AUTO_PULL", "1") in (
        "1",
        "true",
        "True",
    ):
        _WARMUP_STATE["pulling"] = list(missing)
        log.info("Ollama auto-pull: %d missing model(s): %s", len(missing), missing)

        async def _bg_pull() -> None:
            for model in missing:
                try:
                    ok = await asyncio.get_event_loop().run_in_executor(
                        None, lambda m=model: pull_model(host, m, verbose=False)
                    )
                    log.info(
                        "Ollama auto-pull %s -> %s", model, "ok" if ok else "failed"
                    )
                except Exception as e:
                    log.warning("Ollama auto-pull %s error: %s", model, e)
            _WARMUP_STATE["pulling"] = []

        # Fire-and-forget; the cockpit shouldn't block on a multi-GB download.
        # Hold a reference so the task isn't GC'd mid-flight (RUF006).
        _WARMUP_STATE["_pull_task"] = asyncio.create_task(_bg_pull())

    targets = list(snap.get("installed") or [])
    results: list[dict[str, Any]] = []
    async with _httpx.AsyncClient(timeout=120.0) as client:
        for model in targets:
            t0 = time.monotonic()
            try:
                r = await client.post(
                    f"{host}/api/generate",
                    json={
                        "model": model,
                        "prompt": "ok",
                        "stream": False,
                        "options": {"num_predict": 1, "temperature": 0.0},
                    },
                )
                r.raise_for_status()
                results.append({
                    "model": model,
                    "elapsed_s": round(time.monotonic() - t0, 2),
                    "ok": True,
                })
            except Exception as e:
                results.append({
                    "model": model,
                    "elapsed_s": round(time.monotonic() - t0, 2),
                    "ok": False,
                    "error": str(e)[:200],
                })
            # Stream partial results so the UI can show progress mid-run.
            _WARMUP_STATE["results"] = list(results)

    _WARMUP_STATE["in_progress"] = False
    _WARMUP_STATE["finished_at"] = time.time()
    return {
        "ok": all(r["ok"] for r in results),
        "results": results,
        "total_elapsed_s": round(
            (_WARMUP_STATE["finished_at"] - _WARMUP_STATE["started_at"]), 2
        ),
    }


@app.get("/api/ollama/warmup_status")
def api_ollama_warmup_status() -> dict[str, Any]:
    """Poll-friendly snapshot of the current/last warmup. Used by the
    Ollama panel to show 'models ready' once startup pre-warm finishes."""
    state = dict(_WARMUP_STATE)
    state["results"] = list(_WARMUP_STATE["results"])
    return state


# ---------------------------------------------------------------------------
# /api/boot — one-click boot summary
#
# Reads the most-recent run-boot summary written by tools/boot.py. The
# cockpit's first-boot wizard polls this to show which step is still
# running and surface 'fix this' calls-to-action when something is
# degraded. If the boot summary file is missing (older install, or first
# launch before boot.py ran), we trigger a synchronous boot of the
# read-only steps so the wizard always has something to show.
# ---------------------------------------------------------------------------

_BOOT_SUMMARY_PATH = Path("data/cockpit/boot.json")


@app.get("/api/boot")
def api_boot_summary(refresh: bool = False) -> dict[str, Any]:
    """Return the latest boot summary.

    Reads ``data/cockpit/boot.json`` (written by ``tools.boot.run_boot``)
    so a cockpit restart inherits the launcher's snapshot without
    re-running anything. When ``refresh=true``, re-runs the read-only
    steps (env, venv, data_dirs, cockpit_port) inline so the wizard can
    show fresh state after the user updates their .env, kills a port hog,
    etc. We never re-run network-touching steps from this endpoint —
    that's what the explicit Ollama setup button is for.
    """
    if refresh:
        from tools.boot import run_boot

        summary = run_boot(only={"env", "venv", "data_dirs", "cockpit_port"})
        return summary.to_dict()
    try:
        return json.loads(_BOOT_SUMMARY_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        from tools.boot import run_boot

        # First call before the launcher ran: synthesize a minimal snapshot.
        summary = run_boot(only={"env", "venv", "data_dirs", "cockpit_port"})
        return summary.to_dict()


# ---------------------------------------------------------------------------
# /api/onboarding/* -- first-boot wizard (Phase 1C)
#
# Backed by ``packages/cockpit/onboarding.py`` (state) and
# ``packages/cockpit/robinhood_access.py`` (RH waitlist probe). The
# wizard HTML lives at ``/welcome``; these endpoints are how it reads
# and persists user choices.
#
# Design notes:
#   * GET /api/onboarding returns the full state every time. Cheap, and
#     the wizard doesn't render the dashboard so we can afford it.
#   * POST /api/onboarding/check-robinhood is a deliberately tiny probe
#     (4s timeout, fail-open). It updates the persisted status as a side
#     effect so a polling dashboard banner can pick it up.
#   * POST /api/onboarding/complete is the single 'done' commit -- it
#     stamps timestamps, flips ``completed=True``, and is idempotent.
#   * POST /api/onboarding/reset wipes the state file so the wizard re-
#     runs. Exposed for the settings page "Re-run welcome" button.
# ---------------------------------------------------------------------------


class _OnboardingCompleteBody(BaseModel):
    """Payload accepted by ``/api/onboarding/complete``. All fields are
    optional except ``accept_disclaimer`` -- if False, the endpoint 400s
    so we never persist a 'completed' record without acknowledgment."""

    display_name: str = ""
    robinhood_status: str = "unknown"
    live_float_cap_usd: float = 300.0
    accept_disclaimer: bool = False


@app.get("/api/onboarding")
def api_onboarding_get() -> dict[str, Any]:
    """Read the persisted onboarding state. Returns sane defaults when
    the file is missing or corrupt (the wizard treats both the same)."""
    from packages.cockpit.onboarding import load_onboarding

    return load_onboarding().to_dict()


@app.post("/api/onboarding/check-robinhood")
def api_onboarding_check_robinhood() -> dict[str, Any]:
    """Run the Robinhood reachability probe and persist the outcome.

    Never raises -- if the probe blows up we record ``unknown`` so the
    UI can show a neutral state instead of a stack trace. The probe is
    bounded by ``RH_PROBE_TIMEOUT_S`` (4s) so this endpoint can't hang
    the wizard.
    """
    from packages.cockpit.onboarding import (
        load_onboarding,
        save_onboarding,
    )
    from packages.cockpit.robinhood_access import detect_access

    current = load_onboarding()
    result = detect_access(declined_already=current.robinhood_status == "declined")

    # 'offline' is not a persisted status (the onboarding schema doesn't
    # know about it). Leave the stored status untouched in that case --
    # we don't want a flaky wifi blip to overwrite a previously-confirmed
    # 'granted' value.
    if result.outcome in {"granted", "waitlist", "declined"}:
        current.robinhood_status = result.outcome  # type: ignore[assignment]
        save_onboarding(current)

    return {
        "outcome": result.outcome,
        "detail": result.detail,
        "http_status": result.http_status,
        "persisted": result.outcome in {"granted", "waitlist", "declined"},
    }


@app.post("/api/onboarding/complete")
def api_onboarding_complete(body: _OnboardingCompleteBody) -> dict[str, Any]:
    """Stamp completion and persist the final wizard state.

    The disclaimer is the gate: if ``accept_disclaimer`` is False we
    refuse to mark completion. This makes it impossible to bypass the
    acknowledgement by hand-crafting a POST.
    """
    from packages.cockpit.onboarding import (
        VALID_RH_STATUS,
        accept_disclaimer,
        load_onboarding,
        mark_completed,
        mark_started,
        save_onboarding,
    )

    if not body.accept_disclaimer:
        raise HTTPException(
            status_code=400, detail="disclaimer must be accepted"
        )
    if body.robinhood_status not in VALID_RH_STATUS:
        raise HTTPException(
            status_code=400,
            detail=f"invalid robinhood_status: {body.robinhood_status}",
        )
    if body.live_float_cap_usd < 0:
        raise HTTPException(
            status_code=400, detail="live_float_cap_usd must be >= 0"
        )

    state = load_onboarding()
    mark_started(state)  # idempotent
    state.display_name = body.display_name.strip()
    state.robinhood_status = body.robinhood_status  # type: ignore[assignment]
    state.live_float_cap_usd = float(body.live_float_cap_usd)
    accept_disclaimer(state)
    mark_completed(state)
    save_onboarding(state)
    return state.to_dict()


@app.post("/api/onboarding/reset")
def api_onboarding_reset() -> dict[str, Any]:
    """Wipe onboarding state so the wizard re-runs on next visit.

    Used by the settings page 'Re-run welcome' affordance.
    """
    from packages.cockpit.onboarding import load_onboarding, reset

    reset()
    return {"ok": True, "state": load_onboarding().to_dict()}


@app.get("/api/ollama/status")
def api_ollama_status() -> dict[str, Any]:
    """Read-only inventory for the cockpit's Ollama panel.

    Cheap enough to poll every few seconds — never starts the daemon and
    never pulls. The UI uses this to decide whether to show the Setup
    button vs. a green 'ready' badge.
    """
    # Import lazily so cockpit can still boot if tools/ path is funny.
    from tools.check_ollama import status_snapshot

    snap = status_snapshot()
    # Surface the currently-tracked job so the UI can pick up an in-progress
    # setup after a page reload without having to remember it client-side.
    snap["job"] = job_mgr.status(OLLAMA_SETUP_KIND).to_dict()
    return snap


@app.post("/api/ollama/setup")
def api_ollama_setup() -> dict[str, Any]:
    """Kick off ``tools/check_ollama.py --auto`` as a managed background job.

    Reusing the proc registry means the existing /api/jobs/{kind}/stream
    SSE endpoint streams live pull progress to the UI for free, and a
    page reload mid-pull picks the stream back up.
    """
    cmd = [_python_exe(), "tools/check_ollama.py", "--auto"]
    info = job_mgr.start(OLLAMA_SETUP_KIND, cmd)
    return info.to_dict()


@app.post("/api/ollama/stop")
def api_ollama_stop() -> dict[str, Any]:
    """Cancel an in-flight setup. The daemon itself is left running because
    other tools may depend on it; we only kill the pull driver."""
    return job_mgr.stop(OLLAMA_SETUP_KIND).to_dict()


OLLAMA_GPU_FIX_KIND = "ollama_gpu_fix"


@app.post("/api/ollama/gpu_fix")
def api_ollama_gpu_fix() -> dict[str, Any]:
    """One-click GPU enablement for AMD RDNA3 (RX 7900 XT) on Windows.

    Wraps ``tools/fix_ollama_gpu.ps1`` which downloads the matching ROCm
    DLL pack from likelovewant/ollama-for-amd, sets the required env
    vars, restarts the daemon, and verifies GPU detection. Streams
    progress through the standard /api/jobs/{kind}/stream SSE channel
    so the live log shows up in the cockpit Logs panel.

    POSIX returns 400 because the script is Windows-only — the cockpit
    UI hides the button unless the platform is win32.
    """
    if sys.platform != "win32":
        raise HTTPException(
            status_code=400,
            detail="GPU fix script is Windows-only (PowerShell, ROCm DLL pack).",
        )
    script = REPO_ROOT / "tools" / "fix_ollama_gpu.ps1"
    if not script.exists():
        raise HTTPException(
            status_code=404,
            detail=f"GPU fix script not found at {script}",
        )
    cmd = [
        "powershell.exe",
        "-ExecutionPolicy", "Bypass",
        "-NoProfile",
        "-File", str(script),
    ]
    info = job_mgr.start(OLLAMA_GPU_FIX_KIND, cmd)
    return info.to_dict()


@app.get("/api/ollama/gpu_fix")
def api_ollama_gpu_fix_status() -> dict[str, Any]:
    """Status snapshot for the GPU-fix job (used by the cockpit panel)."""
    return job_mgr.status(OLLAMA_GPU_FIX_KIND).to_dict()


@app.post("/api/ollama/gpu_fix/stop")
def api_ollama_gpu_fix_stop() -> dict[str, Any]:
    """Cancel an in-flight GPU fix. Useful if the download stalls or the
    user picked the button by accident — the script is mostly idempotent
    (env vars are already set on a re-run; backup is timestamped)."""
    return job_mgr.stop(OLLAMA_GPU_FIX_KIND).to_dict()


# ---------------------------------------------------------------------------
# Binary pinning -- pick which ollama.exe the cockpit launches
# ---------------------------------------------------------------------------
#
# On AMD machines we sometimes end up with TWO ollama.exe installs on PATH:
#   * Standard installer:   %LOCALAPPDATA%\Programs\Ollama\ollama.exe
#   * Adrenalin AI_Bundle:  %LOCALAPPDATA%\AMD\AI_Bundle\Ollama\ollama.exe
#
# The Adrenalin one ships AMD-blessed ROCm libs that just work on RX 7700+;
# the standard one silently picks CPU on RDNA3. The pin endpoint lets the
# cockpit hardlock the resolver onto the Adrenalin path AND kills any
# already-running ollama.exe so the next daemon-start picks the right binary.
#
# This is a much cheaper fix than the full GPU-fix script: no downloads, no
# DLL surgery, no reboots. Just pin + restart. The script remains the
# fallback for machines that don't have Adrenalin AI_Bundle installed.


def _kill_ollama_processes() -> int:
    """Kill any running ollama.exe on Windows. Returns count killed.

    Uses ``taskkill`` because we don't know which binary spawned the running
    daemon and shutil.which can return either one. On non-Windows this is
    a no-op that returns 0.
    """
    if os.name != "nt":
        return 0
    try:
        result = subprocess.run(
            ["taskkill", "/F", "/IM", "ollama.exe"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        # taskkill returns 128 when no process matches -- that's success for us.
        out = (result.stdout or "") + (result.stderr or "")
        # SUCCESS lines look like: 'SUCCESS: The process "ollama.exe" with PID 1234'
        return out.count("SUCCESS")
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return 0


@app.post("/api/ollama/pin")
def api_ollama_pin(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Pin the cockpit to a specific ollama.exe via COCKPIT_OLLAMA_BIN.

    Payload (all optional):
      * ``flavor``: ``"adrenalin"`` to auto-pick the AI_Bundle path (default),
        ``"standard"`` to pick the ollama.com installer path,
        ``"clear"`` to unset the pin and revert to auto-resolution.
      * ``path``: explicit absolute path -- overrides ``flavor``.

    Side effects (Windows only):
      1. Writes/clears the User-scope ``COCKPIT_OLLAMA_BIN`` env var so the
         pin survives cockpit restarts.
      2. Updates os.environ for the running process so the resolver picks
         up the new pin without needing a cockpit restart.
      3. Kills any running ollama.exe so the next daemon-start uses the
         newly pinned binary.

    Returns the new resolver result so the UI can update the badge.
    """
    from tools.check_ollama import (
        _adrenalin_candidate,
        _standard_candidate,
        resolve_ollama_binary,
    )

    payload = payload or {}
    explicit_path = (payload.get("path") or "").strip()
    flavor_req = (payload.get("flavor") or "adrenalin").strip().lower()

    target: str | None = None
    if explicit_path:
        if not os.path.isfile(explicit_path):
            return {"ok": False, "error": f"path not found: {explicit_path}"}
        target = explicit_path
    elif flavor_req == "clear":
        target = None
    elif flavor_req == "adrenalin":
        target = _adrenalin_candidate()
        if not target:
            return {
                "ok": False,
                "error": "AMD Adrenalin Ollama not found at %LOCALAPPDATA%\\AMD\\AI_Bundle\\Ollama. Install it from the AMD Adrenalin Software AI tab.",
            }
    elif flavor_req == "standard":
        target = _standard_candidate()
        if not target:
            return {"ok": False, "error": "Standard Ollama not found at %LOCALAPPDATA%\\Programs\\Ollama."}
    else:
        return {"ok": False, "error": f"unknown flavor: {flavor_req}"}

    # Update the running process so the next status_snapshot resolves correctly.
    if target:
        os.environ["COCKPIT_OLLAMA_BIN"] = target
    else:
        os.environ.pop("COCKPIT_OLLAMA_BIN", None)

    # Persist to the User-scope env so the pin survives a cockpit restart.
    # setx is Windows-only; on other platforms we just rely on the in-process
    # update + whatever shell rc file the user maintains themselves.
    persisted = False
    if os.name == "nt":
        try:
            value = target or ""
            # setx truncates to 1024 chars and emits a warning to stderr -- not
            # a real path will ever be that long, so we ignore the warning.
            subprocess.run(
                ["setx", "COCKPIT_OLLAMA_BIN", value],
                capture_output=True,
                timeout=5,
                check=False,
            )
            persisted = True
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            persisted = False

    killed = _kill_ollama_processes()

    path, flavor = resolve_ollama_binary()
    return {
        "ok": True,
        "pinned_to": target,
        "persisted": persisted,
        "killed_processes": killed,
        "ollama_binary": path,
        "ollama_flavor": flavor,
    }


@app.post("/api/ollama/warmup")
async def api_ollama_warmup() -> dict[str, Any]:
    """Pre-load each declared model into memory so the first user-facing
    Run isn't sitting through a 30-90s cold-start cascade.

    Fires a one-token generate at every installed model on the active
    profile's chain. Each call is fire-and-forget on a 5s ceiling, so a
    missing model just logs and moves on. Returns a list of per-model
    outcomes for the UI.
    """
    return await _warmup_models()


# --------------------------------------------------------------------------
# Health snapshot (operator-shareable, scrubbed troubleshooting bundle)
# --------------------------------------------------------------------------

HEALTH_SNAPSHOT_PATH = REPO_ROOT / "docs" / "health-snapshot.md"


def _build_snapshot():
    """Collect a fresh snapshot using the cockpit's tracked log paths.

    Imported lazily so the cockpit boots even if a dev removes the module.
    Tests can also patch the log paths via the module-level constants here.
    """
    from packages.cockpit.health_snapshot import collect_snapshot

    return collect_snapshot(
        repo_root=REPO_ROOT,
        paper_log=PAPER_LOG,
        scorecard_path=SCORECARD_LOG,
        promotion_log=SCORECARD_PROMOTION_LOG,
    )


@app.get("/api/health-snapshot")
def api_health_snapshot_preview() -> dict[str, Any]:
    """Return a freshly-rendered snapshot WITHOUT writing it to disk.

    The /errors page calls this so the operator can review exactly what
    would be shared before clicking Save. Body is the markdown plus a
    JSON form for any future tooling.
    """
    from packages.cockpit.health_snapshot import render_markdown

    snap = _build_snapshot()
    return {
        "generated_at": snap.generated_at,
        "markdown": render_markdown(snap),
        "json": snap.to_jsonable(),
        "size_bytes": len(render_markdown(snap).encode("utf-8")),
    }


@app.post("/api/health-snapshot/save")
def api_health_snapshot_save() -> dict[str, Any]:
    """Write the snapshot to ``docs/health-snapshot.md``.

    Path is gitignored by default — the operator opts in explicitly later
    if they want to push it to a private repo.
    """
    from packages.cockpit.health_snapshot import render_markdown, save_markdown

    snap = _build_snapshot()
    body = render_markdown(snap)
    saved = save_markdown(body, HEALTH_SNAPSHOT_PATH)
    return {
        "path": str(saved),
        "size_bytes": len(body.encode("utf-8")),
        "generated_at": snap.generated_at,
    }


# --------------------------------------------------------------------------
# Paper-trade loop control
# --------------------------------------------------------------------------


@app.post("/api/trading/start")
def api_trading_start(req: StartTradingRequest) -> dict[str, Any]:
    cmd = [_python_exe(), "tools/paper_trade.py", "--strategy", req.strategy, "--loop"]
    if req.dry_run:
        cmd.append("--dry-run")
    info = job_mgr.start(PAPER_LOOP_KIND, cmd)
    # Remember intent so we auto-resume on next cockpit boot.
    state = load_state()
    state.paper_loop_intended = True
    state.paper_loop_strategy = req.strategy
    state.paper_loop_dry_run = req.dry_run
    state = record_action(state, f"Paper loop started ({req.strategy}, dry_run={req.dry_run})")
    save_state(state)
    return info.to_dict()


@app.post("/api/trading/stop")
def api_trading_stop() -> dict[str, Any]:
    info = job_mgr.stop(PAPER_LOOP_KIND).to_dict()
    # Clear the auto-resume intent so a future cockpit reboot doesn't
    # re-spawn the loop the user just chose to stop.
    state = load_state()
    state.paper_loop_intended = False
    state = record_action(state, "Paper loop stopped")
    save_state(state)
    return info


@app.get("/api/trading/status")
def api_trading_status() -> dict[str, Any]:
    return job_mgr.status(PAPER_LOOP_KIND).to_dict()


@app.post("/api/trading/liquidate")
async def api_trading_liquidate(req: LiquidateRequest) -> dict[str, Any]:
    """Cancel open orders and close every paper position via Alpaca bulk API.

    Unlike ``/api/flatten`` which submits per-symbol market orders in a
    background task, this hits Alpaca's ``DELETE /v2/positions`` directly
    and returns the result synchronously. Use this to free buying power
    that's tied up in old positions and unblock the soak run.

    Requires ``confirm: true`` in the request body.
    """
    if not req.confirm:
        raise HTTPException(
            status_code=400,
            detail="liquidate requires confirm=true; this closes every open position at market",
        )
    if not (os.getenv("ALPACA_PAPER_KEY_ID") and os.getenv("ALPACA_PAPER_SECRET")):
        raise HTTPException(
            status_code=400,
            detail="Alpaca paper credentials missing -- set ALPACA_PAPER_KEY_ID / ALPACA_PAPER_SECRET",
        )
    # Pause the loop so it can't immediately re-fill what we just closed.
    state = load_state()
    state.paused = True
    state = record_action(state, "Liquidate all positions from cockpit")
    save_state(state)
    broker = AlpacaPaperBroker()
    try:
        try:
            result = await broker.liquidate_all(cancel_orders=req.cancel_orders)
        except BrokerError as e:
            log.error("liquidate failed: %s", e)
            raise HTTPException(status_code=502, detail=f"liquidate failed: {e}") from e
        log.info(
            "liquidate: closed %d positions, cancelled %d orders",
            result.get("closed_positions", 0),
            result.get("cancelled_orders", 0),
        )
        return {
            "ok": True,
            "paused": True,
            "closed_positions": result.get("closed_positions", 0),
            "cancelled_orders": result.get("cancelled_orders", 0),
        }
    finally:
        with contextlib.suppress(Exception):
            await broker.aclose()


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


# Live progress for an in-flight /api/agents/run call. The Run button on
# /agents polls this every second while it awaits the POST so the user
# sees *which* agent is currently calling Ollama, not just an opaque
# "calling Ollama... 69s" ticker. Updated by the run handler before/after
# each agent step; reset between runs.
_AGENT_PROGRESS: dict[str, Any] = {
    "active": False,
    "started_at": None,
    "current_agent": None,  # 'research' | 'strategy' | 'risk' | 'execution' | 'discovery' | None
    "agent_started_at": None,
    "completed": [],  # list of {agent, elapsed_s, status}
    "backend": None,  # 'llm' | 'stub'
    "error": None,
}
_PROGRESS_AGENTS = ("preflight", "research", "strategy", "risk", "execution", "discovery")


def _progress_reset(backend: str) -> None:
    _AGENT_PROGRESS.clear()
    _AGENT_PROGRESS.update({
        "active": True,
        "started_at": time.monotonic(),
        "current_agent": None,
        "agent_started_at": None,
        "completed": [],
        "backend": backend,
        "error": None,
    })


def _progress_begin(agent: str) -> None:
    _AGENT_PROGRESS["current_agent"] = agent
    _AGENT_PROGRESS["agent_started_at"] = time.monotonic()


def _progress_end(agent: str, status: str = "ok") -> None:
    started = _AGENT_PROGRESS.get("agent_started_at") or time.monotonic()
    elapsed = max(0.0, time.monotonic() - float(started))
    _AGENT_PROGRESS["completed"].append({
        "agent": agent,
        "elapsed_s": round(elapsed, 2),
        "status": status,
    })
    _AGENT_PROGRESS["current_agent"] = None
    _AGENT_PROGRESS["agent_started_at"] = None


def _progress_finish(error: str | None = None) -> None:
    _AGENT_PROGRESS["active"] = False
    _AGENT_PROGRESS["current_agent"] = None
    _AGENT_PROGRESS["agent_started_at"] = None
    if error:
        _AGENT_PROGRESS["error"] = error


@app.get("/api/agents/in_flight")
def api_agents_in_flight() -> dict[str, Any]:
    """Live progress for the currently-running pipeline (if any).

    Polled by the /agents Run button every 1s during a run so the user
    sees per-agent progress (e.g. ``strategy ... 12s``) instead of a
    single opaque counter that just keeps ticking up.
    """
    now = time.monotonic()
    snap: dict[str, Any] = {
        "active": bool(_AGENT_PROGRESS.get("active")),
        "current_agent": _AGENT_PROGRESS.get("current_agent"),
        "backend": _AGENT_PROGRESS.get("backend"),
        "completed": list(_AGENT_PROGRESS.get("completed") or []),
        "error": _AGENT_PROGRESS.get("error"),
        "all_agents": list(_PROGRESS_AGENTS),
    }
    started = _AGENT_PROGRESS.get("started_at")
    snap["elapsed_s"] = round(now - float(started), 2) if started else 0.0
    agent_started = _AGENT_PROGRESS.get("agent_started_at")
    snap["current_agent_elapsed_s"] = (
        round(now - float(agent_started), 2) if agent_started else 0.0
    )
    return snap


async def _cleanup_early_discovery(
    task: asyncio.Task | None,
    router: Any,
) -> None:
    """Best-effort cleanup of a parallel discovery kickoff after a run fails.

    If the main graph raised before the discovery block could await the
    early task, we still need to cancel the task and close the dedicated
    LLMRouter so the daemon connection pool doesn't leak. Any error here
    is swallowed — the caller is already raising the real error.
    """
    if task is not None and not task.done():
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):  # pragma: no cover
            await task
    if router is not None:
        with contextlib.suppress(Exception):  # pragma: no cover
            await router.aclose()


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

    # Live progress tracking — polled by the /agents Run button every 1s
    # via /api/agents/in_flight so the user sees *which* agent is calling
    # Ollama, not just an opaque elapsed counter. Reset on every run.
    _progress_reset("llm" if req.use_llm else "stub")

    # Parallel-discovery handoff vars. These are set inside the LLM branch
    # below when discovery is kicked off concurrently with strategy/risk/
    # execution. The discovery block further down checks them to either
    # await the early task or fall back to the sequential path. Declared
    # in the outer scope so the stub branch and exception paths don't
    # NameError when the discovery block reads them.
    _early_discovery_task: asyncio.Task | None = None
    _discovery_router_to_close: Any = None
    try:
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

            # Preflight: fail fast if Ollama isn't reachable at all rather than
            # letting the user sit through a 90s cold-start timeout per agent.
            _progress_begin("preflight")
            try:
                import httpx as _httpx
                _ollama_host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
                async with _httpx.AsyncClient(timeout=3.0) as _hc:
                    _r = await _hc.get(f"{_ollama_host}/api/tags")
                    _r.raise_for_status()
                _progress_end("preflight", "ok")
            except Exception as preflight_err:
                _progress_end("preflight", "failed")
                _progress_finish(error=f"Ollama unreachable: {preflight_err}")
                raise HTTPException(
                    status_code=503,
                    detail=(
                        f"Ollama is not reachable at {_ollama_host}. "
                        f"Start Ollama (or visit /models to install it), "
                        f"then try again. Original error: {preflight_err}"
                    ),
                ) from preflight_err

            # Wrap each runner so we can mark begin/end on the live progress
            # dict without modifying the underlying runner factories.
            def _instrument(agent_name: str, runner):  # type: ignore[no-untyped-def]
                async def _wrapped(inp):  # type: ignore[no-untyped-def]
                    _progress_begin(agent_name)
                    try:
                        out = await runner(inp)
                        _progress_end(agent_name, "ok")
                        return out
                    except Exception:
                        _progress_end(agent_name, "failed")
                        raise
                return _wrapped

            # Discovery agent is advisory-only and only depends on research
            # output — we can run it concurrently with strategy + risk +
            # execution to save ~one LLM-call worth of wall time. The task
            # is kicked off from inside the wrapped research runner as soon
            # as research returns, and awaited later in the discovery block.
            discovery_router = LLMRouter()
            discovery_task: asyncio.Task | None = None

            def _research_with_discovery_kickoff(research_runner):  # type: ignore[no-untyped-def]
                instrumented = _instrument("research", research_runner)
                async def _wrapped(inp):  # type: ignore[no-untyped-def]
                    nonlocal discovery_task
                    research_out = await instrumented(inp)
                    # Fire discovery NOW — it only needs research.thesis +
                    # research.sentiment, both of which are in research_out.
                    try:
                        from packages.agents.runners import build_discovery_runner
                        from packages.shared.schemas import DiscoveryInput
                        from packages.shared.universe import DEFAULT_UNIVERSE
                        disc_universe_local = DEFAULT_UNIVERSE.filter(symbols) or list(symbols)
                        disc_features_local = {
                            "sentiment": float(research_out.sentiment),
                            "regime_bull": 1.0 if regime == "bull" else 0.0,
                            "regime_bear": 1.0 if regime == "bear" else 0.0,
                            "regime_chop": 1.0 if regime == "chop" else 0.0,
                            "regime_crisis": 1.0 if regime == "crisis" else 0.0,
                            "n_signals": 0.0,
                            "n_approved": 0.0,
                        }
                        d_in_early = DiscoveryInput(
                            decision_id=getattr(research_out, "decision_id", ""),
                            regime=regime,  # type: ignore[arg-type]
                            universe=disc_universe_local,
                            features=disc_features_local,
                            recent_thesis=research_out.thesis,
                        )
                        d_runner_early = build_discovery_runner(
                            discovery_router, scorecard_summary=scorecard_summary,
                        )
                        async def _disc_with_progress():
                            _progress_begin("discovery")
                            try:
                                out = await d_runner_early(d_in_early)
                                _progress_end("discovery", "ok")
                                return out
                            except Exception:
                                _progress_end("discovery", "failed")
                                raise
                        discovery_task = asyncio.create_task(_disc_with_progress())
                    except Exception as kick_err:
                        log.warning("discovery kickoff failed (will fall back to sequential): %s", kick_err)
                        discovery_task = None
                    return research_out
                return _wrapped

            router = LLMRouter()
            try:
                graph = AgentGraph(
                    research=_research_with_discovery_kickoff(
                        build_research_runner(router, scorecard_summary=scorecard_summary)
                    ),
                    strategy=_instrument("strategy", build_strategy_runner(router, scorecard_summary=scorecard_summary)),
                    risk=_instrument("risk", build_risk_runner(router, scorecard_summary=scorecard_summary)),
                    execution=_instrument("execution", build_execution_runner(router, scorecard_summary=scorecard_summary)),
                    approval=_auto_approve,
                )
                result = await graph.run(
                    symbols=symbols,
                    regime=regime,
                    positions=[Position(symbol=s, qty=0.0, avg_price=0.0) for s in symbols],
                    features={},
                )
                # Stash the early-fired discovery task on the result so the
                # discovery block below can await it instead of starting a
                # fresh sequential one. We use a function attribute because
                # GraphResult is a frozen dataclass.
                _early_discovery_task = discovery_task
            finally:
                # We close the dedicated discovery router only after the
                # discovery block has read the result, so defer it via a
                # local var the discovery block will clean up.
                _discovery_router_to_close = discovery_router
        else:
            # Stub backend: mark each stub agent in sequence so the UI still
            # shows progression (these usually finish in <100ms total).
            for _stub_agent in ("research", "strategy", "risk", "execution"):
                _progress_begin(_stub_agent)
                _progress_end(_stub_agent, "ok")
            result = await paper_bridge.advise(
                symbols=symbols,
                regime=regime,
                positions=[Position(symbol=s, qty=0.0, avg_price=0.0) for s in symbols],
                target_weights=target_weights,
                sentiment_scores=None,
            )
    except HTTPException:
        _progress_finish(error=_AGENT_PROGRESS.get("error") or "http error")
        await _cleanup_early_discovery(_early_discovery_task, _discovery_router_to_close)
        raise
    except Exception as run_err:
        _progress_finish(error=str(run_err))
        await _cleanup_early_discovery(_early_discovery_task, _discovery_router_to_close)
        raise

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
            # Fast path: discovery was kicked off in parallel with strategy/
            # risk/execution from inside the wrapped research runner. If the
            # kickoff succeeded, just await its result here — saves ~one LLM
            # call of wall time per Run.
            if _early_discovery_task is not None:
                try:
                    d_out = await _early_discovery_task
                finally:
                    if _discovery_router_to_close is not None:
                        try:
                            await _discovery_router_to_close.aclose()
                        except Exception as close_err:  # pragma: no cover
                            log.debug("discovery router close failed: %s", close_err)
            else:
                # Fallback: kickoff didn't fire (e.g. an import failed inside
                # the wrapped research runner). Run the original sequential
                # path so the cockpit still shows a discovery result.
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
                    _progress_begin("discovery")
                    try:
                        d_out = await d_runner(d_in)
                        _progress_end("discovery", "ok")
                    except Exception:
                        _progress_end("discovery", "failed")
                        raise
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
    _progress_finish()
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
# Paper-trading autopilot (§16 60-day soak)
# --------------------------------------------------------------------------
#
# Wires :mod:`packages.cockpit.paper_autopilot` into the running cockpit:
# one shared AutopilotState lives here, the asyncio loop polls it every
# 30s, and the existing paper_loop job slot is what we spawn into.

_AUTOPILOT_STATE = paper_autopilot.AutopilotState(
    pause_checker=lambda: bool(load_state().paused),
    halt_checker=watchdog.is_halt_active,
    job_starter=lambda cmd: job_mgr.start(PAPER_LOOP_KIND, cmd),
)
_AUTOPILOT_TASK: dict[str, Any] = {"_task": None}


class AutopilotConfig(BaseModel):
    """Toggle and tune the paper autopilot."""

    enabled: bool | None = None
    strategy: str | None = None
    dry_run: bool | None = None


def _start_autopilot_task() -> None:
    task = _AUTOPILOT_TASK.get("_task")
    if task is not None and not task.done():
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    _AUTOPILOT_TASK["_task"] = loop.create_task(
        paper_autopilot.autopilot_loop(_AUTOPILOT_STATE, _python_exe)
    )


def _stop_autopilot_task() -> None:
    task = _AUTOPILOT_TASK.get("_task")
    if task is not None and not task.done():
        task.cancel()
    _AUTOPILOT_TASK["_task"] = None


@app.get("/api/autopilot")
def api_autopilot_get() -> dict[str, Any]:
    """Current autopilot config + last-fire status."""
    task = _AUTOPILOT_TASK.get("_task")
    running = bool(task is not None and not task.done())
    history = [
        {
            "trigger": f.trigger,
            "fired_at_utc": f.fired_at_utc,
            "ok": f.ok,
            "note": f.note,
            "job_pid": f.job_pid,
        }
        for f in _AUTOPILOT_STATE.history[-20:]
    ]
    history.reverse()
    return {
        "enabled": _AUTOPILOT_STATE.enabled,
        "running": running,
        "strategy": _AUTOPILOT_STATE.paper_strategy,
        "dry_run": _AUTOPILOT_STATE.dry_run,
        "open_trigger": _AUTOPILOT_STATE.open_trigger.isoformat(timespec="minutes"),
        "close_offset_minutes": _AUTOPILOT_STATE.close_offset_minutes,
        "last_fire_by_trigger": {
            k: v.isoformat() for k, v in _AUTOPILOT_STATE.last_fire_by_trigger.items()
        },
        "last_error": _AUTOPILOT_STATE.last_error,
        "recent_fires": history,
    }


@app.post("/api/autopilot")
async def api_autopilot_set(cfg: AutopilotConfig) -> dict[str, Any]:
    """Enable/disable the autopilot or change its strategy."""
    if cfg.strategy is not None:
        s = cfg.strategy.strip()
        if s:
            _AUTOPILOT_STATE.paper_strategy = s
    if cfg.dry_run is not None:
        _AUTOPILOT_STATE.dry_run = bool(cfg.dry_run)
    if cfg.enabled is not None:
        _AUTOPILOT_STATE.enabled = bool(cfg.enabled)
        if _AUTOPILOT_STATE.enabled:
            _start_autopilot_task()
        else:
            _stop_autopilot_task()
    # Persist autopilot intent so the cockpit can re-arm on reboot
    # without the user re-toggling the GUI (§17 follow-up).
    try:
        cstate = load_state()
        cstate.autopilot_enabled = _AUTOPILOT_STATE.enabled
        cstate.autopilot_strategy = _AUTOPILOT_STATE.paper_strategy
        cstate.autopilot_dry_run = _AUTOPILOT_STATE.dry_run
        save_state(cstate)
    except Exception as e:  # pragma: no cover - never block API on disk hiccup
        log.warning("could not persist autopilot intent: %s", e)
    return api_autopilot_get()


@app.post("/api/autopilot/tick")
async def api_autopilot_tick() -> dict[str, Any]:
    """Manually execute one autopilot tick (tests + ad-hoc fires)."""
    fire = paper_autopilot.run_tick(
        _AUTOPILOT_STATE, datetime.now(UTC), _python_exe()
    )
    if fire is None:
        return {"fired": False, "reason": "no trigger window or autopilot disabled"}
    return {
        "fired": True,
        "trigger": fire.trigger,
        "ok": fire.ok,
        "note": fire.note,
        "job_pid": fire.job_pid,
    }


@app.get("/autopilot", response_class=HTMLResponse)
def autopilot_page() -> HTMLResponse:
    return _render("autopilot.html")


@app.on_event("startup")
async def _autopilot_startup() -> None:  # pragma: no cover
    # Auto-resume autopilot if the user had it enabled before shutdown.
    # Mirrors the paper-loop auto-resume contract: intent persisted in
    # CockpitState wins over the in-process default.
    if os.environ.get("COCKPIT_AUTO_RESUME_AUTOPILOT", "1") in ("1", "true", "True"):
        try:
            cstate = load_state()
            if cstate.autopilot_enabled:
                _AUTOPILOT_STATE.enabled = True
                _AUTOPILOT_STATE.paper_strategy = cstate.autopilot_strategy
                _AUTOPILOT_STATE.dry_run = cstate.autopilot_dry_run
                log.info(
                    "auto-resume: autopilot re-armed strategy=%s dry_run=%s",
                    cstate.autopilot_strategy,
                    cstate.autopilot_dry_run,
                )
        except Exception as e:
            log.warning("auto-resume autopilot: state load failed: %s", e)
    if _AUTOPILOT_STATE.enabled:
        _start_autopilot_task()


@app.on_event("startup")
async def _paper_loop_auto_resume() -> None:  # pragma: no cover
    """If the paper loop was intended to be running when the cockpit shut
    down, re-spawn it now with the same strategy/dry_run combo.

    Honors COCKPIT_AUTO_RESUME_LOOP (default on). Skipped when the bot is
    paused or when a job is already alive (PID still valid).
    """
    if os.environ.get("COCKPIT_AUTO_RESUME_LOOP", "1") not in ("1", "true", "True"):
        return
    try:
        cstate = load_state()
    except Exception as e:
        log.warning("auto-resume: state load failed: %s", e)
        return
    if not cstate.paper_loop_intended:
        return
    if cstate.paused:
        log.info("auto-resume: paper loop intent set but cockpit is paused; skipping")
        return
    existing = job_mgr.status(PAPER_LOOP_KIND)
    if existing.is_running():
        log.info("auto-resume: paper loop already running (pid=%s)", existing.pid)
        return
    cmd = [
        _python_exe(),
        "tools/paper_trade.py",
        "--strategy",
        cstate.paper_loop_strategy,
        "--loop",
    ]
    if cstate.paper_loop_dry_run:
        cmd.append("--dry-run")
    try:
        info = job_mgr.start(PAPER_LOOP_KIND, cmd)
        log.info(
            "auto-resume: paper loop respawned pid=%s strategy=%s dry_run=%s",
            info.pid,
            cstate.paper_loop_strategy,
            cstate.paper_loop_dry_run,
        )
    except Exception as e:
        log.warning("auto-resume: failed to respawn loop: %s", e)


# Background pre-warm so the first Run on /agents doesn't pay a 30-90s
# per-agent cold-start cost. Honors COCKPIT_WARMUP_ON_STARTUP (default on);
# set to 0 to disable for dev. Runs as a fire-and-forget task so the
# cockpit boots immediately — the user can hit pages while warmup churns.
@app.on_event("startup")
async def _ollama_warmup_startup() -> None:  # pragma: no cover
    if os.environ.get("COCKPIT_WARMUP_ON_STARTUP", "1") not in ("1", "true", "True"):
        log.info("Ollama warmup disabled via COCKPIT_WARMUP_ON_STARTUP=0")
        return

    async def _bg_warmup() -> None:
        # Wait a tick so the rest of startup finishes and the daemon (if
        # auto-started elsewhere) has a chance to come up.
        await asyncio.sleep(2.0)
        try:
            result = await _warmup_models()
            if result.get("ok"):
                log.info(
                    "Ollama warmup complete in %.1fs (%d models)",
                    result.get("total_elapsed_s", 0.0),
                    len(result.get("results") or []),
                )
            else:
                log.warning("Ollama warmup finished with errors: %s", result.get("error") or "see /api/ollama/warmup_status")
        except Exception as e:  # don't crash startup over a warmup failure
            log.warning("Ollama warmup skipped: %s", e)

    # Stored on module state so the task isn't GC'd mid-warmup. Cleared
    # automatically when the task finishes.
    global _WARMUP_BG_TASK
    _WARMUP_BG_TASK = asyncio.create_task(_bg_warmup())


# ---------------------------------------------------------------------------
# Background automation: watchdog ticker + daily backups + audit rotation
# + boot doctor. Each loop is opt-out via env var and lives in
# packages/cockpit/automation.py so the logic is testable in isolation.
# ---------------------------------------------------------------------------

_AUTOMATION_TASKS: dict[str, asyncio.Task[Any]] = {}
_BACKUP_STATE: dict[str, Any] = {}


@app.on_event("startup")
async def _automation_startup() -> None:  # pragma: no cover - long-lived tasks
    # Boot doctor: log a one-line readiness summary so the user sees
    # missing keys / empty parquet cache at a glance, not via cryptic
    # downstream agent errors.
    if os.environ.get("COCKPIT_BOOT_DOCTOR", "1") in ("1", "true", "True"):
        try:
            report = automation.boot_doctor_report()
            log.info("boot doctor: %s", automation.summarize_boot_doctor(report))
        except Exception as e:
            log.warning("boot doctor failed: %s", e)

    loop = asyncio.get_running_loop()

    # Watchdog ticker: re-evaluate drawdown every 60s so an overnight
    # breach halts the loop even if no one hits the API.
    if os.environ.get("COCKPIT_WATCHDOG_LOOP", "1") in ("1", "true", "True"):
        def _eval() -> object:
            return watchdog.evaluate_and_persist(equity_curve_points())
        _AUTOMATION_TASKS["watchdog"] = loop.create_task(
            automation.watchdog_loop(evaluator=_eval, poll_seconds=60.0)
        )

    # Daily backup scheduler: zip data/ + logs/ once per UTC day.
    if os.environ.get("COCKPIT_AUTO_BACKUP", "1") in ("1", "true", "True"):
        from tools.backup_daily import build_backup, prune_old

        def _runner() -> Path:
            out = build_backup()
            try:
                prune_old(keep=30)
            except Exception as e:
                log.warning("backup retention prune failed: %s", e)
            return out

        _AUTOMATION_TASKS["backup"] = loop.create_task(
            automation.backup_loop(runner=_runner, state=_BACKUP_STATE)
        )

    # Audit log rotation: once an hour check decisions.jsonl size,
    # gzip-rotate if it exceeds AUDIT_MAX_BYTES (default 50 MB).
    if os.environ.get("COCKPIT_AUDIT_ROTATE", "1") in ("1", "true", "True"):
        from packages.persistence.audit import AUDIT_LOG_PATH

        _AUTOMATION_TASKS["audit_rotate"] = loop.create_task(
            automation.audit_rotate_loop(path=AUDIT_LOG_PATH)
        )


@app.on_event("shutdown")
async def _automation_shutdown() -> None:  # pragma: no cover - shutdown
    for name, task in list(_AUTOMATION_TASKS.items()):
        if task is not None and not task.done():
            task.cancel()
        _AUTOMATION_TASKS.pop(name, None)


@app.get("/api/automation")
def api_automation_status() -> dict[str, Any]:
    """Surface background-loop state so the GUI can prove these are running."""
    out: dict[str, Any] = {"loops": {}, "backup": dict(_BACKUP_STATE)}
    for name, task in _AUTOMATION_TASKS.items():
        out["loops"][name] = {
            "running": bool(task is not None and not task.done()),
            "done": bool(task is not None and task.done()),
        }
    return out


@app.on_event("shutdown")
async def _autopilot_shutdown() -> None:  # pragma: no cover
    _stop_autopilot_task()


# --------------------------------------------------------------------------
# Drawdown watchdog (§16 8% halt)
# --------------------------------------------------------------------------


@app.get("/api/watchdog")
def api_watchdog_get() -> dict[str, Any]:
    """Current watchdog status: live verdict + persistent halt record."""
    verdict = watchdog.evaluate_curve(equity_curve_points())
    return {
        "verdict": {
            "breach": verdict.breach,
            "current_drawdown": verdict.current_drawdown,
            "peak_equity": verdict.peak_equity,
            "current_equity": verdict.current_equity,
            "threshold": verdict.threshold,
            "message": verdict.message,
        },
        "halt": watchdog.read_halt(),
        "halt_active": watchdog.is_halt_active(),
    }


@app.post("/api/watchdog/tick")
def api_watchdog_tick() -> dict[str, Any]:
    """Force one evaluate-and-persist pass; stop paper loop if breached."""
    verdict = watchdog.evaluate_and_persist(equity_curve_points())
    if verdict.breach:
        try:
            job_mgr.stop(PAPER_LOOP_KIND)
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("watchdog: could not stop paper loop: %s", exc)
    return {
        "breach": verdict.breach,
        "message": verdict.message,
        "halt": watchdog.read_halt(),
    }


class ClearHaltRequest(BaseModel):
    acknowledged_by: str = "operator"


@app.post("/api/watchdog/clear")
def api_watchdog_clear(req: ClearHaltRequest) -> dict[str, Any]:
    """Operator acknowledges an active halt and releases it."""
    record = watchdog.clear_halt(acknowledged_by=req.acknowledged_by)
    return {"cleared": True, "record": record}


# --------------------------------------------------------------------------
# /promote -- the live-trading readiness gate
# --------------------------------------------------------------------------


@app.get("/promote", response_class=HTMLResponse)
def promote_page() -> HTMLResponse:
    return _render("promote.html")


@app.get("/api/promote")
def api_promote() -> dict[str, Any]:
    """Return the full live-trading readiness picture.

    Pulls the paper equity curve, runs the §16 readiness gate, and
    surfaces every reason live capital is (or isn't) allowed. Includes
    the Telegram-bot-not-yet-connected line item so the operator knows
    that channel is still required even after the metrics pass.
    """
    import pandas as pd

    from packages.backtests import live_promotion as lp

    points = equity_curve_points(window=200)
    series = pd.Series([float(p.get("equity", 0.0)) for p in points])
    decision = lp.decide_live_capital(series)

    paper_days = int(decision.readiness.metrics.get("paper_days", len(series)))
    days_remaining = max(0, lp.PAPER_MIN_DAYS - paper_days)
    telegram_connected = bool(os.getenv("TELEGRAM_BOT_TOKEN")) and bool(
        os.getenv("TELEGRAM_CHAT_ID")
    )
    enable_flag = os.getenv("ENABLE_LIVE_TRADING", "").strip().lower() in {
        "true",
        "1",
        "yes",
        "on",
    }

    gating_reasons = list(decision.readiness.reasons)
    if not telegram_connected:
        gating_reasons.append(
            "Telegram approval bot is not configured "
            "(set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env)"
        )

    return {
        "live_enabled": bool(decision.live_enabled and telegram_connected),
        "capital_fraction": (
            decision.capital_fraction
            if decision.live_enabled and telegram_connected
            else 0.0
        ),
        "readiness": {
            "ready": decision.readiness.ready and telegram_connected,
            "reasons": gating_reasons,
            "metrics": decision.readiness.metrics,
        },
        "requirements": {
            "paper_min_days": lp.PAPER_MIN_DAYS,
            "paper_max_dd": lp.PAPER_MAX_DD,
            "paper_min_sharpe": lp.PAPER_MIN_SHARPE,
        },
        "progress": {
            "paper_days": paper_days,
            "days_remaining": days_remaining,
            "telegram_connected": telegram_connected,
            "enable_live_flag": enable_flag,
        },
        "canary": (
            {
                "tier_index": decision.canary.tier_index,
                "fraction": decision.canary.fraction,
                "days_in_tier": decision.canary.days_in_tier,
                "dwell_required": decision.canary.dwell_required,
                "next_fraction": decision.canary.next_fraction,
            }
            if decision.canary is not None
            else None
        ),
    }


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
    from packages.agents.price_chain import build_default_chain, provider_summary

    # Build the multi-provider chain. yfinance is always present, so the
    # chain is never empty — paper attribution can run end-to-end on a
    # fresh box with zero env config. Alpaca/Polygon get tried first when
    # configured so we lean on the rate-limit-friendly paid sources.
    chain = build_default_chain()

    n = run_attribution(
        AGENT_LOG,
        SCORECARD_LOG,
        chain.get_close,
        horizons_days=DEFAULT_HORIZONS_DAYS,
        now=None if not force else datetime.now(UTC).replace(year=datetime.now(UTC).year + 10),
    )
    return {
        "appended": n,
        "scorecard_path": str(SCORECARD_LOG),
        "price_chain": provider_summary(chain),
    }


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
def api_errors_list(
    limit: int = 200,
    severity: str | None = None,
    include_resolved: bool = False,
) -> dict[str, Any]:
    # The UI defaults to hiding resolved rows so stale halts that have
    # since recovered don't keep cluttering the page. Pass
    # ``include_resolved=true`` to see the full audit trail.
    return {
        "counts": err_log.count_unresolved(),
        "entries": err_log.list_errors(
            limit=limit, severity=severity, include_resolved=include_resolved
        ),
    }


@app.get("/api/errors/markdown")
def api_errors_markdown(limit: int = 50) -> dict[str, Any]:
    return {"markdown": err_log.to_markdown(limit=limit)}


@app.post("/api/errors/clear")
def api_errors_clear() -> dict[str, Any]:
    n = err_log.clear()
    return {"cleared": n}


@app.post("/api/errors/clear_resolved")
def api_errors_clear_resolved() -> dict[str, Any]:
    n = err_log.clear_resolved()
    return {"cleared": n}


@app.post("/api/errors/{entry_id}/resolve")
def api_errors_resolve(entry_id: str) -> dict[str, Any]:
    ok = err_log.resolve(entry_id)
    return {"ok": ok, "id": entry_id, "counts": err_log.count_unresolved()}


@app.post("/api/errors/{entry_id}/unresolve")
def api_errors_unresolve(entry_id: str) -> dict[str, Any]:
    ok = err_log.unresolve(entry_id)
    return {"ok": ok, "id": entry_id, "counts": err_log.count_unresolved()}


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
