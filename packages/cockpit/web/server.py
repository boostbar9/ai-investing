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
from packages.cockpit.web import autonomy as autonomy_brain
from packages.cockpit.web import bandit as autonomy_bandit
from packages.cockpit.web import brain_memory as autonomy_memory
from packages.cockpit.web import chatter as agent_chatter
from packages.cockpit.web import finnhub_ws as finnhub_ws_mod
from packages.cockpit.web import knowledge_base as autonomy_knowledge
from packages.cockpit.web import live_quotes as live_quotes_mod
from packages.cockpit.web import reflection as autonomy_reflection
from packages.cockpit.web import regime as autonomy_regime
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


def _install_web_file_logging() -> None:
    """Mirror the web-server's log output to a rotating file. Idempotent.

    The Robinhood/OAuth onboarding routes log to *this* uvicorn process.
    Without a file handler that output only reaches the console window, so
    the remote bridge (``/api/remote/weblog``) can't surface connect errors
    to a remote operator. We ADD a RotatingFileHandler to the root logger
    (alongside the existing console handler from basicConfig) so every
    logger that propagates to root — including the robinhood/onboarding
    loggers and uvicorn — is captured. Console logging is unchanged.

    Failures here are swallowed: file logging is observability, never a
    reason to take the server down.
    """
    from logging.handlers import RotatingFileHandler

    from packages.cockpit.web.remote import WEB_LOG_PATH

    root = logging.getLogger()
    for existing in root.handlers:
        if getattr(existing, "_cockpit_web_file", False):
            return
    try:
        WEB_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            WEB_LOG_PATH,
            maxBytes=2 * 1024 * 1024,
            backupCount=2,
            encoding="utf-8",
        )
        handler.setLevel(logging.INFO)
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        handler._cockpit_web_file = True  # type: ignore[attr-defined]
        root.addHandler(handler)
    except OSError:
        # Read-only FS, permissions, etc. — keep console logging only.
        pass


_install_web_file_logging()


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


# ---------------------------------------------------------------------------
# Live Robinhood account snapshot (read-only) for the AI agent context.
#
# Robinhood reads are async (MCP-over-HTTP) but the agent-context + UI
# readers here are sync, so we keep a small in-memory cache refreshed on
# the autonomy loop's quote-warmup tick. Always read-only -- never places
# or cancels an order, so it is safe in shadow mode.
# ---------------------------------------------------------------------------

_RH_SNAPSHOT_CACHE: dict[str, Any] = {"snapshot": None, "ts": None}


def latest_robinhood_snapshot() -> dict[str, Any] | None:
    """Return the most recently cached live Robinhood account snapshot.

    Sync + fast: returns whatever the async warmup tick last fetched
    (or ``None`` before the first refresh / when not connected). The
    agent context + dashboard read this so the AI knows the user's real
    buying power, cash, equity, and current positions when reasoning
    about the market.
    """
    return _RH_SNAPSHOT_CACHE.get("snapshot")


async def _refresh_robinhood_snapshot() -> dict[str, Any] | None:
    """Fetch a fresh read-only Robinhood snapshot and update the cache.

    Never raises -- on any failure the cache is left untouched and the
    previous value (possibly ``None``) is returned. Safe to call from
    the autonomy loop on every warmup tick.
    """
    try:
        from packages.execution.robinhood import (
            is_connected as _rh_is_connected,
        )
        from packages.execution.robinhood import (
            robinhood_account_snapshot as _rh_snapshot,
        )

        if not _rh_is_connected():
            # Not connected -> clear any stale snapshot so the UI/agent
            # don't show data for a disconnected account.
            _RH_SNAPSHOT_CACHE["snapshot"] = None
            _RH_SNAPSHOT_CACHE["ts"] = None
            return None
        snap = await _rh_snapshot()
        _RH_SNAPSHOT_CACHE["snapshot"] = snap
        _RH_SNAPSHOT_CACHE["ts"] = snap.get("as_of")
        return snap
    except Exception as exc:  # pragma: no cover — defensive
        log.debug("robinhood snapshot refresh failed: %s", exc)
        return _RH_SNAPSHOT_CACHE.get("snapshot")


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
    # Phase 25.5 — the paper runs ledger may not carry a regime label
    # (older runs, or strategies that never write one). Fall back to
    # the live autonomy brain so the hero card matches Brain Health
    # instead of showing "—".
    if not auto_regime:
        try:
            snap = autonomy_brain.snapshot() or {}
            # The brain publishes the current regime under `last_regime`
            # (set by `autonomy._sweep` after `regime_module.detect`).
            # Accept `regime` too for forward-compat with future renames.
            brain_reg = (snap.get("last_regime") or snap.get("regime") or {})
            label = brain_reg.get("label")
            if label:
                auto_regime = label
            if confidence is None and brain_reg.get("confidence") is not None:
                confidence = brain_reg["confidence"]
        except Exception:  # pragma: no cover — defensive
            pass
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

# Phase 36c — remote-control bridge. The router is fail-closed: every
# /api/remote/* route returns 503 unless COCKPIT_REMOTE_TOKEN is set to
# a secret of at least 16 characters. See packages/cockpit/web/remote.py
# for the security model and full surface description.
from packages.cockpit.web import remote as _remote_module  # noqa: E402

app.include_router(_remote_module.build_router())

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


@app.get("/preflight", response_class=HTMLResponse)
def preflight_page() -> HTMLResponse:
    """Phase 16: aggregated readiness checklist.

    One URL that lights up green/yellow/red across every precondition
    for going live, with a single 'Arm Live' button at the bottom that
    is only enabled when every gate is green.
    """
    return _render("preflight.html")


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


@app.get("/shadow", response_class=HTMLResponse)
def shadow_page() -> HTMLResponse:
    """Shadow-trading dashboard (Phase 6).

    Renders the static template; the page client-side polls
    ``/api/shadow/snapshot`` every 30s for fresh state.
    """
    return _render("shadow.html")


@app.get("/api/shadow/snapshot")
def api_shadow_snapshot() -> dict[str, Any]:
    """JSON snapshot of pairs + daily PnL + greenlight verdict."""
    from packages.shadow.snapshot import build_snapshot

    snap = build_snapshot()
    return snap.to_payload()


@app.get("/api/shadow/flip-events")
def api_shadow_flip_events(limit: int = 20) -> dict[str, Any]:
    """Recent SHADOW->READY transition events.

    Surfaces these so the cockpit can show a one-time banner and the
    autopilot can fan them out to Telegram / desktop notifications
    without owning the persistence layer.
    """
    from packages.shadow.notify import read_flip_events

    capped = max(1, min(int(limit), 100))
    rows = read_flip_events(limit=capped)
    return {"events": rows, "count": len(rows)}


# ---------------------------------------------------------------------------
# Phase 11 -- decision instrumentation endpoints.
#
# Three sibling routes power the new /shadow page panels:
#   /api/shadow/decisions -- last N per-cycle decision rows (table)
#   /api/shadow/pipeline  -- aggregated candidate funnel (24h window)
#   /api/shadow/window    -- 14-day shadow-trading window progress
# All three are read-only and pure-stdlib so they stay fast under the
# dashboard's 30s polling cadence.
# ---------------------------------------------------------------------------


@app.get("/api/shadow/decisions")
def api_shadow_decisions(limit: int = 50) -> dict[str, Any]:
    """Recent per-cycle decision traces, newest-first.

    Each row exposes the candidate funnel (sweep -> corroborated ->
    agent-approved -> target -> planned -> submitted), halt reasons,
    and a few headline metrics. The /shadow page renders this as the
    'Per-cycle decisions' table.
    """
    from packages.paper.decisions import load_recent

    capped = max(1, min(int(limit), 500))
    rows = load_recent(limit=capped)
    return {"decisions": rows, "count": len(rows), "limit": capped}


@app.get("/api/shadow/pipeline")
def api_shadow_pipeline() -> dict[str, Any]:
    """Aggregated candidate-funnel across the last 24h.

    Returns one entry per stage with totals + average-per-cycle. The
    /shadow page renders this as a horizontal funnel so the user can
    see at a glance which stage is filtering out their candidates.
    """
    from packages.paper.decisions import latest_pipeline

    return latest_pipeline()


@app.get("/api/shadow/window")
def api_shadow_window() -> dict[str, Any]:
    """Shadow-trading window progress.

    Reads the decision log to derive a per-day activity calendar and
    a 'day X of 14' counter. The /shadow page uses this for the window
    progress card at the top.
    """
    from packages.paper.decisions import window_status

    return window_status(target_days=14)


@app.get("/api/shadow/policy")
def api_shadow_policy(limit: int = 50) -> dict[str, Any]:
    """Phase 13: confidence-gated policy decisions from recent cycles.

    Pulls the last ``limit`` cycles that ran the 'policy' strategy and
    flattens their per-symbol decisions into a single newest-first list.
    The /shadow page renders this as the 'Confidence-gated decisions'
    panel with action chips + a confidence histogram.

    Returns a calibration breakdown too: how many BUY/HOLD/SELL
    decisions in each confidence bucket. Once we have enough trade
    outcomes attached we'll layer the realised win-rate on top.
    """
    from packages.paper.decisions import load_recent

    capped = max(1, min(int(limit), 500))
    rows = load_recent(limit=capped)
    decisions: list[dict[str, Any]] = []
    for row in rows:
        for d in row.get("policy_decisions") or []:
            # Attach cycle ts so the dashboard can group / sort.
            d2 = dict(d)
            d2["cycle_ts"] = row.get("ts")
            d2["cycle_regime"] = row.get("regime")
            decisions.append(d2)

    # Bucket by 0.1-wide confidence band x action -> count. Cheap to
    # compute, lets the dashboard render a stacked bar without doing
    # math in JS.
    buckets: dict[str, dict[str, int]] = {}
    for d in decisions:
        try:
            conf = float(d.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        # Bucket lower-bound to one decimal, clamped to [0, 1).
        lb = max(0.0, min(0.9, round(conf - (conf % 0.1), 1)))
        key = f"{lb:.1f}"
        b = buckets.setdefault(key, {"buy": 0, "hold": 0, "sell": 0})
        action = str(d.get("action", "hold"))
        if action in b:
            b[action] += 1

    return {
        "decisions": decisions,
        "count": len(decisions),
        "buckets": buckets,
        "thresholds": {
            "buy": float(__import__("os").environ.get("POLICY_BUY_THRESHOLD", "0.65")),
            "sell": float(__import__("os").environ.get("POLICY_SELL_THRESHOLD", "0.35")),
        },
    }


@app.get("/api/shadow/calibration")
def api_shadow_calibration(
    horizon_days: int = 5, win_threshold: float = 0.0
) -> dict[str, Any]:
    """Phase 14: reliability curve for the confidence-gated policy.

    Joins BUY decisions from the JSONL log with realised forward returns
    over ``horizon_days`` (close-to-close on whatever price data we can
    reach), bins by 0.1-wide predicted-confidence band, and returns the
    reliability table + Brier score + ECE for the dashboard to plot.

    Also reports whether a fitted calibrator is currently active in the
    live policy and includes its diagnostics (raw vs calibrated ECE).

    Best-effort: if price data isn't reachable (offline / yfinance flaky)
    the endpoint returns an empty curve with ``n_samples=0`` rather than
    erroring -- the panel renders a "not enough data yet" placeholder.
    """
    from datetime import UTC, datetime

    from packages.agents.calibration import (
        IsotonicCalibrator,
        ReliabilityCurve,
        extract_calibration_pairs,
    )
    from packages.paper.decisions import iter_decisions

    horizon = max(1, min(int(horizon_days), 30))

    # Pull all decision rows (full history -- typically only a few hundred).
    rows = list(iter_decisions())

    # Collect the symbols + decision timestamps we need forward returns for.
    needed: dict[str, list[str]] = {}
    for row in rows:
        ts = row.get("ts")
        if not ts:
            continue
        for pd_ in row.get("policy_decisions") or []:
            if pd_.get("action") != "buy":
                continue
            sym = pd_.get("symbol")
            if not sym:
                continue
            needed.setdefault(str(sym).upper(), []).append(ts)

    realised: dict[str, dict[str, float]] = {}
    if needed:
        try:
            from tools.paper_trade import load_panel

            panel = load_panel(sorted(needed.keys()))
        except Exception as exc:  # pragma: no cover - network/data dependent
            log.warning("calibration: price panel unavailable: %s", exc)
            panel = None

        if panel is not None and not panel.empty:
            for sym, ts_list in needed.items():
                if sym not in panel.columns:
                    continue
                series = panel[sym].dropna()
                if len(series) < horizon + 1:
                    continue
                for ts in ts_list:
                    try:
                        decision_day = datetime.fromisoformat(
                            str(ts).replace("Z", "+00:00")
                        ).date()
                    except (TypeError, ValueError):
                        continue
                    # Find the first trading day >= decision_day. The
                    # entry price is that day's close; the exit price is
                    # horizon trading days later.
                    idx_dates = [d.date() for d in series.index]
                    entry_i = None
                    for i, d in enumerate(idx_dates):
                        if d >= decision_day:
                            entry_i = i
                            break
                    if entry_i is None or entry_i + horizon >= len(series):
                        continue
                    entry = float(series.iloc[entry_i])
                    exit_ = float(series.iloc[entry_i + horizon])
                    if entry <= 0:
                        continue
                    realised.setdefault(sym, {})[ts] = (exit_ / entry) - 1.0

    pairs = extract_calibration_pairs(
        rows,
        realised,
        horizon_days=horizon,
        win_threshold=float(win_threshold),
    )
    raw_curve = ReliabilityCurve.from_pairs(pairs)

    # If a fitted calibrator exists, also build the post-calibration curve.
    cal = IsotonicCalibrator.load()
    calibrated_curve = None
    if cal.is_fitted and pairs:
        cal_pairs = [(cal(p), y) for p, y in pairs]
        calibrated_curve = ReliabilityCurve.from_pairs(cal_pairs)

    return {
        "horizon_days": horizon,
        "win_threshold": float(win_threshold),
        "n_samples": raw_curve.n_samples,
        "raw": raw_curve.to_dict(),
        "calibrated": calibrated_curve.to_dict() if calibrated_curve else None,
        "calibrator": {
            "is_fitted": cal.is_fitted,
            "n_samples_fit": cal.n_samples_fit,
            "raw_ece": round(cal.raw_ece, 4),
            "calibrated_ece": round(cal.calibrated_ece, 4),
            "raw_brier": round(cal.raw_brier, 4),
            "calibrated_brier": round(cal.calibrated_brier, 4),
            "breakpoints": [
                {"x": round(x, 4), "y": round(y, 4)}
                for x, y in zip(cal.x_breakpoints, cal.y_breakpoints, strict=False)
            ],
        },
        "now": datetime.now(UTC).isoformat(),
    }


@app.get("/api/shadow/sizing")
def api_shadow_sizing(limit: int = 50) -> dict[str, Any]:
    """Phase 15: risk-adaptive sizing diagnostics from the most recent
    'policy' cycle.

    Walks the decision log newest-first and returns the first row whose
    ``sizing`` block is non-empty (i.e. a cycle that actually ran the
    confidence-gated policy with a sizer attached). Also returns a small
    history list of the per-cycle gross_target + dd_multiplier so the
    dashboard can sparkline how aggressively the sizer is exposing the
    book over time.

    Empty-data shape: ``{"latest": {}, "history": [], "count": 0}`` so
    the dashboard renders a "no sizing data yet" placeholder instead of
    erroring on first boot.
    """
    from packages.paper.decisions import load_recent

    capped = max(1, min(int(limit), 500))
    rows = load_recent(limit=capped)

    latest: dict[str, Any] = {}
    history: list[dict[str, Any]] = []
    for row in rows:
        sizing = row.get("sizing") or {}
        # Skip pre-Phase-15 rows and cycles where the sizer didn't run
        # (no per-symbol diagnostics -- nothing to plot).
        if not isinstance(sizing, dict) or not sizing.get("diagnostics"):
            continue
        history.append(
            {
                "ts": row.get("ts"),
                "mode": sizing.get("mode"),
                "equity": sizing.get("equity"),
                "peak_equity": sizing.get("peak_equity"),
                "dd_observed": sizing.get("dd_observed"),
                "dd_exposure_multiplier": sizing.get("dd_exposure_multiplier"),
                "gross_target": sizing.get("gross_target"),
                "gross_actual": sizing.get("gross_actual"),
                "n_positions": len(sizing.get("diagnostics") or []),
            }
        )
        if not latest:
            # First non-empty row is the newest, since load_recent returns
            # newest-first. Attach the cycle's ts/regime so the panel can
            # show when this snapshot was taken.
            latest = dict(sizing)
            latest["cycle_ts"] = row.get("ts")
            latest["cycle_regime"] = row.get("regime")

    return {
        "latest": latest,
        "history": history,
        "count": len(history),
    }


# Phase 12: manual one-shot cycle trigger. Lets the user click a button
# on /shadow and immediately see a new decision row appear in the table.
# Default strategy = 'ensemble' (the same one tools/boot.py drives in the
# background) and dry_run=True so this never accidentally sends a live
# order from the dashboard.
_FORCE_CYCLE_LOCK = asyncio.Lock()


@app.post("/api/shadow/force-cycle")
async def api_shadow_force_cycle(
    strategy: str = "ensemble",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run one paper-trade cycle synchronously and return the record.

    Phase 35c: default flipped to ``dry_run=False`` so the cockpit's
    primary mode is live Alpaca paper trading. Callers that still want
    a dry-run preview can pass ``dry_run=true`` explicitly.

    Single-flight via a module-level lock so a user mashing the button
    doesn't kick off overlapping cycles. The response includes the new
    decision_id so the page can highlight the row that just appeared.
    """
    if _FORCE_CYCLE_LOCK.locked():
        return {
            "ok": False,
            "error": "a cycle is already running; try again in a few seconds",
        }
    async with _FORCE_CYCLE_LOCK:
        try:
            from tools.paper_trade import run as run_cycle
        except ImportError as exc:
            return {"ok": False, "error": f"paper_trade unavailable: {exc}"}
        try:
            record = await run_cycle(strategy, dry_run=bool(dry_run))
        except Exception as exc:
            return {"ok": False, "error": f"{exc.__class__.__name__}: {exc}"}
        return {
            "ok": True,
            "strategy": strategy,
            "dry_run": bool(dry_run),
            "halted": bool(record.get("halted")),
            "reasons": list(record.get("reasons", []) or []),
            "planned": int(record.get("orders_planned", 0) or 0),
            "submitted": int(record.get("orders_submitted", 0) or 0),
            "equity": float(record.get("account_equity", 0) or 0),
            "ts": record.get("ts"),
        }


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


# Phase 26 — News sentiment endpoint. Backed by Finnhub /company-news
# with a 15-min in-process cache. When FINNHUB_API_KEY is unset the
# endpoint still returns 200 with ``label: "neutral"`` and confidence 0,
# so the cockpit UI degrades gracefully instead of erroring out.
@app.get("/api/news-sentiment/{symbol}")
async def api_news_sentiment(symbol: str) -> dict[str, Any]:
    from packages.data.finnhub_news import get_news_client

    client = get_news_client()
    sentiment = await client.score_symbol(symbol)
    return sentiment.to_dict()


@app.get("/api/news-sentiment")
async def api_news_sentiment_batch(symbols: str = "") -> dict[str, Any]:
    """Batch news-sentiment lookup. ``symbols`` is comma-separated.

    Returns a map of upper-cased ticker -> sentiment payload. Each
    lookup hits the same 15-min cache as the singular endpoint, so
    repeated calls on the same set are cheap.
    """
    from packages.data.finnhub_news import get_news_client

    tickers = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    client = get_news_client()
    results: dict[str, Any] = {}
    for sym in tickers:
        sentiment = await client.score_symbol(sym)
        results[sym] = sentiment.to_dict()
    # Capture stats AFTER the lookups so the hit/miss counts reflect
    # this batch (not the state at request entry).
    return {"results": results, "stats": client.stats()}


# Phase 27 — Insider-transactions signal endpoint. Backed by Finnhub
# /stock/insider-transactions with a 30-min in-process cache. Same
# graceful-degrade contract as the news endpoint: no API key → 200
# with ``label: "neutral"`` and confidence 0.
@app.get("/api/insider-signal/{symbol}")
async def api_insider_signal(symbol: str) -> dict[str, Any]:
    from packages.data.finnhub_insider import get_insider_client

    client = get_insider_client()
    signal = await client.score_symbol(symbol)
    return signal.to_dict()


@app.get("/api/insider-signal")
async def api_insider_signal_batch(symbols: str = "") -> dict[str, Any]:
    """Batch insider-signal lookup. ``symbols`` is comma-separated.

    Returns ``{results: {SYM: payload}, stats: {...}}``. The 30-min
    cache means repeated lookups on the same set are essentially free.
    """
    from packages.data.finnhub_insider import get_insider_client

    tickers = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    client = get_insider_client()
    results: dict[str, Any] = {}
    for sym in tickers:
        signal = await client.score_symbol(sym)
        results[sym] = signal.to_dict()
    return {"results": results, "stats": client.stats()}


# Phase 28 — Learning / trade-journal endpoints.
#
# These read directly from ``data/learning/outcomes.jsonl`` (an append-
# only journal written by the outcome labeler). The labeler runs out-of-
# band (a script, cron, or the autonomy loop); the cockpit only *reads*
# the journal so the page is always cheap.

@app.get("/learning", response_class=HTMLResponse)
def learning_page() -> HTMLResponse:
    """Phase 28 — trade journal + per-agent win-rate dashboard."""
    return _render("learning.html")


@app.get("/api/learning/summary")
def api_learning_summary() -> dict[str, Any]:
    """Top-line stats + per-regime breakdown + per-agent scores.

    Returns an empty (but well-formed) payload when no outcomes have
    been labeled yet, so the page can render gracefully on first run.
    """
    from packages.learning.outcome_labeler import (
        DEFAULT_OUTCOMES_PATH,
        load_outcomes,
        per_agent_scores,
        summary_stats,
    )

    rows = load_outcomes(DEFAULT_OUTCOMES_PATH)
    return {
        "summary": summary_stats(rows),
        "agents": [s.to_dict() for s in per_agent_scores(rows)],
        "total_rows": len(rows),
    }


@app.get("/api/learning/picks")
def api_learning_picks(
    limit: int = 200,
    symbol: str = "",
    regime: str = "",
) -> dict[str, Any]:
    """Sortable trade-journal table data.

    Returns the most recent ``limit`` labeled picks (default 200),
    optionally filtered by symbol or regime. Sorted by ts desc.
    """
    from packages.learning.outcome_labeler import (
        DEFAULT_OUTCOMES_PATH,
        load_outcomes,
    )

    rows = load_outcomes(DEFAULT_OUTCOMES_PATH)
    sym_filter = symbol.strip().upper()
    reg_filter = regime.strip().lower()
    if sym_filter:
        rows = [r for r in rows if (r.get("symbol") or "").upper() == sym_filter]
    if reg_filter:
        rows = [r for r in rows if (r.get("regime_at_pick") or "").lower() == reg_filter]
    rows.sort(key=lambda r: r.get("ts") or "", reverse=True)
    return {"picks": rows[: max(1, min(limit, 1000))], "count": len(rows)}


class _BackfillRequest(BaseModel):
    max_picks: int | None = None


@app.post("/api/learning/backfill")
async def api_learning_backfill(req: _BackfillRequest | None = None) -> dict[str, Any]:
    """Trigger the outcome labeler to walk predictions.jsonl now.

    Idempotent: already-labeled picks are skipped. Bounded by
    ``req.max_picks`` so the cockpit can show progress on a one-button
    backfill without blocking the event loop for too long.
    """
    from packages.data.adapters.yfinance import YFinanceAdapter
    from packages.learning.outcome_labeler import backfill_outcomes

    max_picks = (req.max_picks if req else None)
    adapter = YFinanceAdapter()
    try:
        report = await backfill_outcomes(adapter, max_picks=max_picks)
    finally:
        await adapter.aclose()
    return {"report": report.to_dict()}


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


class _FloatCapBody(BaseModel):
    """Payload for setting the Robinhood live float cap."""

    cap_usd: float


@app.get("/api/onboarding/robinhood/cap")
def api_robinhood_cap_get() -> dict[str, Any]:
    """Return the active Robinhood live float cap and its absolute max.

    The cap is the user's hard blast-radius limit on live deployment.
    Reads through the onboarding store so it stays the single source of
    truth (defaults to $300 when unset)."""
    from packages.cockpit.onboarding import (
        ABSOLUTE_MAX_FLOAT_USD,
        clamp_float_cap,
        load_onboarding,
    )

    state = load_onboarding()
    return {
        "cap_usd": clamp_float_cap(state.live_float_cap_usd),
        "absolute_max_usd": ABSOLUTE_MAX_FLOAT_USD,
        "default_usd": 300.0,
    }


@app.post("/api/onboarding/robinhood/cap")
def api_robinhood_cap_set(body: _FloatCapBody) -> dict[str, Any]:
    """Set the Robinhood live float cap, clamped server-side.

    The cap is clamped into ``[0, ABSOLUTE_MAX_FLOAT_USD]``; NaN / inf /
    non-numeric values are rejected with 400. We never persist a value
    that could disable the ceiling. Persisted via the onboarding store so
    ``resolve_float_cap`` (the broker's enforcement path) picks it up."""
    import math

    from packages.cockpit.onboarding import (
        ABSOLUTE_MAX_FLOAT_USD,
        clamp_float_cap,
        load_onboarding,
        save_onboarding,
    )

    try:
        requested = float(body.cap_usd)
    except (TypeError, ValueError) as err:
        raise HTTPException(
            status_code=400, detail="cap_usd must be a number"
        ) from err
    if not math.isfinite(requested):
        raise HTTPException(
            status_code=400, detail="cap_usd must be a finite number"
        )
    if requested < 0:
        raise HTTPException(status_code=400, detail="cap_usd must be >= 0")

    clamped = clamp_float_cap(requested)
    state = load_onboarding()
    state.live_float_cap_usd = clamped
    save_onboarding(state)
    return {
        "cap_usd": clamped,
        "requested_usd": requested,
        "clamped": clamped != requested,
        "absolute_max_usd": ABSOLUTE_MAX_FLOAT_USD,
    }


# ---------------------------------------------------------------------------
# /api/onboarding/robinhood/* -- the "Connect your agent" OAuth flow
#
# Backed by ``packages/execution/robinhood.py`` (begin_auth / complete_auth
# / disconnect / is_connected). The flow is OAuth 2.1 authorization-code
# with PKCE against Robinhood's MCP server. We default the broker to
# SHADOW mode and a $300 float cap regardless of connection state -- this
# only wires up *auth*, never flips the user to live trading.
#
#   POST /api/onboarding/robinhood/connect    -> returns authorize URL
#   GET  /callback (loopback redirect)         -> finishes the exchange
#   POST /api/onboarding/robinhood/finish      -> manual code/state paste
#   POST /api/onboarding/robinhood/disconnect  -> wipe tokens
#   GET  /api/onboarding/robinhood/status      -> connected? + rh_status
# ---------------------------------------------------------------------------


class _RhFinishBody(BaseModel):
    """Manual-paste fallback when the loopback redirect can't reach us
    (some networks / the Cursor-style 403-on-loopback case). The user
    pastes the full redirect URL or the code+state from it."""

    code: str = ""
    state: str = ""
    redirect_url: str = ""


@app.post("/api/onboarding/robinhood/connect")
def api_rh_connect() -> dict[str, Any]:
    """Start the Robinhood OAuth flow and return the authorize URL.

    The cockpit opens this URL in the user's browser; Robinhood prompts
    them to open/confirm their Agentic account and approve, then redirects
    back to our loopback callback. Returns ``ok=False`` with a message on
    discovery/registration failure so the UI shows a clear error instead
    of a dead button.
    """
    from packages.execution.robinhood import begin_auth

    try:
        pending = begin_auth()
    except BrokerError as exc:
        return {"ok": False, "error": str(exc)}
    return {
        "ok": True,
        "authorize_url": pending.authorize_url,
        "redirect_uri": pending.redirect_uri,
        "state": pending.state,
    }


@app.get("/callback", response_class=HTMLResponse)
def api_rh_callback(request: Request) -> HTMLResponse:
    """Loopback redirect target for the OAuth flow.

    Robinhood redirects the browser here with ``?code=...&state=...``.
    We finish the token exchange server-side and render a tiny
    self-contained page telling the user to return to the cockpit.
    """
    from packages.execution.robinhood import complete_auth

    params = request.query_params
    code = params.get("code", "")
    state = params.get("state", "")
    err = params.get("error", "")

    if err:
        return HTMLResponse(
            _rh_callback_page(
                ok=False,
                msg=f"Robinhood returned an error: {err}. "
                "Close this tab and try Connect again.",
            ),
            status_code=400,
        )
    if not code:
        return HTMLResponse(
            _rh_callback_page(
                ok=False, msg="Missing authorization code in callback."
            ),
            status_code=400,
        )
    try:
        complete_auth(code=code, state=state)
    except BrokerError as exc:
        return HTMLResponse(
            _rh_callback_page(ok=False, msg=str(exc)), status_code=400
        )

    # Mark onboarding status granted on a successful connect (the token
    # check on next probe will confirm; this gives instant UI feedback).
    try:
        from packages.cockpit.onboarding import (
            load_onboarding,
            save_onboarding,
        )

        st = load_onboarding()
        st.robinhood_status = "granted"  # type: ignore[assignment]
        save_onboarding(st)
    except Exception:  # pragma: no cover - UI nicety, never fatal
        pass

    return HTMLResponse(
        _rh_callback_page(
            ok=True,
            msg="Your agent is connected to Robinhood. "
            "You can close this tab and return to the cockpit.",
        )
    )


def _rh_callback_page(*, ok: bool, msg: str) -> str:
    """Minimal standalone HTML for the OAuth callback landing page."""
    color = "#16a34a" if ok else "#dc2626"
    icon = "&#10003;" if ok else "&#10007;"
    title = "Connected" if ok else "Connection failed"
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>Robinhood &middot; {title}</title></head>"
        "<body style='font-family:system-ui,sans-serif;background:#0b0b0f;"
        "color:#e5e7eb;display:flex;min-height:100vh;align-items:center;"
        "justify-content:center;margin:0'>"
        "<div style='max-width:420px;text-align:center;padding:32px'>"
        f"<div style='font-size:48px;color:{color}'>{icon}</div>"
        f"<h1 style='font-size:20px;margin:12px 0 8px'>{title}</h1>"
        f"<p style='color:#9ca3af;line-height:1.5'>{msg}</p>"
        "</div></body></html>"
    )


@app.post("/api/onboarding/robinhood/finish")
def api_rh_finish(body: _RhFinishBody) -> dict[str, Any]:
    """Manual-paste completion fallback.

    If the browser redirect can't reach the cockpit's loopback listener,
    the user pastes the full redirect URL (or the code+state) here and we
    finish the exchange. Accepts either ``code``+``state`` directly or a
    ``redirect_url`` we parse them out of.
    """
    from urllib.parse import parse_qs, urlsplit

    from packages.execution.robinhood import complete_auth

    code, state = body.code, body.state
    if body.redirect_url and not (code and state):
        q = parse_qs(urlsplit(body.redirect_url).query)
        code = code or (q.get("code", [""])[0])
        state = state or (q.get("state", [""])[0])
    if not code:
        raise HTTPException(status_code=400, detail="missing code")
    try:
        complete_auth(code=code, state=state)
    except BrokerError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        from packages.cockpit.onboarding import (
            load_onboarding,
            save_onboarding,
        )

        st = load_onboarding()
        st.robinhood_status = "granted"  # type: ignore[assignment]
        save_onboarding(st)
    except Exception:  # pragma: no cover
        pass
    return {"ok": True}


@app.post("/api/onboarding/robinhood/disconnect")
def api_rh_disconnect() -> dict[str, Any]:
    """Wipe Robinhood tokens + client_id. Backs 'Disconnect Robinhood'."""
    from packages.execution.robinhood import disconnect

    disconnect()
    return {"ok": True, "connected": False}


@app.get("/api/onboarding/robinhood/status")
def api_rh_status() -> dict[str, Any]:
    """Report whether a usable Robinhood token is stored, plus the cached
    onboarding status and the ACTIVE-broker safety posture. Read-only;
    never touches the network.

    The ``active_broker`` block surfaces which backend the autonomy loop
    will trade through (default: alpaca_paper), whether it's shadow or
    live, the resolved float cap, and the targeted agentic account number
    MASKED to its last 4 digits -- so the user can confirm at a glance
    that selecting Robinhood didn't silently enable live trading."""
    from packages.cockpit.onboarding import load_onboarding
    from packages.execution.broker_factory import active_broker_status
    from packages.execution.robinhood import is_connected

    return {
        "connected": is_connected(),
        "robinhood_status": load_onboarding().robinhood_status,
        "active_broker": active_broker_status(),
    }


@app.get("/api/onboarding/robinhood/snapshot")
async def api_rh_snapshot(refresh: bool = True) -> dict[str, Any]:
    """Live, read-only snapshot of the connected Robinhood account.

    Surfaces buying power, cash, total equity, and current positions so
    the dashboard can show the user's real account and the AI agent has
    live account context when reasoning about the market.

    ALWAYS read-only -- this never places, reviews, or cancels an order,
    so it is safe even while the bot runs in shadow mode. When
    ``refresh=true`` (default) it fetches fresh data from Robinhood and
    updates the in-memory cache the agent reads; ``refresh=false``
    returns the last cached snapshot without a network call.
    """
    if refresh:
        snap = await _refresh_robinhood_snapshot()
    else:
        snap = latest_robinhood_snapshot()
    if snap is None:
        # Not connected / never fetched -- report a stable empty shape.
        from packages.execution.robinhood import is_connected

        return {
            "connected": is_connected(),
            "mode": "shadow",
            "as_of": None,
            "accounts": [],
            "portfolio": None,
            "positions": [],
            "buying_power": None,
            "cash": None,
            "total_equity": None,
            "errors": [],
        }
    return snap


@app.post("/api/onboarding/robinhood/select-account")
async def api_rh_select_account() -> dict[str, Any]:
    """Discover + persist the agentic-allowed Robinhood account number.

    Calls ``get_accounts`` (read-only) and stores the single account
    flagged ``agentic_allowed=true`` so the broker targets it for reads +
    orders. Robinhood rejects trades on non-agentic accounts at the API
    level, so this MUST succeed before Robinhood live trading is usable.
    Returns the masked account number on success; ``ok=False`` with a
    reason when not connected or no agentic account exists (fail safe)."""
    from packages.execution.robinhood import (
        ensure_agentic_account_number,
        is_connected,
    )

    if not is_connected():
        return {"ok": False, "reason": "not_connected"}
    acct = await ensure_agentic_account_number()
    if not acct:
        return {"ok": False, "reason": "no_agentic_account"}
    return {"ok": True, "account_masked": "••••" + acct[-4:]}


class _BrokerBackendBody(BaseModel):
    """Payload for selecting the active broker backend."""

    backend: str


@app.get("/api/onboarding/broker-backend")
def api_broker_backend_get() -> dict[str, Any]:
    """Return the selected + effective active broker backend and posture.

    Read-only. ``backend`` is what the user selected; the
    ``active_broker`` block reflects what actually resolved (which may
    have failed safe back to alpaca_paper)."""
    from packages.cockpit.onboarding import load_onboarding
    from packages.execution.broker_factory import active_broker_status

    return {
        "backend": load_onboarding().broker_backend,
        "active_broker": active_broker_status(),
    }


@app.post("/api/onboarding/broker-backend")
def api_broker_backend_set(body: _BrokerBackendBody) -> dict[str, Any]:
    """Select the active broker backend (``alpaca_paper`` | ``robinhood``).

    Selecting ``robinhood`` only changes WHICH broker the loop trades
    through -- it does NOT enable live trading (the broker still runs in
    SHADOW unless the resolve_mode gate + ENABLE_LIVE_TRADING authorize).
    Rejects unknown backends with 400."""
    from packages.cockpit.onboarding import (
        VALID_BROKER_BACKENDS,
        load_onboarding,
        save_onboarding,
    )
    from packages.execution.broker_factory import active_broker_status

    backend = body.backend.strip().lower()
    if backend not in VALID_BROKER_BACKENDS:
        raise HTTPException(
            status_code=400,
            detail=f"backend must be one of {list(VALID_BROKER_BACKENDS)}",
        )
    state = load_onboarding()
    state.broker_backend = backend  # type: ignore[assignment]
    save_onboarding(state)
    return {"ok": True, "backend": backend, "active_broker": active_broker_status()}


# ---------------------------------------------------------------------------
# Robinhood go-live readiness + mode control
#
# ``/readiness`` is the Robinhood analogue of ``/api/promote``: an ordered,
# plain-language checklist of every precondition that must hold before a
# real Robinhood order can execute, each with a boolean + human label + a
# "what to do" hint, plus an overall ``ready`` flag and the single most
# important next action. ``/mode`` flips ``rh_mode`` -- going LIVE requires
# an explicit confirm AND server-side re-validation of that checklist;
# going SHADOW is always allowed (turning OFF live never needs a gate).
# ---------------------------------------------------------------------------


def _enable_live_flag() -> bool:
    """True when ENABLE_LIVE_TRADING is armed in the environment."""
    return os.getenv("ENABLE_LIVE_TRADING", "").strip().lower() in {
        "true",
        "1",
        "yes",
        "on",
    }


def _compute_robinhood_readiness() -> dict[str, Any]:
    """Build the ordered Robinhood go-live checklist (read-only, fail-safe).

    Every check is independent and never raises -- a failure to read any
    one precondition is reported as that step being unmet, never a 500.
    The promotion gate is the SAME §16 gate the Alpaca path uses.
    """
    from packages.cockpit.onboarding import load_onboarding
    from packages.execution import robinhood as rh
    from packages.execution.broker_factory import BACKEND_ROBINHOOD

    # Read each source of truth defensively.
    try:
        state = load_onboarding()
    except Exception:  # pragma: no cover - defensive
        state = None
    rh_mode = getattr(state, "rh_mode", "shadow") if state else "shadow"
    backend = getattr(state, "broker_backend", "alpaca_paper") if state else "alpaca_paper"
    cap = 0.0
    try:
        cap = rh.resolve_float_cap()
    except Exception:  # pragma: no cover - defensive
        cap = 0.0

    try:
        connected = rh.is_connected()
    except Exception:  # pragma: no cover - defensive
        connected = False

    try:
        account = rh.resolve_agentic_account_number()
    except Exception:  # pragma: no cover - defensive
        account = None

    # Funded state: only observable from a cached snapshot (never force a
    # network call here -- this endpoint must stay cheap + read-only).
    buying_power: float | None = None
    try:
        snap = latest_robinhood_snapshot()
        if isinstance(snap, dict):
            bp = snap.get("buying_power")
            buying_power = float(bp) if bp is not None else None
    except Exception:  # pragma: no cover - defensive
        buying_power = None

    enable_flag = _enable_live_flag()

    try:
        promote = _compute_promote_payload()
        gate_passed = bool(promote.get("readiness", {}).get("ready", False))
    except Exception:  # pragma: no cover - defensive
        gate_passed = False

    checks: list[dict[str, Any]] = [
        {
            "id": "connected",
            "ok": bool(connected),
            "label": "Robinhood account connected",
            "todo": "Click “Connect Robinhood” and sign in to link your account.",
        },
        {
            "id": "account",
            "ok": bool(account),
            "label": "AI trading account found",
            "todo": "Open your AI (Agentic) account in the Robinhood app, then click “Find my account.”",
        },
        {
            "id": "funded",
            "ok": (buying_power is None) or (buying_power > 0),
            "informational": buying_power is None,
            "label": (
                "Your AI trading account has money to invest"
                if buying_power is not None
                else "Account funding (we'll check once connected)"
            ),
            "todo": "Add money to your AI trading account in the Robinhood app.",
        },
        {
            "id": "backend",
            "ok": backend == BACKEND_ROBINHOOD,
            "label": "Robinhood is the active broker",
            "todo": "Switch the active broker to Robinhood in the control center.",
        },
        {
            "id": "cap",
            "ok": cap > 0,
            "label": "A spending cap is set",
            "todo": "Set a dollar cap (how much real money the AI may use).",
        },
        {
            "id": "enable_live",
            "ok": enable_flag,
            "label": "Live trading is armed",
            "todo": "Arm live trading on the Promote page after the practice checks pass.",
        },
        {
            "id": "promotion_gate",
            "ok": gate_passed,
            "label": "Practice results are good enough to go live",
            "todo": "Keep running in practice mode until the readiness checks on the Promote page pass.",
        },
        {
            "id": "rh_mode_live",
            "ok": rh_mode == "live",
            "label": "Robinhood set to LIVE",
            "todo": "Flip the switch to LIVE (you'll be asked to confirm).",
        },
    ]

    # ``ready`` ignores the final rh_mode flip + informational funding so it
    # reflects "is it SAFE to flip to live now" -- the mode endpoint uses
    # this exact predicate as its server-side gate.
    blocking = [
        c
        for c in checks
        if c["id"] != "rh_mode_live"
        and not c.get("informational")
        and not c["ok"]
    ]
    ready = len(blocking) == 0

    # Next action = first unmet step in order (including the rh_mode flip
    # once everything else is green).
    next_step = "You're ready - flip the switch to LIVE when you want real trading."
    next_id: str | None = None
    for c in checks:
        if c.get("informational"):
            continue
        if not c["ok"]:
            next_step = c["todo"]
            next_id = c["id"]
            break

    # Map the first unmet check to a single concrete UI action so the
    # dashboard can offer ONE button that always does the right next
    # thing. ``kind`` tells the frontend which handler to fire; ``label``
    # is the plain-language button text. Steps the user must do in the
    # Robinhood app (funding) or that require the §16 practice window
    # (promotion_gate) are marked ``manual`` -- the button explains and
    # links rather than pretending to one-click them.
    actions: dict[str, dict[str, str]] = {
        "connected": {"kind": "connect", "label": "Connect Robinhood"},
        "account": {"kind": "find_account", "label": "Find my AI account"},
        "funded": {"kind": "manual", "label": "Add money in the Robinhood app"},
        "backend": {"kind": "select_robinhood", "label": "Make Robinhood the active broker"},
        "cap": {"kind": "scroll_cap", "label": "Set a spending cap"},
        "enable_live": {"kind": "goto_promote", "label": "Arm live trading"},
        "promotion_gate": {"kind": "manual", "label": "Keep practicing to pass the gate"},
    }
    if next_id is None:
        # Everything is green except (optionally) the final LIVE flip.
        next_action = {"kind": "go_live", "label": "Go LIVE (real money)"}
    else:
        next_action = actions.get(
            next_id, {"kind": "manual", "label": "See the checklist"}
        )

    return {
        "ready": ready,
        "next_step": next_step,
        "next_action": next_action,
        "rh_mode": rh_mode,
        "broker_backend": backend,
        "cap_usd": cap,
        "buying_power": buying_power,
        "account_masked": ("••••" + account[-4:]) if account else None,
        "checklist": checks,
    }


@app.get("/api/onboarding/robinhood/readiness")
def api_rh_readiness() -> dict[str, Any]:
    """Ordered Robinhood go-live checklist + overall ready flag + next step.

    Read-only and fail-safe: never raises, never touches the network, and
    reports each precondition as a plain-language step the operator can act
    on. Mirrors ``/api/promote`` but for the Robinhood chain."""
    return _compute_robinhood_readiness()


class _RhModeBody(BaseModel):
    """Payload for setting Robinhood shadow/live mode."""

    mode: str
    confirm: bool = False


@app.post("/api/onboarding/robinhood/mode")
def api_rh_mode_set(body: _RhModeBody) -> dict[str, Any]:
    """Set ``rh_mode`` to ``shadow`` or ``live``.

    SHADOW is always allowed (turning live OFF never needs a gate). Going
    LIVE requires BOTH an explicit ``confirm=true`` AND a server-side
    re-validation of the go-live checklist; if either fails we refuse with
    a clear reason list and leave the mode unchanged. This is the deliberate
    confirmation barrier -- selecting the Robinhood backend alone can never
    flip this; only this endpoint, with confirm + readiness, can."""
    from packages.cockpit.onboarding import (
        VALID_RH_MODES,
        load_onboarding,
        save_onboarding,
    )

    mode = body.mode.strip().lower()
    if mode not in VALID_RH_MODES:
        raise HTTPException(
            status_code=400,
            detail=f"mode must be one of {list(VALID_RH_MODES)}",
        )

    if mode == "live":
        if not body.confirm:
            raise HTTPException(
                status_code=400,
                detail="going live requires explicit confirmation",
            )
        readiness = _compute_robinhood_readiness()
        if not readiness["ready"]:
            unmet = [
                c["label"]
                for c in readiness["checklist"]
                if c["id"] != "rh_mode_live"
                and not c.get("informational")
                and not c["ok"]
            ]
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "not_ready",
                    "reasons": unmet,
                    "next_step": readiness["next_step"],
                },
            )

    state = load_onboarding()
    state.rh_mode = mode  # type: ignore[assignment]
    save_onboarding(state)
    return {"ok": True, "rh_mode": mode}


# ---------------------------------------------------------------------------
# /api/research-sweep -- Phase 1D
#
# The dashboard 'Research Candidates' tile polls these endpoints. GET
# returns the last persisted sweep (read-only, fast). POST kicks off a
# background sweep so the user can refresh on demand without blocking
# the request thread.
# ---------------------------------------------------------------------------


@app.get("/api/research-sweep")
def api_research_sweep_get() -> dict[str, Any]:
    """Return the most recent persisted sweep plus live status.

    Shape: ``{"sweep": <SweepResult dict or None>, "status": {...}}``.
    The dashboard treats ``sweep=None`` as 'no sweep ever ran'
    (showing a 'Run sweep' CTA) versus ``status='failed'`` (showing
    the error).
    """
    from packages.agents.research_sweep import load_status, load_sweep

    return {"sweep": load_sweep(), "status": load_status()}


@app.post("/api/research-sweep/run")
def api_research_sweep_run() -> dict[str, Any]:
    """Fire a background sweep. Returns immediately with current status.

    The actual sweep persists its own status + result file; the
    dashboard polls ``GET /api/research-sweep`` to see progress.
    """
    from packages.agents.research_sweep import (
        kick_off_background,
        load_status,
        save_status,
    )

    # Pre-mark running so the dashboard reflects the kick-off even
    # before the background task gets CPU time.
    save_status("running", detail="kick-off requested")
    kick_off_background()
    return {"ok": True, "status": load_status()}


# ---------------------------------------------------------------------------
# Phase 10: per-source health / contribution dashboard.
# ---------------------------------------------------------------------------


# Stable display order + human-readable labels for each source.
# Sources not in this map still render at the bottom, alphabetically.
_SOURCE_DISPLAY: dict[str, dict[str, str]] = {
    "rss_news":           {"label": "RSS news (Yahoo / MarketWatch / Seeking Alpha)",       "tier": "news"},
    "yahoo_news":         {"label": "Yahoo Finance per-ticker (news + analyst + insider)",  "tier": "news"},
    "sec_form4":          {"label": "SEC EDGAR Form 4 (insider transactions)",              "tier": "filings"},
    "reddit_rich":        {"label": "Reddit (tiered roster: SecurityAnalysis, investing...)","tier": "social"},
    "reddit_per_ticker":  {"label": "Reddit per-ticker subs (discovered)",                  "tier": "social"},
    "stocktwits":         {"label": "StockTwits trending",                                  "tier": "social"},
}


@app.get("/data-sources", response_class=HTMLResponse)
def data_sources_page() -> HTMLResponse:
    """Phase 10: per-source health + contribution dashboard.

    Renders a static template; the page polls
    ``/api/data-sources/snapshot`` every 30s for fresh telemetry.
    """
    return _render("data_sources.html")


@app.get("/api/data-sources/snapshot")
def api_data_sources_snapshot() -> dict[str, Any]:
    """Return per-source telemetry from the most recent sweep.

    Shape:
        {
            "sources": [
                {"name": "yahoo_news", "label": "...", "tier": "news",
                 "ok": bool, "count": int, "latency_ms": float},
                ...
            ],
            "sweep_started_at": iso str | "",
            "sweep_status": "done" | "failed" | "running" | "",
            "candidate_count": int,
            "subreddit_roster": [
                {"name": str, "tier": str, "multiplier": float},
                ...
            ],
        }
    """
    from packages.agents.reddit_trust import (
        DEFAULT_SWEEP_ROSTER,
        quality_for,
    )
    from packages.agents.research_sweep import load_sweep

    sweep = load_sweep() or {}
    sources_meta = sweep.get("sources_meta") or {}

    # Build the source list. Use the canonical order from
    # _SOURCE_DISPLAY; unknown sources tack onto the end.
    rendered: list[dict[str, Any]] = []
    seen: set[str] = set()
    for key, display in _SOURCE_DISPLAY.items():
        meta = sources_meta.get(key) or {}
        rendered.append(
            {
                "name": key,
                "label": display["label"],
                "tier": display["tier"],
                "ok": bool(meta.get("ok", False)),
                "count": int(meta.get("count", 0)),
                "latency_ms": float(meta.get("latency_ms", 0.0)),
                "present": bool(meta),
            }
        )
        seen.add(key)
    for key, meta in sorted(sources_meta.items()):
        if key in seen:
            continue
        rendered.append(
            {
                "name": key,
                "label": key.replace("_", " ").title(),
                "tier": "other",
                "ok": bool(meta.get("ok", False)),
                "count": int(meta.get("count", 0)),
                "latency_ms": float(meta.get("latency_ms", 0.0)),
                "present": True,
            }
        )

    roster = [
        {
            "name": s,
            "tier": quality_for(s).tier,
            "multiplier": round(quality_for(s).multiplier, 2),
        }
        for s in DEFAULT_SWEEP_ROSTER
    ]

    return {
        "sources": rendered,
        "sweep_started_at": sweep.get("started_at", ""),
        "sweep_status": sweep.get("status", ""),
        "candidate_count": len(sweep.get("candidates") or []),
        "subreddit_roster": roster,
    }


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
    # Phase 36b: mark that the user has explicitly touched the controls
    # so future boots use the resume path, not the first-boot auto-start.
    state.paper_loop_user_touched = True
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
    # Phase 36b: also mark touched so we don't auto-start on next boot.
    state.paper_loop_user_touched = True
    state = record_action(state, "Paper loop stopped")
    save_state(state)
    return info


@app.get("/api/trading/status")
def api_trading_status() -> dict[str, Any]:
    return job_mgr.status(PAPER_LOOP_KIND).to_dict()


@app.get("/api/trading/unified-snapshot")
async def api_trading_unified_snapshot(
    decisions_limit: int = 10,
) -> dict[str, Any]:
    """Phase 36a — single payload joining Alpaca state + shadow context.

    The cockpit's primary trading page should show one unified story
    per cycle instead of forcing the operator to flip between /trading
    and /shadow. This endpoint composes:

    * ``account``     — latest Alpaca paper account snapshot (equity, BP)
    * ``positions``   — currently open Alpaca paper positions
    * ``broker``      — reachability + key presence (subset of broker-health)
    * ``loop``        — paper_loop job status (running / stopped / crashed)
    * ``cycles``      — last 5 paper-trade cycle records
    * ``decisions``   — last N per-cycle shadow decisions (the *reasoning*)
    * ``window``      — 14-day shadow window progress (day X of 14)
    * ``streak``      — clean-paper-days streak counter

    Each section is best-effort: if one source raises, we degrade that
    field to ``None`` rather than failing the whole response, so the UI
    can still render the panels that *do* have data.
    """
    out: dict[str, Any] = {
        "account": None,
        "positions": None,
        "broker": None,
        "loop": None,
        "cycles": None,
        "decisions": None,
        "window": None,
        "streak": None,
        "robinhood": None,
        "active_broker": None,
        "errors": {},
    }

    # Which broker backend is active right now + its shadow/live posture
    # (read-only; never enables trading). Powers the dual-pipeline view.
    try:
        from packages.execution.broker_factory import active_broker_status

        out["active_broker"] = active_broker_status()
    except Exception as exc:  # pragma: no cover — defensive
        out["errors"]["active_broker"] = f"{exc.__class__.__name__}: {exc}"

    # Alpaca account + positions (best-effort)
    try:
        out["account"] = latest_account_snapshot()
    except Exception as exc:  # pragma: no cover — defensive
        out["errors"]["account"] = f"{exc.__class__.__name__}: {exc}"

    # Live Robinhood account snapshot (read-only) -- gives the agent the
    # user's REAL buying power / cash / equity / positions when connected.
    # Cached by the autonomy warmup tick; None when not connected.
    try:
        out["robinhood"] = latest_robinhood_snapshot()
    except Exception as exc:  # pragma: no cover — defensive
        out["errors"]["robinhood"] = f"{exc.__class__.__name__}: {exc}"
    try:
        out["positions"] = latest_positions()
    except Exception as exc:  # pragma: no cover
        out["errors"]["positions"] = f"{exc.__class__.__name__}: {exc}"

    # Broker health (subset of /api/trading/broker-health)
    try:
        key_id = os.environ.get("ALPACA_PAPER_KEY_ID", "")
        secret = os.environ.get("ALPACA_PAPER_SECRET", "")
        base_url = os.environ.get(
            "ALPACA_BASE_URL", "https://paper-api.alpaca.markets"
        )
        keys_present = bool(key_id) and bool(secret)
        reachable = False
        if keys_present:
            try:
                reachable = await AlpacaPaperBroker().health()
            except Exception:
                reachable = False
        out["broker"] = {
            "keys_present": keys_present,
            "reachable": reachable,
            "base_url": base_url,
        }
    except Exception as exc:  # pragma: no cover
        out["errors"]["broker"] = f"{exc.__class__.__name__}: {exc}"

    # Paper-loop job status
    try:
        out["loop"] = job_mgr.status(PAPER_LOOP_KIND).to_dict()
    except Exception as exc:  # pragma: no cover
        out["errors"]["loop"] = f"{exc.__class__.__name__}: {exc}"

    # Recent shadow decisions (the per-cycle reasoning ledger). This is
    # what makes the unified view valuable — each fill is paired with
    # the candidate funnel + scoring that produced it.
    try:
        from packages.paper.decisions import load_recent

        n = max(1, min(int(decisions_limit), 50))
        out["decisions"] = load_recent(limit=n)
    except Exception as exc:
        out["errors"]["decisions"] = f"{exc.__class__.__name__}: {exc}"

    # 14-day shadow window progress (day X of 14 counter for soak gating)
    try:
        from packages.paper.decisions import window_status

        out["window"] = window_status()
    except Exception as exc:
        out["errors"]["window"] = f"{exc.__class__.__name__}: {exc}"

    # Clean-paper-days streak
    try:
        out["streak"] = compute_paper_streak(PAPER_LOG).to_dict()
    except Exception as exc:  # pragma: no cover
        out["errors"]["streak"] = f"{exc.__class__.__name__}: {exc}"

    # Last 5 paper-trade cycle records (from runs.jsonl)
    try:
        from packages.paper import iter_paper_runs

        rows = list(iter_paper_runs())
        rows.reverse()
        out["cycles"] = rows[:5]
    except Exception as exc:
        out["errors"]["cycles"] = f"{exc.__class__.__name__}: {exc}"

    return out


@app.get("/api/trading/broker-health")
async def api_trading_broker_health() -> dict[str, Any]:
    """Phase 35c: report Alpaca paper-broker reachability + key presence.

    Lets the UI show a clear status before the operator clicks "Start
    loop" or "Run one cycle". Returns:

    * ``keys_present`` — ALPACA_PAPER_KEY_ID and ALPACA_PAPER_SECRET set
    * ``reachable``    — a GET /v2/account against the paper API returned 200
    * ``base_url``     — the resolved paper-broker URL
    * ``error``        — short string when unreachable (else ``None``)
    """
    key_id = os.environ.get("ALPACA_PAPER_KEY_ID", "")
    secret = os.environ.get("ALPACA_PAPER_SECRET", "")
    base_url = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    keys_present = bool(key_id) and bool(secret)
    reachable = False
    error: str | None = None
    if not keys_present:
        error = "ALPACA_PAPER_KEY_ID / ALPACA_PAPER_SECRET not set in environment"
    else:
        try:
            broker = AlpacaPaperBroker()
            reachable = await broker.health()
            if not reachable:
                error = "Alpaca /v2/account returned non-200 — check keys / outage"
        except Exception as exc:  # pragma: no cover — defensive
            error = f"{exc.__class__.__name__}: {exc}"
    return {
        "ok": keys_present and reachable,
        "keys_present": keys_present,
        "reachable": reachable,
        "base_url": base_url,
        "error": error,
    }


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
    # Fan the run into the rolling chatter feed. Wrapped in a guard so a
    # bug in the feed can never break the trading loop.
    try:
        agent_chatter.ingest_run(payload)
    except Exception as _chatter_err:  # pragma: no cover — defensive
        log.warning("chatter ingest failed (ignored): %s", _chatter_err)
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

# Phase 20 — the scheduler is now **enabled by default** so the AIs
# actually talk in the background. The user can still disable it via
# POST /api/agents/schedule, and the cockpit pause button universally
# skips ticks. The stub backend is still the default so we don't
# require Ollama just to feel alive.
_AGENT_SCHED: dict[str, Any] = {
    "enabled": True,
    "interval_seconds": 1800,  # 30 minutes
    "use_llm": False,
    "symbols": ["SPY", "QQQ", "TLT"],
    "last_run_at": None,
    "last_run_status": None,
    "last_error": None,
    "_task": None,
}


def _agent_sched_set_symbols(symbols: list[str], reason: str) -> None:
    """Curiosity → scheduler bridge.

    Replaces the scheduler's working watchlist with the symbols the
    Curiosity agent picked. Capped at 6 to keep tick latency sane.
    Reason is logged so the audit trail tells us *why* the focus
    changed.
    """
    if not symbols:
        return
    deduped = [str(s).upper().strip() for s in symbols if s]
    deduped = list(dict.fromkeys(deduped))[:6]
    if not deduped:
        return
    prev = list(_AGENT_SCHED.get("symbols") or [])
    if prev == deduped:
        return
    _AGENT_SCHED["symbols"] = deduped
    log.info("curiosity updated scheduler symbols: %s (was %s) — %s",
             deduped, prev, reason)


def _portfolio_symbols_snapshot() -> list[str]:
    """Best-effort portfolio symbols for Curiosity to anchor on.

    Reads from the cached snapshot the dashboard uses so we don't make
    a fresh broker call on every Curiosity tick.
    """
    try:
        snap_path = Path("data/cockpit/snapshot.json")
        if not snap_path.exists():
            return []
        data = json.loads(snap_path.read_text(encoding="utf-8"))
        positions = data.get("positions") or []
        out: list[str] = []
        for p in positions:
            sym = (p.get("symbol") or "").upper().strip()
            if sym and sym not in out:
                out.append(sym)
        return out
    except Exception:
        return []


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


@app.on_event("startup")
async def _autonomy_startup() -> None:  # pragma: no cover
    """Boot the Always-On Brain.

    Wires the Curiosity → scheduler bridge so the next pipeline tick
    focuses on whatever the autonomy loop just decided is interesting,
    then starts the long-lived sweep loop. Disabled when the env var
    ``AUTONOMY_DISABLED=1`` is set (CI / tests / safe-mode).
    """
    if os.environ.get("AUTONOMY_DISABLED") == "1":
        log.info("autonomy brain disabled via AUTONOMY_DISABLED=1")
        return

    # Phase 21 + 25.3: wire the self-improvement loop with best-effort
    # price providers. The LiveQuoteCache is the single seam every
    # Phase 25 hook hits to ask "what's the last price?" — it fronts
    # Finnhub real-time REST quotes and falls back to yfinance daily
    # closes when Finnhub is unavailable (no key / network error /
    # rate limited). The cache is kept warm by the fast loop so the
    # sync `peek()` used here is virtually always within TTL.
    quote_cache = live_quotes_mod.get_default_cache()
    quote_cache.set_fallback(
        lambda symbol: autonomy_regime.default_price_provider(symbol, days=5)
    )
    if os.getenv("FINNHUB_API_KEY"):
        try:
            from packages.data.adapters.finnhub import FinnhubAdapter
            _finnhub_adapter = FinnhubAdapter()
            quote_cache.set_fetcher(
                live_quotes_mod.make_finnhub_fetcher(_finnhub_adapter)
            )
            log.info("live quotes: Finnhub REST primary, yfinance fallback")
        except Exception as exc:  # pragma: no cover — defensive
            log.warning("live quotes: Finnhub adapter init failed: %s", exc)
    else:
        log.info("live quotes: no FINNHUB_API_KEY, using yfinance fallback only")

    # Phase 25.4 — build the WS client that streams live ticks straight
    # into the cache. None when no API key. The REST path keeps working
    # if the WS disconnects, so this is purely additive freshness.
    ws_client = finnhub_ws_mod.build_default_client(
        on_tick=quote_cache.ingest_ws_tick
    )
    if ws_client is not None:
        ws_started = ws_client.start()
        log.info(
            "finnhub_ws: client built (started=%s, max_symbols=%d)",
            ws_started,
            ws_client._max_symbols,
        )
    else:
        log.info("finnhub_ws: no API key, live tick stream disabled")

    def _last_price(sym: str) -> float | None:
        # Synchronous read — the cache is kept warm by the fast loop.
        # On a cold cache (very first call) fall back to a yfinance
        # daily close so we never return None just because the cache
        # hasn't been primed yet.
        cached = quote_cache.peek(sym)
        if cached is not None and cached > 0:
            return cached
        try:
            series = autonomy_regime.default_price_provider(sym, days=5)
            if series:
                price = float(series[-1])
                # Seed the cache so subsequent peeks succeed.
                quote_cache.ingest_ws_tick(sym, price)
                return price
        except Exception:  # pragma: no cover — defensive
            pass
        return None

    # Phase 25: build the exit_rules / dip_watch tick hooks. The hooks
    # capture _last_price + a broker factory so autonomy.py never has
    # to import the broker directly (keeps it testable in isolation).
    #
    # The active broker is obtained through ``resolve_active_broker`` so the
    # loop trades through whichever backend the user selected (default:
    # Alpaca paper). Selecting Robinhood doesn't enable live -- the broker
    # still runs in SHADOW unless the resolve_mode gate + ENABLE_LIVE_TRADING
    # authorize it. The Alpaca-cred gate below only applies when the active
    # backend is Alpaca paper; a Robinhood-backed loop doesn't need Alpaca
    # keys.
    def _loop_broker() -> Any | None:
        """Resolve the active loop broker, gating on Alpaca creds only when
        the effective backend is Alpaca paper. Returns ``None`` (skip the
        tick) when the paper backend has no creds -- existing behavior."""
        from packages.execution.broker_factory import (
            BACKEND_ALPACA_PAPER,
            resolve_broker_selection,
        )

        sel = resolve_broker_selection()
        if sel.effective_backend == BACKEND_ALPACA_PAPER and not (
            os.getenv("ALPACA_PAPER_KEY_ID") and os.getenv("ALPACA_PAPER_SECRET")
        ):
            return None
        return sel.broker

    async def _phase25_exit_rules_tick() -> dict[str, Any]:
        from packages.cockpit.web.dip_watch import arm as dip_arm
        from packages.cockpit.web.exit_rules import run_tick as exit_run_tick

        broker = _loop_broker()
        if broker is None:
            return {"skipped": True, "reason": "no_broker_creds"}
        try:
            async def _submit_sell(symbol: str, qty: float) -> Any:
                from packages.execution.broker import OrderRequest
                return await broker.submit(
                    OrderRequest(symbol=symbol, side="sell", qty=qty)
                )

            # Capture qty from positions so dip_watch can re-arm same size.
            qty_map: dict[str, float] = {}
            try:
                for p in await broker.positions():
                    qty_map[p.symbol] = abs(float(p.qty))
            except Exception:
                pass

            def _on_profit(symbol: str, exit_price: float, pnl_pct: float) -> None:
                dip_arm(
                    symbol=symbol,
                    exit_price=exit_price,
                    exit_pnl_pct=pnl_pct,
                    qty=qty_map.get(symbol, 1.0),
                )

            r = await exit_run_tick(
                positions_getter=broker.positions,
                submit_sell=_submit_sell,
                on_profit_taken=_on_profit,
            )
            return {
                "evaluated": r.evaluated,
                "sells_triggered": r.sells_triggered,
                "sells_executed": r.sells_executed,
                "errors": r.errors,
            }
        finally:
            with contextlib.suppress(Exception):
                await broker.aclose()

    async def _phase25_dip_watch_tick() -> dict[str, Any]:
        from packages.cockpit.web.dip_watch import run_tick as dip_run_tick

        broker = _loop_broker()
        if broker is None:
            return {"skipped": True, "reason": "no_broker_creds"}
        try:
            async def _submit_buy(symbol: str, qty: float) -> Any:
                from packages.execution.broker import OrderRequest
                return await broker.submit(
                    OrderRequest(symbol=symbol, side="buy", qty=qty)
                )

            r = await dip_run_tick(
                price_lookup=_last_price,
                submit_buy=_submit_buy,
            )
            return {
                "checked": r.checked,
                "fired": r.fired,
                "expired": r.expired,
                "errors": r.errors,
            }
        finally:
            with contextlib.suppress(Exception):
                await broker.aclose()

    # Phase 25.3 + 25.4 — keep the live-quote cache warm AND keep the
    # WS subscription set in sync with the active symbols (portfolio +
    # dip-watchers + SPY/VIX baseline). REST refresh seeds the cache
    # immediately; the WS provides sub-second updates after that.
    # ^VIX is excluded from WS (Finnhub WS is equities-only) but still
    # refreshes via REST/yfinance.
    async def _phase25_quote_warmup_tick() -> dict[str, Any]:
        symbols: set[str] = {"SPY", "^VIX"}
        try:
            for s in (_portfolio_symbols_snapshot() or []):
                symbols.add(str(s).upper())
        except Exception:
            pass
        try:
            from packages.cockpit.web.dip_watch import snapshot as dip_snapshot
            for w in (dip_snapshot() or {}).get("watchers", []):
                sym = (w or {}).get("symbol")
                if sym:
                    symbols.add(str(sym).upper())
        except Exception:
            pass
        if not symbols:
            return {"refreshed": 0}
        results = await live_quotes_mod.refresh_symbols(
            quote_cache, sorted(symbols)
        )
        ok = sum(1 for v in results.values() if v is not None)
        ws_result: dict[str, Any] | None = None
        if ws_client is not None:
            # Finnhub WS supports US equities only — strip index symbols
            # (^VIX, ^SPX, etc.) which would just bounce as server errors.
            ws_symbols = sorted(s for s in symbols if not s.startswith("^"))
            try:
                ws_result = await ws_client.set_symbols(ws_symbols)
            except Exception as exc:  # pragma: no cover — defensive
                log.debug("finnhub_ws: set_symbols failed: %s", exc)
                ws_result = {"error": str(exc)[:240]}
        # Keep the live Robinhood account snapshot warm so the agent
        # context + dashboard read fresh buying-power / equity / positions
        # without a per-request network call. Read-only; no-op when the
        # user hasn't connected Robinhood.
        rh_ok = False
        try:
            rh_snap = await _refresh_robinhood_snapshot()
            rh_ok = bool(rh_snap and rh_snap.get("connected"))
        except Exception as exc:  # pragma: no cover — defensive
            log.debug("robinhood snapshot warmup failed: %s", exc)
        return {
            "refreshed": ok,
            "attempted": len(symbols),
            "primary": quote_cache.status()["primary_provider"],
            "ws": ws_result,
            "robinhood": rh_ok,
        }

    # Phase 28-R step 2 — EOD flattener. Closes everything at 15:55-16:05
    # ET so the bot stays pure intraday. Idempotent per session.
    from packages.execution.eod_flattener import make_flatten_tick_hook

    def _eod_broker_factory() -> Any | None:
        # Same selection seam as the autonomy ticks: the EOD flattener
        # operates on whichever backend is active (default Alpaca paper).
        return _loop_broker()

    _eod_flatten_hook = make_flatten_tick_hook(_eod_broker_factory)

    autonomy_brain.configure(
        on_curiosity_focus=_agent_sched_set_symbols,
        portfolio_symbols_getter=_portfolio_symbols_snapshot,
        price_lookup=_last_price,
        regime_price_provider=autonomy_regime.default_price_provider,
        regime_vix_provider=autonomy_regime.default_vix_provider,
        exit_rules_tick=_phase25_exit_rules_tick,
        dip_watch_tick=_phase25_dip_watch_tick,
        quote_warmup_tick=_phase25_quote_warmup_tick,
        eod_flatten_tick=_eod_flatten_hook,
    )
    started = autonomy_brain.start()
    log.info("autonomy brain startup: started=%s", started)


@app.on_event("shutdown")
async def _autonomy_shutdown() -> None:  # pragma: no cover
    autonomy_brain.stop()


@app.on_event("shutdown")
async def _finnhub_ws_shutdown() -> None:  # pragma: no cover
    """Phase 25.4 — close the Finnhub WS connection cleanly."""
    client = finnhub_ws_mod.get_default_client()
    if client is not None:
        await client.stop()


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


@app.get("/api/chatter")
def api_chatter(limit: int = 25) -> dict[str, Any]:
    """Rolling feed of recent agent narrations.

    Returns the most-recent ``limit`` entries (newest first), each
    shaped like ``{ts, agent, status, message, decision_id, regime,
    used_llm}``. Backed by an in-memory ring buffer
    (``packages.cockpit.web.chatter``), so a process restart clears it
    — the durable record is the agent log on disk.
    """
    # Clamp limit defensively so a hostile caller can't ask for a huge
    # response. The ring buffer is bounded anyway, but be explicit.
    safe = max(0, min(int(limit or 0), agent_chatter.CHATTER_MAX))
    items = agent_chatter.recent(safe)
    return {
        "items": items,
        "count": len(items),
        "max": agent_chatter.CHATTER_MAX,
    }


@app.get("/api/autonomy")
def api_autonomy() -> dict[str, Any]:
    """Always-On Brain status.

    Surfaces what the autonomous research + Curiosity loop is doing
    without exposing internal task handles. Powers the dashboard's
    autonomy pill.
    """
    return autonomy_brain.snapshot()


@app.post("/api/autonomy/tick")
async def api_autonomy_tick() -> dict[str, Any]:
    """Force one autonomous research + curiosity tick right now.

    Useful for the user button on the dashboard, and as a deterministic
    entry point for tests. Honors the pause flag like the scheduled
    loop does.
    """
    return await autonomy_brain.run_one_tick()


@app.post("/api/autonomy/enable")
def api_autonomy_enable() -> dict[str, Any]:
    """Start the autonomy loop if it isn't already running."""
    started = autonomy_brain.start()
    return {"started": bool(started), **autonomy_brain.snapshot()}


@app.post("/api/autonomy/disable")
def api_autonomy_disable() -> dict[str, Any]:
    """Stop the autonomy loop."""
    autonomy_brain.stop()
    return autonomy_brain.snapshot()


# ---------------------------------------------------------------------------
# Phase 21 — Self-improving brain endpoints
# ---------------------------------------------------------------------------


@app.get("/api/agent_status")
def api_agent_status() -> dict[str, Any]:
    """Phase 33: per-agent narration for the Agent Status panel.

    Returns the latest ``AgentStatus`` row per actor so the cockpit
    can render "what is each lane doing right now?". Always 200 —
    a missing log degrades to ``rows: {}``.
    """
    from dataclasses import asdict

    try:
        from packages.agents.narration import read_latest

        latest = read_latest()
        return {"ok": True, "rows": {a: asdict(s) for a, s in latest.items()}}
    except Exception as exc:  # pragma: no cover - defensive
        return {"ok": False, "rows": {}, "error": str(exc)[:200]}


@app.get("/api/brain")
def api_brain() -> dict[str, Any]:
    """Self-improving brain status: accuracy, regime, weights, reflection.

    Aggregates the four Phase 21 components so the dashboard can render
    a single "Brain Health" card. Always returns 200 — errors degrade
    to empty sections rather than crashing the page.
    """
    out: dict[str, Any] = {
        "ok": True,
        "memory": {},
        "regime": None,
        "bandit": {},
        "reflection": None,
        "recent_picks": [],
        "recent_reflections": [],
        "knowledge": {},
        "storage": {},
    }
    try:
        out["memory"] = autonomy_memory.accuracy_stats()
    except Exception as exc:
        out["memory"] = {"error": str(exc)[:200]}
    try:
        out["recent_picks"] = autonomy_memory.recent_picks(limit=15)
    except Exception:
        out["recent_picks"] = []
    try:
        out["bandit"] = autonomy_bandit.snapshot()
    except Exception as exc:
        out["bandit"] = {"error": str(exc)[:200]}
    try:
        out["reflection"] = autonomy_reflection.latest()
    except Exception:
        out["reflection"] = None
    try:
        out["recent_reflections"] = autonomy_reflection.recent(limit=8)
    except Exception:
        out["recent_reflections"] = []
    # Phase 22: knowledge base + storage health for the Memory Health card.
    try:
        out["knowledge"] = autonomy_knowledge.snapshot()
    except Exception as exc:
        out["knowledge"] = {"error": str(exc)[:200]}
    try:
        out["storage"] = {
            "brain_memory": autonomy_memory.store_info(),
            "bandit": autonomy_bandit.store_info(),
            "reflections": autonomy_reflection.store_info(),
            "knowledge_base": autonomy_knowledge.store_info(),
        }
    except Exception as exc:
        out["storage"] = {"error": str(exc)[:200]}
    snap = autonomy_brain.snapshot()
    out["regime"] = snap.get("last_regime")
    return out


@app.post("/api/brain/reset")
def api_brain_reset() -> dict[str, Any]:
    """Wipe brain learning state — picks ledger, bandit weights,
    reflections. The autonomy loop keeps running; it just starts
    learning from scratch.
    """
    removed: dict[str, Any] = {}
    try:
        autonomy_memory.reset_for_tests()
        removed["memory"] = True
    except Exception as exc:
        removed["memory_error"] = str(exc)[:200]
    try:
        autonomy_bandit.reset_for_tests()
        removed["bandit"] = True
    except Exception as exc:
        removed["bandit_error"] = str(exc)[:200]
    try:
        autonomy_reflection.reset_for_tests()
        removed["reflections"] = True
    except Exception as exc:
        removed["reflections_error"] = str(exc)[:200]
    try:
        autonomy_knowledge.reset_for_tests()
        removed["knowledge"] = True
    except Exception as exc:
        removed["knowledge_error"] = str(exc)[:200]
    return {"ok": True, **removed}


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
async def _cleanup_stale_tmp_files() -> None:  # pragma: no cover
    """Phase 15b: sweep orphan tmp*.tmp files left by crashed atomic writes.

    Stale temps accumulate when a worker crashes between
    ``NamedTemporaryFile.close()`` and ``os.replace()``. They are harmless
    but make ``data/cockpit/`` noisy over time. Anything older than the
    default age threshold (1 hour) is definitionally stale -- atomic
    writes complete in milliseconds.
    """
    try:
        from packages.shared.atomic_io import cleanup_stale_tmp_files

        for sweep_dir in (
            REPO_ROOT / "data" / "cockpit",
            REPO_ROOT / "data" / "paper_log",
            REPO_ROOT / "data" / "calibration",
            REPO_ROOT / "data" / "params",
        ):
            removed = cleanup_stale_tmp_files(sweep_dir)
            if removed:
                log.info(
                    "startup tmp-sweep: removed %d stale temp(s) from %s",
                    len(removed),
                    sweep_dir,
                )
    except Exception as e:
        log.warning("startup tmp-sweep failed: %s", e)


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


def _decide_paper_loop_autostart(
    cstate: Any,
    *,
    job_is_running: bool,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Pure decision function for paper-loop auto-start / auto-resume on boot.

    Returns a dict ``{"action": str, "strategy": str|None, "dry_run": bool,
    "reason": str}`` where ``action`` is one of ``"start"`` (first-boot
    auto-start path), ``"resume"`` (operator had it running before), or
    ``"skip"`` (do nothing — ``reason`` explains why).

    Extracted so the startup hook stays a thin wrapper around testable
    logic. Honors ``COCKPIT_AUTO_RESUME_LOOP`` (gates both branches) and
    ``COCKPIT_AUTO_START_LOOP`` (gates only the first-boot auto-start).
    """
    env = env if env is not None else dict(os.environ)
    if env.get("COCKPIT_AUTO_RESUME_LOOP", "1") not in ("1", "true", "True"):
        return {"action": "skip", "strategy": None, "dry_run": False, "reason": "COCKPIT_AUTO_RESUME_LOOP off"}
    if getattr(cstate, "paused", False):
        return {"action": "skip", "strategy": None, "dry_run": False, "reason": "cockpit paused"}
    if job_is_running:
        return {"action": "skip", "strategy": None, "dry_run": False, "reason": "already running"}

    is_resume = bool(getattr(cstate, "paper_loop_intended", False))
    if is_resume:
        return {
            "action": "resume",
            "strategy": getattr(cstate, "paper_loop_strategy", None) or "ensemble",
            "dry_run": bool(getattr(cstate, "paper_loop_dry_run", False)),
            "reason": "resume prior intent",
        }

    if env.get("COCKPIT_AUTO_START_LOOP", "1") not in ("1", "true", "True"):
        return {"action": "skip", "strategy": None, "dry_run": False, "reason": "COCKPIT_AUTO_START_LOOP off"}
    if getattr(cstate, "paper_loop_user_touched", False):
        return {"action": "skip", "strategy": None, "dry_run": False, "reason": "user has touched controls"}
    if not (env.get("ALPACA_PAPER_KEY_ID") and env.get("ALPACA_PAPER_SECRET")):
        return {"action": "skip", "strategy": None, "dry_run": False, "reason": "alpaca paper keys missing"}
    return {
        "action": "start",
        "strategy": "ensemble",
        "dry_run": False,
        "reason": "first-boot auto-start",
    }


@app.on_event("startup")
async def _paper_loop_auto_resume() -> None:  # pragma: no cover
    """Auto-start / auto-resume the paper-trade loop on cockpit boot.

    Thin wrapper around :func:`_decide_paper_loop_autostart` — see that
    function for the policy. Honors ``COCKPIT_AUTO_RESUME_LOOP`` (gates
    both branches) and ``COCKPIT_AUTO_START_LOOP`` (gates only the
    Phase 36b first-boot auto-start path).
    """
    try:
        cstate = load_state()
    except Exception as e:
        log.warning("auto-resume: state load failed: %s", e)
        return
    existing = job_mgr.status(PAPER_LOOP_KIND)
    decision = _decide_paper_loop_autostart(
        cstate, job_is_running=existing.is_running()
    )
    if decision["action"] == "skip":
        log.info("auto-resume: skipping (%s)", decision["reason"])
        return

    strategy = decision["strategy"] or "ensemble"
    dry_run = bool(decision["dry_run"])

    if decision["action"] == "start":
        # First-boot auto-start: persist intent so reboots resume.
        cstate.paper_loop_intended = True
        cstate.paper_loop_strategy = strategy
        cstate.paper_loop_dry_run = dry_run
        cstate = record_action(
            cstate,
            f"Auto-started paper loop ({strategy}, LIVE PAPER) on first boot",
        )
        save_state(cstate)
        log.info("auto-start: first-boot auto-start — strategy=%s LIVE PAPER", strategy)

    cmd = [
        _python_exe(),
        "tools/paper_trade.py",
        "--strategy",
        strategy,
        "--loop",
    ]
    if dry_run:
        cmd.append("--dry-run")
    try:
        info = job_mgr.start(PAPER_LOOP_KIND, cmd)
        log.info(
            "auto-%s: paper loop spawned pid=%s strategy=%s dry_run=%s",
            decision["action"],
            info.pid,
            strategy,
            dry_run,
        )
    except Exception as e:
        log.warning("auto-resume/start: failed to spawn loop: %s", e)


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

    # Flip-event notifier: poll data/cockpit/shadow_flips.jsonl every 60s
    # and deliver any new rows to configured sinks (desktop toast +
    # webhook). Env-gated so it can be disabled in CI.
    if os.environ.get("COCKPIT_FLIP_NOTIFY", "1") in ("1", "true", "True"):
        try:
            from packages.shadow.notify_loop import flip_notify_loop

            _AUTOMATION_TASKS["flip_notify"] = loop.create_task(
                flip_notify_loop(poll_seconds=60.0)
            )
        except Exception as e:  # never crash boot over a notifier dep
            log.warning("flip notify loop skipped: %s", e)


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


def _compute_promote_payload() -> dict[str, Any]:
    """Build the same payload ``/api/promote`` returns.

    Extracted as a module function so ``arm_live`` can re-validate the
    gate server-side without going through HTTP. Tests can monkey-patch
    this whole function to inject a synthetic verdict.
    """
    import pandas as pd

    from packages.backtests import live_promotion as lp
    from packages.shadow.greenlight import (
        GREENLIGHT_DAYS_REQUIRED,
    )
    from packages.shadow.greenlight import (
        read_status as read_shadow_status,
    )

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

    shadow_payload = read_shadow_status() or {}
    shadow_status = str(shadow_payload.get("status", "shadow"))
    shadow_streak = int(shadow_payload.get("streak_days", 0))
    shadow_ready = shadow_status == "ready"

    gating_reasons = list(decision.readiness.reasons)
    if not telegram_connected:
        gating_reasons.append(
            "Telegram approval bot is not configured "
            "(set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env)"
        )
    if not shadow_ready:
        gating_reasons.append(
            f"Shadow soak not complete "
            f"({shadow_streak}/{GREENLIGHT_DAYS_REQUIRED} non-negative days)"
        )

    all_clear = (
        decision.live_enabled
        and telegram_connected
        and shadow_ready
    )

    return {
        "live_enabled": bool(all_clear),
        "capital_fraction": (
            decision.capital_fraction if all_clear else 0.0
        ),
        "readiness": {
            "ready": decision.readiness.ready and telegram_connected and shadow_ready,
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
            "shadow_status": shadow_status,
            "shadow_streak_days": shadow_streak,
            "shadow_days_required": GREENLIGHT_DAYS_REQUIRED,
            "shadow_ready": shadow_ready,
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


@app.get("/api/promote")
def api_promote() -> dict[str, Any]:
    """Return the full live-trading readiness picture.

    Pulls the paper equity curve, runs the §16 readiness gate, and
    surfaces every reason live capital is (or isn't) allowed. Includes
    the Telegram-bot-not-yet-connected line item so the operator knows
    that channel is still required even after the metrics pass.
    """
    return _compute_promote_payload()


@app.post("/api/arm-live")
def api_arm_live(body: dict[str, Any] | None = None) -> dict[str, Any]:
    """One-click promote to live trading.

    Re-validates every gate server-side; the button is just a hint.
    Writes ENABLE_LIVE_TRADING=true to .env, mirrors to os.environ,
    and appends an immutable audit row.
    """
    from packages.cockpit.web.arm_live import arm_live

    note = None
    if isinstance(body, dict):
        raw_note = body.get("note")
        if isinstance(raw_note, str):
            note = raw_note.strip()[:500] or None

    result = arm_live(
        actor="operator",
        gate_evaluator=_compute_promote_payload,
        note=note,
    )
    return result.to_response()


@app.post("/api/disarm-live")
def api_disarm_live(body: dict[str, Any] | None = None) -> dict[str, Any]:
    """Operator panic button: flip live trading off without a gate."""
    from packages.cockpit.web.arm_live import disarm_live

    reason = ""
    if isinstance(body, dict):
        raw = body.get("reason")
        if isinstance(raw, str):
            reason = raw.strip()[:500]
    result = disarm_live(actor="operator", reason=reason)
    return result.to_response()


@app.get("/api/arm-live/audit")
def api_arm_live_audit(limit: int = 50) -> dict[str, Any]:
    """Tail the arm/disarm audit log."""
    from packages.cockpit.web.arm_live import read_audit

    capped = max(1, min(int(limit), 500))
    rows = read_audit(limit=capped)
    return {"events": rows, "count": len(rows)}


# --------------------------------------------------------------------------
# Phase 15a: one-click risk-adaptive sizing activation
# --------------------------------------------------------------------------


@app.get("/api/sizing/config")
def api_sizing_config() -> dict[str, Any]:
    """Return the active sizing config + available presets.

    Used by the Settings page to render preset buttons and a current-mode
    badge.
    """
    from packages.cockpit.web.sizing_control import current_config

    return current_config()


@app.post("/api/sizing/configure")
def api_sizing_configure(body: dict[str, Any] | None = None) -> dict[str, Any]:
    """Apply a sizing preset (and optional per-key overrides) atomically.

    Writes the configured ``POLICY_*`` env vars to ``.env`` and mirrors
    them to ``os.environ`` so the next paper_trade cycle picks them up
    without a worker restart. Empty string values delete the key, which
    is how the "Off" preset reverts to Phase 13 equal-weight.
    """
    from packages.cockpit.web.sizing_control import configure

    preset: str | None = None
    overrides: dict[str, str] | None = None
    note: str | None = None
    if isinstance(body, dict):
        raw_preset = body.get("preset")
        if isinstance(raw_preset, str):
            preset = raw_preset.strip() or None
        raw_overrides = body.get("overrides")
        if isinstance(raw_overrides, dict):
            overrides = {
                str(k): str(v) for k, v in raw_overrides.items() if isinstance(k, str)
            }
        raw_note = body.get("note")
        if isinstance(raw_note, str):
            note = raw_note.strip()[:500] or None

    result = configure(
        preset=preset,
        overrides=overrides,
        actor="operator",
        note=note,
    )
    return result.to_response()


@app.get("/api/sizing/audit")
def api_sizing_audit(limit: int = 50) -> dict[str, Any]:
    """Tail the sizing-config audit log."""
    from packages.cockpit.web.sizing_control import read_audit

    capped = max(1, min(int(limit), 500))
    rows = read_audit(limit=capped)
    return {"events": rows, "count": len(rows)}


# --------------------------------------------------------------------------
# Phase 25: active profit-taking + dip-watch buy-back
# --------------------------------------------------------------------------


@app.get("/api/exit-rules")
def api_exit_rules() -> dict[str, Any]:
    """Snapshot of active exit thresholds, peaks, and recent audit."""
    from packages.cockpit.web.exit_rules import snapshot

    return snapshot()


@app.post("/api/exit-rules/tick")
async def api_exit_rules_tick() -> dict[str, Any]:
    """Manual trigger for the exit-rules tick (autonomy normally drives it).

    Evaluates every open Alpaca paper position; submits sells when any
    rule fires. Use this to test the loop without waiting for autonomy.
    """
    from packages.cockpit.web.dip_watch import arm as dip_arm
    from packages.cockpit.web.exit_rules import run_tick

    broker = AlpacaPaperBroker()
    try:
        async def _submit_sell(symbol: str, qty: float) -> Any:
            from packages.execution.broker import OrderRequest
            return await broker.submit(
                OrderRequest(symbol=symbol, side="sell", qty=qty)
            )

        def _on_profit(symbol: str, exit_price: float, pnl_pct: float) -> None:
            # Find the qty we just closed so dip_watch can re-arm same size.
            # Best-effort — if we can't introspect, default to 1 share.
            dip_arm(
                symbol=symbol,
                exit_price=exit_price,
                exit_pnl_pct=pnl_pct,
                qty=1.0,
            )

        result = await run_tick(
            positions_getter=broker.positions,
            submit_sell=_submit_sell,
            on_profit_taken=_on_profit,
        )
        return {
            "evaluated": result.evaluated,
            "sells_triggered": result.sells_triggered,
            "sells_executed": result.sells_executed,
            "decisions": [
                {
                    "symbol": d.symbol,
                    "action": d.action,
                    "reason": d.reason,
                    "pnl_pct": d.pnl_pct,
                    "peak_pct": d.peak_pct,
                }
                for d in result.decisions
            ],
            "errors": result.errors,
        }
    finally:
        with contextlib.suppress(Exception):
            await broker.aclose()


@app.get("/api/dip-watch")
def api_dip_watch() -> dict[str, Any]:
    """Snapshot of armed dip-watchers + config."""
    from packages.cockpit.web.dip_watch import snapshot

    return snapshot()


@app.post("/api/dip-watch/clear")
def api_dip_watch_clear(symbol: str | None = None) -> dict[str, Any]:
    """Manually cancel one watcher (symbol=...) or all (no arg)."""
    from packages.cockpit.web.dip_watch import clear

    removed = clear(symbol)
    return {"removed": removed, "symbol": symbol}


@app.post("/api/dip-watch/tick")
async def api_dip_watch_tick() -> dict[str, Any]:
    """Manual trigger for the dip-watch tick."""
    from packages.cockpit.web.dip_watch import run_tick

    # Cheap price lookup via yfinance fast_info.
    def _price(symbol: str) -> float | None:
        try:
            import yfinance as yf  # type: ignore[import-not-found]
            t = yf.Ticker(symbol)
            p = t.fast_info.get("last_price") if hasattr(t, "fast_info") else None
            return float(p) if p else None
        except Exception:
            return None

    broker = AlpacaPaperBroker()
    try:
        async def _submit_buy(symbol: str, qty: float) -> Any:
            from packages.execution.broker import OrderRequest
            return await broker.submit(
                OrderRequest(symbol=symbol, side="buy", qty=qty)
            )

        result = await run_tick(
            price_lookup=_price,
            submit_buy=_submit_buy,
        )
        return {
            "checked": result.checked,
            "fired": result.fired,
            "expired": result.expired,
            "errors": result.errors,
        }
    finally:
        with contextlib.suppress(Exception):
            await broker.aclose()


# --------------------------------------------------------------------------
# Phase 16: aggregated pre-flight readiness checklist
# --------------------------------------------------------------------------


@app.get("/api/preflight")
def api_preflight() -> dict[str, Any]:
    """Run every readiness check and return the aggregated verdict.

    The page poller hits this every few seconds. Checks are best-effort
    (any individual check that raises is downgraded to a failure row
    rather than killing the whole snapshot), so this endpoint should
    never return a 5xx in practice.
    """
    from packages.cockpit.web.preflight import compute_preflight

    return compute_preflight().to_dict()


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


# ---------------------------------------------------------------------------
# Phase 25.3 — live data-feed status
# ---------------------------------------------------------------------------


@app.get("/api/data-feed")
def api_data_feed() -> dict[str, Any]:
    """Snapshot of the live-quote cache + WS stream.

    Driven by :mod:`packages.cockpit.web.live_quotes` for the cache
    half and :mod:`packages.cockpit.web.finnhub_ws` for the live
    tick stream. The dashboard polls this so the user can confirm
    Finnhub is live (REST + WS), tell at a glance whether the WS
    stream is connected, and see per-symbol last-tick times.
    """
    cache = live_quotes_mod.get_default_cache()
    out = cache.status()
    ws_client = finnhub_ws_mod.get_default_client()
    if ws_client is not None:
        out["websocket"] = ws_client.status()
    else:
        out["websocket"] = {
            "enabled": False,
            "reason": "no_api_key_or_disabled",
        }
    return out


@app.post("/api/data-feed/refresh")
async def api_data_feed_refresh(symbol: str | None = None) -> dict[str, Any]:
    """Force-refresh one symbol (or the active portfolio when omitted)."""
    cache = live_quotes_mod.get_default_cache()
    if symbol:
        price = await cache.lookup(symbol.upper())
        return {"symbol": symbol.upper(), "price": price}
    syms: set[str] = {"SPY", "^VIX"}
    try:
        for s in (_portfolio_symbols_snapshot() or []):
            syms.add(str(s).upper())
    except Exception:
        pass
    results = await live_quotes_mod.refresh_symbols(cache, sorted(syms))
    return {"refreshed": dict(results)}


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
