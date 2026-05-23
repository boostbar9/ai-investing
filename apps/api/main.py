"""FastAPI entrypoint for the ai-investing cockpit + bot backend.

Phase 3 endpoints:
- GET  /health             — liveness for Grafana + /health bot command
- GET  /version            — spec + phase metadata
- GET  /regime             — current 4-state HMM badge
- GET  /positions          — current positions from broker
- GET  /agents/status      — agent status lights (research/strategy/risk/exec)
- GET  /approvals/pending  — items waiting on operator (Telegram bot polls this)
- POST /approvals/{id}     — operator approve/deny
- GET  /audit/{decision_id} — Decision Trace (§20 "open Decision Trace")
- GET  /strategies         — registered strategy catalogue
- GET  /strategies/modes   — per-strategy execution mode (paper/shadow/live)
- POST /strategies/{name}/mode — set a strategy's execution mode
- GET  /broker/account     — paper broker account summary
- GET  /activity           — recent audit events (activity feed module)
- GET  /health/detail      — broker, LLM router, regime cache, DB health panel
- GET  /live/promotion     — Phase 5 live readiness + canary capital tier
- POST /security/rotation-reminder — n8n quarterly key-rotation event
- GET  /security/audit     — list recent security audit events
- POST /auth/passkey/register/options    — issue WebAuthn registration challenge
- POST /auth/passkey/register/verify     — verify + persist a new passkey
- POST /auth/passkey/authenticate/options — issue WebAuthn sign-in challenge
- POST /auth/passkey/authenticate/verify  — verify sign-in + mint session
"""
from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from packages.shared.passkeys import (
    PasskeyStore,
    PasskeyVerificationError,
    build_authentication_options,
    build_registration_options,
    verify_authentication,
    verify_registration,
)

app = FastAPI(title="ai-investing API", version="0.1.0")

# --- In-memory stubs (real impls land when DB + Temporal are wired) ---
_PENDING: dict[UUID, dict[str, Any]] = {}
_AUDIT: dict[UUID, list[dict[str, Any]]] = {}


class ApprovalDecision(BaseModel):
    approve: bool
    note: str | None = None


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok", "ts": datetime.now(UTC).isoformat()}


@app.get("/version")
async def version() -> dict[str, str]:
    return {"spec": "v3.1", "phase": "3-agents"}


@app.get("/regime")
async def regime() -> dict[str, Any]:
    # Stub: real impl reads cached HMM output from DragonflyDB.
    return {"regime": "bull", "confidence": 0.78, "as_of": datetime.now(UTC).isoformat()}


@app.get("/positions")
async def positions() -> dict[str, Any]:
    """Live positions from the broker, with PnL %% per name.

    When ``ALPACA_PAPER_KEY_ID`` is unset (local dev / CI) we return an
    empty list rather than 500ing — callers should see "no positions"
    instead of a broken cockpit.
    """
    from packages.execution.broker import (
        AlpacaPaperBroker,
        BrokerError,
        BrokerRouter,
    )

    if not os.getenv("ALPACA_PAPER_KEY_ID"):
        return {"positions": [], "as_of": datetime.now(UTC).isoformat()}

    broker = AlpacaPaperBroker()
    try:
        router = BrokerRouter([broker])
        try:
            ps = await router.positions()
            return {
                "positions": [p.to_dict() for p in ps],
                "as_of": datetime.now(UTC).isoformat(),
            }
        except BrokerError as e:
            return {
                "positions": [],
                "as_of": datetime.now(UTC).isoformat(),
                "error": str(e),
            }
    finally:
        await broker.aclose()


@app.get("/agents/status")
async def agent_status() -> dict[str, Any]:
    return {
        "research": {"ok": True, "last_run": None, "model": "deepseek-r1:70b"},
        "strategy": {"ok": True, "last_run": None, "model": "qwen2.5:72b"},
        "risk":     {"ok": True, "last_run": None, "model": "deepseek-r1:70b"},
        "execution": {"ok": True, "last_run": None, "model": "llama3.3:70b"},
    }


@app.get("/approvals/pending")
async def approvals_pending() -> dict[str, Any]:
    return {"pending": list(_PENDING.values())}


@app.post("/approvals/{decision_id}")
async def approvals_decide(decision_id: UUID, body: ApprovalDecision) -> dict[str, Any]:
    item = _PENDING.pop(decision_id, None)
    if not item:
        raise HTTPException(status_code=404, detail="approval not found")
    _AUDIT.setdefault(decision_id, []).append(
        {
            "actor": "operator",
            "event_type": "approval",
            "approve": body.approve,
            "note": body.note,
            "ts": datetime.now(UTC).isoformat(),
        }
    )
    return {"ok": True, "decision_id": str(decision_id), "approved": body.approve}


@app.get("/audit/{decision_id}")
async def audit_trace(decision_id: UUID) -> dict[str, Any]:
    """Decision Trace — every event flowing through ``decision_id`` (§20)."""
    events = _AUDIT.get(decision_id, [])
    if not events:
        raise HTTPException(status_code=404, detail="no audit trail for decision")
    return {"decision_id": str(decision_id), "events": events}


# --- Dev helper to seed a fake pending approval for cockpit smoke tests ---
@app.post("/_dev/seed-approval")
async def dev_seed_approval() -> dict[str, Any]:
    did = uuid4()
    _PENDING[did] = {
        "decision_id": str(did),
        "symbol": "SPY",
        "side": "buy",
        "qty": 1,
        "thesis": "20d momentum positive, regime bull",
        "ts": datetime.now(UTC).isoformat(),
    }
    _AUDIT.setdefault(did, []).append(
        {
            "actor": "system",
            "event_type": "seed",
            "ts": datetime.now(UTC).isoformat(),
        }
    )
    return _PENDING[did]


@app.post("/_dev/push-test")
async def dev_push_test() -> dict[str, Any]:
    """Send a test push so operators can verify OneSignal wiring (§12 / #6).

    Returns ``{"skipped": true}`` when OneSignal isn't configured locally.
    """
    from packages.shared.push import PushClient, PushPayload

    client = PushClient()
    try:
        return await client.send(
            PushPayload(
                title="ai-investing push test",
                body="✅ cockpit ↔ OneSignal wiring works",
                dedupe_key=f"push-test-{datetime.now(UTC).isoformat()}",
            )
        )
    finally:
        await client.aclose()


# --- Strategies / Activity / Health Detail ---


@app.get("/strategies")
async def list_strategies() -> dict[str, Any]:
    """Strategy catalogue — backs the cockpit Strategies panel."""
    from packages.strategies import all_strategies

    return {
        "strategies": [
            {
                "name": name,
                "description": (cls.__doc__ or "").strip().splitlines()[0]
                if cls.__doc__
                else "",
            }
            for name, cls in all_strategies().items()
        ]
    }


class StrategyModeUpdate(BaseModel):
    mode: str  # "paper" | "shadow" | "live"


@app.get("/strategies/modes")
async def list_strategy_modes() -> dict[str, Any]:
    """Per-strategy execution-mode snapshot for the cockpit toggle UI."""
    from packages.execution.modes import ExecutionMode, all_modes, get_mode
    from packages.strategies import all_strategies

    modes: dict[str, str] = {}
    for name in all_strategies():
        modes[name] = get_mode(name).value
    # Include any non-strategy entries the operator may have set explicitly.
    for k, v in all_modes().items():
        modes.setdefault(k, v.value)
    return {"modes": modes, "available": [m.value for m in ExecutionMode]}


@app.post("/strategies/{name}/mode")
async def set_strategy_mode(name: str, body: StrategyModeUpdate) -> dict[str, Any]:
    """Update a strategy's execution mode.

    Setting ``live`` is permitted, but the runner will still downgrade to
    ``paper`` at execution time unless the live-promotion gate has cleared
    AND ``ENABLE_LIVE_TRADING=true``. See :mod:`packages.execution.modes`.
    """
    from packages.execution.modes import ExecutionMode, set_mode
    from packages.strategies import all_strategies

    if name not in all_strategies():
        raise HTTPException(status_code=404, detail=f"unknown strategy: {name}")
    try:
        mode = ExecutionMode(body.mode.lower())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"invalid mode: {body.mode}") from e
    set_mode(name, mode)
    return {"strategy": name, "mode": mode.value}


@app.get("/broker/account")
async def broker_account() -> dict[str, Any]:
    """Paper broker account summary — equity, cash, buying power.

    Used by the cockpit training view to show how the agents are doing on
    the fake account at a glance.
    """
    from packages.execution.broker import AlpacaPaperBroker, BrokerError

    broker = AlpacaPaperBroker()
    try:
        try:
            data = await broker.account()
        except BrokerError as e:
            raise HTTPException(status_code=502, detail=str(e)) from e
    finally:
        await broker.aclose()
    return {
        "broker": broker.name,
        "equity": data.get("equity"),
        "cash": data.get("cash"),
        "buying_power": data.get("buying_power"),
        "status": data.get("status"),
    }


@app.get("/activity")
async def activity_feed(limit: int = 50) -> dict[str, Any]:
    """Flattened recent audit events across decisions (§10 activity feed)."""
    flat: list[dict[str, Any]] = []
    for did, events in _AUDIT.items():
        for e in events:
            flat.append({**e, "decision_id": str(did)})
    flat.sort(key=lambda e: e.get("ts", ""), reverse=True)
    return {"events": flat[:limit]}


class RotationReminder(BaseModel):
    ts: str
    scope: str
    runbook: str
    channel: str = "n8n-quarterly"


# Append-only in-memory store; a real impl writes to the immutable audit table.
_SECURITY_AUDIT: list[dict[str, Any]] = []


@app.post("/security/rotation-reminder")
async def rotation_reminder(body: RotationReminder) -> dict[str, Any]:
    """Receive the n8n quarterly key-rotation reminder event.

    Writes to the security audit log (§13). Returns the recorded audit id
    so the n8n workflow can chain a Telegram confirmation.
    """
    entry = {
        "audit_id": str(uuid4()),
        "recorded_at": datetime.now(UTC).isoformat(),
        **body.model_dump(),
    }
    _SECURITY_AUDIT.append(entry)
    return {"ok": True, "audit_id": entry["audit_id"]}


@app.get("/security/audit")
async def list_security_audit(limit: int = 50) -> dict[str, Any]:
    """List recent security audit events (rotation reminders, etc.)."""
    return {"events": _SECURITY_AUDIT[-limit:][::-1]}


@app.get("/live/promotion")
async def live_promotion() -> dict[str, Any]:
    """Phase 5 (§15) live-trading readiness + current canary capital tier.

    Reads paper/live equity curves from optional JSON files pointed to by
    ``PAPER_EQUITY_PATH`` and ``LIVE_EQUITY_PATH``. Each file is a JSON
    object: ``{"equity": [100.0, 100.1, ...]}``. Missing files are treated
    as empty curves — the gate fails closed.
    """
    import json
    from pathlib import Path

    import pandas as pd

    from packages.backtests.live_promotion import decide_live_capital

    def _load(env_key: str) -> pd.Series:
        p = os.getenv(env_key)
        if not p:
            return pd.Series(dtype=float)
        try:
            data = json.loads(Path(p).read_text())
            return pd.Series(data.get("equity", []), dtype=float)
        except (OSError, json.JSONDecodeError):
            return pd.Series(dtype=float)

    paper = _load("PAPER_EQUITY_PATH")
    live = _load("LIVE_EQUITY_PATH")
    decision = decide_live_capital(paper, live)
    canary_payload = (
        {
            "tier_index": decision.canary.tier_index,
            "fraction": decision.canary.fraction,
            "days_in_tier": decision.canary.days_in_tier,
            "dwell_required": decision.canary.dwell_required,
            "next_fraction": decision.canary.next_fraction,
            "reasons": decision.canary.reasons,
        }
        if decision.canary is not None
        else None
    )
    return {
        "live_enabled": decision.live_enabled,
        "capital_fraction": decision.capital_fraction,
        "readiness": {
            "ready": decision.readiness.ready,
            "reasons": decision.readiness.reasons,
            "metrics": decision.readiness.metrics,
        },
        "canary": canary_payload,
    }


@app.get("/health/detail")
async def health_detail() -> dict[str, Any]:
    """Per-subsystem health for the cockpit Health Panel module.

    These are intentionally cheap checks; Grafana is the source of truth.
    """
    return {
        "api":     {"ok": True, "ts": datetime.now(UTC).isoformat()},
        "broker":  {"ok": True, "name": os.getenv("BROKER_PRIMARY", "alpaca-paper")},
        "llm":     {"ok": True, "host": os.getenv("OLLAMA_HOST", "http://localhost:11434")},
        "regime":  {"ok": True, "source": "hmm-or-heuristic"},
        "db":      {"ok": True, "driver": "timescale"},
        "cache":   {"ok": True, "driver": "dragonfly"},
    }


@app.get("/data/sources")
async def data_sources() -> dict[str, Any]:
    """Status of every data source (free-tier first).

    Reads the Parquet cache mtimes to report freshness without re-fetching
    upstream APIs. The cockpit polls this every ~15s.
    """
    from pathlib import Path

    from packages.data.registry import FREE_ADAPTER_NAMES

    parquet_root = Path(os.getenv("DATA_PARQUET_ROOT", "data/parquet"))

    def _scan(subdir: str) -> dict[str, Any]:
        d = parquet_root / subdir
        if not d.exists():
            return {"ok": False, "files": 0, "last_update": None, "path": str(d)}
        files = list(d.glob("*.parquet"))
        if not files:
            return {"ok": False, "files": 0, "last_update": None, "path": str(d)}
        latest = max(f.stat().st_mtime for f in files)
        return {
            "ok": True,
            "files": len(files),
            "last_update": datetime.fromtimestamp(latest, tz=UTC).isoformat(),
            "path": str(d),
        }

    daily = _scan("daily")
    intraday = _scan("intraday")
    macro = _scan("macro")

    # Sentiment lives as a single JSON snapshot, not Parquet.
    sent_path = parquet_root / "sentiment" / "latest.json"
    if sent_path.exists():
        sent = {
            "ok": True,
            "files": 1,
            "last_update": datetime.fromtimestamp(
                sent_path.stat().st_mtime, tz=UTC
            ).isoformat(),
            "path": str(sent_path),
        }
    else:
        sent = {"ok": False, "files": 0, "last_update": None, "path": str(sent_path)}

    return {
        "sources": {
            "daily_bars":    {**daily,    "adapter": "alpaca_data / yfinance"},
            "intraday_bars": {**intraday, "adapter": "alpaca_data"},
            "macro":         {**macro,    "adapter": "fred"},
            "sentiment":     {**sent,     "adapter": "sentiment (reddit + rss)"},
        },
        "free_tier": list(FREE_ADAPTER_NAMES),
    }


# ---------------------------------------------------------------------------
# §12 Phase 4 — passkey biometric login (issue #7)
# ---------------------------------------------------------------------------
#
# Tailscale is the network gate; passkeys are the per-user lock inside it.
# The server only trusts the `X-Tailscale-User` header set by the upstream
# tsnsrv reverse proxy. In dev (no Tailscale), the header is absent and we
# fall back to a single hardcoded operator id.

_PASSKEY_STORE = PasskeyStore()


def _rp_id() -> str:
    return os.getenv("WEBAUTHN_RP_ID", "localhost")


def _expected_origin() -> str:
    return os.getenv("WEBAUTHN_ORIGIN", "http://localhost:3000")


def _resolve_user(x_tailscale_user: str | None) -> str:
    """Resolve the caller's identity.

    In production this header is set by tsnsrv. In dev we fall back to
    ``WEBAUTHN_DEV_USER`` (default ``devin``) so the flow is testable.
    """
    if x_tailscale_user:
        return x_tailscale_user
    return os.getenv("WEBAUTHN_DEV_USER", "devin")


class RegistrationOptionsRequest(BaseModel):
    user_display_name: str = ""


class RegistrationVerifyRequest(BaseModel):
    response: dict[str, Any]
    label: str = ""


class AuthenticationVerifyRequest(BaseModel):
    response: dict[str, Any]


@app.post("/auth/passkey/register/options")
async def passkey_register_options(
    body: RegistrationOptionsRequest,
    x_tailscale_user: str | None = Header(default=None, alias="X-Tailscale-User"),
) -> dict[str, Any]:
    user_id = _resolve_user(x_tailscale_user)
    return build_registration_options(
        user_id,
        rp_id=_rp_id(),
        rp_name="ai-investing cockpit",
        user_display_name=body.user_display_name or user_id,
        existing_credentials=_PASSKEY_STORE.credentials_for_user(user_id),
        store=_PASSKEY_STORE,
    )


@app.post("/auth/passkey/register/verify")
async def passkey_register_verify(
    body: RegistrationVerifyRequest,
) -> dict[str, Any]:
    try:
        cred = verify_registration(
            response=body.response,
            expected_origin=_expected_origin(),
            expected_rp_id=_rp_id(),
            store=_PASSKEY_STORE,
            label=body.label,
        )
    except PasskeyVerificationError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {
        "ok": True,
        "credential_id": cred.credential_id,
        "user_id": cred.user_id,
        "label": cred.label,
    }


@app.post("/auth/passkey/authenticate/options")
async def passkey_authenticate_options(
    x_tailscale_user: str | None = Header(default=None, alias="X-Tailscale-User"),
) -> dict[str, Any]:
    user_id = _resolve_user(x_tailscale_user)
    return build_authentication_options(
        user_id, rp_id=_rp_id(), store=_PASSKEY_STORE
    )


@app.post("/auth/passkey/authenticate/verify")
async def passkey_authenticate_verify(
    body: AuthenticationVerifyRequest,
) -> dict[str, Any]:
    try:
        cred = verify_authentication(
            response=body.response,
            expected_origin=_expected_origin(),
            expected_rp_id=_rp_id(),
            store=_PASSKEY_STORE,
        )
    except PasskeyVerificationError as e:
        raise HTTPException(status_code=401, detail=str(e)) from e
    # A real impl mints a signed session token here. For the skeleton we
    # return a placeholder the cockpit can stash in HttpOnly cookie via its
    # own NextAuth handler.
    return {
        "ok": True,
        "user_id": cred.user_id,
        "session_hint": str(uuid4()),
    }
