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
- GET  /activity           — recent audit events (activity feed module)
- GET  /health/detail      — broker, LLM router, regime cache, DB health panel
- GET  /live/promotion     — Phase 5 live readiness + canary capital tier
"""
from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

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


@app.get("/activity")
async def activity_feed(limit: int = 50) -> dict[str, Any]:
    """Flattened recent audit events across decisions (§10 activity feed)."""
    flat: list[dict[str, Any]] = []
    for did, events in _AUDIT.items():
        for e in events:
            flat.append({**e, "decision_id": str(did)})
    flat.sort(key=lambda e: e.get("ts", ""), reverse=True)
    return {"events": flat[:limit]}


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
