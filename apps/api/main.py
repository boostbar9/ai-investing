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
"""
from __future__ import annotations

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
    # Stub: real impl reads from broker (read-only key) and joins last marks.
    return {"positions": [], "as_of": datetime.now(UTC).isoformat()}


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
    return _PENDING[did]
