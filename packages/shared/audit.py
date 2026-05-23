"""Immutable audit log (v3.1 §3).

Every approval and every order writes one row. Rows are append-only;
DELETE / UPDATE are forbidden at the DB level via a trigger (see migration).
"""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class AuditEvent(BaseModel):
    audit_id: UUID = Field(default_factory=uuid4)
    decision_id: UUID
    actor: str  # "research" | "strategy" | "risk" | "execution" | "operator"
    event_type: str  # "agent_call" | "approval" | "order" | "fill" | "halt"
    payload: dict[str, Any]
    ts: datetime = Field(default_factory=lambda: datetime.now(UTC))


def to_row(evt: AuditEvent) -> dict[str, Any]:
    return {
        "audit_id": str(evt.audit_id),
        "decision_id": str(evt.decision_id),
        "actor": evt.actor,
        "event_type": evt.event_type,
        "payload": evt.payload,
        "ts": evt.ts.isoformat(),
    }
