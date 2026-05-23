"""Shared Pydantic schemas — the v3.1 agent contract.

Every agent MUST accept ``decision_id``, return within ``max_tokens``, and emit
a JSON response that validates against its schema or fail closed.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

Regime = Literal["bull", "bear", "chop", "crisis"]
Side = Literal["buy", "sell"]


class AgentRequest(BaseModel):
    decision_id: UUID
    max_tokens: int = Field(default=4096, ge=128, le=32768)
    timeout_seconds: int = Field(default=30, ge=1, le=300)


class ResearchInput(AgentRequest):
    symbols: list[str]
    lookback_days: int = 30


class ResearchOutput(BaseModel):
    decision_id: UUID
    thesis: str
    sentiment: float = Field(ge=-1.0, le=1.0)
    citations: list[str]


class StrategyInput(AgentRequest):
    regime: Regime
    universe: list[str]
    features: dict[str, float]


class Signal(BaseModel):
    symbol: str
    side: Side
    strength: float = Field(ge=0.0, le=1.0)
    rationale: str


class StrategyOutput(BaseModel):
    decision_id: UUID
    signals: list[Signal]


class Position(BaseModel):
    symbol: str
    qty: float
    avg_price: float


class RiskInput(AgentRequest):
    positions: list[Position]
    candidates: list[Signal]


class RiskOutput(BaseModel):
    decision_id: UUID
    approved: list[Signal]
    rejected: list[Signal]
    halted: bool = False
    halt_reason: str | None = None


class Order(BaseModel):
    symbol: str
    side: Side
    qty: float
    limit_price: float | None = None


class ExecutionInput(AgentRequest):
    approved_orders: list[Order]


class Fill(BaseModel):
    symbol: str
    side: Side
    qty: float
    price: float
    timestamp: datetime


class ExecutionOutput(BaseModel):
    decision_id: UUID
    fills: list[Fill]
    audit_id: UUID
