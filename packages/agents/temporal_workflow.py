"""Temporal Workflow for the agent graph (issue #3, §5).

The in-process :class:`AgentGraph` runs the topology in one async loop, which
is great for tests but bad for durability. This module expresses the same
topology as a Temporal Workflow so we get:

  - Durable history → resume after crash / restart
  - Per-activity retry policies (research/strategy/risk/execution have
    different failure modes)
  - Human-in-the-loop via :func:`temporalio.workflow.wait_condition` instead
    of an in-memory callback
  - Signals from the cockpit / Telegram bot land in workflow history

Activity functions are kept side-effect-light so they can be unit-tested
in isolation; ``apps/api`` (or a dedicated worker) registers them against
a running Temporal cluster.

Determinism rules:
  - Inside ``@workflow.defn`` code we MUST NOT touch real clocks, env vars,
    random, network, or third-party libraries. All side effects happen in
    ``@activity.defn`` functions.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from temporalio import activity, workflow
from temporalio.common import RetryPolicy

# Heavy / non-deterministic imports happen ONLY inside ``with
# workflow.unsafe.imports_passed_through()`` so the workflow sandbox stays
# small and deterministic.
with workflow.unsafe.imports_passed_through():
    from packages.shared.schemas import (
        ExecutionInput,
        ExecutionOutput,
        Order,
        Position,
        ResearchInput,
        ResearchOutput,
        RiskInput,
        RiskOutput,
        Signal,
        StrategyInput,
        StrategyOutput,
    )


# ---------------------------------------------------------------------------
# Workflow input / output (dataclasses → Temporal serializes via its default
# data converter)
# ---------------------------------------------------------------------------


@dataclass
class TradeCycleInput:
    decision_id: str  # str for serialization; converted back to UUID inside
    symbols: list[str]
    regime: str
    positions: list[dict[str, Any]]  # serialized Position
    features: dict[str, float]
    approval_timeout_seconds: int = 600


@dataclass
class TradeCycleResult:
    decision_id: str
    halted: bool
    halt_reason: str | None = None
    n_signals: int = 0
    n_approved: int = 0
    n_fills: int = 0
    audit: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Activities (overridable in tests via the activity registry)
# ---------------------------------------------------------------------------


@activity.defn(name="agent.research")
async def research_activity(payload: dict[str, Any]) -> dict[str, Any]:
    """Default research activity — calls the LLM-backed runner. Worker code
    replaces this in production with a router-bound implementation."""
    from packages.agents.llm_router import LLMRouter
    from packages.agents.runners import build_research_runner

    router = LLMRouter()
    try:
        run = build_research_runner(router)
        out = await run(ResearchInput.model_validate(payload))
        return out.model_dump(mode="json")
    finally:
        await router.aclose()


@activity.defn(name="agent.strategy")
async def strategy_activity(payload: dict[str, Any]) -> dict[str, Any]:
    from packages.agents.llm_router import LLMRouter
    from packages.agents.runners import build_strategy_runner

    router = LLMRouter()
    try:
        run = build_strategy_runner(router)
        out = await run(StrategyInput.model_validate(payload))
        return out.model_dump(mode="json")
    finally:
        await router.aclose()


@activity.defn(name="agent.risk")
async def risk_activity(payload: dict[str, Any]) -> dict[str, Any]:
    from packages.agents.llm_router import LLMRouter
    from packages.agents.runners import build_risk_runner

    router = LLMRouter()
    try:
        run = build_risk_runner(router)
        out = await run(RiskInput.model_validate(payload))
        return out.model_dump(mode="json")
    finally:
        await router.aclose()


@activity.defn(name="agent.execution")
async def execution_activity(payload: dict[str, Any]) -> dict[str, Any]:
    from packages.agents.llm_router import LLMRouter
    from packages.agents.runners import build_execution_runner

    router = LLMRouter()
    try:
        run = build_execution_runner(router)
        out = await run(ExecutionInput.model_validate(payload))
        return out.model_dump(mode="json")
    finally:
        await router.aclose()


ALL_ACTIVITIES = [
    research_activity,
    strategy_activity,
    risk_activity,
    execution_activity,
]


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------


@workflow.defn(name="TradeCycleWorkflow")
class TradeCycleWorkflow:
    """One end-to-end trade decision (Research → Strategy → Risk → HITL →
    Execution).

    Signals:
      - ``approve`` (list of symbols to approve)
      - ``deny`` (forces halt)
    """

    def __init__(self) -> None:
        self._approved_symbols: set[str] | None = None
        self._denied: bool = False

    # --- Signals ---
    @workflow.signal(name="approve")
    def on_approve(self, symbols: list[str]) -> None:
        self._approved_symbols = set(symbols)

    @workflow.signal(name="deny")
    def on_deny(self) -> None:
        self._denied = True

    # --- Main ---
    @workflow.run
    async def run(self, payload: TradeCycleInput) -> TradeCycleResult:
        # ---- Research ----
        research_payload = {
            "decision_id": payload.decision_id,
            "symbols": payload.symbols,
        }
        # Research output isn't gating downstream agents in v3.1 — it's
        # recorded in the audit trail by the activity itself. Awaited for
        # ordering / OTel spans.
        _research_out = await workflow.execute_activity(
            research_activity,
            research_payload,
            start_to_close_timeout=timedelta(seconds=120),
            retry_policy=RetryPolicy(
                initial_interval=timedelta(seconds=2),
                maximum_attempts=3,
            ),
        )

        # ---- Strategy ----
        strategy_payload = {
            "decision_id": payload.decision_id,
            "regime": payload.regime,
            "universe": payload.symbols,
            "features": payload.features,
        }
        strategy_out = await workflow.execute_activity(
            strategy_activity,
            strategy_payload,
            start_to_close_timeout=timedelta(seconds=120),
            retry_policy=RetryPolicy(
                initial_interval=timedelta(seconds=2),
                maximum_attempts=3,
            ),
        )
        signals = strategy_out["signals"]
        if not signals:
            return TradeCycleResult(
                decision_id=payload.decision_id,
                halted=False,
                n_signals=0,
            )

        # ---- Risk ----
        risk_payload = {
            "decision_id": payload.decision_id,
            "positions": payload.positions,
            "candidates": signals,
        }
        risk_out = await workflow.execute_activity(
            risk_activity,
            risk_payload,
            start_to_close_timeout=timedelta(seconds=60),
            retry_policy=RetryPolicy(
                initial_interval=timedelta(seconds=1),
                maximum_attempts=2,  # risk should be conservative on retry
            ),
        )
        if risk_out.get("halted") or not risk_out.get("approved"):
            return TradeCycleResult(
                decision_id=payload.decision_id,
                halted=bool(risk_out.get("halted")),
                halt_reason=risk_out.get("halt_reason"),
                n_signals=len(signals),
                n_approved=0,
            )

        # ---- HITL approval (wait for signal or timeout) ----
        approved_candidates: list[dict[str, Any]] = risk_out["approved"]
        # Treat timeout as no-go (handled by `_approved_symbols is None` branch).
        with contextlib.suppress(TimeoutError):
            await workflow.wait_condition(
                lambda: self._approved_symbols is not None or self._denied,
                timeout=timedelta(seconds=payload.approval_timeout_seconds),
            )
        if self._denied:
            return TradeCycleResult(
                decision_id=payload.decision_id,
                halted=False,
                halt_reason="operator denied",
                n_signals=len(signals),
                n_approved=0,
            )
        if self._approved_symbols is None:
            # Implicit timeout → treat as no-go (safer than auto-approve).
            return TradeCycleResult(
                decision_id=payload.decision_id,
                halted=False,
                halt_reason="approval timeout",
                n_signals=len(signals),
                n_approved=0,
            )

        final = [s for s in approved_candidates if s["symbol"] in self._approved_symbols]
        if not final:
            return TradeCycleResult(
                decision_id=payload.decision_id,
                halted=False,
                halt_reason="no symbols approved by operator",
                n_signals=len(signals),
                n_approved=0,
            )

        # ---- Execution ----
        orders = [
            {"symbol": s["symbol"], "side": s["side"], "qty": 1.0}
            for s in final
        ]
        exec_payload = {
            "decision_id": payload.decision_id,
            "approved_orders": orders,
        }
        exec_out = await workflow.execute_activity(
            execution_activity,
            exec_payload,
            start_to_close_timeout=timedelta(seconds=60),
            retry_policy=RetryPolicy(
                initial_interval=timedelta(seconds=1),
                maximum_attempts=3,
            ),
        )

        return TradeCycleResult(
            decision_id=payload.decision_id,
            halted=False,
            n_signals=len(signals),
            n_approved=len(final),
            n_fills=len(exec_out.get("fills", [])),
        )


# ---------------------------------------------------------------------------
# Re-exports for the worker entrypoint
# ---------------------------------------------------------------------------


ALL_WORKFLOWS = [TradeCycleWorkflow]


# Silence unused-import warnings for runtime types referenced via schemas.
__all__ = [
    "ALL_ACTIVITIES",
    "ALL_WORKFLOWS",
    "ExecutionInput",
    "ExecutionOutput",
    "Order",
    "Position",
    "ResearchInput",
    "ResearchOutput",
    "RiskInput",
    "RiskOutput",
    "Signal",
    "StrategyInput",
    "StrategyOutput",
    "TradeCycleInput",
    "TradeCycleResult",
    "TradeCycleWorkflow",
    "execution_activity",
    "research_activity",
    "risk_activity",
    "strategy_activity",
]
