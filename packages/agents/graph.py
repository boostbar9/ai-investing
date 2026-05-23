"""LangGraph + Temporal agent graph (§5).

Phase 3 wiring: Research → Strategy → Risk → (HITL approval) → Execution.

This module exposes a pure-Python orchestrator so unit tests don't need a
running Temporal worker. ``apps/api`` registers the same nodes as Temporal
activities; the topology is identical.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Awaitable, Callable
from uuid import UUID, uuid4

from packages.shared.audit import AuditEvent
from packages.shared.otel import span
from packages.shared.schemas import (
    ExecutionInput,
    ExecutionOutput,
    Position,
    ResearchInput,
    ResearchOutput,
    RiskInput,
    RiskOutput,
    Signal,
    StrategyInput,
    StrategyOutput,
)

# Each callable is an agent that emits a span and returns a typed schema.
ResearchFn = Callable[[ResearchInput], Awaitable[ResearchOutput]]
StrategyFn = Callable[[StrategyInput], Awaitable[StrategyOutput]]
RiskFn = Callable[[RiskInput], Awaitable[RiskOutput]]
ExecutionFn = Callable[[ExecutionInput], Awaitable[ExecutionOutput]]
ApprovalFn = Callable[[list[Signal], UUID], Awaitable[list[Signal]]]


@dataclass
class GraphResult:
    decision_id: UUID
    research: ResearchOutput
    strategy: StrategyOutput
    risk: RiskOutput
    execution: ExecutionOutput | None
    audit: list[AuditEvent] = field(default_factory=list)
    halted: bool = False


@dataclass
class AgentGraph:
    research: ResearchFn
    strategy: StrategyFn
    risk: RiskFn
    execution: ExecutionFn
    approval: ApprovalFn

    async def run(
        self,
        *,
        symbols: list[str],
        regime: str,
        positions: list[Position],
        features: dict[str, float],
    ) -> GraphResult:
        decision_id = uuid4()
        audit: list[AuditEvent] = []

        with span("graph.run", {"decision_id": str(decision_id), "regime": regime}):
            # 1. Research
            r_in = ResearchInput(decision_id=decision_id, symbols=symbols)
            research = await self.research(r_in)
            audit.append(
                AuditEvent(
                    decision_id=decision_id,
                    actor="research",
                    event_type="agent_call",
                    payload={"sentiment": research.sentiment, "n_citations": len(research.citations)},
                )
            )

            # 2. Strategy
            s_in = StrategyInput(
                decision_id=decision_id,
                regime=regime,  # type: ignore[arg-type]
                universe=symbols,
                features=features,
            )
            strategy = await self.strategy(s_in)
            audit.append(
                AuditEvent(
                    decision_id=decision_id,
                    actor="strategy",
                    event_type="agent_call",
                    payload={"n_signals": len(strategy.signals)},
                )
            )

            # 3. Risk
            risk = await self.risk(
                RiskInput(decision_id=decision_id, positions=positions, candidates=strategy.signals)
            )
            audit.append(
                AuditEvent(
                    decision_id=decision_id,
                    actor="risk",
                    event_type="agent_call",
                    payload={
                        "approved": len(risk.approved),
                        "rejected": len(risk.rejected),
                        "halted": risk.halted,
                    },
                )
            )
            if risk.halted or not risk.approved:
                return GraphResult(decision_id, research, strategy, risk, None, audit, halted=risk.halted)

            # 4. HITL approval (Telegram / Discord)
            approved = await self.approval(risk.approved, decision_id)
            audit.append(
                AuditEvent(
                    decision_id=decision_id,
                    actor="operator",
                    event_type="approval",
                    payload={"approved": len(approved), "denied": len(risk.approved) - len(approved)},
                )
            )
            if not approved:
                return GraphResult(decision_id, research, strategy, risk, None, audit, halted=False)

            # 5. Execution
            from packages.shared.schemas import Order  # local import to avoid cycle

            orders = [Order(symbol=s.symbol, side=s.side, qty=1.0) for s in approved]
            execution = await self.execution(
                ExecutionInput(decision_id=decision_id, approved_orders=orders)
            )
            audit.append(
                AuditEvent(
                    decision_id=decision_id,
                    actor="execution",
                    event_type="order",
                    payload={"n_fills": len(execution.fills)},
                )
            )
            return GraphResult(decision_id, research, strategy, risk, execution, audit, halted=False)
