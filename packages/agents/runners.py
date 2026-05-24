"""LLM-backed agent runners (§5, issue #4).

Each runner is a thin shell around :class:`LLMRouter`:

  1. Build the prompt via ``packages.agents.prompts``.
  2. Ask the router for JSON.
  3. Validate against the Pydantic output schema. Force-override
     ``decision_id`` to the trusted server-side value so a hallucinated id
     can never leak into the audit log.
  4. On any validation or transport failure, return a SAFE DEFAULT
     (sentiment 0, no signals, no approvals, etc.). The orchestrator's
     risk halt then ensures we never trade on a broken agent.

These runners are designed to be drop-in substitutes for the stub
callables wired into :class:`AgentGraph`.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from uuid import uuid4

from pydantic import ValidationError

from packages.agents.llm_router import LLMError, LLMRouter
from packages.agents.prompts import (
    discovery_prompt,
    execution_prompt,
    research_prompt,
    risk_prompt,
    strategy_prompt,
)
from packages.shared.otel import span
from packages.shared.schemas import (
    DiscoveryInput,
    DiscoveryOutput,
    ExecutionInput,
    ExecutionOutput,
    Fill,
    PatternCandidate,
    ResearchInput,
    ResearchOutput,
    RiskInput,
    RiskOutput,
    StrategyInput,
    StrategyOutput,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Safe defaults — used when the model fails or returns invalid JSON.
# ---------------------------------------------------------------------------

def _safe_research(payload: ResearchInput) -> ResearchOutput:
    return ResearchOutput(
        decision_id=payload.decision_id,
        thesis="(agent fallback — neutral; no live signal)",
        sentiment=0.0,
        citations=[],
    )


def _safe_strategy(payload: StrategyInput) -> StrategyOutput:
    return StrategyOutput(decision_id=payload.decision_id, signals=[])


def _safe_risk(payload: RiskInput) -> RiskOutput:
    return RiskOutput(
        decision_id=payload.decision_id,
        approved=[],
        rejected=list(payload.candidates),
        halted=True,
        halt_reason="risk-agent fallback: rejecting all candidates",
    )


def _safe_execution(payload: ExecutionInput) -> ExecutionOutput:
    # Execution fallback never invents fills.
    return ExecutionOutput(
        decision_id=payload.decision_id,
        fills=[],
        audit_id=uuid4(),
    )


def _safe_discovery(payload: DiscoveryInput) -> DiscoveryOutput:
    # Discovery falls back to “no new patterns” — the order path doesn't
    # depend on us so the safest output is silence.
    return DiscoveryOutput(
        decision_id=payload.decision_id,
        patterns=[],
        notes="(discovery fallback \u2014 no patterns proposed)",
    )


# ---------------------------------------------------------------------------
# Generic runner
# ---------------------------------------------------------------------------

async def _run(
    *,
    router: LLMRouter,
    agent: str,
    decision_id_str: str,
    prompt: str,
    output_model: type,
    safe_default,  # callable that returns a fallback instance
    payload,
):
    with span(f"agent.{agent}", {"decision_id": decision_id_str}) as s:
        try:
            raw = await router.generate_json(
                agent=agent,
                prompt=prompt,
                decision_id=decision_id_str,
            )
        except LLMError as e:
            log.warning("agent %s LLM chain failed: %s", agent, e)
            s.set_attribute("agent.fallback", "llm_error")
            return safe_default(payload)

        # Trust server-side decision_id; never accept the model's value.
        raw["decision_id"] = decision_id_str
        try:
            return output_model.model_validate(raw)
        except ValidationError as e:
            log.warning("agent %s invalid JSON: %s", agent, e)
            s.set_attribute("agent.fallback", "validation_error")
            return safe_default(payload)


# ---------------------------------------------------------------------------
# Public factories — match the callable shapes wired into AgentGraph.
# ---------------------------------------------------------------------------

def build_research_runner(router: LLMRouter) -> Callable[[ResearchInput], Awaitable[ResearchOutput]]:
    async def _research(payload: ResearchInput) -> ResearchOutput:
        return await _run(
            router=router,
            agent="research",
            decision_id_str=str(payload.decision_id),
            prompt=research_prompt(payload),
            output_model=ResearchOutput,
            safe_default=_safe_research,
            payload=payload,
        )

    return _research


def build_strategy_runner(router: LLMRouter) -> Callable[[StrategyInput], Awaitable[StrategyOutput]]:
    async def _strategy(payload: StrategyInput) -> StrategyOutput:
        return await _run(
            router=router,
            agent="strategy",
            decision_id_str=str(payload.decision_id),
            prompt=strategy_prompt(payload),
            output_model=StrategyOutput,
            safe_default=_safe_strategy,
            payload=payload,
        )

    return _strategy


def build_risk_runner(router: LLMRouter) -> Callable[[RiskInput], Awaitable[RiskOutput]]:
    async def _risk(payload: RiskInput) -> RiskOutput:
        return await _run(
            router=router,
            agent="risk",
            decision_id_str=str(payload.decision_id),
            prompt=risk_prompt(payload),
            output_model=RiskOutput,
            safe_default=_safe_risk,
            payload=payload,
        )

    return _risk


def build_discovery_runner(router: LLMRouter) -> Callable[[DiscoveryInput], Awaitable[DiscoveryOutput]]:
    """Advisory pattern-discovery runner.

    Unlike the four order-path agents, Discovery is allowed to fail loudly
    (returning the safe default ``patterns=[]``) without halting trading.
    The caller is expected to log the output for offline review.
    """

    async def _discovery(payload: DiscoveryInput) -> DiscoveryOutput:
        raw = await _run(
            router=router,
            agent="discovery",
            decision_id_str=str(payload.decision_id),
            prompt=discovery_prompt(payload),
            output_model=DiscoveryOutput,
            safe_default=_safe_discovery,
            payload=payload,
        )
        # Strict universe + feature gate — drop any hallucinated symbols or
        # feature keys so the dashboard never shows a fictional ticker.
        allowed_symbols = {s.upper() for s in payload.universe}
        allowed_features = set(payload.features.keys())
        clean: list[PatternCandidate] = []
        for p in raw.patterns:
            syms = [s.upper() for s in p.symbols if s and s.upper() in allowed_symbols]
            feats = [k for k in p.feature_keys if k in allowed_features]
            if not syms or not feats:
                continue  # drop — nothing grounded to act on
            clean.append(p.model_copy(update={"symbols": syms, "feature_keys": feats}))
        return raw.model_copy(update={"patterns": clean})

    return _discovery


def build_execution_runner(router: LLMRouter) -> Callable[[ExecutionInput], Awaitable[ExecutionOutput]]:
    async def _execution(payload: ExecutionInput) -> ExecutionOutput:
        # We let the LLM plan slicing, but always fill audit_id locally so
        # downstream consumers can rely on it.
        out = await _run(
            router=router,
            agent="execution",
            decision_id_str=str(payload.decision_id),
            prompt=execution_prompt(payload),
            output_model=ExecutionOutput,
            safe_default=_safe_execution,
            payload=payload,
        )
        # Force a server-side audit_id.
        out = out.model_copy(update={"audit_id": uuid4()})
        return out

    return _execution


# ---------------------------------------------------------------------------
# Optional: a tiny helper for tests / call paths that need a synthetic Fill
# (used by the broker-side path; LLM never invents fills).
# ---------------------------------------------------------------------------

def synthetic_fill(symbol: str, side: str, qty: float, price: float) -> Fill:
    return Fill(
        symbol=symbol,
        side=side,  # type: ignore[arg-type]
        qty=qty,
        price=price,
        timestamp=datetime.now(UTC),
    )
