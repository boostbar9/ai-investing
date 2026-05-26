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
from typing import Any
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
from packages.persistence.audit import log_decision
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

def _validation_repair_hint(err: ValidationError, output_model: type) -> str:
    """Build a short repair instruction from a Pydantic validation error.

    Used to give the LLM a second chance: we re-prompt with the original
    instructions plus an explicit list of the fields it got wrong. Keeps
    the hint short so we don't blow up the context window.
    """
    missing: list[str] = []
    other: list[str] = []
    for e in err.errors()[:6]:  # cap at 6 issues to keep hint small
        loc = ".".join(str(p) for p in e.get("loc", ()))
        if e.get("type") == "missing":
            missing.append(loc)
        else:
            other.append(f"{loc}: {e.get('msg', 'invalid')}")
    parts: list[str] = []
    if missing:
        parts.append("missing required field(s): " + ", ".join(missing))
    if other:
        parts.append("invalid: " + "; ".join(other))
    schema_name = getattr(output_model, "__name__", "the output schema")
    detail = "; ".join(parts) if parts else "schema mismatch"
    return (
        f"Your previous response did not match {schema_name} ({detail}). "
        "Respond again with valid JSON that includes every required field. "
        "Do not omit empty arrays -- return [] explicitly."
    )


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
        # ---- Attempt 1 -----------------------------------------------------
        try:
            raw = await router.generate_json(
                agent=agent,
                prompt=prompt,
                decision_id=decision_id_str,
            )
        except LLMError as e:
            log.warning("agent %s LLM chain failed (will retry once): %s", agent, e)
            s.set_attribute("agent.retry_reason", "llm_error")
            log_decision(
                decision_id=decision_id_str, agent=agent, attempt=1,
                prompt=prompt, raw_response=None,
                validation_ok=False, validation_error=f"llm_error: {e}",
            )
            try:
                raw = await router.generate_json(
                    agent=agent,
                    prompt=prompt,
                    decision_id=decision_id_str,
                )
            except LLMError as e2:
                log.warning("agent %s LLM retry also failed: %s", agent, e2)
                s.set_attribute("agent.fallback", "llm_error")
                log_decision(
                    decision_id=decision_id_str, agent=agent, attempt=2,
                    prompt=prompt, raw_response=None,
                    validation_ok=False, validation_error=f"llm_error: {e2}",
                    extra={"fallback": "llm_error"},
                )
                return safe_default(payload)

        # Trust server-side decision_id; never accept the model's value.
        raw["decision_id"] = decision_id_str
        try:
            out = output_model.model_validate(raw)
            log_decision(
                decision_id=decision_id_str, agent=agent, attempt=1,
                prompt=prompt, raw_response=raw, validation_ok=True,
            )
            return out
        except ValidationError as e:
            log.warning(
                "agent %s invalid JSON (will retry once with repair hint): %s", agent, e
            )
            s.set_attribute("agent.retry_reason", "validation_error")
            log_decision(
                decision_id=decision_id_str, agent=agent, attempt=1,
                prompt=prompt, raw_response=raw,
                validation_ok=False, validation_error=str(e),
            )
            # ---- Attempt 2 (validation repair) -----------------------------
            repair_prompt = prompt + "\n\n" + _validation_repair_hint(e, output_model)
            try:
                raw2 = await router.generate_json(
                    agent=agent,
                    prompt=repair_prompt,
                    decision_id=decision_id_str,
                )
            except LLMError as e_llm:
                log.warning("agent %s repair retry LLM error: %s", agent, e_llm)
                s.set_attribute("agent.fallback", "llm_error_on_repair")
                log_decision(
                    decision_id=decision_id_str, agent=agent, attempt=2,
                    prompt=repair_prompt, raw_response=None,
                    validation_ok=False, validation_error=f"llm_error: {e_llm}",
                    extra={"fallback": "llm_error_on_repair"},
                )
                return safe_default(payload)
            raw2["decision_id"] = decision_id_str
            try:
                out = output_model.model_validate(raw2)
                s.set_attribute("agent.recovered_via_repair", True)
                log.info("agent %s recovered via repair retry", agent)
                log_decision(
                    decision_id=decision_id_str, agent=agent, attempt=2,
                    prompt=repair_prompt, raw_response=raw2, validation_ok=True,
                    extra={"recovered_via_repair": True},
                )
                return out
            except ValidationError as e2:
                log.warning("agent %s repair retry still invalid: %s", agent, e2)
                s.set_attribute("agent.fallback", "validation_error")
                log_decision(
                    decision_id=decision_id_str, agent=agent, attempt=2,
                    prompt=repair_prompt, raw_response=raw2,
                    validation_ok=False, validation_error=str(e2),
                    extra={"fallback": "validation_error"},
                )
                return safe_default(payload)


# ---------------------------------------------------------------------------
# Public factories — match the callable shapes wired into AgentGraph.
# ---------------------------------------------------------------------------

def build_research_runner(
    router: LLMRouter,
    *,
    scorecard_summary: dict[str, Any] | None = None,
) -> Callable[[ResearchInput], Awaitable[ResearchOutput]]:
    async def _research(payload: ResearchInput) -> ResearchOutput:
        return await _run(
            router=router,
            agent="research",
            decision_id_str=str(payload.decision_id),
            prompt=research_prompt(payload, scorecard_summary=scorecard_summary),
            output_model=ResearchOutput,
            safe_default=_safe_research,
            payload=payload,
        )

    return _research


def build_strategy_runner(
    router: LLMRouter,
    *,
    scorecard_summary: dict[str, Any] | None = None,
) -> Callable[[StrategyInput], Awaitable[StrategyOutput]]:
    async def _strategy(payload: StrategyInput) -> StrategyOutput:
        return await _run(
            router=router,
            agent="strategy",
            decision_id_str=str(payload.decision_id),
            prompt=strategy_prompt(payload, scorecard_summary=scorecard_summary),
            output_model=StrategyOutput,
            safe_default=_safe_strategy,
            payload=payload,
        )

    return _strategy


def build_risk_runner(
    router: LLMRouter,
    *,
    scorecard_summary: dict[str, Any] | None = None,
) -> Callable[[RiskInput], Awaitable[RiskOutput]]:
    async def _risk(payload: RiskInput) -> RiskOutput:
        return await _run(
            router=router,
            agent="risk",
            decision_id_str=str(payload.decision_id),
            prompt=risk_prompt(payload, scorecard_summary=scorecard_summary),
            output_model=RiskOutput,
            safe_default=_safe_risk,
            payload=payload,
        )

    return _risk


def build_discovery_runner(
    router: LLMRouter,
    *,
    scorecard_summary: dict[str, Any] | None = None,
) -> Callable[[DiscoveryInput], Awaitable[DiscoveryOutput]]:
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
            prompt=discovery_prompt(payload, scorecard_summary=scorecard_summary),
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


def build_execution_runner(
    router: LLMRouter,
    *,
    scorecard_summary: dict[str, Any] | None = None,
) -> Callable[[ExecutionInput], Awaitable[ExecutionOutput]]:
    async def _execution(payload: ExecutionInput) -> ExecutionOutput:
        # We let the LLM plan slicing, but always fill audit_id locally so
        # downstream consumers can rely on it.
        out = await _run(
            router=router,
            agent="execution",
            decision_id_str=str(payload.decision_id),
            prompt=execution_prompt(payload, scorecard_summary=scorecard_summary),
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
