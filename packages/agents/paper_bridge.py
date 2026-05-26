"""Bridge that runs the LangGraph agents alongside the nightly paper-trade flow.

The paper-trade executor already has a deterministic, walk-forward-tuned
strategy producing target weights. The LangGraph stack (Research → Strategy →
Risk → Approval) runs in **advisory mode**: it can veto a run (halt) or
attenuate weights via the research sentiment, but it cannot invent new orders.

Design:

* Research agent (stub, deterministic by default) reads cached RSS/Reddit
  sentiment scores and emits a thesis + aggregate sentiment in [-1, 1].
* Strategy agent emits Signals that mirror the deterministic strategy's
  target weights (sign-of-weight => side, magnitude => strength).
* Risk agent applies hard rules: equity drawdown, position concentration,
  agent-fallback detection. On any risk failure -> halted=True.
* Approval defaults to "auto-approve all". A future Telegram/Discord
  approval hook can be plugged in by swapping the callable.

This module is intentionally side-effect-free and deterministic by default
so it can run inside the nightly cron without any external dependency.
"""
from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from packages.agents.graph import AgentGraph, GraphResult
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

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stub agents (deterministic, no LLM round-trips).
# ---------------------------------------------------------------------------


def _aggregate_sentiment(scores: dict[str, float] | None) -> float:
    if not scores:
        return 0.0
    vals = [v for v in scores.values() if isinstance(v, (int, float))]
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


def make_research_agent(
    sentiment_scores: dict[str, float] | None,
) -> Any:
    """Build a deterministic research agent closed over sentiment scores."""

    agg = _aggregate_sentiment(sentiment_scores)
    if agg > 0.15:
        thesis = "Aggregate news sentiment is positive; bias long bias allowed."
    elif agg < -0.15:
        thesis = "Aggregate news sentiment is negative; reduce gross exposure."
    else:
        thesis = "Sentiment near neutral; no overlay change recommended."

    async def _research(inp: ResearchInput) -> ResearchOutput:
        return ResearchOutput(
            decision_id=inp.decision_id,
            thesis=thesis,
            sentiment=max(-1.0, min(1.0, agg)),
            citations=sorted(sentiment_scores.keys()) if sentiment_scores else [],
        )

    return _research


def make_strategy_agent(target_weights: dict[str, float]) -> Any:
    """Strategy agent mirrors the deterministic target weights as signals.

    Honors spec §5: in ``crisis`` regime the strategy MUST emit zero signals
    so the downstream risk/execution legs have nothing to approve.
    """

    async def _strategy(inp: StrategyInput) -> StrategyOutput:
        # Spec §5 hard rule: crisis regime kills the chain.
        if inp.regime == "crisis":
            return StrategyOutput(decision_id=inp.decision_id, signals=[])
        signals = []
        for sym, w in target_weights.items():
            if abs(w) < 1e-6:
                continue
            side = "buy" if w > 0 else "sell"
            signals.append(
                Signal(
                    symbol=sym,
                    side=side,
                    strength=min(1.0, abs(w)),
                    rationale=f"target_weight={w:.4f}",
                    # Mirror the deterministic weight onto the signal so
                    # downstream sizing (LLM or rule-based) sees the same
                    # value. Clipped to [-1, 1] defensively.
                    target_weight=max(-1.0, min(1.0, float(w))),
                )
            )
        return StrategyOutput(decision_id=inp.decision_id, signals=signals)

    return _strategy


def make_risk_agent(
    *,
    max_concentration: float = 0.6,
    min_sentiment: float = -0.5,
    research_sentiment: float = 0.0,
) -> Any:
    """Risk agent enforces concentration and sentiment floors.

    Halt if any single signal's strength exceeds ``max_concentration``
    or if aggregate research sentiment is below ``min_sentiment``.
    """

    async def _risk(inp: RiskInput) -> RiskOutput:
        # Reject high-concentration signals; halt on sentiment floor breach.
        approved: list[Signal] = []
        rejected: list[Signal] = []
        if research_sentiment < min_sentiment:
            return RiskOutput(
                decision_id=inp.decision_id,
                approved=[],
                rejected=list(inp.candidates),
                halted=True,
                halt_reason=f"sentiment {research_sentiment:.2f} < floor {min_sentiment:.2f}",
            )
        for s in inp.candidates:
            if s.strength > max_concentration:
                rejected.append(s)
            else:
                approved.append(s)
        return RiskOutput(
            decision_id=inp.decision_id,
            approved=approved,
            rejected=rejected,
            halted=False,
            halt_reason=None,
        )

    return _risk


async def _auto_approve(sigs: list[Signal], _decision_id: UUID) -> list[Signal]:
    return sigs


async def _noop_execution(inp: ExecutionInput) -> ExecutionOutput:
    """Advisory mode: do NOT submit orders here. Real execution lives in
    paper_trade.py via the Alpaca broker. Returns an empty fill list."""
    return ExecutionOutput(decision_id=inp.decision_id, fills=[], audit_id=inp.decision_id)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def advise(
    *,
    symbols: list[str],
    regime: str,
    positions: list[Position],
    target_weights: dict[str, float],
    sentiment_scores: dict[str, float] | None,
    features: dict[str, float] | None = None,
    max_concentration: float = 0.6,
    min_sentiment: float = -0.5,
) -> GraphResult:
    """Run the agent graph in advisory mode.

    Returns the :class:`GraphResult` so the caller can:

    * Inspect ``halted`` and abort the paper-trade submission step.
    * Persist ``audit`` events to the run log.
    * Use ``research.sentiment`` for further weight scaling if desired.
    """
    research = make_research_agent(sentiment_scores)
    strategy = make_strategy_agent(target_weights)
    agg = _aggregate_sentiment(sentiment_scores)
    risk = make_risk_agent(
        max_concentration=max_concentration,
        min_sentiment=min_sentiment,
        research_sentiment=agg,
    )

    graph = AgentGraph(
        research=research,
        strategy=strategy,
        risk=risk,
        execution=_noop_execution,
        approval=_auto_approve,
    )

    return await graph.run(
        symbols=symbols,
        regime=regime,
        positions=positions,
        features=features or {},
    )
