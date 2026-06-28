"""Real local-LLM thesis enrichment for the top research candidates.

After the rule-based sweep ranks candidates, this module asks a local
Ollama model (via the established :class:`LLMRouter.generate_json`
pattern) to write a genuine plain-language thesis for the TOP N
candidates and to sanity-check them with a small, bounded confidence
nudge.

Design contract (every bullet is load-bearing):

  * ON BY DEFAULT via ``ENABLE_AGENT_LLM`` (truthy by default), but it
    only actually runs when a quick reachability probe says Ollama is
    alive AND at least one model in the agent's chain is installed
    (reuses :func:`installed_matches` + the router's ``/api/tags``
    inventory — no re-implementation).
  * FAIL SAFE everywhere. A cold / missing / erroring / non-JSON LLM
    must NEVER block or slow a cycle, never be treated as bearish, and
    never fabricate. On ANY uncertainty we keep the existing rule-based
    thesis and a ``confidence_adjustment`` of 0.0.
  * Bounded. ``confidence_adjustment`` is CLAMPED to [-0.10, +0.10] so
    the model can nudge, never override. A hard per-call timeout
    (``LLM_THESIS_TIMEOUT_S``) plus a total-budget guard keep the
    enrichment from ever delaying the loop more than a bounded amount.

The model REASONS over the signals the pipeline already collected; it
does NOT fetch new data.
"""

from __future__ import annotations

import contextlib
import logging
import os
import time
from typing import TYPE_CHECKING, Any

from packages.agents.llm_router import (
    EMERGENCY_FALLBACK_MODELS,
    LLMError,
    LLMRouter,
    installed_matches,
)
from packages.agents.model_profiles import chain_for

if TYPE_CHECKING:  # pragma: no cover - import only for typing
    from packages.agents.research_sweep import Candidate

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunables (env-overridable). Defaults chosen so the feature is ON but the
# cost/time is tightly bounded.
# ---------------------------------------------------------------------------

# How many of the (already-ranked) candidates get a real LLM thesis.
LLM_THESIS_TOP_N = int(os.getenv("LLM_THESIS_TOP_N") or 5)

# Hard per-call ceiling handed to the router's asyncio.wait_for.
LLM_THESIS_TIMEOUT_S = int(os.getenv("LLM_THESIS_TIMEOUT_S") or 20)

# Total wall-clock budget across ALL top-N enrichments. On exhaustion the
# remaining candidates keep their rule-based thesis. Defaults to TOP_N *
# per-call timeout so a single slow model can't blow the whole loop.
LLM_THESIS_TOTAL_BUDGET_S = float(
    os.getenv("LLM_THESIS_TOTAL_BUDGET_S")
    or float(LLM_THESIS_TIMEOUT_S * max(1, LLM_THESIS_TOP_N))
)

# The model can NUDGE, never override. Bound in [-CLAMP, +CLAMP].
CONFIDENCE_ADJ_CLAMP = 0.10

# Which agent chain to use for thesis generation. Research is the natural
# fit (heavy-reasoning chain on capable hardware).
THESIS_AGENT = "research"

_VALID_DIRECTIONS = ("bull", "bear", "neutral")


def llm_enabled() -> bool:
    """Return whether the ``ENABLE_AGENT_LLM`` flag is ON.

    Defaults to ON. ``false`` / ``0`` / ``no`` / ``off`` (case-insensitive)
    explicitly disable; everything else (including unset) is ON.
    """
    raw = os.getenv("ENABLE_AGENT_LLM")
    if raw is None:
        return True
    return raw.strip().lower() not in {"false", "0", "no", "off", ""}


async def active_model(router: LLMRouter, agent: str = THESIS_AGENT) -> str | None:
    """Return the model that would actually serve ``agent``, or ``None``.

    ``None`` means "LLM not ready" — either Ollama is unreachable (the
    ``/api/tags`` probe failed) or no model in the agent's declared chain
    (nor the emergency fallback tier) is installed. NEVER raises.

    The walk order mirrors the router's own
    :meth:`LLMRouter.generate_json`: declared chain first, then the
    emergency fallback tier — so the reported model matches what the
    router would pick.
    """
    try:
        installed = await router._installed_models()
    except Exception as exc:  # pragma: no cover - defensive
        log.debug("active_model probe failed: %s", exc)
        return None
    if not installed:
        # None (unreachable) or empty (no models) → not ready.
        return None
    try:
        chain = chain_for(agent)
    except Exception as exc:  # pragma: no cover - defensive
        log.debug("chain_for(%s) failed: %s", agent, exc)
        return None
    candidates = (chain.primary, chain.backup, chain.quantized, *EMERGENCY_FALLBACK_MODELS)
    for model in candidates:
        if model and installed_matches(model, installed):
            return model
    return None


def _build_prompt(c: Candidate) -> str:
    """Render a candidate's already-gathered signals into a plain-language
    prompt that asks for a STRUCTURED JSON verdict.

    No numbers are invented — every field comes off the candidate the
    pipeline already produced. The instruction is deliberately
    non-technical so the resulting thesis is "easy for someone that
    doesn't know what anything does".
    """
    analyst = ""
    if c.analyst_mean_rating:
        analyst = (
            f"analyst mean rating {c.analyst_mean_rating:.2f} "
            f"(1=Strong Buy, 5=Strong Sell) from {c.analyst_num} analysts"
        )
        if c.analyst_recent_action:
            analyst += f", recent {c.analyst_recent_action}"
            if c.analyst_recent_firm:
                analyst += f" by {c.analyst_recent_firm}"
    insider = ""
    if c.insider_form4_30d:
        net = c.insider_net_shares
        side = "net buying" if net > 0 else ("net selling" if net < 0 else "mixed")
        insider = (
            f"{c.insider_form4_30d} insider Form 4 filings in 30d ({side}, "
            f"buys={c.insider_buy_count}, sells={c.insider_sell_count})"
        )
    headlines = "; ".join((c.sample_headlines or [])[:5]) or "none gathered"

    signals = [
        f"ticker: {c.symbol}",
        f"crowd sentiment score: {c.sentiment_score:+.2f} (range -1..+1)",
        f"headline mentions: {c.mentions}",
        f"reddit author trust: {c.reddit_trust:.2f} (range 0..1)",
        f"news-corroborated: {'yes' if c.corroborated else 'no'} "
        f"(score {c.corroboration_score:.2f})",
    ]
    if analyst:
        signals.append(f"analysts: {analyst}")
    if insider:
        signals.append(f"insiders: {insider}")
    if c.stocktwits_trending:
        signals.append(f"trending on StockTwits (watchlist {c.stocktwits_watchlist})")
    if c.yahoo_news_count:
        signals.append(f"fresh Yahoo headlines: {c.yahoo_news_count}")
    signals.append(f"sample headlines: {headlines}")

    signal_block = "\n".join(f"- {s}" for s in signals)

    return (
        "You are a careful, plain-spoken investing assistant. You are given "
        "signals that were ALREADY gathered for one stock. Reason over ONLY "
        "these signals — do not invent prices, dates, or facts that are not "
        "listed.\n\n"
        f"SIGNALS:\n{signal_block}\n\n"
        "Write a short verdict for a non-expert who does not know any jargon. "
        "Respond with a SINGLE JSON object and nothing else, with EXACTLY "
        "these keys:\n"
        '  "thesis": one or two plain-language sentences explaining what the '
        "signals suggest and why, in everyday words.\n"
        '  "direction": one of "bull", "bear", or "neutral".\n'
        '  "confidence_adjustment": a small number between -0.10 and 0.10. '
        "Positive if the signals look more encouraging than the crowd score "
        "implies, negative if they look worse (a useful contrarian check), "
        "near zero if they roughly agree.\n"
        '  "risk_flag": a SHORT phrase naming the single biggest risk (e.g. '
        '"earnings tomorrow", "very wide spread", "thin volume"), or null if '
        "nothing stands out.\n"
    )


def _coerce_verdict(raw: Any) -> dict[str, Any] | None:
    """Validate + normalize an LLM JSON verdict into a safe dict.

    Returns ``None`` when the payload is unusable (so the caller keeps the
    rule-based thesis). Enforces the clamp and makes the
    ``confidence_adjustment`` sign consistent with ``direction`` so a
    "bear" verdict can only ever LOWER the score and a "bull" verdict can
    only ever raise it — the contrarian sanity-check value, fail-safe.
    """
    if not isinstance(raw, dict):
        return None
    thesis = raw.get("thesis")
    if not isinstance(thesis, str) or not thesis.strip():
        return None
    thesis = thesis.strip()

    direction = str(raw.get("direction") or "").strip().lower()
    if direction not in _VALID_DIRECTIONS:
        direction = "neutral"

    adj_raw = raw.get("confidence_adjustment", 0.0)
    try:
        adj = float(adj_raw)
    except (TypeError, ValueError):
        adj = 0.0
    if adj != adj or adj in (float("inf"), float("-inf")):  # NaN / inf guard
        adj = 0.0
    # Clamp to the bounded nudge window.
    adj = max(-CONFIDENCE_ADJ_CLAMP, min(CONFIDENCE_ADJ_CLAMP, adj))
    # Make sign consistent with direction so a bearish/neutral verdict can
    # never accidentally raise a bullish-scored candidate (and vice versa).
    if direction == "bear":
        adj = min(0.0, adj)
    elif direction == "bull":
        adj = max(0.0, adj)

    risk = raw.get("risk_flag")
    if risk is None:
        risk_flag = ""
    elif isinstance(risk, str):
        risk_flag = risk.strip()[:120]
    else:
        risk_flag = str(risk)[:120]

    return {
        "thesis": thesis[:600],
        "direction": direction,
        "confidence_adjustment": round(adj, 4),
        "risk_flag": risk_flag,
    }


async def enrich_top_candidates(
    candidates: list[Candidate],
    *,
    router: LLMRouter | None = None,
    agent: str = THESIS_AGENT,
    top_n: int | None = None,
    timeout_s: int | None = None,
    total_budget_s: float | None = None,
) -> dict[str, Any]:
    """Enrich the top-N candidates IN PLACE with a real LLM thesis.

    Returns a small status dict the dashboard can render::

        {"enabled": bool, "active": bool, "model": str|None,
         "enriched": int, "attempted": int, "reason": str}

    NEVER raises. On any failure (flag off, Ollama unreachable, model
    missing, timeout, non-JSON, budget exhausted) the affected candidates
    keep their rule-based thesis (``thesis_source == "rule"``,
    ``confidence_adjustment == 0.0``) and ``active`` reflects whether real
    reasoning ran at all.
    """
    meta: dict[str, Any] = {
        "enabled": llm_enabled(),
        "active": False,
        "model": None,
        "enriched": 0,
        "attempted": 0,
        "reason": "",
    }
    if not candidates:
        meta["reason"] = "no candidates"
        return meta
    if not meta["enabled"]:
        meta["reason"] = "ENABLE_AGENT_LLM is off"
        return meta

    top_n = LLM_THESIS_TOP_N if top_n is None else top_n
    timeout_s = LLM_THESIS_TIMEOUT_S if timeout_s is None else timeout_s
    total_budget_s = (
        LLM_THESIS_TOTAL_BUDGET_S if total_budget_s is None else total_budget_s
    )
    if top_n <= 0:
        meta["reason"] = "top_n <= 0"
        return meta

    own_router = router is None
    if own_router:
        router = LLMRouter()

    try:
        model = await active_model(router, agent)
        if model is None:
            meta["reason"] = "ollama unreachable or model not installed"
            return meta
        meta["active"] = True
        meta["model"] = model

        deadline = time.monotonic() + max(0.0, total_budget_s)
        targets = candidates[:top_n]
        for c in targets:
            if time.monotonic() >= deadline:
                meta["reason"] = "total budget exhausted"
                break
            meta["attempted"] += 1
            try:
                raw = await router.generate_json(
                    agent=agent,
                    prompt=_build_prompt(c),
                    decision_id=f"thesis-{c.symbol}",
                    timeout_seconds=timeout_s,
                )
            except LLMError as exc:
                log.info("thesis LLM failed for %s (keeping rule thesis): %s", c.symbol, exc)
                continue
            except Exception as exc:  # pragma: no cover - belt-and-braces
                log.warning("thesis LLM unexpected error for %s: %s", c.symbol, exc)
                continue

            verdict = _coerce_verdict(raw)
            if verdict is None:
                log.info("thesis LLM returned unusable JSON for %s; keeping rule thesis", c.symbol)
                continue

            c.thesis = verdict["thesis"]
            c.direction = verdict["direction"]
            c.risk_flag = verdict["risk_flag"]
            c.confidence_adjustment = verdict["confidence_adjustment"]
            c.thesis_source = f"llm:{model}"
            meta["enriched"] += 1

        if not meta["reason"]:
            meta["reason"] = f"enriched {meta['enriched']}/{meta['attempted']}"
        return meta
    except Exception as exc:  # pragma: no cover - last-ditch fail-safe
        log.warning("enrich_top_candidates crashed (ignored): %s", exc)
        meta["reason"] = f"error: {exc.__class__.__name__}"
        return meta
    finally:
        if own_router and router is not None:
            with contextlib.suppress(Exception):  # pragma: no cover - defensive
                await router.aclose()
