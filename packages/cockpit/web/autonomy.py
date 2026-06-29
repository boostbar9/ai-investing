"""Always-On Brain: continuous autonomous research + curiosity loop.

This module makes the cockpit *think on its own* between manual triggers.
It runs two cooperating background tasks plus a small "Curiosity" agent:

  1. **Research sweep loop** \u2014 every ``sweep_market_seconds`` during
     market hours, every ``sweep_off_seconds`` otherwise. Calls
     ``packages.agents.research_sweep.run_sweep()`` to pull fresh
     Reddit + Yahoo News + StockTwits + insider data, applies trust
     scoring, persists a fresh ``data/research_sweep.json``, and
     pushes a chatter entry summarizing the top candidates.

  2. **Curiosity agent** \u2014 after each sweep, scans the new candidate
     list and recent chatter to pick the most *interesting* symbol to
     dig deeper on. "Interesting" = high confidence OR high
     corroboration OR fresh insider action OR analyst upgrade. It
     then nudges the agent-scheduler's symbol watchlist toward that
     symbol so the next pipeline tick focuses there. It also drops
     its reasoning into the chatter feed so the user can literally
     watch the brain make choices.

Both loops respect the cockpit pause button and the watchdog halt
flag (the same kill-switches the trading autopilot honors). State is
in-memory so a process restart starts fresh \u2014 the durable record is
``data/research_sweep.json`` plus the agent log.

By default this module **does not auto-start**. ``server.py`` opts in
during startup (Phase 20) once the operator has confirmed they want
the always-on brain. The pause button still universally halts work.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from datetime import time as dtime
from typing import Any
from zoneinfo import ZoneInfo

from packages.cockpit.web import bandit as agent_bandit
from packages.cockpit.web import brain_memory
from packages.cockpit.web import chatter as agent_chatter
from packages.cockpit.web import knowledge_base as agent_knowledge
from packages.cockpit.web import reflection as agent_reflection
from packages.cockpit.web import regime as regime_module

log = logging.getLogger("autonomy")

ET = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# Config + state
# ---------------------------------------------------------------------------


@dataclass
class AutonomyConfig:
    """Tunables for the always-on brain.

    Cadences are intentionally generous \u2014 the agents are advisory and
    we don't want to thrash data sources. The user can override via
    env vars (mostly for tests).
    """

    sweep_market_seconds: int = int(
        os.environ.get("AUTONOMY_SWEEP_MARKET_S") or 900
    )  # every 15min during the regular session (Phase 25 — was 2h, too slow
    # to catch intraday profit-take opportunities)
    sweep_off_seconds: int = int(
        os.environ.get("AUTONOMY_SWEEP_OFF_S") or 6 * 3600
    )  # every 6h overnight / weekends
    # Skip a tick when the most recent sweep finished less than this ago
    # (e.g. a manual /api/research/sweep/run just landed).
    min_sweep_gap_seconds: int = 600
    # Curiosity considers the most-recent N candidates and picks 1\u20133 to
    # focus on; the rest of the symbol watchlist stays untouched if
    # nothing interesting beats the current pick.
    curiosity_top_n: int = 8
    curiosity_focus_count: int = 3
    # Optional Curiosity hook that lets server.py push the chosen
    # focus symbols into the agent scheduler. Filled in at wire-up.
    on_curiosity_focus: Callable[[list[str], str], None] | None = None
    # Optional hook to fetch portfolio symbols so we always keep them
    # in the focus list (don't drop the user's actual holdings).
    portfolio_symbols_getter: Callable[[], list[str]] | None = None
    # ---- Phase 21: self-improvement knobs --------------------------
    # When True, the tick judges prior picks, updates the bandit, runs
    # regime detection, and writes a reflection. Disable in tests that
    # don't want side-effects.
    self_improve_enabled: bool = True
    # Holding horizon for outcome judgment. 24h is one full session.
    judgment_horizon_hours: int = int(
        os.environ.get("AUTONOMY_JUDGMENT_HOURS") or 24
    )
    # Price provider for outcome judgment + regime detection. Tests
    # inject deterministic providers; the default uses yfinance.
    price_lookup: Callable[[str], float | None] | None = None
    regime_price_provider: Callable[[str], list[float] | None] | None = None
    regime_vix_provider: Callable[[], float | None] | None = None
    # ---- Phase 25: active profit-taking + dip-watch hooks ---------
    # When True, each market-hours tick also evaluates exit_rules on
    # every open position and checks armed dip-watchers for re-entry.
    # Disabled outside market hours to avoid spurious quote calls.
    exit_rules_enabled: bool = os.environ.get("AUTONOMY_EXIT_RULES", "1") != "0"
    dip_watch_enabled: bool = os.environ.get("AUTONOMY_DIP_WATCH", "1") != "0"
    # Phase 28-R step 2 — end-of-day flattener. Closes every position
    # in the 15:55-16:05 ET window so the bot never holds overnight.
    # The hook is idempotent per-session; safe to call every 60s.
    eod_flatten_enabled: bool = os.environ.get("AUTONOMY_EOD_FLATTEN", "1") != "0"
    # Async hooks supplied by the cockpit at wire-up time so the
    # autonomy module never imports the broker directly (testability).
    exit_rules_tick: Callable[[], Any] | None = None
    dip_watch_tick: Callable[[], Any] | None = None
    eod_flatten_tick: Callable[[], Any] | None = None
    # Trading-controls guardrail. After each sweep the chosen candidates are
    # passed through the user's budget + confidence gate: qualifying trades
    # go down the (shadow) execution path, the rest are queued as pending.
    # Injected by the cockpit (default None = no-op) so the autonomy module
    # stays broker-agnostic and existing tests see zero new side effects.
    trading_controls_enabled: bool = True
    trading_controls_tick: Callable[[list[dict[str, Any]]], Any] | None = None
    # ---- Phase 25.3: live-quote cache warmer ---------------------
    # Called at the head of every fast tick (and exposed via run_tick)
    # to refresh the LiveQuoteCache for the active symbol set before
    # exit_rules / dip_watch read prices. None-safe: when missing,
    # the fast tick proceeds with the existing cache contents.
    quote_warmup_tick: Callable[[], Any] | None = None
    # ---- Phase 25.1: fast loop ------------------------------------
    # Exit-rules + dip-watch are price-sensitive and must not wait a
    # full research-sweep cycle (15min) to fire. The fast loop runs
    # ONLY those two hooks every ``fast_loop_seconds`` during market
    # hours. Set to 0 to disable and fall back to the slow loop only.
    fast_loop_seconds: int = int(
        os.environ.get("AUTONOMY_FAST_LOOP_S") or 60
    )
    # Phase 35 — adaptive cadence. When any open position is "hot"
    # (peak >= trail_arm_pct), fire the next fast tick after this many
    # seconds instead of ``fast_loop_seconds``. Closer to the wire =
    # faster profit-take/trailing-stop execution. 10s is roughly the
    # round-trip cost of a paper-Alpaca quote + exit decision, leaving
    # plenty of headroom under the API rate limits.
    fast_loop_hot_seconds: int = int(
        os.environ.get("AUTONOMY_FAST_LOOP_HOT_S") or 10
    )


# Phase 35 — autonomy watchdog threshold (module-level so curiosity
# can import it without pulling the whole config object). If the most
# recent fast-tick heartbeat is older than this during market hours,
# something has stalled the loop and curiosity should narrate it.
FAST_LOOP_STALE_S: int = 300


@dataclass
class AutonomyState:
    """Mutable runtime state. Lives in-process, not on disk."""

    enabled: bool = False
    last_sweep_started_at: str | None = None
    last_sweep_finished_at: str | None = None
    last_sweep_status: str | None = None
    last_sweep_candidates: int = 0
    last_curiosity_at: str | None = None
    last_curiosity_focus: list[str] = field(default_factory=list)
    last_curiosity_reason: str = ""
    last_error: str = ""
    # Phase 21 — self-improvement state surfaced for /api/brain.
    last_regime: dict[str, Any] | None = None
    last_reflection: dict[str, Any] | None = None
    last_judged_count: int = 0
    last_bandit_weights: dict[str, float] = field(default_factory=dict)
    # Background asyncio task handles; managed by start()/stop().
    _sweep_task: asyncio.Task[Any] | None = None
    _fast_task: asyncio.Task[Any] | None = None
    _config: AutonomyConfig = field(default_factory=AutonomyConfig)
    # Phase 25.1 — last fast-tick observability.
    last_fast_tick_at: str | None = None
    last_fast_tick_status: str | None = None
    last_fast_tick_exit: dict[str, Any] | None = None
    last_fast_tick_dip: dict[str, Any] | None = None
    # Phase 28-R step 2 — last EOD-flatten result.
    last_fast_tick_eod_flatten: dict[str, Any] | None = None
    # Phase 35 — hot-position flag drives adaptive fast-loop cadence.
    # Set True when any open position's peak >= trail_arm_pct so the
    # next fast tick fires in ``fast_loop_hot_seconds`` instead of
    # ``fast_loop_seconds``. Reset by run_tick every cycle.
    any_position_hot: bool = False
    # Phase 35 — autonomy watchdog: ISO ts of the last successful
    # fast tick. Stale > FAST_LOOP_STALE_S during market hours → alert.
    last_fast_tick_heartbeat_at: str | None = None


# Module-level singleton. The cockpit owns at most one autonomy loop.
STATE = AutonomyState()


def configure(config: AutonomyConfig | None = None, **kwargs: Any) -> AutonomyConfig:
    """Merge fresh config into the running state.

    Call this once at startup to wire the curiosity callback and any
    overrides. Returns the active config.
    """
    if config is not None:
        STATE._config = config
    for k, v in kwargs.items():
        if hasattr(STATE._config, k):
            setattr(STATE._config, k, v)
    return STATE._config


# ---------------------------------------------------------------------------
# Calendar helpers \u2014 "are US equities open right now?"
# ---------------------------------------------------------------------------


def is_market_open(now: datetime | None = None) -> bool:
    """Cheap US-equities open check: Mon\u2013Fri, 09:30\u201316:00 ET.

    Holidays are not considered \u2014 the worst case is one extra sweep
    on a holiday, which is harmless. The trading autopilot owns the
    full market calendar.
    """
    if now is None:
        now = datetime.now(UTC)
    et = now.astimezone(ET)
    if et.weekday() >= 5:  # 5 = Saturday
        return False
    return dtime(9, 30) <= et.time() < dtime(16, 0)


def current_sweep_interval(now: datetime | None = None) -> int:
    """Pick the active sweep cadence based on market hours."""
    cfg = STATE._config
    return cfg.sweep_market_seconds if is_market_open(now) else cfg.sweep_off_seconds


# ---------------------------------------------------------------------------
# Curiosity \u2014 pick the next thing worth investigating
# ---------------------------------------------------------------------------


# Feature labels emitted by the scorer. Kept in sync with bandit
# DEFAULT_ARMS and regime.SCORE_MULTIPLIERS — add a new label here when
# you want the bandit + regime modifier to apply to a new signal.
FEATURE_LABELS: tuple[str, ...] = (
    "corroborated",
    "reddit_trust",
    "analyst_bullish",
    "analyst_action",
    "insider",
    "stocktwits",
    "yahoo_news",
    "ranker",         # Phase 34: supervised LightGBM probability arm
    "fundamentals",   # Robinhood read-only fundamentals momentum signal
)


def _score_candidate(
    c: dict[str, Any],
    *,
    weights: dict[str, float] | None = None,
    multipliers: dict[str, float] | None = None,
    agent_weights: dict[str, float] | None = None,
) -> tuple[float, list[str], list[str]]:
    """Score a sweep candidate for "investigate-worthiness".

    Returns ``(score, reasons, features)``. ``score`` blends
    confidence with corroboration and per-source enrichment.
    ``reasons`` is the human chatter copy. ``features`` is the list
    of stable feature IDs that fired — the bandit credits/blames
    these IDs after outcomes are judged.

    Per-feature contributions are multiplied by:
      * ``weights[feature]``  — the bandit's adapted weight (default 1.0)
      * ``multipliers[feature]`` — the current regime's tilt (default 1.0)

    When both are omitted the scorer behaves exactly like Phase 20.
    """
    if not isinstance(c, dict):
        return (0.0, [], [])
    reasons: list[str] = []
    features: list[str] = []
    weights = weights or {}
    multipliers = multipliers or {}

    def gain(label: str, base: float) -> float:
        """Apply bandit weight * regime multiplier to a base bonus."""
        w = float(weights.get(label, 1.0))
        m = float(multipliers.get(label, 1.0))
        return base * w * m

    # Base = confidence (already in [0, 1]).
    score = float(c.get("confidence") or 0.0)

    if c.get("corroborated"):
        score += gain("corroborated", 0.20)
        reasons.append("news-corroborated")
        features.append("corroborated")
    cs = float(c.get("corroboration_score") or 0.0)
    if cs >= 0.5:
        score += gain("corroborated", 0.10)
        if "news-corroborated" not in reasons:
            reasons.append(f"corroboration {cs:.2f}")
            if "corroborated" not in features:
                features.append("corroborated")

    rt = float(c.get("reddit_trust") or 0.0)
    if rt >= 0.6:
        score += gain("reddit_trust", 0.10)
        reasons.append(f"reddit-trust {rt:.2f}")
        features.append("reddit_trust")

    # Analyst rating: 1 = Strong Buy \u2026 5 = Strong Sell. Anything <2.5
    # is bullish, >3.5 bearish.
    amr = float(c.get("analyst_mean_rating") or 0.0)
    if amr and amr <= 2.2 and int(c.get("analyst_num") or 0) >= 5:
        score += gain("analyst_bullish", 0.08)
        reasons.append("analysts bullish")
        features.append("analyst_bullish")
    elif amr and amr >= 3.8 and int(c.get("analyst_num") or 0) >= 5:
        score += gain("analyst_bullish", 0.08)
        reasons.append("analysts bearish")
        features.append("analyst_bullish")

    action = (c.get("analyst_recent_action") or "").lower()
    if action in {"upgrade", "downgrade"}:
        score += gain("analyst_action", 0.06)
        reasons.append(f"analyst {action}")
        features.append("analyst_action")

    # Insider Form 4 activity in the last 30 days.
    if int(c.get("insider_form4_30d") or 0) >= 3:
        net = float(c.get("insider_net_shares") or 0.0)
        side = "buying" if net > 0 else ("selling" if net < 0 else "rotation")
        score += gain("insider", 0.08)
        reasons.append(f"insider {side}")
        features.append("insider")

    if c.get("stocktwits_trending"):
        score += gain("stocktwits", 0.04)
        reasons.append("stocktwits trending")
        features.append("stocktwits")

    yc = int(c.get("yahoo_news_count") or 0)
    if yc >= 5:
        score += gain("yahoo_news", 0.03)
        reasons.append(f"{yc} fresh headlines")
        features.append("yahoo_news")

    # Phase 34: supervised LightGBM ranker as a contribution arm.
    # P(EOD >= +0.5%) is in [0, 1]; we center on 0.5 so a neutral
    # prediction contributes nothing, and a strongly confident model
    # tilts the score by up to ±0.20 before bandit/regime modulation.
    # ``ranker_proba`` is plumbed in by the live caller (paper_trade
    # populates it after computing inference); if absent we skip
    # gracefully so unit tests and synthetic candidates still work.
    try:
        proba = c.get("ranker_proba")
        if proba is not None:
            tilt = (float(proba) - 0.5) * 0.40   # ±0.20 at extremes
            if abs(tilt) > 1e-4:
                score += gain("ranker", tilt)
                reasons.append(f"ranker {float(proba):.2f}")
                features.append("ranker")
    except (TypeError, ValueError):
        pass

    # Ensemble reweighting: scale the score by the average historical
    # influence of the agents that voted for this candidate (learned from
    # resolved outcomes). Fail-safe — no attribution or no learned weights
    # leaves the score untouched (multiplier 1.0), and the multiplier is
    # bounded so a quietened agent is never fully silenced.
    if agent_weights:
        voted = c.get("agents_voted") or c.get("agents") or []
        if isinstance(voted, dict):
            voted = list(voted.keys())
        if isinstance(voted, (list, tuple)) and voted:
            try:
                from packages.learning.agent_weights import (
                    agent_influence_multiplier,
                )

                mult = agent_influence_multiplier(
                    [str(a) for a in voted], agent_weights
                )
                score *= mult
            except Exception:  # pragma: no cover — defensive
                pass

    # Robinhood fundamentals: GUARDRAIL-FIRST safety + a modest momentum
    # signal. Fires ONLY when the read-only fundamentals phase actually ran
    # (``fundamentals_source`` is set); otherwise it contributes EXACTLY 0.0
    # and the rule-based scoring output is byte-for-byte unchanged (protects
    # the yfinance-ordering + scorer tests). The net contribution is clamped
    # to a protective-skewed band [-0.20, +0.08] so it can dampen or veto a
    # pick (push it below min_confidence) but only ever nudge it up. Safety
    # PENALTIES are applied at FULL strength (not bandit-weighted) so the
    # bandit can never learn to weaken a delisting / earnings guardrail; the
    # small POSITIVES flow through the bandit like every other signal. A
    # missing field stays at its neutral default => no contribution, never
    # bearish.
    if c.get("fundamentals_source"):
        fund = 0.0
        # --- SAFETY (never boosts) ---
        if c.get("compliance_ok") is False:
            fund -= 0.15
            reasons.append("delisting/compliance risk")
        dte = c.get("days_to_earnings")
        if isinstance(dte, (int, float)) and not isinstance(dte, bool) and dte <= 1:
            fund -= 0.08
            reasons.append("earnings imminent")
        try:
            relv = float(c.get("rel_volume") or 0.0)
            mcap = float(c.get("market_cap") or 0.0)
            pfh = float(c.get("pct_from_52w_high") or 0.0)
        except (TypeError, ValueError):
            relv = mcap = pfh = 0.0
        # Extreme illiquidity AND micro-cap: a small protective penalty.
        if 0.0 < relv < 0.3 and 0.0 < mcap < 5e7:
            fund -= 0.05
            reasons.append("thin liquidity")
        # --- MODEST POSITIVE (bandit-weighted, bounded) ---
        if relv >= 1.5:
            fund += gain("fundamentals", 0.04)
            reasons.append(f"volume {relv:.1f}x normal")
            features.append("fundamentals")
        if pfh <= -0.05:
            fund += gain("fundamentals", 0.02)
            reasons.append("not extended vs 52w high")
            if "fundamentals" not in features:
                features.append("fundamentals")
        score += max(-0.20, min(0.08, fund))

    # LLM thesis sanity-check nudge. Applied LAST — AFTER scoring, bandit,
    # regime, and agent-weight steps — and CLAMPED to [-0.10, +0.10] so the
    # model can only nudge, never override. Absent / rule-based candidates
    # carry confidence_adjustment == 0.0, so this is a no-op for them and the
    # rule-based scoring output stays byte-for-byte unchanged. A bear/neutral
    # verdict on a bullish-scored candidate lowers the score (within clamp);
    # the risk_flag is surfaced separately on the candidate for the UI.
    try:
        adj = float(c.get("confidence_adjustment") or 0.0)
    except (TypeError, ValueError):
        adj = 0.0
    if adj:
        adj = max(-0.10, min(0.10, adj))
        score += adj
        if adj > 0:
            reasons.append(f"AI thesis +{adj:.2f}")
        else:
            reasons.append(f"AI thesis {adj:.2f}")

    return (round(score, 4), reasons, features)


def pick_focus(
    candidates: list[dict[str, Any]],
    *,
    top_n: int = 8,
    focus_count: int = 3,
    weights: dict[str, float] | None = None,
    multipliers: dict[str, float] | None = None,
    agent_weights: dict[str, float] | None = None,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Curiosity's decision: which symbols are worth a deep dive next?

    Looks at the ``top_n`` candidates, scores each, keeps the
    ``focus_count`` highest-scoring symbols. Deterministic for tests.

    Optional ``weights`` (from the bandit) and ``multipliers`` (from
    the regime detector) tilt the scoring. Without them this
    behaves identically to the Phase 20 implementation.

    Returns ``(symbols, details)`` where each ``details`` entry is
    ``{symbol, score, reasons, features, candidate}``. The raw
    candidate dict is included so callers can capture ``last_price``
    or other context for outcome judgment.
    """
    if not candidates:
        return ([], [])

    # Phase 34: load the supervised ranker once per sweep and inject
    # ``ranker_proba`` into each candidate dict so ``_score_candidate``
    # picks it up. Cached on disk; load_model returns None when no
    # model is fitted yet or when lightgbm is unavailable — in that
    # case the scorer's ranker arm contributes nothing.
    try:
        from packages.learning.ranker import load_model as _load_ranker

        _ranker_model = _load_ranker()
    except Exception as exc:  # pragma: no cover - defensive
        log.debug("ranker load failed: %s", exc)
        _ranker_model = None

    scored: list[tuple[float, str, list[str], list[str], dict[str, Any]]] = []
    for c in candidates[:top_n]:
        sym = str(c.get("symbol") or "").upper().strip()
        if not sym:
            continue
        # Inject ranker probability before scoring (idempotent: if
        # already present, e.g. unit-test seeded, we leave it alone).
        if _ranker_model is not None and "ranker_proba" not in c:
            try:
                from packages.learning.feature_snapshot import (
                    extract_features_from_candidate,
                )

                c["ranker_proba"] = _ranker_model.predict_proba(
                    extract_features_from_candidate(c)
                )
            except Exception as exc:  # pragma: no cover - defensive
                log.debug("ranker inference failed for %s: %s", sym, exc)
        s, r, feats = _score_candidate(
            c, weights=weights, multipliers=multipliers, agent_weights=agent_weights
        )
        scored.append((s, sym, r, feats, c))
    # Highest score first; ties broken alphabetically for determinism.
    scored.sort(key=lambda x: (-x[0], x[1]))
    chosen = scored[:focus_count]
    syms: list[str] = []
    details: list[dict[str, Any]] = []
    seen: set[str] = set()
    for s, sym, reasons, feats, cand in chosen:
        if sym in seen:
            continue
        seen.add(sym)
        syms.append(sym)
        details.append(
            {
                "symbol": sym,
                "score": s,
                "reasons": reasons,
                "features": feats,
                "candidate": cand,
            }
        )
    return (syms, details)


def _curiosity_message(focus: list[dict[str, Any]]) -> str:
    """Render the Curiosity decision as a single chatter line."""
    if not focus:
        return "Nothing stood out this sweep \u2014 keeping the existing watchlist."
    pieces = []
    for f in focus:
        sym = f["symbol"]
        reasons = f.get("reasons") or []
        if reasons:
            pieces.append(f"{sym} ({', '.join(reasons[:3])})")
        else:
            pieces.append(sym)
    return "Next focus: " + "; ".join(pieces)


def _sweep_summary_message(result_dict: dict[str, Any]) -> str:
    """Render a sweep result as a single chatter line."""
    cands = result_dict.get("candidates") or []
    if not cands:
        return "Sweep complete \u2014 no actionable candidates this cycle."
    top = cands[:3]
    pieces = []
    for c in top:
        sym = str(c.get("symbol") or "?").upper()
        conf = float(c.get("confidence") or 0.0)
        kind = c.get("signal_kind") or ""
        pieces.append(f"{sym} {kind} ({conf:.2f})")
    extra = "" if len(cands) <= 3 else f" +{len(cands) - 3} more"
    return "Fresh sweep: " + ", ".join(pieces) + extra


# ---------------------------------------------------------------------------
# The sweep + curiosity tick
# ---------------------------------------------------------------------------


# Injection points so tests can mock the heavy network call without
# rewriting the loop logic.
SweepCallable = Callable[[], Awaitable[Any]]
PauseCheck = Callable[[], bool]


async def _default_sweep_runner() -> Any:
    """Default sweep: real research_sweep.run_sweep, persisted to disk."""
    # Imported lazily so importing this module is cheap (and so unit
    # tests can mock without pulling the full sweep dependency tree).
    from packages.agents.research_sweep import run_sweep, save_sweep

    result = await run_sweep()
    try:
        save_sweep(result)
    except Exception as e:  # pragma: no cover \u2014 disk errors are non-fatal
        log.warning("save_sweep failed (ignored): %s", e)
    return result


def _default_pause_check() -> bool:
    """Returns True when the cockpit pause flag is set."""
    try:
        from packages.cockpit.state import load_state

        return bool(load_state().paused)
    except Exception:  # pragma: no cover \u2014 don't take down the loop
        return False


async def _run_phase25_hooks(
    cfg: AutonomyConfig,
) -> tuple[
    dict[str, Any] | None,
    dict[str, Any] | None,
    dict[str, Any] | None,
]:
    """Run exit_rules_tick + dip_watch_tick + eod_flatten_tick in parallel.

    Returns ``(exit_result, dip_result, eod_flatten_result)``. Any
    element is ``None`` if its hook is disabled or unset. Exceptions
    are caught per-hook and surfaced via chatter — the other hooks
    still run.
    """

    async def _wrap_exit() -> dict[str, Any] | None:
        if not (cfg.exit_rules_enabled and cfg.exit_rules_tick is not None):
            return None
        try:
            return await cfg.exit_rules_tick()
        except Exception as exc:  # pragma: no cover — defensive
            log.warning("exit_rules tick failed: %s", exc)
            agent_chatter.push(
                agent="exit_rules",
                status="warn",
                message=f"Exit-rules tick failed: {exc}",
            )
            return {"error": str(exc)[:240]}

    async def _wrap_dip() -> dict[str, Any] | None:
        if not (cfg.dip_watch_enabled and cfg.dip_watch_tick is not None):
            return None
        try:
            return await cfg.dip_watch_tick()
        except Exception as exc:  # pragma: no cover — defensive
            log.warning("dip_watch tick failed: %s", exc)
            agent_chatter.push(
                agent="dip_watch",
                status="warn",
                message=f"Dip-watch tick failed: {exc}",
            )
            return {"error": str(exc)[:240]}

    async def _wrap_eod_flatten() -> dict[str, Any] | None:
        if not (cfg.eod_flatten_enabled and cfg.eod_flatten_tick is not None):
            return None
        try:
            return await cfg.eod_flatten_tick()
        except Exception as exc:  # pragma: no cover — defensive
            log.warning("eod_flatten tick failed: %s", exc)
            agent_chatter.push(
                agent="eod_flatten",
                status="warn",
                message=f"EOD-flatten tick failed: {exc}",
            )
            return {"error": str(exc)[:240]}

    return await asyncio.gather(_wrap_exit(), _wrap_dip(), _wrap_eod_flatten())


async def run_fast_tick() -> dict[str, Any]:
    """Phase 25.1 — the fast loop.

    Runs ONLY the price-sensitive hooks (exit_rules + dip_watch) in
    parallel. Skips outside market hours and when the cockpit is paused.
    Cheap enough to call every 60s; the heavy research sweep stays on
    the slow loop.
    """
    cfg = STATE._config
    now_iso = datetime.now(UTC).isoformat(timespec="seconds")
    STATE.last_fast_tick_at = now_iso

    if not is_market_open():
        STATE.last_fast_tick_status = "skipped_closed"
        return {"skipped": True, "reason": "market_closed"}
    if _default_pause_check():
        STATE.last_fast_tick_status = "skipped_paused"
        return {"skipped": True, "reason": "paused"}

    # Phase 25.3 — refresh the live-quote cache before reading prices.
    # Best-effort: any failure is swallowed so a flaky data feed never
    # blocks exit-rules / dip-watch.
    warmup_result: dict[str, Any] | None = None
    if cfg.quote_warmup_tick is not None:
        try:
            warmup_result = await cfg.quote_warmup_tick()
        except Exception as exc:  # pragma: no cover — defensive
            log.debug("quote_warmup_tick failed: %s", exc)
            warmup_result = {"error": str(exc)[:240]}

    exit_result, dip_result, eod_flatten_result = await _run_phase25_hooks(cfg)
    STATE.last_fast_tick_exit = exit_result
    STATE.last_fast_tick_dip = dip_result
    STATE.last_fast_tick_eod_flatten = eod_flatten_result
    STATE.last_fast_tick_status = "ok"
    # Phase 35 — watchdog heartbeat. Curiosity inspects this to detect
    # a stalled fast loop and narrate the blocker.
    STATE.last_fast_tick_heartbeat_at = now_iso
    return {
        "skipped": False,
        "ok": True,
        "quote_warmup": warmup_result,
        "exit_rules": exit_result,
        "dip_watch": dip_result,
        "eod_flatten": eod_flatten_result,
        "ts": now_iso,
    }


async def run_one_tick(
    *,
    sweep_runner: SweepCallable | None = None,
    pause_check: PauseCheck | None = None,
) -> dict[str, Any]:
    """Execute one autonomous research + curiosity pass.

    Returns a dict describing what happened. Never raises \u2014 errors are
    captured into ``STATE.last_error`` so the loop stays alive.
    """
    sweep_runner = sweep_runner or _default_sweep_runner
    pause_check = pause_check or _default_pause_check
    cfg = STATE._config
    now_iso = datetime.now(UTC).isoformat(timespec="seconds")

    if pause_check():
        STATE.last_sweep_status = "skipped_paused"
        agent_chatter.push(
            agent="autonomy",
            status="warn",
            message="Cockpit paused \u2014 skipped autonomous research tick.",
            ts=now_iso,
        )
        return {"skipped": True, "reason": "paused"}

    STATE.last_sweep_started_at = now_iso
    try:
        result = await sweep_runner()
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"[:240]
        STATE.last_error = msg
        STATE.last_sweep_status = "failed"
        STATE.last_sweep_finished_at = datetime.now(UTC).isoformat(timespec="seconds")
        log.warning("autonomous sweep failed: %s", e)
        agent_chatter.push(
            agent="autonomy",
            status="warn",
            message=f"Sweep failed (ignored): {msg}",
            ts=STATE.last_sweep_finished_at,
        )
        return {"skipped": False, "ok": False, "error": msg}

    # The sweep may return a dataclass or a dict; normalize.
    if hasattr(result, "to_dict"):
        result_dict = result.to_dict()
    elif isinstance(result, dict):
        result_dict = result
    else:
        result_dict = {"status": "unknown", "candidates": []}

    STATE.last_sweep_finished_at = datetime.now(UTC).isoformat(timespec="seconds")
    STATE.last_sweep_status = result_dict.get("status") or "ok"
    STATE.last_sweep_candidates = len(result_dict.get("candidates") or [])

    # Push a Research-flavored chatter line so the homepage feed lights up.
    agent_chatter.push(
        agent="research",
        status=STATE.last_sweep_status if STATE.last_sweep_status == "ok" else "warn",
        message=_sweep_summary_message(result_dict),
        ts=STATE.last_sweep_finished_at,
    )

    # ------------------------------------------------------------------
    # Phase 21-A: judge prior picks → reward the bandit.
    # ------------------------------------------------------------------
    judged_count = 0
    if cfg.self_improve_enabled and cfg.price_lookup is not None:
        try:
            judged = brain_memory.judge_picks(
                cfg.price_lookup,
                horizon_hours=cfg.judgment_horizon_hours,
            )
            judged_count = len(judged)
            for jp in judged:
                status = jp.get("status")
                if status not in {"hit", "miss", "flat"}:
                    continue
                # Reward: +1 hit, -1 miss, -0.25 flat (penalise weak picks).
                reward = (
                    1.0 if status == "hit"
                    else -1.0 if status == "miss"
                    else -0.25
                )
                feats = jp.get("features") or []
                if feats:
                    agent_bandit.update_with_outcome(feats, reward)
            # Fold judged picks into the durable knowledge base so
            # learnings survive ledger compaction.
            try:
                if judged:
                    agent_knowledge.apply_judged(judged)
            except Exception as kb_exc:  # pragma: no cover — defensive
                log.warning("knowledge_base apply_judged failed: %s", kb_exc)
            if judged_count:
                hits = sum(1 for j in judged if j.get("status") == "hit")
                misses = sum(1 for j in judged if j.get("status") == "miss")
                agent_chatter.push(
                    agent="reflection",
                    status="ok" if hits >= misses else "warn",
                    message=(
                        f"Judged {judged_count} prior picks: {hits} hits, "
                        f"{misses} misses — weights adapted."
                    ),
                    ts=STATE.last_sweep_finished_at,
                )
        except Exception as exc:  # pragma: no cover — defensive
            log.warning("brain_memory judge_picks failed: %s", exc)
    STATE.last_judged_count = judged_count

    # ------------------------------------------------------------------
    # Close the confidence loop: refit the bounded calibrator from the
    # outcome journal. Network-free (reads data/learning/outcomes.jsonl),
    # cold-start-safe (identity map until enough resolved trades), and
    # fully guarded so a calibration hiccup can never break the tick.
    # The labeling half of the loop (which needs price bars) runs on the
    # /api/learning/backfill endpoint and the schedule.
    # ------------------------------------------------------------------
    if cfg.self_improve_enabled:
        try:
            from packages.learning.feedback import recalibrate_from_outcomes

            recalibrate_from_outcomes()
        except Exception as exc:  # pragma: no cover — defensive
            log.warning("learning recalibration failed: %s", exc)

        # Auto-reweight the ensemble's agents from resolved outcomes: shift
        # influence toward agents that have actually been right. Network-free
        # (reads data/learning/outcomes.jsonl), cold-start-safe (equal weights
        # until agents clear the per-agent sample floor), bounded so nobody is
        # ever silenced, and guarded so a hiccup can't break the tick.
        try:
            from packages.learning.agent_weights import reweight_from_outcomes

            reweight_from_outcomes()
        except Exception as exc:  # pragma: no cover — defensive
            log.warning("learning agent reweight failed: %s", exc)

    # ------------------------------------------------------------------
    # Phase 21-B: detect current market regime.
    # ------------------------------------------------------------------
    regime_snap = None
    multipliers: dict[str, float] = {}
    if cfg.self_improve_enabled:
        try:
            regime_snap = regime_module.detect(
                price_provider=cfg.regime_price_provider,
                vix_provider=cfg.regime_vix_provider,
            )
            multipliers = dict(regime_snap.multipliers)
            STATE.last_regime = {
                "label": regime_snap.label,
                "confidence": regime_snap.confidence,
                "reasons": list(regime_snap.reasons),
                "vix": regime_snap.vix,
                "spy_trend_20d": regime_snap.spy_trend_20d,
                "spy_drawdown_60d": regime_snap.spy_drawdown_60d,
                "breadth": regime_snap.breadth,
                "ts": regime_snap.ts,
            }
        except Exception as exc:  # pragma: no cover — defensive
            log.warning("regime detect failed: %s", exc)
            STATE.last_regime = {"label": "neutral", "reasons": [str(exc)[:120]]}

    # Pull current bandit weights (cheap read-only call).
    try:
        weights = agent_bandit.current_weights() if cfg.self_improve_enabled else {}
    except Exception as exc:  # pragma: no cover — defensive
        log.warning("bandit weights load failed: %s", exc)
        weights = {}
    STATE.last_bandit_weights = dict(weights)

    # Pull current per-agent influence weights (learned from outcomes).
    try:
        from packages.learning.agent_weights import current_agent_weights

        agent_weights = current_agent_weights() if cfg.self_improve_enabled else {}
    except Exception as exc:  # pragma: no cover — defensive
        log.warning("agent weights load failed: %s", exc)
        agent_weights = {}

    # ------------------------------------------------------------------
    # Curiosity step — now tilted by bandit + regime + agent influence.
    # ------------------------------------------------------------------
    focus_syms, focus_details = pick_focus(
        result_dict.get("candidates") or [],
        top_n=cfg.curiosity_top_n,
        focus_count=cfg.curiosity_focus_count,
        weights=weights or None,
        multipliers=multipliers or None,
        agent_weights=agent_weights or None,
    )
    STATE.last_curiosity_at = STATE.last_sweep_finished_at
    STATE.last_curiosity_focus = focus_syms
    STATE.last_curiosity_reason = _curiosity_message(focus_details)

    # ------------------------------------------------------------------
    # Phase 21-C: record picks into brain memory for future judgment.
    # ------------------------------------------------------------------
    if cfg.self_improve_enabled:
        regime_label = (STATE.last_regime or {}).get("label")
        for fd in focus_details:
            cand = fd.get("candidate") or {}
            entry_price = None
            for key in ("last_price", "price", "close", "current_price"):
                val = cand.get(key)
                if val is not None:
                    try:
                        entry_price = float(val)
                        break
                    except (TypeError, ValueError):
                        continue
            if entry_price is None and cfg.price_lookup is not None:
                try:
                    entry_price = cfg.price_lookup(fd["symbol"])
                except Exception:  # pragma: no cover — defensive
                    entry_price = None
            try:
                brain_memory.record_pick(
                    fd["symbol"],
                    score=fd.get("score") or 0.0,
                    reasons=fd.get("reasons") or [],
                    features=fd.get("features") or [],
                    entry_price=entry_price,
                    regime=regime_label,
                )
            except Exception as exc:  # pragma: no cover — defensive
                log.warning("record_pick failed for %s: %s", fd.get("symbol"), exc)

    agent_chatter.push(
        agent="curiosity",
        status="ok" if focus_syms else "info",
        message=STATE.last_curiosity_reason,
        ts=STATE.last_sweep_finished_at,
    )

    # ------------------------------------------------------------------
    # Phase 21-D: compose + persist reflection.
    # ------------------------------------------------------------------
    if cfg.self_improve_enabled:
        try:
            stats = brain_memory.accuracy_stats()
            prior = (STATE.last_reflection or {}).get("stats", {}).get("hit_rate")
            bandit_snap = agent_bandit.snapshot()
            refl = agent_reflection.compose(
                stats=stats,
                regime=STATE.last_regime,
                bandit_snapshot=bandit_snap,
                prior_hit_rate=prior,
            )
            agent_reflection.append(refl)
            STATE.last_reflection = {
                "ts": refl.ts,
                "headline": refl.headline,
                "paragraph": refl.paragraph,
                "lessons": list(refl.lessons),
                "stats": dict(refl.stats),
                "regime": refl.regime,
            }
            agent_chatter.push(
                agent="reflection",
                status="info",
                message=refl.headline,
                ts=refl.ts,
            )
        except Exception as exc:  # pragma: no cover — defensive
            log.warning("reflection failed: %s", exc)

    # Push focus into the agent scheduler if a hook is wired up. Keep
    # the user's portfolio symbols at the front so we never drop them.
    if cfg.on_curiosity_focus is not None and focus_syms:
        try:
            portfolio: list[str] = []
            if cfg.portfolio_symbols_getter is not None:
                try:
                    portfolio = [
                        str(s).upper() for s in (cfg.portfolio_symbols_getter() or [])
                    ]
                except Exception:  # pragma: no cover
                    portfolio = []
            merged = list(dict.fromkeys(portfolio + focus_syms))[:6]
            cfg.on_curiosity_focus(merged, STATE.last_curiosity_reason)
        except Exception as e:  # pragma: no cover \u2014 don't break the loop
            log.warning("curiosity focus hook failed: %s", e)

    # ------------------------------------------------------------------
    # Phase 25 / 25.1: active profit-taking + dip-watch buy-back.
    # Only fires during market hours — outside hours, quotes are stale
    # and Alpaca rejects orders anyway, so the calls would be wasted.
    #
    # Phase 25.1: the two hooks are independent and both make remote
    # calls. Run them in parallel via asyncio.gather so the tick
    # finishes in max(exit, dip) instead of exit+dip. The fast loop
    # uses the same helper.
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Trading-controls guardrail. Push the freshly-chosen candidates
    # through the user's budget + confidence limits: qualifying ones
    # proceed down the existing SHADOW execution path, the rest are
    # recorded as pending with plain-language reasons. The hook also
    # re-evaluates previously-held trades so they auto-proceed once they
    # qualify. Fully defensive — a failure here never breaks the tick.
    # ------------------------------------------------------------------
    trading_controls_result: dict[str, Any] | None = None
    if cfg.trading_controls_enabled and cfg.trading_controls_tick is not None:
        try:
            trading_controls_result = await cfg.trading_controls_tick(focus_details)
        except Exception as exc:  # pragma: no cover — defensive
            log.warning("trading-controls tick failed: %s", exc)

    exit_tick_result: dict[str, Any] | None = None
    dip_tick_result: dict[str, Any] | None = None
    eod_flatten_tick_result: dict[str, Any] | None = None
    if is_market_open():
        exit_tick_result, dip_tick_result, eod_flatten_tick_result = (
            await _run_phase25_hooks(cfg)
        )

    return {
        "skipped": False,
        "ok": True,
        "status": STATE.last_sweep_status,
        "candidates": STATE.last_sweep_candidates,
        "focus": focus_syms,
        "focus_details": focus_details,
        "judged": judged_count,
        "regime": (STATE.last_regime or {}).get("label"),
        "exit_rules": exit_tick_result,
        "dip_watch": dip_tick_result,
        "eod_flatten": eod_flatten_tick_result,
        "trading_controls": trading_controls_result,
    }


# ---------------------------------------------------------------------------
# Long-lived loop
# ---------------------------------------------------------------------------


async def _sweep_loop() -> None:  # pragma: no cover \u2014 exercised via run_one_tick
    """Wake up periodically and run one autonomous tick.

    Sleep interval adapts: shorter during market hours, longer
    overnight / weekends. Resilient to transport errors via the same
    try/except wrapping the tick.
    """
    backoff = 30
    while STATE.enabled:
        try:
            await run_one_tick()
            backoff = 30
        except asyncio.CancelledError:
            raise
        except Exception as e:  # defensive \u2014 run_one_tick already catches
            STATE.last_error = str(e)[:300]
            log.warning("autonomy loop tick crashed: %s", e)
            backoff = min(backoff * 2, 600)
            await asyncio.sleep(backoff)
            continue
        # Sleep until the next scheduled tick, adapting to market hours.
        try:
            await asyncio.sleep(current_sweep_interval())
        except asyncio.CancelledError:
            raise


async def _fast_loop() -> None:  # pragma: no cover — exercised via run_fast_tick
    """Phase 25.1 — wake every ``fast_loop_seconds`` and run a fast tick.

    Idle outside market hours (sleeps the same fast interval and bails
    immediately on the next check). Resilient to transport errors via
    backoff like the sweep loop.
    """
    backoff = 30
    while STATE.enabled:
        cfg = STATE._config
        # Allow disabling fast loop via cfg.fast_loop_seconds = 0.
        interval = cfg.fast_loop_seconds
        if interval <= 0:
            await asyncio.sleep(60)
            continue
        try:
            await run_fast_tick()
            backoff = 30
        except asyncio.CancelledError:
            raise
        except Exception as e:  # defensive — run_fast_tick already catches
            STATE.last_error = str(e)[:300]
            log.warning("autonomy fast loop tick crashed: %s", e)
            backoff = min(backoff * 2, 600)
            await asyncio.sleep(backoff)
            continue
        # Phase 35 — adaptive cadence. exit_rules.run_tick() sets the
        # any_position_hot flag whenever any open position's peak has
        # crossed trail_arm_pct. When hot, we drop the interval from
        # 60s to ~10s so a profit-take or trailing-stop fires within a
        # tick of the spike instead of waiting up to a minute.
        hot = bool(STATE.any_position_hot)
        hot_interval = max(1, int(cfg.fast_loop_hot_seconds or 0)) if hot else 0
        effective_interval = hot_interval if (hot and hot_interval > 0) else interval
        try:
            await asyncio.sleep(effective_interval)
        except asyncio.CancelledError:
            raise


def start() -> bool:
    """Start the long-lived loops if not already running.

    Returns True if a fresh sweep task was started, False if one was
    already in flight. Safe to call multiple times. The fast loop is
    started alongside whenever a fresh event loop is available.
    """
    STATE.enabled = True
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No event loop \u2014 will be retried on the next start() call.
        return False
    # Fast loop: start independently of the sweep loop.
    fast_task = STATE._fast_task
    if fast_task is None or fast_task.done():
        STATE._fast_task = loop.create_task(_fast_loop())
    sweep_task = STATE._sweep_task
    if sweep_task is not None and not sweep_task.done():
        return False
    STATE._sweep_task = loop.create_task(_sweep_loop())
    return True


def stop() -> None:
    """Stop both loops. Safe even if nothing is running."""
    STATE.enabled = False
    for attr in ("_sweep_task", "_fast_task"):
        task = getattr(STATE, attr)
        if task is not None and not task.done():
            task.cancel()
        setattr(STATE, attr, None)


def snapshot() -> dict[str, Any]:
    """Lightweight status for the dashboard / API."""
    cfg = STATE._config
    return {
        "enabled": bool(STATE.enabled),
        "running": bool(
            STATE._sweep_task is not None and not STATE._sweep_task.done()
        ),
        "market_open": is_market_open(),
        "current_interval_s": current_sweep_interval(),
        "last_sweep_started_at": STATE.last_sweep_started_at,
        "last_sweep_finished_at": STATE.last_sweep_finished_at,
        "last_sweep_status": STATE.last_sweep_status,
        "last_sweep_candidates": STATE.last_sweep_candidates,
        "last_curiosity_at": STATE.last_curiosity_at,
        "last_curiosity_focus": list(STATE.last_curiosity_focus),
        "last_curiosity_reason": STATE.last_curiosity_reason,
        "last_error": STATE.last_error,
        "last_regime": STATE.last_regime,
        "last_reflection": STATE.last_reflection,
        "last_judged_count": STATE.last_judged_count,
        "last_bandit_weights": dict(STATE.last_bandit_weights),
        # Phase 25.1 fast-loop telemetry.
        "fast_loop": {
            "running": bool(
                STATE._fast_task is not None and not STATE._fast_task.done()
            ),
            "interval_s": cfg.fast_loop_seconds,
            "last_tick_at": STATE.last_fast_tick_at,
            "last_tick_status": STATE.last_fast_tick_status,
            "last_exit": STATE.last_fast_tick_exit,
            "last_dip": STATE.last_fast_tick_dip,
            "last_eod_flatten": STATE.last_fast_tick_eod_flatten,
        },
        "config": {
            "sweep_market_seconds": cfg.sweep_market_seconds,
            "sweep_off_seconds": cfg.sweep_off_seconds,
            "fast_loop_seconds": cfg.fast_loop_seconds,
            "curiosity_top_n": cfg.curiosity_top_n,
            "curiosity_focus_count": cfg.curiosity_focus_count,
            "self_improve_enabled": cfg.self_improve_enabled,
            "judgment_horizon_hours": cfg.judgment_horizon_hours,
        },
    }


def reset_for_tests() -> None:
    """Wipe state and stop the task. Tests only."""
    stop()
    STATE.last_sweep_started_at = None
    STATE.last_sweep_finished_at = None
    STATE.last_sweep_status = None
    STATE.last_sweep_candidates = 0
    STATE.last_curiosity_at = None
    STATE.last_curiosity_focus = []
    STATE.last_curiosity_reason = ""
    STATE.last_error = ""
    STATE.last_regime = None
    STATE.last_reflection = None
    STATE.last_judged_count = 0
    STATE.last_bandit_weights = {}
    STATE.last_fast_tick_at = None
    STATE.last_fast_tick_status = None
    STATE.last_fast_tick_exit = None
    STATE.last_fast_tick_dip = None
    STATE.last_fast_tick_eod_flatten = None
    # Phase 35 fields.
    STATE.any_position_hot = False
    STATE.last_fast_tick_heartbeat_at = None
    STATE._config = AutonomyConfig()
