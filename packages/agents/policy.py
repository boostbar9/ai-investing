"""Confidence-gated trading policy (Phase 13).

This module implements a discrete BUY / HOLD / SELL policy that only
acts when a composite confidence score crosses a threshold. The
explicit "HOLD" zone is the killer feature: instead of always having
an opinion (and trading on noise), the policy abstains when its
signals don't agree strongly enough to overcome trading friction.

Composite confidence per symbol is a weighted blend of:

    candidate.confidence      (research sweep heuristic in [0, 1])
    catalyst                  (event quality: earnings/news/volume/analyst,
                               fail-safe, gated by a fundamentals red flag)
    regime_confidence         (HMM posterior probability of current state)
    trust_score               (reddit_trust + corroboration boost)
    base_ensemble_alignment   (does the ensemble also want this name?)

The blend weights live in CONFIDENCE_WEIGHTS so we can tune (or later
fit) them from real evidence. Symbols with composite >= BUY_THRESHOLD
become full positions; composite <= SELL_THRESHOLD become exits;
anything between is left untouched. Position sizing is full-equal-weight
across all BUY symbols at decision time -- proportional sizing is a
follow-up once we have calibration data.

The policy is deliberately a *third* strategy run in parallel to the
existing ensemble: it does NOT replace anything. Phase 13's whole point
is to have an apples-to-apples paper-trade comparison over the same
14-day soak.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from typing import Any, Literal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds (env-overridable so users can tune without redeploying)
# ---------------------------------------------------------------------------

# Threshold to enter a position. 0.65 means "we want at least 65% composite
# confidence before paying spread + slippage on a new trade." Starting on the
# higher side; if the policy never trades we can drop it after 1 week of
# evidence.
BUY_THRESHOLD = float(os.getenv("POLICY_BUY_THRESHOLD", "0.65"))

# Threshold to *exit* an existing position. Asymmetric on purpose -- once
# we're in a trade, we hold through some signal weakness (we already paid
# the spread). Only exit when confidence really erodes.
SELL_THRESHOLD = float(os.getenv("POLICY_SELL_THRESHOLD", "0.35"))

# Maximum number of concurrent BUY positions. Caps both diversification
# floor (so we don't go all-in on one name) and capital efficiency on a
# $300 float (so each position is at least $30).
MAX_POSITIONS = int(os.getenv("POLICY_MAX_POSITIONS", "10"))

# Floor on portfolio cash. Even with 10 perfect signals we keep 5% cash
# so the rebalancer has wiggle room.
CASH_FLOOR = float(os.getenv("POLICY_CASH_FLOOR", "0.05"))


# Composite-confidence blend weights. Must sum to roughly 1.0 -- the
# composite is clamped to [0, 1] at the end so small drift is fine.
#
# Phase 3 (catalyst-swing) rebalance. Every weight is env-overridable via the
# ``POLICY_WEIGHT_<COMPONENT>`` pattern so the blend can be tuned in production
# without a redeploy (e.g. raise ``POLICY_WEIGHT_TRUST`` the day Reddit's feed
# comes back). Rationale for the new defaults vs. the original
# (candidate .40 / regime .20 / trust .20 / ensemble .20):
#
#   * ``trust`` 0.20 -> 0.05. Reddit is 403-blocked live, so the trust feed is
#     effectively dead weight that always degrades toward 0 and just dilutes
#     the score. We keep it NON-ZERO (not deleted) so the moment Reddit
#     recovers, bumping POLICY_WEIGHT_TRUST back up revives it with no code
#     change.
#   * ``ensemble_alignment`` 0.20 -> 0.15. It's a coarse binary vote; in a
#     catalyst-driven discovery flow many genuine event names simply aren't in
#     the ensemble yet, so leaning on it this hard suppressed good catalysts.
#   * ``catalyst`` 0.00 -> 0.20 (NEW). The 0.20 freed from trust + ensemble
#     funds an explicit event-quality component (earnings proximity, fresh-news
#     recency/corroboration, volume expansion, analyst/insider) gated by a
#     fundamentals red-flag cap. This is the whole point of the strategy: score
#     event quality directly instead of inferring it from dead Reddit trust.
#   * ``candidate`` 0.40 and ``regime`` 0.20 are UNCHANGED -- the dominant
#     direct research signal and the crisis-gating regime semantics are
#     preserved.
CONFIDENCE_WEIGHTS = {
    # The research-sweep heuristic captures mention volume + sentiment.
    # Weighted highest because it's the most direct per-symbol signal.
    "candidate": float(os.getenv("POLICY_WEIGHT_CANDIDATE", "0.40")),
    # Event-quality blend built ONLY from signals already on the candidate
    # record (earnings proximity, fresh news, volume expansion, analyst/
    # insider), capped by a fundamentals sanity gate. Each sub-signal degrades
    # to 0 when its feed is missing -- never bearish, never fabricated.
    "catalyst": float(os.getenv("POLICY_WEIGHT_CATALYST", "0.20")),
    # Regime posterior: a 0.9-confident bull regime makes us trust BUYs
    # more. In crisis the same candidate confidence buys nothing.
    "regime": float(os.getenv("POLICY_WEIGHT_REGIME", "0.20")),
    # Does the existing ensemble also have a non-zero weight on this
    # name? If yes, full vote into composite; the ensemble has independent
    # logic so agreement is meaningful.
    "ensemble_alignment": float(
        os.getenv("POLICY_WEIGHT_ENSEMBLE_ALIGNMENT", "0.15")
    ),
    # Reddit trust + corroboration: high-karma authors corroborated by
    # news = strong vote. Missing data degrades to 0 (handled below).
    # Default is low because Reddit is 403-blocked live; raise the env
    # override to revive it when the feed recovers.
    "trust": float(os.getenv("POLICY_WEIGHT_TRUST", "0.05")),
}


# ---------------------------------------------------------------------------
# Catalyst sub-signal knobs (env-overridable). The catalyst component is a
# weighted blend of four sub-signals, each pulled straight off the candidate /
# sweep record produced by ``research_sweep`` and each fail-safe: a missing or
# stale feed contributes EXACTLY 0 (neutral), never a negative/bearish value
# and never a fabricated one.
# ---------------------------------------------------------------------------

# Internal weights of the catalyst sub-signals. Sum ~1.0 so the catalyst
# component itself stays in [0, 1] before the fundamentals cap.
CATALYST_SUBWEIGHTS = {
    "earnings": 0.25,         # proximity to next earnings report
    "news": 0.30,             # fresh-news recency + corroboration
    "volume": 0.15,           # relative-volume expansion
    "analyst_insider": 0.30,  # analyst rating/upgrade + insider buying
}

# Earnings within this many calendar days counts as a near-term catalyst;
# day 0 scores 1.0 and it ramps linearly to 0 at the window edge.
CATALYST_EARNINGS_WINDOW_DAYS = float(
    os.getenv("POLICY_CATALYST_EARNINGS_WINDOW_DAYS", "14")
)
# rel_volume (volume / 30d-average) at/above this multiple = full volume score.
CATALYST_VOLUME_FULL = float(os.getenv("POLICY_CATALYST_VOLUME_FULL", "2.0"))
# News-headline count (rss + yahoo) that saturates the freshness sub-signal.
CATALYST_NEWS_SATURATION = float(
    os.getenv("POLICY_CATALYST_NEWS_SATURATION", "5.0")
)
# Form-4 filing count in the last 30d that saturates the insider-activity nudge.
CATALYST_FORM4_SATURATION = float(
    os.getenv("POLICY_CATALYST_FORM4_SATURATION", "5.0")
)
# Fundamentals sanity gate: a Noncompliant / delisting-risk name has its
# catalyst score CAPPED at this ceiling (default 0.0 = no catalyst credit at
# all). It can only ever REDUCE the score, never boost it.
CATALYST_NONCOMPLIANT_CAP = float(
    os.getenv("POLICY_CATALYST_NONCOMPLIANT_CAP", "0.0")
)


Action = Literal["buy", "hold", "sell"]


# Regime-specific multiplier on BUY_THRESHOLD. In crisis we want MUCH
# higher confidence before opening anything new; in bull we can be
# slightly less picky. Multiplier > 1 = harder to trigger BUY.
REGIME_BUY_MULTIPLIER: dict[str, float] = {
    "bull": 0.95,    # easier to buy
    "chop": 1.05,    # slightly harder (range-bound markets eat trend bets)
    "bear": 1.30,    # much harder
    "crisis": 1.80,  # almost never (only exceptionally strong signals)
}


@dataclass(frozen=True)
class PolicyDecision:
    """One symbol's BUY/HOLD/SELL decision with its composite confidence.

    Persisted into the decision log so we can plot a calibration curve
    after enough trades (predicted-confidence vs. realised-win-rate).

    Phase 14 adds ``raw_confidence``: when a calibrator is active,
    ``composite_confidence`` is the *calibrated* score used for threshold
    gating, and ``raw_confidence`` keeps the uncalibrated composite for
    diagnostics. When no calibrator is fitted the two are equal.
    """

    symbol: str
    action: Action
    composite_confidence: float  # in [0, 1] -- post-calibration
    components: dict[str, float] = field(default_factory=dict)
    reason: str = ""
    # Phase 14: uncalibrated composite (additive; identical to
    # composite_confidence when no calibrator is fitted).
    raw_confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "symbol": self.symbol,
            "action": self.action,
            "confidence": round(float(self.composite_confidence), 4),
            "components": {k: round(float(v), 4) for k, v in self.components.items()},
            "reason": self.reason,
        }
        if self.raw_confidence is not None:
            out["raw_confidence"] = round(float(self.raw_confidence), 4)
        return out


def _safe_float(v: Any, default: float = 0.0) -> float:
    """Coerce ``v`` to float, returning ``default`` on any failure.

    Used because candidate / sweep payloads come from JSON and may
    legitimately carry ``None`` for unavailable enrichment fields.
    """
    try:
        if v is None:
            return default
        f = float(v)
    except (TypeError, ValueError):
        return default
    if math.isnan(f) or math.isinf(f):
        return default
    return f


def _clip01(x: float) -> float:
    """Clamp into [0, 1]. Composite math can drift slightly outside.

    NaN/inf are treated as 0.0 -- garbage in produces a neutral signal
    rather than poisoning downstream weights.
    """
    if not math.isfinite(x):
        return 0.0
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _catalyst_earnings(cand: dict[str, Any]) -> float:
    """Near-term earnings proximity sub-signal in [0, 1].

    ``days_to_earnings`` is ``None`` when the feed didn't run / no future
    report is known -> 0.0 (unknown is NEUTRAL, never bearish). A report today
    scores 1.0 and the score ramps linearly to 0 at the window edge; a date
    past the window (or in the past) is no longer a forward catalyst -> 0.0.
    """
    days = cand.get("days_to_earnings")
    if days is None:
        return 0.0
    d = _safe_float(days, default=-1.0)
    window = CATALYST_EARNINGS_WINDOW_DAYS
    if d < 0.0 or window <= 0.0 or d >= window:
        return 0.0
    return _clip01((window - d) / window)


def _catalyst_news(cand: dict[str, Any]) -> float:
    """Fresh-news recency + corroboration sub-signal in [0, 1].

    Built from the corroboration score (already in [0, 1]) and the volume of
    fresh headlines (RSS + Yahoo). All inputs default to 0 / False, so a
    candidate that never saw a news feed scores 0.0 (NEUTRAL). A corroborated
    story floors the sub-signal at 0.5 even when the raw headline count is thin
    -- corroboration is the strongest freshness evidence we have.
    """
    corro = _clip01(_safe_float(cand.get("corroboration_score"), 0.0))
    headlines = max(0.0, _safe_float(cand.get("news_headlines"), 0.0)) + max(
        0.0, _safe_float(cand.get("yahoo_news_count"), 0.0)
    )
    saturation = (
        _clip01(headlines / CATALYST_NEWS_SATURATION)
        if CATALYST_NEWS_SATURATION > 0.0
        else 0.0
    )
    score = max(corro, saturation)
    if bool(cand.get("corroborated", False)):
        score = max(score, 0.5)
    return _clip01(score)


def _catalyst_volume(cand: dict[str, Any]) -> float:
    """Relative-volume expansion sub-signal in [0, 1].

    ``rel_volume`` is volume / 30d-average; it defaults to 0.0 when the
    fundamentals feed didn't run (unknown) and is <= 1.0 when there is simply
    no expansion. BOTH map to 0.0 (NEUTRAL) -- a quiet or unknown tape is never
    bearish. Expansion ramps linearly from 1x up to ``CATALYST_VOLUME_FULL``.
    """
    rv = _safe_float(cand.get("rel_volume"), 0.0)
    if rv <= 1.0:
        return 0.0
    full = CATALYST_VOLUME_FULL if CATALYST_VOLUME_FULL > 1.0 else 2.0
    return _clip01((rv - 1.0) / (full - 1.0))


def _catalyst_analyst_insider(cand: dict[str, Any]) -> float:
    """Analyst + insider (Form 4) sub-signal in [0, 1].

    Averages whichever of these are actually present (each absent input is
    simply skipped, NOT scored 0-and-averaged-in, so a single strong present
    signal isn't diluted by missing feeds):

      * analyst mean rating (1=Strong Buy .. 5=Strong Sell): mapped so 1 -> 1.0
        and >= 3 (hold/sell) -> 0.0. Only counted when at least one analyst
        covers the name.
      * a recent analyst UPGRADE -> 1.0. A downgrade is NOT scored negative
        (catalyst never goes bearish); it just doesn't contribute.
      * insider buy/sell mix: net buying -> toward 1.0.
      * recent Form-4 filing activity: a mild presence nudge.

    Returns 0.0 when none of the feeds are present (NEUTRAL).
    """
    parts: list[float] = []

    rating = _safe_float(cand.get("analyst_mean_rating"), 0.0)
    num = _safe_float(cand.get("analyst_num"), 0.0)
    if rating > 0.0 and num > 0.0:
        parts.append(_clip01((3.0 - rating) / 2.0))

    if str(cand.get("analyst_recent_action") or "").strip().lower() == "upgrade":
        parts.append(1.0)

    buys = max(0.0, _safe_float(cand.get("insider_buy_count"), 0.0))
    sells = max(0.0, _safe_float(cand.get("insider_sell_count"), 0.0))
    if buys + sells > 0.0:
        parts.append(_clip01(buys / (buys + sells)))

    form4 = _safe_float(cand.get("insider_form4_30d"), 0.0)
    if form4 > 0.0 and CATALYST_FORM4_SATURATION > 0.0:
        parts.append(_clip01(form4 / CATALYST_FORM4_SATURATION))

    if not parts:
        return 0.0
    return _clip01(sum(parts) / len(parts))


def catalyst_score(candidate: dict[str, Any] | None) -> float:
    """Composite event-quality score in [0, 1] from the candidate record.

    Weighted blend of the four catalyst sub-signals (earnings proximity, fresh
    news, volume expansion, analyst/insider). Each sub-signal is independently
    fail-safe: a missing/stale feed contributes 0 (NEUTRAL), so a candidate
    with no enrichment scores exactly 0.0 and is never penalised below zero.

    Fundamentals sanity gate: when the candidate is flagged Noncompliant /
    delisting-risk (``compliance_ok`` is False -- only ever set by a feed that
    actually ran), the score is CAPPED at ``CATALYST_NONCOMPLIANT_CAP``
    (default 0.0). The gate can only REDUCE the score, never boost it. A
    candidate whose fundamentals feed never ran keeps ``compliance_ok=True`` and
    is therefore never capped -- absence is not a red flag.
    """
    cand = candidate or {}
    subs = {
        "earnings": _catalyst_earnings(cand),
        "news": _catalyst_news(cand),
        "volume": _catalyst_volume(cand),
        "analyst_insider": _catalyst_analyst_insider(cand),
    }
    score = _clip01(
        sum(subs[k] * CATALYST_SUBWEIGHTS.get(k, 0.0) for k in subs)
    )
    if not bool(cand.get("compliance_ok", True)):
        score = min(score, _clip01(CATALYST_NONCOMPLIANT_CAP))
    return _clip01(score)


def composite_confidence(
    *,
    candidate: dict[str, Any] | None,
    regime: str,
    regime_confidence: float,
    ensemble_weight: float,
) -> tuple[float, dict[str, float]]:
    """Blend the per-symbol signals into one composite confidence in [0, 1].

    Returns (composite, components_breakdown). The breakdown is logged
    into the decision record so we can later inspect which input drove
    a given action -- vital for debugging false signals.
    """
    cand = candidate or {}

    # Component 1: research sweep heuristic. Already in [0, 1].
    cand_conf = _clip01(_safe_float(cand.get("confidence"), 0.0))

    # Component 2: regime posterior, modulated by direction. A
    # high-confidence BEAR regime should NOT increase BUY confidence
    # for risk assets, so we invert in non-bull regimes.
    reg_conf_raw = _clip01(_safe_float(regime_confidence, 0.0))
    if regime == "bull":
        reg_component = reg_conf_raw
    elif regime == "chop":
        # Chop is neutral -- contribute 0.5 regardless of posterior.
        reg_component = 0.5
    else:
        # Bear / crisis: a confident bearish regime SUBTRACTS from our
        # buy confidence. Mapped so 0.9 posterior crisis -> 0.1 score.
        reg_component = _clip01(1.0 - reg_conf_raw)

    # Component 3: trust + corroboration. Reddit trust is in [0, 1];
    # corroboration is a boolean we collapse to 0.0 or 0.3. The blend
    # is intentionally lossy -- a strong corroborated story should
    # pull this above 0.7 even if reddit_trust alone is modest.
    reddit_trust = _clip01(_safe_float(cand.get("reddit_trust"), 0.0))
    corroborated_bonus = 0.30 if bool(cand.get("corroborated", False)) else 0.0
    trust_component = _clip01(reddit_trust + corroborated_bonus)

    # Component 4: ensemble alignment. If the existing ensemble has a
    # non-trivial weight on this name (>= 1bps), treat as full vote.
    # If the ensemble is silent on the name, this contributes 0 -- which
    # makes pure-research candidates (no ensemble support) much harder
    # to trigger. Intentional: we don't want to act on Reddit-only signal.
    aligned = abs(_safe_float(ensemble_weight, 0.0)) >= 1e-4
    align_component = 1.0 if aligned else 0.0

    # Component 5: catalyst (event quality). Built ONLY from signals already on
    # the candidate record; fully fail-safe (missing feed -> 0, never bearish,
    # never fabricated) and capped by the fundamentals red-flag gate. See
    # ``catalyst_score``.
    catalyst_component = catalyst_score(cand)

    components = {
        "candidate": cand_conf,
        "catalyst": catalyst_component,
        "regime": reg_component,
        "trust": trust_component,
        "ensemble_alignment": align_component,
    }
    weights = CONFIDENCE_WEIGHTS
    composite = sum(components[k] * weights.get(k, 0.0) for k in components)
    return _clip01(composite), components


@dataclass
class ConfidenceGatedPolicy:
    """Discrete-action policy that abstains when confidence is moderate.

    Construction takes overrides so tests can pin thresholds. Defaults
    pull from the module constants (which themselves pull from env).

    Phase 14: optional ``calibrator`` (any callable mapping a float to a
    float in [0, 1]). When provided, raw composite confidence is mapped
    through the calibrator BEFORE threshold gating. This means the
    BUY/SELL thresholds always speak in true-probability space, not in
    whatever-units the raw composite happens to drift into.
    """

    buy_threshold: float = BUY_THRESHOLD
    sell_threshold: float = SELL_THRESHOLD
    max_positions: int = MAX_POSITIONS
    cash_floor: float = CASH_FLOOR
    # Optional calibration map. Typing is intentionally loose (any callable
    # taking and returning a float) so this module doesn't import
    # calibration.py -- keeps the dep graph one-way to avoid cycles.
    calibrator: Any = None

    def __post_init__(self) -> None:
        # Sanity: SELL must be strictly below BUY, or every name in the
        # "between" zone gets two opposing actions.
        if not (0.0 <= self.sell_threshold < self.buy_threshold <= 1.0):
            raise ValueError(
                f"invalid thresholds: sell={self.sell_threshold} buy={self.buy_threshold}"
            )

    def decide(
        self,
        *,
        sweep_candidates: list[dict[str, Any]] | None,
        ensemble_weights: dict[str, float] | None,
        current_holdings: set[str] | None,
        regime: str,
        regime_confidence: float,
    ) -> list[PolicyDecision]:
        """Produce one decision per symbol we have any opinion on.

        Symbols considered: union of (sweep candidates, ensemble names,
        current holdings). For each we compute composite confidence,
        apply regime-modulated thresholds, and emit BUY/HOLD/SELL.

        Critically, **only emits SELL for symbols we actually hold** --
        a SELL for an empty position is a no-op and would pollute the
        decision log with noise. Conversely we emit BUY for new names
        even if ensemble didn't pick them, as long as composite clears.
        """
        sweep_by_symbol: dict[str, dict[str, Any]] = {}
        for c in (sweep_candidates or []):
            sym = str(c.get("symbol", "")).upper().strip()
            if sym:
                sweep_by_symbol[sym] = c

        ensemble_by_symbol: dict[str, float] = {
            str(k).upper(): float(v) for k, v in (ensemble_weights or {}).items()
        }
        held = {s.upper() for s in (current_holdings or set())}

        universe = set(sweep_by_symbol) | set(ensemble_by_symbol) | held
        regime_mult = REGIME_BUY_MULTIPLIER.get(regime, 1.0)
        effective_buy = _clip01(self.buy_threshold * regime_mult)

        decisions: list[PolicyDecision] = []
        for sym in sorted(universe):
            raw_composite, components = composite_confidence(
                candidate=sweep_by_symbol.get(sym),
                regime=regime,
                regime_confidence=regime_confidence,
                ensemble_weight=ensemble_by_symbol.get(sym, 0.0),
            )

            # Phase 14: pipe through calibrator if one is fitted. The
            # calibrator is monotone so action ordering is preserved, but
            # the absolute confidence level may shift -- which is the
            # whole point: thresholds now mean true probability.
            if self.calibrator is not None:
                try:
                    composite = _clip01(float(self.calibrator(raw_composite)))
                except Exception as exc:  # never let calibration crash a cycle
                    logger.warning(
                        "calibrator failed on %.3f for %s: %s; using raw",
                        raw_composite, sym, exc,
                    )
                    composite = raw_composite
            else:
                composite = raw_composite

            is_held = sym in held

            if composite >= effective_buy:
                # New BUY (or maintain). Reason carries the cap state so
                # we can see in logs when good signals got cut by MAX_POSITIONS.
                action: Action = "buy"
                reason = (
                    f"composite {composite:.2f} >= {effective_buy:.2f} "
                    f"({regime}, mult {regime_mult:.2f})"
                )
            elif composite <= self.sell_threshold and is_held:
                # SELL: only emit when we actually have something to sell.
                action = "sell"
                reason = (
                    f"composite {composite:.2f} <= {self.sell_threshold:.2f} "
                    f"and currently held"
                )
            else:
                action = "hold"
                if is_held:
                    reason = (
                        f"composite {composite:.2f} in HOLD zone; staying long"
                    )
                else:
                    reason = (
                        f"composite {composite:.2f} below buy threshold {effective_buy:.2f}; "
                        f"no position"
                    )

            decisions.append(
                PolicyDecision(
                    symbol=sym,
                    action=action,
                    composite_confidence=composite,
                    components=components,
                    reason=reason,
                    raw_confidence=(
                        raw_composite if self.calibrator is not None else None
                    ),
                )
            )

        return decisions

    def to_target_weights(
        self,
        decisions: list[PolicyDecision],
        *,
        sizer: Any = None,
        equity: float = 0.0,
        peak_equity: float = 0.0,
        realised_vols: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Convert discrete decisions to the weight dict paper_trade expects.

        BUY -> sized weight (subject to MAX_POSITIONS cap and CASH_FLOOR).
        HOLD -> not in dict (rebalancer treats absence as "keep current").
        SELL -> 0.0 (rebalancer interprets as flat-the-position).

        Phase 15: an optional ``sizer`` (any ``packages.agents.sizing.RiskSizer``
        instance) takes over BUY weight computation. When provided, it
        sees the BUYs + the current equity/peak/vol context and applies
        confidence-proportional or fractional-Kelly sizing with an
        optional drawdown taper. Missing sizer -> Phase 13 equal-weight
        behaviour (full back-compat).
        """
        # Sort BUYs by composite descending so the cap keeps the best names.
        buys = sorted(
            (d for d in decisions if d.action == "buy"),
            key=lambda d: -d.composite_confidence,
        )
        sells = [d for d in decisions if d.action == "sell"]

        weights: dict[str, float] = {}
        if sizer is not None and buys:
            # Delegate BUY sizing to the risk-adaptive sizer. It owns the
            # max_positions cap + cash_floor internally so we don't
            # double-apply either constraint here.
            try:
                result = sizer.size(
                    buy_decisions=buys,
                    max_positions=self.max_positions,
                    equity=equity,
                    peak_equity=peak_equity,
                    realised_vols=realised_vols,
                )
                for sym, w in result.weights.items():
                    weights[sym] = round(float(w), 6)
                if result.notes:
                    logger.info("policy sizer notes: %s", "; ".join(result.notes))
            except Exception as exc:  # never let sizing crash the cycle
                logger.warning(
                    "policy sizer failed (%s); falling back to equal-weight", exc
                )
                weights = self._equal_weight_buys(buys)
        else:
            weights = self._equal_weight_buys(buys)

        # Emit explicit zero for SELLs so the rebalancer knows to flatten.
        for d in sells:
            weights[d.symbol] = 0.0
        return weights

    def _equal_weight_buys(self, buys: list[PolicyDecision]) -> dict[str, float]:
        """Phase 13 equal-weight sizing, kept as the fallback path so any
        sizer failure degrades gracefully instead of submitting zero-size
        orders into a hot cycle."""
        kept_buys = buys[: self.max_positions]
        if buys[self.max_positions :]:
            logger.info(
                "policy capped %d additional BUY candidates (max_positions=%d)",
                len(buys) - self.max_positions, self.max_positions,
            )
        weights: dict[str, float] = {}
        if kept_buys:
            per = (1.0 - self.cash_floor) / len(kept_buys)
            for d in kept_buys:
                weights[d.symbol] = round(per, 6)
        return weights


__all__ = [
    "BUY_THRESHOLD",
    "CASH_FLOOR",
    "CATALYST_SUBWEIGHTS",
    "CONFIDENCE_WEIGHTS",
    "MAX_POSITIONS",
    "REGIME_BUY_MULTIPLIER",
    "SELL_THRESHOLD",
    "Action",
    "ConfidenceGatedPolicy",
    "PolicyDecision",
    "catalyst_score",
    "composite_confidence",
]
