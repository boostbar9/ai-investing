"""Confidence-gated trading policy (Phase 13).

This module implements a discrete BUY / HOLD / SELL policy that only
acts when a composite confidence score crosses a threshold. The
explicit "HOLD" zone is the killer feature: instead of always having
an opinion (and trading on noise), the policy abstains when its
signals don't agree strongly enough to overcome trading friction.

Composite confidence per symbol is a weighted blend of:

    candidate.confidence      (research sweep heuristic in [0, 1])
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
CONFIDENCE_WEIGHTS = {
    # The research-sweep heuristic captures mention volume + sentiment.
    # Weighted highest because it's the most direct per-symbol signal.
    "candidate": 0.40,
    # Regime posterior: a 0.9-confident bull regime makes us trust BUYs
    # more. In crisis the same candidate confidence buys nothing.
    "regime": 0.20,
    # Reddit trust + corroboration: high-karma authors corroborated by
    # news = strong vote. Missing data degrades to 0 (handled below).
    "trust": 0.20,
    # Does the existing ensemble also have a non-zero weight on this
    # name? If yes, +20% to composite; the ensemble has independent
    # logic so agreement is meaningful.
    "ensemble_alignment": 0.20,
}


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

    components = {
        "candidate": cand_conf,
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
    "CONFIDENCE_WEIGHTS",
    "MAX_POSITIONS",
    "REGIME_BUY_MULTIPLIER",
    "SELL_THRESHOLD",
    "Action",
    "ConfidenceGatedPolicy",
    "PolicyDecision",
    "composite_confidence",
]
