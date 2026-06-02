"""Intraday morning setup finder — Phase 28-R step 4.

Selects the top-K intraday day-trade setups at the start of each US
cash session by combining four orthogonal signals:

  1. **ORB breakout** \u2014 has the symbol's last close cleared the
     opening-range high within the entry window?
  2. **VWAP alignment** \u2014 is price trading above the session VWAP
     (i.e. on the right side of value)?
  3. **News-sentiment tilt** \u2014 trailing-window aggregate sentiment
     score from packages.data.adapters.sentiment (Phase 26).
  4. **Insider-cluster tilt** \u2014 InsiderSignal score from
     packages.data.finnhub_insider (Phase 27).

A **liquidity filter** rejects names with < $50M/20d average dollar
volume. Position sizing splits ``min($300, equity * 0.01)`` evenly
across the surviving top-K (default K=3).

This module is intentionally pure: the public entry point
``rank_candidates`` takes a list of pre-built ``CandidateInput``s and
returns ranked ``RankedSetup``s. The orchestrator
``find_morning_setups`` accepts injected providers (price, sentiment,
insider, liquidity) so unit tests never need a live data feed.

The module is **opt-in via INTRADAY_MODE=1**; the runtime never
imports it unless that flag is set. Until the user flips the switch
this code is dormant.
"""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

log = logging.getLogger("intraday.setup_finder")


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------


# Liquidity floor. The user trades ~$300/day on a $99k account, so
# anything below $50M/20d ADV would be illiquid relative to even our
# tiny clip and could be hard to exit by 15:55 ET.
DEFAULT_LIQUIDITY_FLOOR_USD = 50_000_000.0

# Top-K to ship to the router per session.
DEFAULT_TOP_K = 3

# Daily float ceiling \u2014 we never put more than this in fresh intraday
# names, regardless of equity.
DAILY_FLOAT_CEILING_USD = 300.0

# Fraction-of-equity cap. Whichever floor (ceiling / equity*pct) is
# smaller wins. 1% of equity matches the user's risk preference.
EQUITY_FRACTION = 0.01

# Per-position minimum after the split. Below this we drop the slot
# rather than ship a sub-$50 order that gets eaten by spread/fees.
MIN_PER_POSITION_USD = 25.0

# Signal weights for the composite score. ORB and VWAP are the price
# truth; news + insider are decorations that tilt ranking.
WEIGHTS: dict[str, float] = {
    "orb_breakout": 0.40,
    "vwap_align": 0.30,
    "news_sentiment": 0.20,
    "insider_cluster": 0.10,
}


# ---------------------------------------------------------------------------
# Input / output dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidateInput:
    """Pre-built signal bundle for one symbol at one decision instant.

    All values are precomputed by the orchestrator (or stubbed by tests)
    so ``rank_candidates`` stays pure.

    Attributes:
        symbol: Upper-cased ticker.
        close: Most recent intraday close.
        orb_high: High of the opening-range window (5-min bars over
            the first 30 min of the session).
        vwap: Session VWAP at the decision moment.
        adv_usd_20d: Trailing 20-day average dollar volume. Used for
            the liquidity filter.
        news_score: Aggregate news-sentiment score in [-1, 1] from
            packages.data.adapters.sentiment.aggregate_sentiment.
            ``None`` means "no news, treat as 0".
        news_n: Headline count contributing to ``news_score``. Acts as
            a confidence proxy.
        insider_score: InsiderSignal.score in [-1, 1] from Phase 27.
            ``None`` means "no signal, treat as 0".
        insider_confidence: InsiderSignal.confidence in [0, 1].
    """

    symbol: str
    close: float
    orb_high: float
    vwap: float
    adv_usd_20d: float
    news_score: float | None = None
    news_n: int = 0
    insider_score: float | None = None
    insider_confidence: float = 0.0


@dataclass(frozen=True)
class RankedSetup:
    """One ranked setup ready for the router."""

    symbol: str
    score: float
    components: dict[str, float]
    notional_usd: float
    reason: str


@dataclass(frozen=True)
class SetupFinderResult:
    """Output of ``find_morning_setups``."""

    setups: list[RankedSetup]
    rejected: list[dict[str, Any]] = field(default_factory=list)
    ts: str = ""


# ---------------------------------------------------------------------------
# Component scorers \u2014 pure functions
# ---------------------------------------------------------------------------


def score_orb_breakout(close: float, orb_high: float) -> float:
    """+1.0 when close is meaningfully above the ORB high, 0.0 otherwise.

    A 0.25% margin above orb_high registers as a full breakout; below
    or equal returns 0. We saturate at 1.5% above ORB so a screaming
    runner doesn't dominate.
    """
    if orb_high <= 0 or close <= 0:
        return 0.0
    edge = (close - orb_high) / orb_high
    if edge <= 0:
        return 0.0
    # 0.25%..1.5% linear ramp \u2192 [0.2, 1.0]; >1.5% caps at 1.0.
    if edge < 0.0025:
        return max(0.0, edge / 0.0025 * 0.2)
    if edge >= 0.015:
        return 1.0
    return 0.2 + (edge - 0.0025) / (0.015 - 0.0025) * 0.8


def score_vwap_align(close: float, vwap: float) -> float:
    """+1.0 when close is comfortably above VWAP, 0.0 when below."""
    if vwap <= 0 or close <= 0:
        return 0.0
    edge = (close - vwap) / vwap
    if edge <= 0:
        return 0.0
    # 0..1% maps to [0, 1.0] linearly; above 1% caps at 1.0.
    return min(1.0, edge / 0.01)


def score_news_sentiment(
    sentiment: float | None, n_headlines: int
) -> float:
    """Map signed sentiment in [-1, 1] to [0, 1].

    n_headlines acts as a multiplier (1+ -> full strength, 0 -> 0).
    We clip to [0, 1] so bearish headlines just zero out the tilt
    rather than penalising the composite \u2014 the price-side scorers
    already vetoed any bearish setup before we got here.
    """
    if sentiment is None or n_headlines <= 0:
        return 0.0
    s = max(0.0, min(1.0, (sentiment + 1.0) / 2.0))
    # Confidence ramp: 1 headline -> 0.5, 5+ headlines -> 1.0
    confidence = min(1.0, 0.5 + 0.125 * (n_headlines - 1))
    return s * confidence


def score_insider_cluster(
    insider_score: float | None, confidence: float
) -> float:
    """Insider score weighted by its own confidence.

    Phase 27's InsiderSignal already encodes "single-buyer caps near
    0.15 confidence" so we just multiply through.
    """
    if insider_score is None:
        return 0.0
    s = max(0.0, min(1.0, (insider_score + 1.0) / 2.0))
    c = max(0.0, min(1.0, confidence))
    return s * c


def composite_score(components: dict[str, float]) -> float:
    """Weighted sum of the four components."""
    total = 0.0
    for key, weight in WEIGHTS.items():
        total += weight * components.get(key, 0.0)
    return total


# ---------------------------------------------------------------------------
# Ranking + sizing
# ---------------------------------------------------------------------------


def _per_position_notional(equity_usd: float, k: int) -> float:
    """Cap = min($300, equity * 1%), split evenly across k slots."""
    if k <= 0:
        return 0.0
    cap = min(DAILY_FLOAT_CEILING_USD, max(0.0, equity_usd) * EQUITY_FRACTION)
    return cap / float(k)


def _build_reason(components: dict[str, float], cand: CandidateInput) -> str:
    """Human-readable reason string for chatter + audit."""
    parts: list[str] = []
    if components["orb_breakout"] > 0:
        edge = (cand.close - cand.orb_high) / cand.orb_high * 100
        parts.append(f"ORB+{edge:.2f}%")
    if components["vwap_align"] > 0:
        parts.append(f"VWAP+{((cand.close - cand.vwap) / cand.vwap) * 100:.2f}%")
    if components["news_sentiment"] > 0 and cand.news_score is not None:
        parts.append(f"news {cand.news_score:+.2f}/{cand.news_n}")
    if components["insider_cluster"] > 0 and cand.insider_score is not None:
        parts.append(f"insider {cand.insider_score:+.2f}")
    return " | ".join(parts) if parts else "composite"


def rank_candidates(
    candidates: Iterable[CandidateInput],
    *,
    equity_usd: float,
    top_k: int = DEFAULT_TOP_K,
    liquidity_floor_usd: float = DEFAULT_LIQUIDITY_FLOOR_USD,
) -> tuple[list[RankedSetup], list[dict[str, Any]]]:
    """Pure ranker. Returns (top-K setups, rejected reasons).

    Steps:
      1. Drop names below the liquidity floor.
      2. Drop names with zero ORB+VWAP price signal (the price-side
         scorers must BOTH register above 0 \u2014 no fundamentals-only
         entries).
      3. Score the remainder, sort descending.
      4. Take the top ``top_k``.
      5. Drop any whose split notional falls below MIN_PER_POSITION_USD.
    """
    rejected: list[dict[str, Any]] = []
    scored: list[tuple[float, dict[str, float], CandidateInput]] = []

    for cand in candidates:
        if cand.adv_usd_20d < liquidity_floor_usd:
            rejected.append(
                {
                    "symbol": cand.symbol,
                    "reason": "illiquid",
                    "adv_usd_20d": cand.adv_usd_20d,
                }
            )
            continue
        components = {
            "orb_breakout": score_orb_breakout(cand.close, cand.orb_high),
            "vwap_align": score_vwap_align(cand.close, cand.vwap),
            "news_sentiment": score_news_sentiment(
                cand.news_score, cand.news_n
            ),
            "insider_cluster": score_insider_cluster(
                cand.insider_score, cand.insider_confidence
            ),
        }
        # Price-side gate \u2014 both ORB and VWAP must be positive.
        if components["orb_breakout"] <= 0 or components["vwap_align"] <= 0:
            rejected.append(
                {
                    "symbol": cand.symbol,
                    "reason": "no_price_signal",
                    "components": components,
                }
            )
            continue
        scored.append((composite_score(components), components, cand))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:top_k]

    # Honest sizing: if 2 names pass, split between 2; if 1, get the
    # full ceiling (still capped at $300 or 1% equity).
    k_active = len(top)
    notional = _per_position_notional(equity_usd, k_active)

    if notional < MIN_PER_POSITION_USD:
        # All slots would be too small \u2014 fail open with zero setups so
        # the router doesn't fire postage-stamp orders.
        for _, _, cand in top:
            rejected.append(
                {
                    "symbol": cand.symbol,
                    "reason": "sub_min_per_position",
                    "notional_usd": round(notional, 2),
                }
            )
        return [], rejected

    setups: list[RankedSetup] = []
    for score, components, cand in top:
        setups.append(
            RankedSetup(
                symbol=cand.symbol,
                score=round(score, 4),
                components={k: round(v, 4) for k, v in components.items()},
                notional_usd=round(notional, 2),
                reason=_build_reason(components, cand),
            )
        )
    return setups, rejected


# ---------------------------------------------------------------------------
# Orchestrator \u2014 wraps the ranker with injectable providers
# ---------------------------------------------------------------------------


# Provider callables. Each is a synchronous callable returning a value
# or None. Async providers should be awaited by the caller before
# being adapted into these. Keeping them sync simplifies the ranker.
PriceLookup = Callable[[str], dict[str, float] | None]
SentimentLookup = Callable[[str], tuple[float | None, int]]
InsiderLookup = Callable[[str], tuple[float | None, float]]


def find_morning_setups(
    universe: Iterable[str],
    *,
    equity_usd: float,
    price_lookup: PriceLookup,
    sentiment_lookup: SentimentLookup | None = None,
    insider_lookup: InsiderLookup | None = None,
    top_k: int = DEFAULT_TOP_K,
    liquidity_floor_usd: float = DEFAULT_LIQUIDITY_FLOOR_USD,
    now: datetime | None = None,
) -> SetupFinderResult:
    """Build CandidateInputs for ``universe`` and rank them.

    Args:
        universe: Symbols to consider.
        equity_usd: Current account equity for sizing.
        price_lookup: ``symbol \u2192 {close, orb_high, vwap, adv_usd_20d}``
            or None when no bars are available.
        sentiment_lookup: optional ``symbol \u2192 (score, n_headlines)``.
        insider_lookup: optional ``symbol \u2192 (score, confidence)``.
        top_k: K for the ranker.
        liquidity_floor_usd: Min trailing dollar-volume to consider.
        now: clock override for tests.

    Side effects: none. Logs warnings on per-symbol provider errors but
    never raises.
    """
    ts = (now or datetime.now(UTC)).astimezone(UTC).isoformat(timespec="seconds")
    candidates: list[CandidateInput] = []
    rejected: list[dict[str, Any]] = []

    for raw in universe:
        symbol = str(raw).upper()
        try:
            price_bundle = price_lookup(symbol)
        except Exception as exc:  # pragma: no cover \u2014 defensive
            log.warning("price_lookup failed for %s: %s", symbol, exc)
            rejected.append({"symbol": symbol, "reason": "price_lookup_error"})
            continue
        if not price_bundle:
            rejected.append({"symbol": symbol, "reason": "no_bars"})
            continue

        try:
            close = float(price_bundle["close"])
            orb_high = float(price_bundle["orb_high"])
            vwap = float(price_bundle["vwap"])
            adv_usd_20d = float(price_bundle["adv_usd_20d"])
        except (KeyError, TypeError, ValueError):
            rejected.append({"symbol": symbol, "reason": "malformed_bars"})
            continue

        if math.isnan(close) or math.isnan(vwap) or math.isnan(orb_high):
            rejected.append({"symbol": symbol, "reason": "nan_in_bars"})
            continue

        news_score: float | None = None
        news_n = 0
        if sentiment_lookup is not None:
            try:
                news_score, news_n = sentiment_lookup(symbol)
            except Exception as exc:  # pragma: no cover \u2014 defensive
                log.warning("sentiment_lookup failed for %s: %s", symbol, exc)
                news_score, news_n = None, 0

        insider_score: float | None = None
        insider_conf = 0.0
        if insider_lookup is not None:
            try:
                insider_score, insider_conf = insider_lookup(symbol)
            except Exception as exc:  # pragma: no cover \u2014 defensive
                log.warning("insider_lookup failed for %s: %s", symbol, exc)
                insider_score, insider_conf = None, 0.0

        candidates.append(
            CandidateInput(
                symbol=symbol,
                close=close,
                orb_high=orb_high,
                vwap=vwap,
                adv_usd_20d=adv_usd_20d,
                news_score=news_score,
                news_n=news_n,
                insider_score=insider_score,
                insider_confidence=insider_conf,
            )
        )

    setups, ranker_rejected = rank_candidates(
        candidates,
        equity_usd=equity_usd,
        top_k=top_k,
        liquidity_floor_usd=liquidity_floor_usd,
    )
    rejected.extend(ranker_rejected)
    return SetupFinderResult(setups=setups, rejected=rejected, ts=ts)


# ---------------------------------------------------------------------------
# Mode flag helper
# ---------------------------------------------------------------------------


def is_intraday_mode_enabled() -> bool:
    """True iff ``INTRADAY_MODE=1`` in the environment."""
    return os.environ.get("INTRADAY_MODE", "0") == "1"
