"""Under-the-radar catalyst lane — classifier, liquidity gate, conviction sizing.

This module backs the dedicated discovery lane that deliberately surfaces
under-followed small/micro-cap names WITH a real, detectable catalyst (e.g. a
~$1.50 biotech that gaps on an FDA approval). It is the opposite of the
news-headline-frequency funnel that over-indexes on mega-caps.

Three pure, fail-safe building blocks live here (no network, no globals — every
function is unit-testable with plain inputs):

1. :func:`classify_catalyst` — reads text (headline/summary) and returns a
   ``CatalystSignal`` with a ``catalyst_type`` (fda / m&a / contract / earnings /
   analyst / none), a ``catalyst_score`` in ``[0, 1]``, and a short
   ``catalyst_detail``. **Never fabricates**: no recognised catalyst language ->
   ``type="none"``, ``score=0.0``. Absence of data is ``none``, not a guess.

2. :func:`liquidity_gate` — the MANDATORY tradability/liquidity gate the user
   agreed to keep. A name is EXCLUDED when it is below the price floor, below the
   minimum average daily DOLLAR volume, or above the maximum bid/ask spread — and
   **fail-safe excluded** when any of those inputs is missing/unparseable (we
   never assume a name is tradable). This protects shadow-data integrity.

3. :func:`conviction_notional` — lets a high ``catalyst_score`` earn full
   RELATIVE conviction weight, but the result is STRICTLY clamped by the absolute
   caps: ``$50``/trade, the ``$300`` default budget (remaining), and the
   ``$10,000`` ceiling. Conviction only scales WITHIN those caps; it can never
   bypass them.

All thresholds are env-overridable so the operator can retune without code
changes. Nothing in this module places, mutates, or simulates an order, and
nothing here can enable live trading.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Env-overridable knobs. Read at call time (via the module-level getters) so a
# test's monkeypatched env actually takes effect.
# ---------------------------------------------------------------------------

# Price floor — reaches down into micro-cap territory but keeps out sub-penny
# junk that pollutes shadow fills. Default $0.50 (user-chosen).
DEFAULT_MIN_PRICE = 0.50
# Minimum average daily DOLLAR volume so we never surface untradeable names.
# Default ~$1,000,000/day (user-chosen).
DEFAULT_MIN_DOLLAR_VOL = 1_000_000.0
# Maximum estimated bid/ask spread (as a fraction of mid) so slippage doesn't
# eat the edge / poison shadow fills. Default 0.035 (3.5%, user-chosen 3-4%).
DEFAULT_MAX_SPREAD_PCT = 0.035
# "Under-the-radar" small/micro-cap ceiling. A name with a KNOWN market cap
# above this is a mainstream large-cap and is NOT an under-radar candidate.
# Default $2B (the conventional small-cap line). Env-overridable.
DEFAULT_MAX_MARKET_CAP = 2_000_000_000.0
# When market cap is unknown, fall back to a price ceiling to keep mega-cap
# priced names (AAPL/SPY/...) out of the under-radar lane. Default $50.
DEFAULT_MAX_PRICE = 50.0


def _env_float(name: str, default: float) -> float:
    """Parse a positive float from env; fall back to ``default`` on anything
    missing/garbage/non-finite. Never raises."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return default
    if v != v or v in (float("inf"), float("-inf")) or v < 0:
        return default
    return v


def min_price() -> float:
    return _env_float("RADAR_MIN_PRICE", DEFAULT_MIN_PRICE)


def min_dollar_vol() -> float:
    return _env_float("RADAR_MIN_DOLLAR_VOL", DEFAULT_MIN_DOLLAR_VOL)


def max_spread_pct() -> float:
    return _env_float("RADAR_MAX_SPREAD_PCT", DEFAULT_MAX_SPREAD_PCT)


def max_market_cap() -> float:
    return _env_float("RADAR_MAX_MARKET_CAP", DEFAULT_MAX_MARKET_CAP)


def max_price() -> float:
    return _env_float("RADAR_MAX_PRICE", DEFAULT_MAX_PRICE)


# ---------------------------------------------------------------------------
# Catalyst classifier
# ---------------------------------------------------------------------------

# Catalyst type labels. ``none`` is the fail-safe default — absence of any
# recognised catalyst language, never a guess.
CATALYST_TYPES = ("fda", "m&a", "contract", "earnings", "analyst", "none")

# Each catalyst type maps to a list of compiled keyword patterns and a base
# score. The score reflects how directly the language implies a discrete, dated,
# price-moving event: a confirmed FDA approval / completed acquisition is the
# strongest; an analyst note the weakest. Ordering matters — the first type with
# a hit wins, so the highest-conviction catalysts are checked first.
_CATALYST_RULES: list[tuple[str, float, tuple[str, ...]]] = [
    (
        "fda",
        0.90,
        (
            r"\bfda\b",
            r"\bpdufa\b",
            r"\bphase\s*(?:1|2|3|i|ii|iii)\b",
            r"\bclinical\s+(?:trial|readout|data|results?)\b",
            r"\btopline\s+(?:data|results?)\b",
            r"\bbreakthrough\s+therapy\b",
            r"\borphan\s+drug\b",
            r"\b510\(k\)\b",
            r"\bnda\b",
            r"\bbla\b",
            r"\beua\b",
            r"\bapprov(?:al|es|ed)\b.*\b(?:drug|therapy|treatment|indication)\b",
        ),
    ),
    (
        "m&a",
        0.85,
        (
            r"\bacqui(?:re|res|red|sition)\b",
            r"\bmerg(?:e|er|ers|ing)\b",
            r"\bbuyout\b",
            r"\btakeover\b",
            r"\bto\s+be\s+acquired\b",
            r"\bdefinitive\s+agreement\b",
            r"\btender\s+offer\b",
            r"\bgo[\s-]?private\b",
            r"\bstrategic\s+alternatives\b",
        ),
    ),
    (
        "contract",
        0.70,
        (
            r"\bawarded\b",
            r"\bcontract\s+(?:win|award|worth)\b",
            r"\bwins?\s+(?:contract|deal|order)\b",
            r"\bpartnership\b",
            r"\bcollaboration\s+agreement\b",
            r"\bsupply\s+agreement\b",
            r"\bgovernment\s+contract\b",
            r"\bdefense\s+contract\b",
            r"\b(?:secures?|lands?)\s+(?:deal|order|contract)\b",
        ),
    ),
    (
        "earnings",
        0.65,
        (
            r"\bearnings\s+(?:beat|surprise|miss)\b",
            r"\bbeats?\s+(?:estimates|expectations|eps|revenue)\b",
            r"\braises?\s+(?:guidance|outlook|forecast)\b",
            r"\brecord\s+(?:revenue|quarter|profit)\b",
            r"\btops?\s+(?:estimates|expectations|forecasts?)\b",
            r"\bquarterly\s+results\b",
            r"\bpre[\s-]?announce",
        ),
    ),
    (
        "analyst",
        0.50,
        (
            r"\bupgrade[sd]?\b",
            r"\bdowngrade[sd]?\b",
            r"\binitiate[sd]?\s+coverage\b",
            r"\bprice\s+target\b",
            r"\boutperform\b",
            r"\boverweight\b",
            r"\bbuy\s+rating\b",
            r"\breiterate[sd]?\b",
        ),
    ),
]

# Pre-compile every pattern once at import for speed and determinism.
_COMPILED_RULES: list[tuple[str, float, list[re.Pattern[str]]]] = [
    (kind, base, [re.compile(p, re.IGNORECASE) for p in pats])
    for kind, base, pats in _CATALYST_RULES
]


@dataclass(frozen=True)
class CatalystSignal:
    """The classifier's verdict for one piece of text.

    ``catalyst_type`` is one of :data:`CATALYST_TYPES`; ``catalyst_score`` is in
    ``[0, 1]`` (0.0 when ``type == "none"``); ``catalyst_detail`` is a short,
    human-readable phrase pulled from the matched language (never fabricated —
    empty when there is no match).
    """

    catalyst_type: str = "none"
    catalyst_score: float = 0.0
    catalyst_detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "catalyst_type": self.catalyst_type,
            "catalyst_score": self.catalyst_score,
            "catalyst_detail": self.catalyst_detail,
        }


# Dated / confirmed language nudges the score up: a catalyst with a concrete
# date or a "confirmed/approved" verb is more actionable than vague speculation.
_DATED_BONUS = re.compile(
    r"\b(?:approv(?:ed|al)|completed|confirmed|granted|signed|"
    r"\d{1,2}/\d{1,2}|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b",
    re.IGNORECASE,
)
# Speculative hedging language pulls the score down — a rumour is not a catalyst
# we should size into.
_SPECULATIVE_PENALTY = re.compile(
    r"\b(?:rumou?r|speculat|could|may|might|reportedly|potential|explor(?:e|ing)|"
    r"considering|in\s+talks)\b",
    re.IGNORECASE,
)


def classify_catalyst(text: str | None) -> CatalystSignal:
    """Classify the catalyst (if any) implied by ``text``.

    Pure and deterministic. Returns ``CatalystSignal(type="none", score=0.0)``
    when the text is empty or contains no recognised catalyst language — we
    NEVER fabricate a catalyst. Speculative/hedged language is recognised but
    scored lower (a rumour is not a dated event); confirmed/dated language is
    scored higher.
    """
    if not text or not isinstance(text, str):
        return CatalystSignal()
    blob = text.strip()
    if not blob:
        return CatalystSignal()

    for kind, base, patterns in _COMPILED_RULES:
        for pat in patterns:
            m = pat.search(blob)
            if not m:
                continue
            score = base
            if _DATED_BONUS.search(blob):
                score = min(1.0, score + 0.08)
            if _SPECULATIVE_PENALTY.search(blob):
                # Halve the *edge above the floor* so a pure rumour can't earn
                # full conviction, but a recognised type still scores > none.
                score = 0.30 + (score - 0.30) * 0.5
            detail = _extract_detail(blob, m)
            return CatalystSignal(
                catalyst_type=kind,
                catalyst_score=round(max(0.0, min(1.0, score)), 4),
                catalyst_detail=detail,
            )
    return CatalystSignal()


def _extract_detail(text: str, match: re.Match[str]) -> str:
    """A short, human-readable snippet around the matched catalyst keyword.

    Pulls the sentence/clause containing the match so the operator can see WHY
    the name was flagged. Never fabricates — it only quotes the source text.
    """
    start = match.start()
    # Grab up to ~80 chars of context centred-ish on the match.
    lo = max(0, start - 30)
    hi = min(len(text), match.end() + 50)
    snippet = text[lo:hi].strip()
    return re.sub(r"\s+", " ", snippet)[:120]


# ---------------------------------------------------------------------------
# Liquidity / spread / price gate (MANDATORY, fail-safe)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GateResult:
    """Outcome of the tradability/liquidity gate. ``passed`` is the only thing
    callers must honour; ``reason`` explains an exclusion (for logs/UI)."""

    passed: bool
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "reason": self.reason}


def _pos_float(v: Any) -> float | None:
    """Best-effort positive float, else ``None`` (missing / garbage / <= 0)."""
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f or f in (float("inf"), float("-inf")) or f <= 0:
        return None
    return f


def liquidity_gate(
    *,
    price: Any,
    avg_dollar_volume: Any,
    spread_pct: Any,
    min_price_floor: float | None = None,
    min_dollar_volume: float | None = None,
    max_spread: float | None = None,
) -> GateResult:
    """The MANDATORY tradability gate. EXCLUDES (``passed=False``) a name that:

      * has a price below the floor (``RADAR_MIN_PRICE``, default $0.50),
      * has average daily dollar volume below ``RADAR_MIN_DOLLAR_VOL``
        (default $1,000,000), or
      * has an estimated bid/ask spread above ``RADAR_MAX_SPREAD_PCT``
        (default 3.5%).

    **Fail-safe:** if ANY of ``price`` / ``avg_dollar_volume`` / ``spread_pct``
    is missing or unparseable, the name is EXCLUDED — we never assume a name is
    tradable. ``spread_pct`` is a fraction of mid (0.02 == 2%); a zero spread is
    allowed (it is not "missing"). This gate is not removable and protects the
    integrity of the shadow measurement.
    """
    floor = min_price_floor if min_price_floor is not None else min_price()
    min_dv = min_dollar_volume if min_dollar_volume is not None else min_dollar_vol()
    max_sp = max_spread if max_spread is not None else max_spread_pct()

    px = _pos_float(price)
    if px is None:
        return GateResult(False, "price missing/invalid -> excluded (fail-safe)")
    dv = _pos_float(avg_dollar_volume)
    if dv is None:
        return GateResult(
            False, "avg dollar volume missing/invalid -> excluded (fail-safe)"
        )
    # Spread may legitimately be 0.0, so we accept >= 0 here (not _pos_float).
    if spread_pct is None or isinstance(spread_pct, bool):
        return GateResult(False, "spread missing -> excluded (fail-safe)")
    try:
        sp = float(spread_pct)
    except (TypeError, ValueError):
        return GateResult(False, "spread unparseable -> excluded (fail-safe)")
    if sp != sp or sp < 0 or sp in (float("inf"), float("-inf")):
        return GateResult(False, "spread invalid -> excluded (fail-safe)")

    if px < floor:
        return GateResult(False, f"price ${px:.4f} below floor ${floor:.2f}")
    if dv < min_dv:
        return GateResult(
            False, f"avg $vol ${dv:,.0f} below min ${min_dv:,.0f}"
        )
    if sp > max_sp:
        return GateResult(
            False, f"spread {sp*100:.2f}% above max {max_sp*100:.2f}%"
        )
    return GateResult(True, "")


def is_under_radar(
    *,
    market_cap: Any = None,
    price: Any = None,
    cap_ceiling: float | None = None,
    price_ceiling: float | None = None,
) -> bool:
    """True when a name looks small/micro-cap (under-followed), not mega-cap.

    Uses the KNOWN market cap against ``RADAR_MAX_MARKET_CAP`` (default $2B) when
    available; otherwise falls back to a price ceiling (``RADAR_MAX_PRICE``,
    default $50) so mega-cap priced names stay out of the lane. With neither
    signal present we return ``False`` — this is a *membership* test, not the
    safety gate, so when in doubt we keep a name OUT of the under-radar lane
    rather than mislabel a mega-cap.
    """
    cap_max = cap_ceiling if cap_ceiling is not None else max_market_cap()
    px_max = price_ceiling if price_ceiling is not None else max_price()
    mc = _pos_float(market_cap)
    if mc is not None:
        return mc <= cap_max
    px = _pos_float(price)
    if px is not None:
        return px <= px_max
    return False


# ---------------------------------------------------------------------------
# Conviction sizing — high catalyst_score earns full RELATIVE weight, but the
# result is STRICTLY clamped by the ABSOLUTE caps. These caps are NON-NEGOTIABLE
# and conviction can NEVER bypass them.
# ---------------------------------------------------------------------------

# Absolute, non-negotiable dollar caps mirrored from the trading guardrails.
# These are duplicated here as named constants so the clamp is self-documenting;
# callers SHOULD pass the live values, but the defaults are the spec's hard caps.
#   * ABSOLUTE_PER_TRADE_USD: $50  — never size a single trade above this.
#   * ABSOLUTE_BUDGET_USD:    $300 — default total budget.
#   * ABSOLUTE_CEILING_USD:   $10,000 — the float ceiling that is NEVER raised.
ABSOLUTE_PER_TRADE_USD = 50.0
ABSOLUTE_BUDGET_USD = 300.0
ABSOLUTE_CEILING_USD = 10_000.0

# Floor fraction of the per-trade cap that even a zero-score (but already
# catalyst-gated, so a real catalyst exists) name receives. Conviction scales
# the size from this floor up to the full per-trade cap at score 1.0.
DEFAULT_CONVICTION_FLOOR = 0.40


def conviction_notional(
    catalyst_score: float,
    *,
    per_trade_cap: float = ABSOLUTE_PER_TRADE_USD,
    budget_remaining: float = ABSOLUTE_BUDGET_USD,
    ceiling: float = ABSOLUTE_CEILING_USD,
    conviction_floor: float = DEFAULT_CONVICTION_FLOOR,
) -> float:
    """Dollar notional for a catalyst pick, scaled by conviction but HARD-CLAMPED.

    A high ``catalyst_score`` (in ``[0, 1]``) earns full RELATIVE conviction
    weight: the size scales linearly from ``conviction_floor * per_trade_cap`` at
    score 0 up to the full ``per_trade_cap`` at score 1. The result is then
    STRICTLY clamped by EVERY absolute cap::

        notional = min(notional, per_trade_cap, budget_remaining, ceiling)

    so it can NEVER exceed $50/trade, the remaining $300 budget, or the $10,000
    ceiling — regardless of how high the score is (scores above 1.0 or below 0.0
    are clamped first). Conviction only redistributes size WITHIN the caps; it
    never bypasses them. Returns a non-negative dollar amount (0.0 when any cap
    is non-positive, e.g. the budget is fully allocated).
    """
    # Clamp inputs into their valid ranges first so a garbage score can't
    # produce a size outside the caps.
    try:
        score = float(catalyst_score)
    except (TypeError, ValueError):
        score = 0.0
    if score != score:  # NaN
        score = 0.0
    score = max(0.0, min(1.0, score))

    # The absolute caps. A non-finite / negative cap collapses to 0 (safe).
    caps = []
    for cap in (per_trade_cap, budget_remaining, ceiling):
        c = _pos_float(cap)
        caps.append(c if c is not None else 0.0)
    hard_cap = min(caps) if caps else 0.0
    if hard_cap <= 0:
        return 0.0

    floor = max(0.0, min(1.0, conviction_floor))
    # Relative weight in [floor, 1.0]; full conviction at score 1.0.
    weight = floor + (1.0 - floor) * score
    # Conviction scales WITHIN the per-trade cap...
    notional = weight * _pos_float(per_trade_cap) if _pos_float(per_trade_cap) else 0.0
    # ...then is hard-clamped by EVERY absolute cap. min() guarantees no cap is
    # ever exceeded even if per_trade_cap > budget_remaining or > ceiling.
    notional = min(notional, hard_cap)
    return round(max(0.0, notional), 4)


__all__ = [
    "ABSOLUTE_BUDGET_USD",
    "ABSOLUTE_CEILING_USD",
    "ABSOLUTE_PER_TRADE_USD",
    "CATALYST_TYPES",
    "CatalystSignal",
    "GateResult",
    "classify_catalyst",
    "conviction_notional",
    "is_under_radar",
    "liquidity_gate",
    "max_market_cap",
    "max_price",
    "max_spread_pct",
    "min_dollar_vol",
    "min_price",
]
