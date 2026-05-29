"""Tiered subreddit roster + per-sub quality weights — Phase 10.

The original sweep treated r/wallstreetbets and r/SecurityAnalysis as
equally credible sources. That's the wrong default: WSB is fun but
pump-prone, SecurityAnalysis is mostly long-form DD with sourced
citations. Phase 10 introduces a per-subreddit quality multiplier that
feeds the trust scorer alongside karma, age, engagement, and history.

The roster is intentionally hand-curated rather than data-driven. We'd
need months of label history to learn these multipliers from data, and
the manual values reflect the consensus of the investing community
(r/SecurityAnalysis and r/Bogleheads are widely regarded as
high-quality; r/wallstreetbets and r/pennystocks are not).

Quality multipliers are bounded to ``[0.4, 1.0]``: a multiplier of 0.5
on a WSB post means the post's trust weight is *halved* before the
corroboration gate sees it, but never zeroed (legit WSB DD does exist).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

# Tier definitions. Each tier maps a multiplier in [0.4, 1.0] applied
# to the trust weight produced by the karma/age/engagement/history
# scorers. The fallback for unknown subs is the GENERAL tier weight.

TIER_HIGH_QUALITY = 1.0     # rigorous DD subs
TIER_GENERAL = 0.85         # mainstream investing communities
TIER_SECTOR = 0.80          # focused but broad audience
TIER_PER_TICKER = 0.90      # ticker-specific subs (often early news)
TIER_WSB = 0.55             # WSB and similar high-noise venues
TIER_PENNY = 0.45           # penny/microcap subs (extreme pump risk)


# Hand-curated roster. Add to this map rather than spreading the
# weights across the codebase.
SUBREDDIT_QUALITY: dict[str, float] = {
    # ---- High quality (DD-heavy, citation culture) -----------------
    "SecurityAnalysis": TIER_HIGH_QUALITY,
    "ValueInvesting": TIER_HIGH_QUALITY,
    "investing": TIER_HIGH_QUALITY,
    "Bogleheads": TIER_HIGH_QUALITY,

    # ---- General investing -----------------------------------------
    "stocks": TIER_GENERAL,
    "StockMarket": TIER_GENERAL,
    "stockmarket": TIER_GENERAL,    # case-insensitive lookup handled below
    "dividends": TIER_GENERAL,
    "ETFs": TIER_GENERAL,
    "personalfinance": TIER_GENERAL,

    # ---- Sector / strategy specific --------------------------------
    "options": TIER_SECTOR,
    "thetagang": TIER_SECTOR,
    "Daytrading": TIER_SECTOR,
    "swingtrading": TIER_SECTOR,
    "CanadianInvestor": TIER_SECTOR,
    "EuropeanInvestor": TIER_SECTOR,
    "algotrading": TIER_SECTOR,

    # ---- Per-ticker subs (high signal, often early) ----------------
    # Most active per-ticker communities. The discovery module probes
    # additional ones dynamically per candidate; entries here just
    # ensure they get the right weight when seen.
    "NVDA_Stock": TIER_PER_TICKER,
    "teslainvestorsclub": TIER_PER_TICKER,
    "TSLA": TIER_PER_TICKER,
    "AMD_Stock": TIER_PER_TICKER,
    "Palantir_Investors": TIER_PER_TICKER,
    "PLTR": TIER_PER_TICKER,
    "AMC_Stock": TIER_PER_TICKER,
    "GME": TIER_PER_TICKER,
    "Superstonk": TIER_WSB,         # GME-related but pump-prone
    "Spacstocks": TIER_SECTOR,
    "Vitards": TIER_SECTOR,
    "RKLB": TIER_PER_TICKER,
    "SOFIInvestors": TIER_PER_TICKER,

    # ---- WSB tier (entertaining, low credibility) ------------------
    "wallstreetbets": TIER_WSB,
    "WallStreetbetsELITE": TIER_WSB,
    "Shortsqueeze": TIER_WSB,

    # ---- Penny / microcap (heavy pump risk) ------------------------
    "pennystocks": TIER_PENNY,
    "Robinhoodpennystocks": TIER_PENNY,
    "RobinHoodPennyStocks": TIER_PENNY,
}


# Tiers used by _gather_rich_reddit to construct the fetch roster.
# Order matters for fetch priority when budget is tight.
DEFAULT_SWEEP_ROSTER: tuple[str, ...] = (
    "SecurityAnalysis",
    "ValueInvesting",
    "investing",
    "stocks",
    "StockMarket",
    "wallstreetbets",
    "options",
    "thetagang",
    "Bogleheads",
    "dividends",
)


@dataclass(frozen=True)
class SubredditQuality:
    """Resolved quality info for one subreddit name."""

    subreddit: str
    multiplier: float
    tier: str


def _classify_tier(multiplier: float) -> str:
    if multiplier >= TIER_HIGH_QUALITY - 1e-6:
        return "high_quality"
    if multiplier >= TIER_PER_TICKER - 1e-6:
        return "per_ticker"
    if multiplier >= TIER_GENERAL - 1e-6:
        return "general"
    if multiplier >= TIER_SECTOR - 1e-6:
        return "sector"
    if multiplier >= TIER_WSB - 1e-6:
        return "wsb"
    return "penny"


def quality_for(subreddit: str) -> SubredditQuality:
    """Resolve quality multiplier for ``subreddit``.

    Lookup is case-insensitive; unknown subs fall back to the GENERAL
    tier (we'd rather under-weight a great new sub than over-weight a
    pump farm we haven't catalogued yet).
    """
    if not subreddit:
        return SubredditQuality("", TIER_GENERAL, _classify_tier(TIER_GENERAL))
    # Direct hit first (preserves declared casing).
    if subreddit in SUBREDDIT_QUALITY:
        m = SUBREDDIT_QUALITY[subreddit]
        return SubredditQuality(subreddit, m, _classify_tier(m))
    # Case-insensitive fallback.
    lowered = subreddit.lower()
    for name, mult in SUBREDDIT_QUALITY.items():
        if name.lower() == lowered:
            return SubredditQuality(subreddit, mult, _classify_tier(mult))
    return SubredditQuality(
        subreddit, TIER_GENERAL, _classify_tier(TIER_GENERAL)
    )


def fetch_roster(extra: Iterable[str] = ()) -> tuple[str, ...]:
    """Return the default sweep roster, optionally extended with per-
    ticker subs discovered for the current candidate set.

    Duplicates are removed while preserving roster order so the
    high-quality subs always get fetched first (we hit them under
    Reddit's anonymous rate-limit before risking 429s on the long tail).
    """
    seen: set[str] = set()
    out: list[str] = []
    for s in (*DEFAULT_SWEEP_ROSTER, *extra):
        key = s.lower()
        if key in seen or not s:
            continue
        seen.add(key)
        out.append(s)
    return tuple(out)
