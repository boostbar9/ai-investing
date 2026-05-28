"""Trust scorer for Reddit posts.

Given a :class:`RedditPost` (and optional history), produce a weight in
``[0, 1]`` that downstream consumers multiply into the raw sentiment
signal. The score is decomposed into four components so the cockpit can
explain "why" -- never a single opaque number.

  * **karma_component**       -- log-scaled total karma; minimum gates
    out fresh accounts no matter how loud they are.
  * **age_component**          -- account age in days; a 3-day-old
    burner gets 0 even with 10k karma (almost certainly farmed).
  * **engagement_component**   -- balanced upvotes + comments. Low
    upvote ratio with high comment count = controversial (lower trust).
  * **history_component**      -- the author's prior signal accuracy
    from :class:`TrustHistory`. Defaults to 0.5 (neutral) when unknown.

We also surface a list of **pump flags** -- pattern matches that have
historically preceded coordinated pumps. A post can have a high overall
weight and still get flagged; flags don't *force* a weight reduction
(some legitimate posts trigger them) but the corroboration gate
upstream uses them as a tiebreaker.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from packages.agents.reddit_trust.history import TrustHistory
    from packages.agents.reddit_trust.schema import RedditPost

from packages.agents.reddit_trust.schema import PostTrust

# ---------------------------------------------------------------------------
# Tunables -- exposed as module constants so they can be swept in
# Phase 5's pretraining. NEVER hardcode these inside functions.
# ---------------------------------------------------------------------------

# Karma at which the karma component saturates to ~0.95. log-scale.
KARMA_SATURATION = 50_000
# Account age (days) at which age component saturates.
AGE_SATURATION_DAYS = 365
# Minimum age before the component starts crediting at all.
AGE_MIN_DAYS = 7
# Component weights -- must sum to 1.0.
W_KARMA = 0.25
W_AGE = 0.25
W_ENGAGEMENT = 0.20
W_HISTORY = 0.30
# Floor on history_component when we have no labeled data on the author.
HISTORY_UNKNOWN_NEUTRAL = 0.5
# Pump-detection knobs
PUMP_EXCLAMATION_THRESHOLD = 3  # "BUY NOW!!! TO THE MOON!!!"
PUMP_TICKER_DENSITY_THRESHOLD = 3  # 3+ distinct tickers in one post
# Min net score (upvotes - downvotes) before an account is treated as
# engaged. Posts below this get engagement_component = 0.1.
ENGAGEMENT_MIN_SCORE = 5


_PUMP_PHRASES = re.compile(
    r"\b(to the moon|going parabolic|yolo all in|squeeze incoming|"
    r"buy now|last chance|do not miss|guaranteed|sure thing|"
    r"pump|moonshot|10\s*x|100\s*x|1000\s*x)\b",
    re.IGNORECASE,
)

_ALL_CAPS_RUN = re.compile(r"\b[A-Z]{4,}\b")


@dataclass
class PumpFlag:
    """One detected pump-style pattern with a short explanation."""

    code: str
    detail: str


@dataclass
class TrustBreakdown:
    """Full per-component score so the UI can explain itself."""

    karma: float
    age: float
    engagement: float
    history: float
    weight: float
    pump_flags: list[PumpFlag] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Component scorers (pure functions for easy unit testing)
# ---------------------------------------------------------------------------


def score_karma(karma: int | None) -> float:
    """Log-scaled karma score in [0, 1]. ``None`` -> 0.1 (low, not zero
    -- account exists, we just couldn't fetch /about.json)."""
    if karma is None:
        return 0.1
    if karma <= 0:
        return 0.0
    # log10 saturation: 0 karma -> 0, KARMA_SATURATION -> ~0.95.
    val = math.log10(1 + karma) / math.log10(1 + KARMA_SATURATION)
    return max(0.0, min(1.0, val))


def score_age(account_age_days: float | None) -> float:
    """Linear-ramp account age score.

    * ``None``                       -> 0.1 (unknown, treat as suspect)
    * ``< AGE_MIN_DAYS``             -> 0.0 (almost certainly farmed)
    * ``[AGE_MIN_DAYS, SATURATION]`` -> linear ramp
    * ``> SATURATION``               -> 1.0
    """
    if account_age_days is None:
        return 0.1
    if account_age_days < AGE_MIN_DAYS:
        return 0.0
    if account_age_days >= AGE_SATURATION_DAYS:
        return 1.0
    span = AGE_SATURATION_DAYS - AGE_MIN_DAYS
    return (account_age_days - AGE_MIN_DAYS) / span


def score_engagement(
    *, score: int, num_comments: int, upvote_ratio: float | None
) -> float:
    """Composite engagement score.

    * Low net score = low engagement (nobody read it).
    * High score + low upvote-ratio = controversial (lower trust).
    * High score + high ratio = well-received (boost).
    """
    if score < ENGAGEMENT_MIN_SCORE:
        return 0.1
    # Score component: log-saturated, hits 0.95 at ~10k upvotes.
    score_part = math.log10(1 + score) / math.log10(1 + 10_000)
    score_part = max(0.0, min(1.0, score_part))
    # Comment-to-score ratio: very high comments relative to upvotes
    # indicates an argument fest, not consensus signal.
    if score > 0:
        c_to_s = num_comments / score
        if c_to_s > 2.0:
            score_part *= 0.7  # noisy thread
    # Upvote ratio: anything below 0.7 is controversial-grade.
    if upvote_ratio is not None:
        if upvote_ratio < 0.55:
            score_part *= 0.5
        elif upvote_ratio < 0.7:
            score_part *= 0.8
        else:
            # Reward strong consensus (≥0.9 ratio).
            if upvote_ratio >= 0.9:
                score_part = min(1.0, score_part * 1.1)
    return max(0.0, min(1.0, score_part))


def score_history(
    history: TrustHistory | None, author: str | None
) -> tuple[float, int]:
    """Look up the author's prior accuracy.

    Returns ``(component, sample_size)``. When we have no labeled
    history (or no ``TrustHistory`` was passed) we return the neutral
    floor :data:`HISTORY_UNKNOWN_NEUTRAL` -- punishing unknowns would
    bake in a forever cold-start problem.
    """
    if history is None or not author:
        return (HISTORY_UNKNOWN_NEUTRAL, 0)
    acc, n = history.author_accuracy(author)
    if acc is None or n == 0:
        return (HISTORY_UNKNOWN_NEUTRAL, 0)
    # Bayesian-ish shrink toward the neutral prior when sample is small.
    # k = pseudo-count; with k=10, an author with 1/1 isn't crowned
    # 100% accurate.
    k = 10.0
    shrunk = (acc * n + HISTORY_UNKNOWN_NEUTRAL * k) / (n + k)
    return (max(0.0, min(1.0, shrunk)), n)


# ---------------------------------------------------------------------------
# Pump detection -- structural patterns, not vibes.
# ---------------------------------------------------------------------------


def detect_pump_flags(post: RedditPost) -> list[PumpFlag]:
    """Return zero or more pump-style patterns spotted in ``post``."""
    flags: list[PumpFlag] = []
    text = f"{post.title}\n{post.selftext}"

    # 1. Catchphrase match.
    if _PUMP_PHRASES.search(text):
        m = _PUMP_PHRASES.search(text)
        flags.append(
            PumpFlag("pump_phrase", f"matched pump phrase '{m.group(0)}'")
        )

    # 2. Excessive exclamation.
    exclaim = text.count("!")
    if exclaim >= PUMP_EXCLAMATION_THRESHOLD:
        flags.append(
            PumpFlag(
                "exclamation_spam",
                f"{exclaim} '!' marks (threshold {PUMP_EXCLAMATION_THRESHOLD})",
            )
        )

    # 3. Many distinct tickers crammed into one post = spray-and-pray.
    if len(post.tickers) >= PUMP_TICKER_DENSITY_THRESHOLD:
        flags.append(
            PumpFlag(
                "ticker_density",
                f"{len(post.tickers)} tickers in one post",
            )
        )

    # 4. All-caps runs in the title.
    caps = _ALL_CAPS_RUN.findall(post.title)
    # Exclude the tickers themselves (they're legitimately caps).
    caps_non_ticker = [c for c in caps if c not in post.tickers]
    if len(caps_non_ticker) >= 2:
        flags.append(
            PumpFlag(
                "all_caps", f"{len(caps_non_ticker)} all-caps words in title"
            )
        )

    # 5. Young account + strong directional claim = classic burner pump.
    age = post.account_age_days()
    if age is not None and age < AGE_MIN_DAYS and (
        post.score >= 50 or post.num_comments >= 25
    ):
        flags.append(
            PumpFlag(
                "young_account_high_engagement",
                f"author {age:.1f}d old with {post.score} upvotes",
            )
        )

    return flags


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class RedditTrustScorer:
    """Combines the component scorers into a single weight.

    The class is stateless except for the (optional) history reference,
    so a single instance can be shared across the whole sweep.
    """

    def __init__(self, history: TrustHistory | None = None) -> None:
        self._history = history
        # Belt-and-braces: ensure component weights sum to 1.0.
        total = W_KARMA + W_AGE + W_ENGAGEMENT + W_HISTORY
        if not math.isclose(total, 1.0, abs_tol=1e-6):  # pragma: no cover
            raise ValueError(
                f"Component weights must sum to 1.0 (got {total})"
            )

    def score(self, post: RedditPost) -> TrustBreakdown:
        karma_c = score_karma(post.author_karma)
        age_c = score_age(post.account_age_days())
        eng_c = score_engagement(
            score=post.score,
            num_comments=post.num_comments,
            upvote_ratio=post.upvote_ratio,
        )
        hist_c, hist_n = score_history(self._history, post.author)
        pump_flags = detect_pump_flags(post)

        weight = (
            W_KARMA * karma_c
            + W_AGE * age_c
            + W_ENGAGEMENT * eng_c
            + W_HISTORY * hist_c
        )

        # Pump-flag penalty: each flag shaves 15%, capped at 60% total.
        # We do NOT zero it out -- some legit DD posts use exclamation.
        penalty = min(0.60, 0.15 * len(pump_flags))
        weight = max(0.0, weight * (1.0 - penalty))

        reasons: list[str] = []
        if post.author is None:
            reasons.append("deleted/automod author -- karma & age unknown")
        if hist_n == 0 and post.author:
            reasons.append("no labeled history for this author yet")
        if pump_flags:
            reasons.append(
                f"pump heuristics fired: {[f.code for f in pump_flags]} "
                f"(penalty {int(penalty * 100)}%)"
            )
        return TrustBreakdown(
            karma=karma_c,
            age=age_c,
            engagement=eng_c,
            history=hist_c,
            weight=weight,
            pump_flags=pump_flags,
            reasons=reasons,
        )

    def score_to_post_trust(self, post: RedditPost) -> PostTrust:
        """Compatibility wrapper -- returns the schema-side ``PostTrust``
        dataclass for serialization."""
        br = self.score(post)
        return PostTrust(
            post_id=post.id,
            weight=br.weight,
            karma_component=br.karma,
            age_component=br.age,
            history_component=br.history,
            engagement_component=br.engagement,
            pump_flags=[f.code for f in br.pump_flags],
            reasons=br.reasons,
        )
