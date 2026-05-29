"""Reddit trust-scoring + news-corroboration gate (Phase 3).

The existing :mod:`packages.data.adapters.sentiment` module is fast and
free, but it deliberately throws away every trust-relevant Reddit field
(``author``, ``score``, ``num_comments``, ``upvote_ratio``,
``author_created_utc``) because it was designed as a *signal*, not as
an arbiter of credibility.

This subpackage layers on top:

  * :mod:`schema`         -- enriched ``RedditPost`` + ``PostTrust`` shapes
  * :mod:`fetcher`        -- Reddit JSON pull that *keeps* the trust fields
  * :mod:`scorer`         -- karma / account-age / signal-accuracy / pump
                             heuristics -> weight in [0, 1]
  * :mod:`history`        -- persistent author scoring history (JSONL)
  * :mod:`corroboration`  -- gate that requires news headlines to back
                             any Reddit-only candidate above a floor

It is intentionally additive. No existing caller of ``SentimentAdapter``
needs to change for Phase 3 to ship.
"""

from packages.agents.reddit_trust.corroboration import (
    CorroborationResult,
    NewsCorroborationGate,
)
from packages.agents.reddit_trust.fetcher import (
    RICH_REDDIT_URL,
    fetch_rich_reddit,
)
from packages.agents.reddit_trust.history import TrustHistory
from packages.agents.reddit_trust.schema import PostTrust, RedditPost
from packages.agents.reddit_trust.scorer import (
    PumpFlag,
    RedditTrustScorer,
    TrustBreakdown,
)
from packages.agents.reddit_trust.subreddit_quality import (
    DEFAULT_SWEEP_ROSTER,
    SUBREDDIT_QUALITY,
    SubredditQuality,
    fetch_roster,
    quality_for,
)
from packages.agents.reddit_trust.ticker_discovery import (
    candidate_subreddits,
    discover_for_tickers,
)

__all__ = [
    "DEFAULT_SWEEP_ROSTER",
    "RICH_REDDIT_URL",
    "SUBREDDIT_QUALITY",
    "CorroborationResult",
    "NewsCorroborationGate",
    "PostTrust",
    "PumpFlag",
    "RedditPost",
    "RedditTrustScorer",
    "SubredditQuality",
    "TrustBreakdown",
    "TrustHistory",
    "candidate_subreddits",
    "discover_for_tickers",
    "fetch_rich_reddit",
    "fetch_roster",
    "quality_for",
]
