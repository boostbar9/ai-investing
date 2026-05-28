"""Enriched Reddit post + trust schema.

The existing :class:`packages.data.adapters.base.NewsItem` is great for
sentiment aggregation but lossy for credibility -- once a post is
flattened into a headline + ticker + timestamp, you can't tell a
year-old account with 800k karma from a day-old burner.

We keep the new schema *separate* from ``NewsItem`` rather than
extending it so Phase 3's risk surface stays small.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass(frozen=True)
class RedditPost:
    """A single hot post with every trust-relevant field intact.

    All numeric fields are optional because Reddit occasionally returns
    nulls (deleted authors, restricted subs). The scorer must handle
    missing values gracefully.
    """

    # Identity
    id: str
    permalink: str
    subreddit: str
    title: str
    selftext: str

    # Author (the credibility-relevant bits)
    author: str | None
    author_created_utc: float | None  # epoch seconds
    author_karma: int | None  # link + comment karma summed when available

    # Engagement
    score: int  # net upvotes
    num_comments: int
    upvote_ratio: float | None  # 0..1
    created_utc: float  # epoch seconds

    # Derived (filled by the fetcher)
    tickers: tuple[str, ...] = ()

    def age_hours(self, *, now: float | None = None) -> float:
        """How old the post is, in hours, clamped at 0."""
        ref = now if now is not None else datetime.now(UTC).timestamp()
        return max(0.0, (ref - self.created_utc) / 3600.0)

    def account_age_days(self, *, now: float | None = None) -> float | None:
        """Author account age in days, or None if Reddit didn't tell us."""
        if self.author_created_utc is None:
            return None
        ref = now if now is not None else datetime.now(UTC).timestamp()
        return max(0.0, (ref - self.author_created_utc) / 86400.0)


@dataclass
class PostTrust:
    """The scorer's verdict on a single post.

    ``weight`` is the *only* number downstream consumers need to know
    about -- multiply per-post sentiment by ``weight`` and you've got
    a trust-weighted signal. The rest of the fields exist so the
    cockpit can explain itself ("we ignored this post because the
    author is 3 days old").
    """

    post_id: str
    weight: float  # [0, 1] -- multiply raw sentiment by this
    karma_component: float  # [0, 1]
    age_component: float  # [0, 1] -- author account age
    history_component: float  # [0, 1] -- prior signal accuracy
    engagement_component: float  # [0, 1] -- upvotes/comments balance
    pump_flags: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "post_id": self.post_id,
            "weight": round(self.weight, 4),
            "karma_component": round(self.karma_component, 4),
            "age_component": round(self.age_component, 4),
            "history_component": round(self.history_component, 4),
            "engagement_component": round(self.engagement_component, 4),
            "pump_flags": list(self.pump_flags),
            "reasons": list(self.reasons),
        }
