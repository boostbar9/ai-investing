"""Rich Reddit post fetcher (trust-aware).

This deliberately does NOT reuse :class:`SentimentAdapter.fetch_reddit`
because that method drops the credibility-relevant fields the moment
it flattens to ``NewsItem``. The two paths run in parallel during a
sweep -- the cheap one feeds sentiment aggregation, the rich one feeds
the trust scorer + corroboration gate.

We also pull ``/about.json`` for each unique author so we can score
account age + karma. That doubles the request count, but Reddit's
public JSON endpoint is unauthenticated and we rate-limit it via the
shared ``BUCKETS`` bag.
"""

from __future__ import annotations

import contextlib
import logging
import os
from typing import Any

import httpx

from packages.data.adapters.sentiment import extract_tickers
from packages.shared.otel import span

logger = logging.getLogger(__name__)

# Public Reddit JSON endpoints. Override in tests via env or kwargs.
RICH_REDDIT_URL = os.getenv(
    "RICH_REDDIT_URL", "https://www.reddit.com/r/{subreddit}/hot.json"
)
RICH_REDDIT_USER_URL = os.getenv(
    "RICH_REDDIT_USER_URL", "https://www.reddit.com/user/{username}/about.json"
)

# Honest about who we are, for the Reddit folks reading their logs.
USER_AGENT = os.getenv(
    "RICH_REDDIT_UA",
    "ai-investing/0.3 (+https://github.com/boostbar9/ai-investing)",
)

DEFAULT_TIMEOUT_S = 8.0


def _author_present(name: str | None) -> bool:
    """``[deleted]`` and ``AutoModerator`` are not real authors for
    credibility purposes."""
    if not name:
        return False
    return name not in ("[deleted]", "AutoModerator")


async def _fetch_author_meta(
    client: httpx.AsyncClient, username: str
) -> tuple[float | None, int | None]:
    """Return ``(created_utc, total_karma)`` or ``(None, None)`` on any
    failure. We never raise -- a missing author profile just downgrades
    that post's trust score, it doesn't break the sweep."""
    url = RICH_REDDIT_USER_URL.format(username=username)
    try:
        r = await client.get(
            url,
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
            timeout=DEFAULT_TIMEOUT_S,
        )
    except Exception as exc:  # pragma: no cover - network varies
        logger.debug("author meta fetch failed for %s: %s", username, exc)
        return (None, None)
    if r.status_code != 200:
        return (None, None)
    try:
        data = (r.json().get("data") or {})
    except (ValueError, AttributeError):
        return (None, None)
    created = data.get("created_utc")
    # Reddit returns ``total_karma`` (modern) and/or ``link_karma`` +
    # ``comment_karma`` (legacy). Sum them if needed, capped at int.
    karma = data.get("total_karma")
    if karma is None:
        lk = data.get("link_karma")
        ck = data.get("comment_karma")
        if lk is not None or ck is not None:
            karma = int(lk or 0) + int(ck or 0)
    try:
        created_f = float(created) if created is not None else None
    except (TypeError, ValueError):
        created_f = None
    try:
        karma_i = int(karma) if karma is not None else None
    except (TypeError, ValueError):
        karma_i = None
    return (created_f, karma_i)


async def fetch_rich_reddit(
    subreddit: str,
    *,
    limit: int = 25,
    client: httpx.AsyncClient | None = None,
    include_author_meta: bool = True,
) -> list[dict[str, Any]]:
    """Pull ``limit`` hot posts from ``r/<subreddit>`` with full author
    metadata. Returns plain dicts (the schema layer wraps them into
    :class:`RedditPost` objects). Never raises: returns ``[]`` on any
    transport failure so the sweep stays best-effort.

    Each dict has the keys: ``id``, ``permalink``, ``subreddit``,
    ``title``, ``selftext``, ``author``, ``author_created_utc``,
    ``author_karma``, ``score``, ``num_comments``, ``upvote_ratio``,
    ``created_utc``, ``tickers``.
    """
    owned = client is None
    if owned:
        client = httpx.AsyncClient(timeout=DEFAULT_TIMEOUT_S)
    try:
        with span("reddit_trust.fetch", {"subreddit": subreddit}):
            url = RICH_REDDIT_URL.format(subreddit=subreddit)
            try:
                r = await client.get(
                    url,
                    params={"limit": limit},
                    headers={
                        "User-Agent": USER_AGENT,
                        "Accept": "application/json",
                    },
                )
            except Exception as exc:
                logger.warning(
                    "rich reddit fetch failed for r/%s: %s",
                    subreddit,
                    exc.__class__.__name__,
                )
                return []
            if r.status_code != 200:
                logger.warning(
                    "rich reddit r/%s: HTTP %s", subreddit, r.status_code
                )
                return []
            try:
                payload = r.json()
            except ValueError:
                return []
            children = (payload.get("data") or {}).get("children") or []

            # First pass: build records with the post-level fields. Defer
            # author-meta calls so we can dedupe by username.
            records: list[dict[str, Any]] = []
            unique_authors: set[str] = set()
            for ch in children:
                d = ch.get("data") or {}
                title = d.get("title") or ""
                if not title:
                    continue
                author = d.get("author")
                if _author_present(author):
                    unique_authors.add(author)
                rec = {
                    "id": d.get("id") or "",
                    "permalink": (
                        f"https://www.reddit.com{d.get('permalink', '')}"
                    ),
                    "subreddit": subreddit,
                    "title": title,
                    "selftext": d.get("selftext") or "",
                    "author": author if _author_present(author) else None,
                    "author_created_utc": None,
                    "author_karma": None,
                    "score": int(d.get("score") or 0),
                    "num_comments": int(d.get("num_comments") or 0),
                    "upvote_ratio": (
                        float(d["upvote_ratio"])
                        if d.get("upvote_ratio") is not None
                        else None
                    ),
                    "created_utc": float(d.get("created_utc") or 0.0),
                    "tickers": tuple(
                        extract_tickers(
                            title + " " + (d.get("selftext") or "")
                        )
                    ),
                }
                records.append(rec)

            # Second pass: enrich with author meta. Single-flight per author.
            if include_author_meta and unique_authors:
                meta: dict[str, tuple[float | None, int | None]] = {}
                for username in unique_authors:
                    meta[username] = await _fetch_author_meta(client, username)
                for rec in records:
                    a = rec["author"]
                    if a and a in meta:
                        created, karma = meta[a]
                        rec["author_created_utc"] = created
                        rec["author_karma"] = karma

            return records
    finally:
        if owned:
            with contextlib.suppress(Exception):
                await client.aclose()
