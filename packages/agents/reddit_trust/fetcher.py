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
import re
import xml.etree.ElementTree as ET
from typing import Any

import httpx

from packages.agents.reddit_trust.oauth import get_oauth
from packages.data.adapters.sentiment import extract_tickers
from packages.shared.otel import span

logger = logging.getLogger(__name__)

# Public Reddit JSON endpoints. Override in tests via env or kwargs.
#
# The primary host (www.reddit.com) routinely 403s anonymous JSON requests
# from datacenter / residential IPs that Reddit's anti-bot fingerprint has
# flagged. The fallbacks below cycle through alternate hosts (old.reddit.com
# is run on a separate infra slice that's almost never blocked) and finally
# to the RSS endpoint, which has a different rate-limit pool entirely.
# We try them IN ORDER until one returns 200 -- the corroboration gate just
# needs *some* community signal, not necessarily the richest one.
# Phase 25.5 — when ``REDDIT_CLIENT_ID`` is set, we hit oauth.reddit.com
# with a bearer token instead of the public hosts (which 403 from
# datacenter IPs). oauth.reddit.com is the *only* host that gives the
# 100/min/per-OAuth-app rate limit and never 403s on a valid token.
RICH_REDDIT_OAUTH_URL = os.getenv(
    "RICH_REDDIT_OAUTH_URL", "https://oauth.reddit.com/r/{subreddit}/hot"
)
RICH_REDDIT_URL = os.getenv(
    "RICH_REDDIT_URL", "https://www.reddit.com/r/{subreddit}/hot.json"
)
RICH_REDDIT_FALLBACKS = [
    # old.reddit.com serves the same JSON shape from a different infra path.
    "https://old.reddit.com/r/{subreddit}/hot.json",
    # api.reddit.com is the unauthenticated public API endpoint; same shape.
    "https://api.reddit.com/r/{subreddit}/hot",
]
RICH_REDDIT_RSS_URL = os.getenv(
    "RICH_REDDIT_RSS_URL", "https://www.reddit.com/r/{subreddit}/hot.rss"
)
RICH_REDDIT_USER_URL = os.getenv(
    "RICH_REDDIT_USER_URL", "https://www.reddit.com/user/{username}/about.json"
)

# Honest about who we are, for the Reddit folks reading their logs.
# Note: Reddit's anti-bot prefers UA strings that look like real browsers OR
# very-explicit bot UAs with contact info -- the explicit form below is what
# the Reddit API docs themselves recommend for unauthenticated polling.
USER_AGENT = os.getenv(
    "RICH_REDDIT_UA",
    "ai-investing/0.4 by /u/boostbar9 (+https://github.com/boostbar9/ai-investing)",
)

DEFAULT_TIMEOUT_S = 8.0


# RSS <item> shape: title + link + pubDate + author (in dc:creator). Reddit
# does not expose karma/upvote_ratio over RSS, so the trust scorer downgrades
# RSS-derived posts gracefully (missing fields -> neutral score). That's the
# right trade-off: a partial signal beats a complete blackout.
_RSS_CREATOR_RE = re.compile(r"/u/([A-Za-z0-9_\-]+)")
_DC_NS = "{http://purl.org/dc/elements/1.1/}"


def _parse_rss_to_records(
    xml_text: str, subreddit: str, *, limit: int
) -> list[dict[str, Any]]:
    """Convert Reddit RSS to the same record shape ``fetch_rich_reddit`` returns.

    Missing fields (karma, upvote_ratio, score, num_comments) come back as
    ``None`` / ``0`` so downstream trust scoring degrades smoothly. This is
    intentionally a last-resort path -- it gives the corroboration gate
    *something* when the JSON endpoints are blanket-403'd.
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []
    records: list[dict[str, Any]] = []
    # Atom feed (Reddit uses Atom-flavoured RSS at /.rss).
    atom_ns = "{http://www.w3.org/2005/Atom}"
    entries = root.findall(f"{atom_ns}entry")
    if not entries:
        # Fall back to plain RSS <item> shape just in case.
        entries = root.findall(".//item")
    for entry in entries[:limit]:
        # NOTE: ``Element.find`` returns ``None`` when missing but an Element
        # is also falsy if it has no children. Always use ``is not None``.
        title_el = entry.find(f"{atom_ns}title")
        if title_el is None:
            title_el = entry.find("title")
        title = (title_el.text or "").strip() if title_el is not None else ""
        if not title:
            continue
        link_el = entry.find(f"{atom_ns}link")
        if link_el is None:
            link_el = entry.find("link")
        if link_el is not None and link_el.get("href"):
            permalink = link_el.get("href") or ""
        elif link_el is not None and link_el.text:
            permalink = link_el.text
        else:
            permalink = ""
        # Author: Atom feed puts it inside <author><name>/u/foo</name></author>.
        # We accept either the namespaced or bare <name> child so the parser
        # works against both feed shapes seen in the wild.
        author: str | None = None
        author_el = entry.find(f"{atom_ns}author/{atom_ns}name")
        if author_el is None:
            author_el = entry.find("author/name")
        if author_el is not None and author_el.text:
            m = _RSS_CREATOR_RE.search(author_el.text)
            author = m.group(1) if m else None
        # Body (content of the post if present).
        body_el = entry.find(f"{atom_ns}content")
        if body_el is None:
            body_el = entry.find("description")
        selftext = (body_el.text or "")[:2000] if body_el is not None else ""
        records.append(
            {
                "id": entry.findtext(f"{atom_ns}id", default="") or "",
                "permalink": permalink,
                "subreddit": subreddit,
                "title": title,
                "selftext": selftext,
                "author": author if author and author not in ("[deleted]", "AutoModerator") else None,
                "author_created_utc": None,
                "author_karma": None,
                "score": 0,
                "num_comments": 0,
                "upvote_ratio": None,
                "created_utc": 0.0,
                "tickers": tuple(extract_tickers(title + " " + selftext)),
            }
        )
    return records


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
            # Phase 25.5 — prefer OAuth (oauth.reddit.com) when configured.
            # That host returns 200 even from datacenter IPs as long as the
            # bearer is valid; the public hosts 403 us anonymously. We still
            # keep the unauthenticated fallback chain for the no-OAuth case.
            oauth = get_oauth()
            bearer = await oauth.get_token() if oauth.enabled else None
            headers = {
                "User-Agent": oauth.user_agent if oauth.enabled else USER_AGENT,
                "Accept": "application/json",
            }
            if bearer:
                headers["Authorization"] = f"bearer {bearer}"
                json_hosts = [
                    RICH_REDDIT_OAUTH_URL,
                    RICH_REDDIT_URL,
                    *RICH_REDDIT_FALLBACKS,
                ]
            else:
                json_hosts = [RICH_REDDIT_URL, *RICH_REDDIT_FALLBACKS]

            payload: dict[str, Any] | None = None
            last_status = 0
            for host_template in json_hosts:
                url = host_template.format(subreddit=subreddit)
                try:
                    r = await client.get(url, params={"limit": limit}, headers=headers)
                except Exception as exc:
                    logger.debug(
                        "rich reddit transport fail %s: %s",
                        host_template,
                        exc.__class__.__name__,
                    )
                    continue
                last_status = r.status_code
                # If OAuth token went stale (401), invalidate and let the
                # next host attempt mint a fresh one on the subsequent call.
                if r.status_code == 401 and bearer and "oauth.reddit.com" in url:
                    logger.info("reddit oauth: 401 — invalidating cached token")
                    oauth.invalidate()
                    continue
                if r.status_code != 200:
                    continue
                try:
                    payload = r.json()
                    break
                except ValueError:
                    continue

            # If every JSON host failed, fall back to the RSS feed -- it has
            # its own rate-limit pool and almost never 403s. Returns a
            # reduced-fidelity record (no karma/score) which the trust scorer
            # handles via neutral defaults.
            if payload is None:
                rss_url = RICH_REDDIT_RSS_URL.format(subreddit=subreddit)
                try:
                    rr = await client.get(
                        rss_url,
                        params={"limit": limit},
                        headers={
                            "User-Agent": USER_AGENT,
                            "Accept": "application/rss+xml, application/xml",
                        },
                    )
                    if rr.status_code == 200 and rr.text:
                        records = _parse_rss_to_records(
                            rr.text, subreddit, limit=limit
                        )
                        if records:
                            logger.info(
                                "rich reddit r/%s: served from RSS (%d posts) after JSON 403",
                                subreddit, len(records),
                            )
                            return records
                    last_status = rr.status_code or last_status
                except Exception as exc:
                    logger.debug(
                        "rich reddit RSS fail for r/%s: %s",
                        subreddit, exc.__class__.__name__,
                    )
                logger.warning(
                    "rich reddit r/%s: HTTP %s (JSON + RSS exhausted)",
                    subreddit, last_status,
                )
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
