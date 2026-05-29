"""Per-ticker subreddit discovery — Phase 10.

When a candidate ticker shows up in the sweep, the most active
discussion is often *not* on r/wallstreetbets or r/stocks but on the
ticker's own subreddit (r/NVDA_Stock, r/teslainvestorsclub, r/RKLB).
Those subs frequently surface earnings whispers, product announcements,
and management interviews hours before the news hits mainstream feeds.

This module probes a small set of candidate subreddit-name patterns
for each ticker, keeps the ones that respond ``200``, and feeds them
into the sweep's roster on top of the static high-quality set.

Probing happens with HEAD requests so we don't burn the rate-limit
budget actually fetching posts from subs that don't exist. We cache
positive hits in process memory; the cache is small (one bool per
probed sub) and persists for the lifetime of the cockpit.
"""

from __future__ import annotations

import contextlib
import logging
import os
from collections.abc import Iterable

import httpx

from packages.shared.rate_limit import BUCKETS

logger = logging.getLogger(__name__)

# Probe URL: Reddit's about.json for a sub returns 200 if it exists,
# 404/403 otherwise. Lighter than fetching /hot.json.
ABOUT_URL = os.getenv(
    "REDDIT_ABOUT_URL", "https://www.reddit.com/r/{sub}/about.json"
)

USER_AGENT = os.getenv(
    "RICH_REDDIT_UA",
    "ai-investing/0.4 (+https://github.com/boostbar9/ai-investing)",
)

# Process-local cache: ``{subreddit_name: exists_bool}``. Populated by
# :func:`probe_subreddit`; consulted by :func:`candidate_subreddits` to
# skip names we've already proven don't exist.
_EXISTENCE_CACHE: dict[str, bool] = {}

# Common name patterns. Many per-ticker subs follow one of these.
# Order matters: we yield them in priority order so callers can stop
# probing once they have enough hits.
NAME_PATTERNS: tuple[str, ...] = (
    "{ticker}_Stock",
    "{ticker}Stock",
    "{ticker}",
    "{ticker}_Investors",
    "{ticker}Investors",
)

# Symbol -> hand-curated subreddit overrides. These either don't match
# any pattern (e.g. teslainvestorsclub for TSLA) or are the *primary*
# community even when a pattern-matched sub exists.
HAND_OVERRIDES: dict[str, tuple[str, ...]] = {
    "TSLA": ("teslainvestorsclub", "TSLA"),
    "GME": ("GME", "Superstonk"),
    "AMC": ("AMC_Stock",),
    "PLTR": ("PLTR", "Palantir_Investors"),
    "AMD": ("AMD_Stock",),
    "NVDA": ("NVDA_Stock",),
    "RKLB": ("RKLB",),
    "SOFI": ("SOFIInvestors",),
}


def candidate_subreddits(ticker: str) -> tuple[str, ...]:
    """Return the ordered list of subreddit names worth probing for
    ``ticker``. Pure function — no network calls.
    """
    if not ticker or not ticker.isascii():
        return ()
    t = ticker.upper()
    seen: set[str] = set()
    out: list[str] = []
    for sub in HAND_OVERRIDES.get(t, ()):
        if sub.lower() not in seen:
            seen.add(sub.lower())
            out.append(sub)
    for pattern in NAME_PATTERNS:
        name = pattern.format(ticker=t)
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(name)
    return tuple(out)


async def probe_subreddit(
    name: str, *, client: httpx.AsyncClient
) -> bool:
    """Return True if r/<name> exists and serves public JSON.

    Cached: a sub once proven to exist (or not) stays decided for the
    process lifetime. Anonymous Reddit handles roughly 1 req/s before
    serving 429s, so we go through the shared ``reddit`` bucket.
    """
    if name in _EXISTENCE_CACHE:
        return _EXISTENCE_CACHE[name]
    await BUCKETS["reddit"].acquire()
    url = ABOUT_URL.format(sub=name)
    try:
        r = await client.get(
            url,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "application/json",
            },
            timeout=6.0,
        )
    except Exception as exc:
        logger.debug("probe %s failed: %s", name, exc)
        # Don't cache transient failures — try again next sweep.
        return False
    # 200 means it exists. 403/404 means it doesn't (or is banned).
    exists = r.status_code == 200
    if exists:
        # Some banned subs return 200 with {"reason": "banned"} body.
        try:
            payload = r.json()
        except ValueError:
            payload = {}
        if (payload.get("data") or {}).get("over18") is None and not (
            payload.get("data")
        ):
            exists = False
    _EXISTENCE_CACHE[name] = exists
    return exists


async def discover_for_tickers(
    tickers: Iterable[str],
    *,
    client: httpx.AsyncClient | None = None,
    max_per_ticker: int = 2,
    max_total: int = 8,
) -> tuple[str, ...]:
    """Discover up to ``max_total`` existing per-ticker subreddits
    across the supplied ``tickers``. Caps per-ticker fan-out at
    ``max_per_ticker`` so a 10-ticker candidate set can't fan out into
    50 probes.

    Never raises. Returns a tuple of subreddit names guaranteed to
    exist (or at least to have responded 200 once during this run).
    """
    owned = client is None
    if owned:
        client = httpx.AsyncClient(timeout=8.0)
    found: list[str] = []
    try:
        for ticker in tickers:
            if len(found) >= max_total:
                break
            hits_for_ticker = 0
            for cand in candidate_subreddits(ticker):
                if hits_for_ticker >= max_per_ticker:
                    break
                if len(found) >= max_total:
                    break
                if await probe_subreddit(cand, client=client):
                    found.append(cand)
                    hits_for_ticker += 1
    finally:
        if owned:
            with contextlib.suppress(Exception):
                await client.aclose()
    return tuple(found)


def reset_cache() -> None:
    """Test seam — drop the existence cache so probes re-fire."""
    _EXISTENCE_CACHE.clear()
