"""Phase 26 — Finnhub news sentiment, per-ticker with caching.

The Reddit sentiment path (``packages/data/adapters/sentiment.py``) was
the project's only sentiment source. After the Reddit API gate
(Phase 25.5) we need a *primary* sentiment source that doesn't depend
on Reddit approval. Finnhub already ships per-ticker company news; we
score the headlines locally with the lexicon already proven in the
Reddit path, cache aggressively to stay under the 60/min rate limit,
and surface a single bullish/neutral/bearish score the cockpit can
display.

Design choices:
  * **Caching by symbol + day-window.** Most users hit the same ~20
    tickers many times per minute; without a cache we'd burn the rate
    limit instantly. Default TTL is 15min and a hard cap of 256
    entries keeps memory bounded.
  * **Recency-weighted scoring.** Headlines from today count more than
    headlines from a week ago. Half-life is 48h by default; tunable.
  * **Source diversity.** We surface the unique source count so
    callers can down-weight signals that come from a single outlet
    (a 100% positive score sourced only from PR Newswire is suspect).
  * **Graceful degradation.** No FINNHUB_API_KEY → adapter reports
    ``enabled=False`` and ``score_symbol()`` returns a neutral payload.
    No retries on transport errors; the caller treats the absence of
    fresh news as neutral, not as a strong negative.
"""
from __future__ import annotations

import logging
import math
import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from packages.data.adapters.base import DataAdapterError, NewsItem
from packages.data.adapters.finnhub import FinnhubAdapter
from packages.data.adapters.sentiment import score_headline

logger = logging.getLogger(__name__)


# --- Tunables -----------------------------------------------------------------

# Cache TTL — Finnhub publishes news within minutes, but a 15-min cache
# keeps us well under the 60/min rate limit without lagging signals.
DEFAULT_CACHE_TTL_S = 15 * 60

# Hard cap on cached symbols to prevent unbounded memory growth.
DEFAULT_CACHE_MAX = 256

# Lookback window for "recent" company news. Anything older than this
# is excluded entirely from the score.
DEFAULT_LOOKBACK_DAYS = 7

# Recency half-life — a headline from 48h ago contributes half as
# much as one from now.
DEFAULT_HALFLIFE_HOURS = 48.0


@dataclass(frozen=True)
class NewsSentiment:
    """Aggregated per-ticker sentiment from recent news.

    All fields are derived; callers should treat this as immutable.

    Attributes:
        symbol: Ticker the score is for (upper-cased).
        score: Recency-weighted mean in [-1, 1]. Positive = bullish.
        confidence: How much we trust the score, in [0, 1]. Driven by
            article count + source diversity. Below ~0.2 → treat as
            "no signal".
        label: ``"bullish"`` | ``"neutral"`` | ``"bearish"`` — derived
            from ``score`` thresholds. Saves the UI from duplicating
            the bucket logic.
        article_count: Number of headlines that fed the score.
        source_count: Number of distinct ``source`` values.
        fresh_at: When the score was computed (UTC, naive).
        sample_headlines: Up to 3 top-contributing headlines, for
            debugging + the cockpit tooltip.
    """

    symbol: str
    score: float
    confidence: float
    label: str
    article_count: int
    source_count: int
    fresh_at: datetime
    sample_headlines: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_bullish(self) -> bool:
        return self.label == "bullish"

    @property
    def is_bearish(self) -> bool:
        return self.label == "bearish"

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "score": round(self.score, 4),
            "confidence": round(self.confidence, 4),
            "label": self.label,
            "article_count": self.article_count,
            "source_count": self.source_count,
            "fresh_at": self.fresh_at.isoformat(),
            "sample_headlines": list(self.sample_headlines),
        }


def _label_for(score: float, confidence: float) -> str:
    """Bucket a score into bullish/neutral/bearish.

    Confidence below 0.2 forces ``neutral`` — we don't want a single
    Yahoo Finance headline to flip the brain on a thinly-covered name.
    """
    if confidence < 0.2:
        return "neutral"
    if score > 0.15:
        return "bullish"
    if score < -0.15:
        return "bearish"
    return "neutral"


def _confidence_from(article_count: int, source_count: int) -> float:
    """Soft confidence model: more articles + more sources = more trust.

    Saturates near 1.0 around 20+ articles from 5+ sources. A single
    article from one source pegs confidence near 0.1 — barely above
    the "no signal" floor — so the score is treated as noise unless
    corroborated.
    """
    if article_count <= 0:
        return 0.0
    # Each contributes a logistic-like ramp; combine multiplicatively
    # so BOTH must be present to clear the floor.
    # Tuned so 3 articles + 3 sources ≈ 0.5, saturating near 1.0 at
    # ~15 articles from 5+ sources.
    article_factor = 1 - math.exp(-article_count / 4.0)
    source_factor = 1 - math.exp(-max(0, source_count) / 2.0)
    return round(article_factor * source_factor, 4)


def aggregate_news_sentiment(
    symbol: str,
    items: list[NewsItem],
    *,
    now: datetime | None = None,
    halflife_hours: float = DEFAULT_HALFLIFE_HOURS,
) -> NewsSentiment:
    """Score a batch of news items into a single recency-weighted score.

    Pure function — easy to unit-test without httpx mocks. The
    network-aware ``FinnhubNewsClient.score_symbol`` calls this after
    fetching the raw items.
    """
    sym = symbol.upper()
    now = now or datetime.now(UTC)
    # Normalize the comparison clock: items carry tz-aware ts, so
    # ``now`` must match.
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)

    if not items:
        return NewsSentiment(
            symbol=sym,
            score=0.0,
            confidence=0.0,
            label="neutral",
            article_count=0,
            source_count=0,
            fresh_at=now.replace(tzinfo=None),
            sample_headlines=(),
        )

    # Weighted sum: each headline contributes (score x recency_weight).
    # Recency weight is exponential decay with half-life ``halflife_hours``.
    total_weight = 0.0
    weighted_score = 0.0
    sources: set[str] = set()
    scored: list[tuple[float, str]] = []  # (abs_contribution, headline)

    for item in items:
        ts = item.ts
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        age_h = max(0.0, (now - ts).total_seconds() / 3600.0)
        if age_h > 24 * DEFAULT_LOOKBACK_DAYS:
            continue
        recency = 0.5 ** (age_h / halflife_hours)
        s = score_headline(f"{item.headline} {item.summary or ''}")
        if s == 0.0 and not item.headline:
            continue
        weighted_score += s * recency
        total_weight += recency
        if item.source:
            sources.add(item.source)
        scored.append((abs(s) * recency, item.headline))

    if total_weight <= 0:
        return NewsSentiment(
            symbol=sym,
            score=0.0,
            confidence=0.0,
            label="neutral",
            article_count=0,
            source_count=0,
            fresh_at=now.replace(tzinfo=None),
            sample_headlines=(),
        )

    score = weighted_score / total_weight
    # Clamp defensively — score_headline returns [-1, 1] but recency
    # weighting can't push us out of that band; this is a guardrail.
    score = max(-1.0, min(1.0, score))

    article_count = len(scored)
    source_count = len(sources)
    confidence = _confidence_from(article_count, source_count)
    label = _label_for(score, confidence)

    # Top 3 headlines by absolute weighted contribution.
    scored.sort(reverse=True)
    sample_headlines = tuple(h for _, h in scored[:3] if h)

    return NewsSentiment(
        symbol=sym,
        score=score,
        confidence=confidence,
        label=label,
        article_count=article_count,
        source_count=source_count,
        fresh_at=now.replace(tzinfo=None),
        sample_headlines=sample_headlines,
    )


# --- Network-aware client -----------------------------------------------------


@dataclass
class _CacheEntry:
    sentiment: NewsSentiment
    expires_at: float  # monotonic seconds


class FinnhubNewsClient:
    """Caching client for per-ticker news sentiment.

    Thread-safety: not asyncio-locked because the underlying
    ``FinnhubAdapter.get_company_news`` is idempotent and a double
    fetch on a cache miss is harmless (last writer wins). Network
    cost is bounded by the rate-limit bucket in
    ``packages.shared.rate_limit``.
    """

    def __init__(
        self,
        adapter: FinnhubAdapter | None = None,
        *,
        cache_ttl_s: float = DEFAULT_CACHE_TTL_S,
        cache_max: int = DEFAULT_CACHE_MAX,
        lookback_days: int = DEFAULT_LOOKBACK_DAYS,
        halflife_hours: float = DEFAULT_HALFLIFE_HOURS,
        clock: Any = None,
    ) -> None:
        self._adapter = adapter or FinnhubAdapter()
        self._cache: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._cache_ttl_s = cache_ttl_s
        self._cache_max = cache_max
        self._lookback_days = lookback_days
        self._halflife_hours = halflife_hours
        self._clock = clock or time.monotonic
        self._hits = 0
        self._misses = 0
        self._errors = 0

    @property
    def enabled(self) -> bool:
        return self._adapter.has_key

    def stats(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "cached_symbols": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "errors": self._errors,
            "cache_ttl_s": self._cache_ttl_s,
            "cache_max": self._cache_max,
        }

    def invalidate(self, symbol: str | None = None) -> None:
        if symbol is None:
            self._cache.clear()
        else:
            self._cache.pop(symbol.upper(), None)

    async def score_symbol(
        self, symbol: str, *, now: datetime | None = None
    ) -> NewsSentiment:
        """Return a cached or freshly-computed sentiment for ``symbol``.

        Never raises — on transport / API failure returns a
        neutral payload with ``confidence=0``.
        """
        sym = symbol.upper()
        wall_now = now or datetime.now(UTC)
        cache_now = self._clock()

        cached = self._cache.get(sym)
        if cached is not None and cached.expires_at > cache_now:
            self._hits += 1
            # LRU touch.
            self._cache.move_to_end(sym)
            return cached.sentiment

        self._misses += 1
        if not self.enabled:
            # No API key → return a neutral payload but do NOT cache
            # it; the moment a key shows up we should pick it up.
            return aggregate_news_sentiment(
                sym, items=[], now=wall_now,
                halflife_hours=self._halflife_hours,
            )

        # Fetch fresh news items for the lookback window.
        to_d = wall_now.date()
        frm_d = (wall_now - timedelta(days=self._lookback_days)).date()
        try:
            items = await self._adapter.get_company_news(
                sym, frm=frm_d.isoformat(), to=to_d.isoformat()
            )
        except DataAdapterError as exc:
            self._errors += 1
            logger.warning("finnhub news %s: %s", sym, exc)
            return aggregate_news_sentiment(
                sym, items=[], now=wall_now,
                halflife_hours=self._halflife_hours,
            )
        except Exception as exc:
            self._errors += 1
            logger.warning("finnhub news %s: transport %s", sym, exc)
            return aggregate_news_sentiment(
                sym, items=[], now=wall_now,
                halflife_hours=self._halflife_hours,
            )

        sentiment = aggregate_news_sentiment(
            sym, items, now=wall_now,
            halflife_hours=self._halflife_hours,
        )
        # Only cache *meaningful* responses. A zero-article response is
        # likely a transient — refetch on the next call rather than
        # serving stale emptiness for 15 minutes.
        if sentiment.article_count > 0:
            self._put(sym, sentiment, cache_now)
        return sentiment

    async def aclose(self) -> None:
        await self._adapter.aclose()

    # --- internals --------------------------------------------------

    def _put(self, sym: str, sentiment: NewsSentiment, now: float) -> None:
        self._cache[sym] = _CacheEntry(
            sentiment=sentiment, expires_at=now + self._cache_ttl_s
        )
        self._cache.move_to_end(sym)
        while len(self._cache) > self._cache_max:
            self._cache.popitem(last=False)


# --- Module-level singleton (mirrors the reddit_trust.oauth pattern) ----------


_default_client: FinnhubNewsClient | None = None


def get_news_client() -> FinnhubNewsClient:
    """Return the process-wide news sentiment client."""
    global _default_client
    if _default_client is None:
        _default_client = FinnhubNewsClient()
    return _default_client


def reset_news_client_for_tests() -> None:
    global _default_client
    _default_client = None


__all__ = [
    "FinnhubNewsClient",
    "NewsSentiment",
    "aggregate_news_sentiment",
    "get_news_client",
    "reset_news_client_for_tests",
]


# Side-effect: prove the env var is read at module load time.
_ = os.getenv("FINNHUB_API_KEY", "")
