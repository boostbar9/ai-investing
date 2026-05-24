"""Pull real per-symbol sentiment scores from Reddit + RSS.

Output: dict ``{symbol: score in [-1, 1]}`` for the requested symbols.

Used by the paper-trade runner with ``--use-sentiment``. Also handy as a
standalone CLI to inspect what the noise actually looks like::

    PYTHONPATH=. python3 tools/fetch_sentiment.py SPY QQQ AAPL NVDA

Implementation notes
--------------------
- Uses the existing :class:`SentimentAdapter` (Reddit JSON + RSS feeds).
- Sentiment scoring is lexicon-based, not LLM. Good enough for a noisy
  contrarian signal -- which is exactly how the overlay treats it.
- Symbols with zero matched headlines get a neutral 0.0 score.
- Persistent cache at ``data/sentiment_cache.json`` so repeated calls in
  the same trading day don't hammer Reddit.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from packages.data.adapters.sentiment import (
    SentimentAdapter,
    aggregate_sentiment,
)

log = logging.getLogger("sentiment")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

CACHE_PATH = Path("data/sentiment_cache.json")
CACHE_TTL_HOURS = 4


@dataclass
class SymbolScore:
    symbol: str
    score: float
    n_headlines: int


def _load_cache() -> dict | None:
    if not CACHE_PATH.exists():
        return None
    try:
        data = json.loads(CACHE_PATH.read_text())
        ts = datetime.fromisoformat(data.get("ts", ""))
        if datetime.now(UTC) - ts < timedelta(hours=CACHE_TTL_HOURS):
            return data
    except (json.JSONDecodeError, ValueError, OSError, TypeError):
        pass
    return None


def _save_cache(scores: dict[str, float], n_headlines: dict[str, int]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(
        json.dumps(
            {
                "ts": datetime.now(UTC).isoformat(),
                "scores": scores,
                "n_headlines": n_headlines,
            },
            indent=2,
        )
    )


async def fetch_scores(
    symbols: list[str],
    *,
    use_cache: bool = True,
    max_per_source: int = 25,
) -> dict[str, float]:
    """Return ``{symbol: score in [-1, 1]}`` for each requested symbol.

    Symbols with no recent matched headlines get 0.0 (neutral).
    """
    if use_cache:
        cached = _load_cache()
        if cached is not None:
            scores = cached.get("scores") or {}
            log.info("using cached sentiment (%d symbols, age < %dh)", len(scores), CACHE_TTL_HOURS)
            return {s: float(scores.get(s, 0.0)) for s in symbols}

    adapter = SentimentAdapter()
    try:
        items = await adapter.fetch_all(max_per_source=max_per_source)
    finally:
        await adapter.aclose()
    log.info("pulled %d headlines/posts total", len(items))

    agg = aggregate_sentiment(items, window_hours=24)
    scores: dict[str, float] = {}
    n_headlines: dict[str, int] = {}
    for sym in symbols:
        if sym in agg:
            scores[sym] = float(agg[sym]["score"])
            n_headlines[sym] = int(agg[sym]["n"])
        else:
            scores[sym] = 0.0
            n_headlines[sym] = 0

    _save_cache(scores, n_headlines)
    return scores


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("symbols", nargs="+", help="Symbols to score (e.g. SPY QQQ AAPL).")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    scores = asyncio.run(fetch_scores(args.symbols, use_cache=not args.no_cache))
    cache = _load_cache() or {}
    n_headlines = cache.get("n_headlines", {})
    print(f"{'Symbol':<8} {'Score':>7} {'N':>5}")
    print("-" * 24)
    for sym in args.symbols:
        s = scores.get(sym, 0.0)
        n = n_headlines.get(sym, 0)
        print(f"{sym:<8} {s:>7.3f} {n:>5}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
