"""Async token-bucket rate limiter — used by every data source adapter.

Honors the hard caps listed in §14. Each source gets its own bucket.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass


@dataclass
class TokenBucket:
    rate_per_second: float
    capacity: int
    _tokens: float = 0.0
    _last: float = 0.0
    _lock: asyncio.Lock = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self._tokens = float(self.capacity)
        self._last = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, n: int = 1) -> None:
        async with self._lock:
            while True:
                now = time.monotonic()
                elapsed = now - self._last
                self._last = now
                self._tokens = min(self.capacity, self._tokens + elapsed * self.rate_per_second)
                if self._tokens >= n:
                    self._tokens -= n
                    return
                wait = (n - self._tokens) / self.rate_per_second
                await asyncio.sleep(wait)


# Buckets sized to §14 free tiers (conservative — paid users can override).
BUCKETS: dict[str, TokenBucket] = {
    "polygon": TokenBucket(rate_per_second=5 / 60, capacity=5),       # 5/min free
    "alpha_vantage": TokenBucket(rate_per_second=25 / 86400, capacity=25),  # 25/day
    # Finnhub free tier is 60/min average **but** the
    # /stock/insider-transactions endpoint enforces a separate ~10 req/s
    # sub-limit. The live log on 2026-06-01 22:36:26 showed a 6-symbol
    # burst (XLI/SPY/XLK/XLU/XLB/XLE) tripping 429s because the previous
    # capacity=60 let a cold bucket fire 60 calls back-to-back. Cap the
    # burst at 8 while keeping the 60/min average refill.
    "finnhub": TokenBucket(rate_per_second=60 / 60, capacity=8),
    "sec_edgar": TokenBucket(rate_per_second=10, capacity=10),       # SEC asks <10 req/s
    "fred": TokenBucket(rate_per_second=5, capacity=20),             # generous default
    "reddit": TokenBucket(rate_per_second=1, capacity=10),
    # yfinance unofficial endpoint — stay polite, 2 req/s, burst 10.
    "yfinance": TokenBucket(rate_per_second=2, capacity=10),
    # Alpaca market data free tier: ~200 req/min for IEX feed.
    "alpaca_data": TokenBucket(rate_per_second=200 / 60, capacity=20),
    # Generic RSS poller — 1 req/sec per feed is plenty.
    "rss": TokenBucket(rate_per_second=1, capacity=5),
}
