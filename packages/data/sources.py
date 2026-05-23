"""Phase 1 starter: rate-limit-aware source registry from §14."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Source:
    name: str
    use: str
    free_tier: str
    paid_plan: str
    hard_cap: str


SOURCES: list[Source] = [
    Source("Polygon.io", "Bars/ticks", "5/min", "Stocks Starter $29", "100/sec"),
    Source("Alpha Vantage", "Fundamentals", "25/day", "Premium $50", "75/min"),
    Source("Finnhub", "News, alt", "60/min", "Free OK", "60/min"),
    Source("SEC EDGAR", "Filings", "Unlimited", "n/a", "self-throttle"),
    Source("FRED", "Macro", "Unlimited", "n/a", "none"),
    Source("Reddit / X", "Sentiment (opt)", "varies", "optional", "self-throttle"),
]
