"""AI Sentiment Overlay (§6).

Multiplier-style overlay: scales base weights by a sentiment factor from the
Research Agent. In Phase 2 we stub the factor to 1.0 (neutral); the LangGraph
wiring in Phase 3 plugs in real LLM-derived sentiment per symbol.
"""
from __future__ import annotations

import pandas as pd

from .base import Strategy, StrategyMeta


class SentimentOverlay(Strategy):
    meta = StrategyMeta(
        name="sentiment-overlay",
        description="Scales another strategy's weights by per-symbol AI sentiment in [0, 1.25].",
        universe=[],
    )

    def __init__(self, base: Strategy, sentiment: dict[str, float] | None = None) -> None:
        self.base = base
        self.sentiment = sentiment or {}
        self.meta = StrategyMeta(
            name=f"{base.meta.name}+sentiment",
            description=self.meta.description,
            universe=base.meta.universe,
        )

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        w = self.base.generate_signals(prices)
        scales = pd.Series(
            {c: max(0.0, min(1.25, self.sentiment.get(c, 1.0))) for c in w.columns}
        )
        scaled = w.mul(scales, axis=1)
        # Re-cap row-sum at 1.0 so the overlay can't lever us up.
        rsum = scaled.sum(axis=1)
        over = rsum > 1.0
        if over.any():
            scaled.loc[over] = scaled.loc[over].div(rsum[over], axis=0)
        return scaled
