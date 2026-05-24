"""Strategy stack (§6)."""
from .base import Strategy, StrategyMeta  # noqa: F401
from .intraday_trend import IntradayTrendFollowing
from .mean_reversion import MeanReversion
from .sector_rotation import SectorRotation
from .sentiment_overlay import SentimentOverlay
from .trend_following import TrendFollowing


def all_strategies() -> dict[str, type[Strategy]]:
    return {
        "trend-following": TrendFollowing,
        "sector-rotation": SectorRotation,
        "mean-reversion": MeanReversion,
        "sentiment-overlay": SentimentOverlay,
        "intraday-trend": IntradayTrendFollowing,
    }
