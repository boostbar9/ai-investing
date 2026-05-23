"""Strategy stack (§6)."""
from .base import Strategy, StrategyMeta  # noqa: F401
from .mean_reversion import MeanReversion  # noqa: F401
from .sector_rotation import SectorRotation  # noqa: F401
from .sentiment_overlay import SentimentOverlay  # noqa: F401
from .trend_following import TrendFollowing  # noqa: F401


def all_strategies() -> dict[str, type[Strategy]]:
    return {
        "trend-following": TrendFollowing,
        "sector-rotation": SectorRotation,
        "mean-reversion": MeanReversion,
        "sentiment-overlay": SentimentOverlay,
    }
