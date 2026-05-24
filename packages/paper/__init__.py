"""Paper-trading observability utilities (streak counter, KPIs)."""

from packages.paper.streak import (
    PaperDayStats,
    StreakSummary,
    compute_paper_streak,
    iter_paper_runs,
    summarise_paper_days,
)

__all__ = [
    "PaperDayStats",
    "StreakSummary",
    "compute_paper_streak",
    "iter_paper_runs",
    "summarise_paper_days",
]
