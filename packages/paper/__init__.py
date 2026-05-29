"""Paper-trading observability utilities (streak counter, KPIs, decisions)."""

from packages.paper.decisions import (
    DecisionRecord,
    PipelineStage,
    append_decision,
    build_record,
    iter_decisions,
    latest_pipeline,
    load_recent,
    window_status,
)
from packages.paper.predictions import (
    REGIME_EXPECTED_RETURN_5D,
    append_predictions,
    iter_predictions,
    load_predictions,
    predicted_pnl_for_symbol,
)
from packages.paper.sim_pnl import (
    daily_equity_curve,
    merge_real_and_synth,
    synth_trades_from_runs,
)
from packages.paper.streak import (
    PaperDayStats,
    StreakSummary,
    compute_paper_streak,
    iter_paper_runs,
    summarise_paper_days,
)

__all__ = [
    "REGIME_EXPECTED_RETURN_5D",
    "DecisionRecord",
    "PaperDayStats",
    "PipelineStage",
    "StreakSummary",
    "append_decision",
    "append_predictions",
    "build_record",
    "compute_paper_streak",
    "daily_equity_curve",
    "iter_decisions",
    "iter_paper_runs",
    "iter_predictions",
    "latest_pipeline",
    "load_predictions",
    "load_recent",
    "merge_real_and_synth",
    "predicted_pnl_for_symbol",
    "summarise_paper_days",
    "synth_trades_from_runs",
    "window_status",
]
