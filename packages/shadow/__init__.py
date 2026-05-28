"""Shadow-trading dashboard + auto-greenlight (Phase 6).

This package owns three responsibilities:

1. **Pairing**: turn the JSONL shadow-trade audit log into round-trip
   trades (buy + matching sell) so we have something to compute PnL on.
2. **PnL aggregation**: roll those round-trips into daily PnL series.
3. **Greenlight**: decide whether we've observed 14 consecutive trading
   days of non-negative PnL -- the trigger that flips us from
   ``shadow`` mode toward live capital.

The package is import-light: importing it does not touch the
filesystem, the broker, or the cockpit. Everything is composed by the
``snapshot`` orchestrator and exposed to the web UI via a single
``/shadow`` route.
"""
from __future__ import annotations

from packages.shadow.greenlight import (
    GREENLIGHT_DAYS_REQUIRED,
    GreenlightVerdict,
    evaluate_greenlight,
    read_status,
    write_status,
)
from packages.shadow.notify import (
    FlipEvent,
    append_flip_event,
    detect_flip,
    read_flip_events,
)
from packages.shadow.pairing import (
    PairedTrade,
    pair_round_trips,
)
from packages.shadow.pnl import (
    DailyPnL,
    PredictedVsActual,
    aggregate_daily,
    predicted_vs_actual,
)
from packages.shadow.snapshot import ShadowDashboard, build_snapshot

__all__ = [
    "GREENLIGHT_DAYS_REQUIRED",
    "DailyPnL",
    "FlipEvent",
    "GreenlightVerdict",
    "PairedTrade",
    "PredictedVsActual",
    "ShadowDashboard",
    "aggregate_daily",
    "append_flip_event",
    "build_snapshot",
    "detect_flip",
    "evaluate_greenlight",
    "pair_round_trips",
    "predicted_vs_actual",
    "read_flip_events",
    "read_status",
    "write_status",
]
