"""Nightly data refresh (Temporal cron @ 03:00 UTC).

Pulls yesterday's bars + the current sentiment snapshot and updates the
local Parquet cache. Designed to be:

  - **Idempotent**: re-running the same day is a no-op.
  - **Resilient**: per-symbol failures don't abort the whole run.
  - **Cheap**: only touches a small "tail" of history.

Triggered by the Temporal worker. The activity ``data.nightly_refresh``
is registered alongside the trading workflow.
"""
from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from packages.data import pretrain
from packages.data.adapters.sentiment import SentimentAdapter, aggregate_sentiment

log = logging.getLogger("nightly_refresh")


def _sentiment_path() -> Path:
    import os

    root = Path(os.getenv("DATA_PARQUET_ROOT", "data/parquet"))
    return root / "sentiment" / "latest.json"


async def refresh_sentiment(adapter: SentimentAdapter | None = None) -> dict[str, Any]:
    """Pull recent posts/headlines and write an aggregated snapshot to disk."""
    adapter = adapter or SentimentAdapter()
    try:
        items = await adapter.fetch_all()
    finally:
        if adapter is not None:
            await adapter.aclose()
    agg = aggregate_sentiment(items, window_hours=24)
    out_path = _sentiment_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "n_items": len(items),
        "by_symbol": agg,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    return payload


async def run() -> dict[str, Any]:
    """Top-level nightly job: refresh bars + sentiment, return summary."""
    log.info("nightly_refresh starting")
    bars = await pretrain.run(refresh_after_days=0.5, include_intraday=True)
    sentiment = await refresh_sentiment()
    summary = {
        "ran_at": datetime.now(UTC).isoformat(),
        "bars": bars,
        "sentiment": {"n_items": sentiment["n_items"], "n_symbols": len(sentiment["by_symbol"])},
    }
    log.info("nightly_refresh complete: %s", summary)
    return summary


# ---------------------------------------------------------------------------
# Temporal activity wrapper
# ---------------------------------------------------------------------------


def temporal_activities() -> list[Any]:
    """Return the Temporal activity registrations for this module.

    Wrapped in a function so importing this module never accidentally
    instantiates Temporal stubs in test envs.
    """
    from temporalio import activity

    @activity.defn(name="data.nightly_refresh")
    async def nightly_refresh_activity() -> dict[str, Any]:
        return await run()

    return [nightly_refresh_activity]
