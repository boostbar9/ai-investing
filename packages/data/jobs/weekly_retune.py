"""Weekly walk-forward retune (Temporal cron @ Sunday 05:00 UTC).

Loops the universe, refits strategy parameters per-symbol on a 2-year window,
and promotes any challenger that clears the promotion gate. The new champion
is persisted under ``data/params/champion.json``.

Per-symbol results are also written to ``data/params/retune_log.jsonl`` for
audit / cockpit display.
"""
from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from packages.backtests.walk_forward import retune

log = logging.getLogger("weekly_retune")

# Keep this small — the per-symbol refit is fast but full universes blow up
# cron runtimes. Start with the index ETFs; extend by env var.
DEFAULT_RETUNE_UNIVERSE = ("SPY", "QQQ", "IWM")


def _retune_log_path() -> Path:
    import os

    root = Path(os.getenv("DATA_PARAMS_ROOT", "data/params"))
    return root / "retune_log.jsonl"


def _append_log(entry: dict[str, Any]) -> None:
    p = _retune_log_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as fh:
        fh.write(json.dumps(entry) + "\n")


async def run(universe: tuple[str, ...] | None = None) -> dict[str, Any]:
    universe = universe or DEFAULT_RETUNE_UNIVERSE
    summary: dict[str, Any] = {
        "ran_at": datetime.now(UTC).isoformat(),
        "n_symbols": 0,
        "n_promoted": 0,
        "results": [],
        "errors": [],
    }
    for sym in universe:
        try:
            result = retune(sym)
        except FileNotFoundError as e:
            summary["errors"].append(f"{sym}: {e}")
            continue
        except Exception as e:
            log.warning("retune %s failed: %s", sym, e)
            summary["errors"].append(f"{sym}: {e}")
            continue
        entry = {
            "symbol": sym,
            "ran_at": summary["ran_at"],
            "promoted": result.promoted,
            "challenger": result.challenger.as_dict(),
            "in_sample_sharpe": result.in_sample_sharpe,
            "out_of_sample_sharpe": result.out_of_sample_sharpe,
            "reasons": result.reasons,
        }
        _append_log(entry)
        summary["results"].append(entry)
        summary["n_symbols"] += 1
        if result.promoted:
            summary["n_promoted"] += 1
    log.info(
        "weekly_retune complete: %d symbols, %d promoted",
        summary["n_symbols"],
        summary["n_promoted"],
    )
    return summary


def temporal_activities() -> list[Any]:
    from temporalio import activity

    @activity.defn(name="data.weekly_retune")
    async def weekly_retune_activity() -> dict[str, Any]:
        return await run()

    return [weekly_retune_activity]


def main() -> None:  # pragma: no cover - CLI entry point
    import asyncio

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    s = asyncio.run(run())
    log.info("weekly_retune summary: %s", s)


if __name__ == "__main__":  # pragma: no cover
    main()
