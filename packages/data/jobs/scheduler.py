"""Programmatic Temporal Schedule installer for the data cron jobs.

Run ``python -m packages.data.jobs.scheduler`` after the worker is up to
register two schedules:

  - ``data-nightly-refresh``  — daily @ 03:00 UTC
  - ``data-weekly-retune``    — Sundays @ 05:00 UTC

Idempotent: re-running updates the schedule spec in place.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

log = logging.getLogger("scheduler")


# Schedules declared as plain dicts so this module is importable even when
# ``temporalio`` isn't installed in the test sandbox.
SCHEDULES: dict[str, dict[str, Any]] = {
    "data-nightly-refresh": {
        "activity": "data.nightly_refresh",
        "cron": "0 3 * * *",  # 03:00 UTC every day
        "description": "Pull yesterday's bars + sentiment, update Parquet cache.",
    },
    "data-weekly-retune": {
        "activity": "data.weekly_retune",
        "cron": "0 5 * * 0",  # 05:00 UTC every Sunday
        "description": "Walk-forward retune of strategy params.",
    },
}


async def install() -> None:  # pragma: no cover — touches the live Temporal cluster
    from temporalio.client import Client, ScheduleActionStartWorkflow

    host = os.getenv("TEMPORAL_HOST", "localhost:7233")
    namespace = os.getenv("TEMPORAL_NAMESPACE", "default")
    task_queue = os.getenv("TEMPORAL_TASK_QUEUE", "ai-investing")

    client = await Client.connect(host, namespace=namespace)

    # Imports kept local so the module is importable without temporalio.
    from temporalio.client import Schedule, ScheduleSpec, ScheduleState

    for sched_id, cfg in SCHEDULES.items():
        spec = ScheduleSpec(cron_expressions=[cfg["cron"]])
        action = ScheduleActionStartWorkflow(
            "DataJobWorkflow",
            cfg["activity"],
            id=f"{sched_id}-workflow",
            task_queue=task_queue,
        )
        schedule = Schedule(
            action=action, spec=spec, state=ScheduleState(note=cfg["description"])
        )
        try:
            handle = client.get_schedule_handle(sched_id)
            await handle.update(lambda _, ns=schedule: ns)
            log.info("updated schedule %s", sched_id)
        except Exception:
            await client.create_schedule(sched_id, schedule)
            log.info("created schedule %s", sched_id)


def main() -> None:  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    asyncio.run(install())


if __name__ == "__main__":  # pragma: no cover
    main()
