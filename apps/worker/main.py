"""Temporal worker entrypoint (§5, issue #3).

Run with::

    python -m apps.worker.main

Reads ``TEMPORAL_HOST`` (default ``localhost:7233``) and ``TEMPORAL_TASK_QUEUE``
(default ``ai-investing``). Registers the agent activities and workflows.
"""

from __future__ import annotations

import asyncio
import logging
import os

from temporalio.client import Client
from temporalio.worker import Worker

from packages.agents.temporal_workflow import ALL_ACTIVITIES, ALL_WORKFLOWS

log = logging.getLogger("worker")


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    host = os.getenv("TEMPORAL_HOST", "localhost:7233")
    namespace = os.getenv("TEMPORAL_NAMESPACE", "default")
    task_queue = os.getenv("TEMPORAL_TASK_QUEUE", "ai-investing")

    log.info("connecting to temporal at %s namespace=%s", host, namespace)
    client = await Client.connect(host, namespace=namespace)

    log.info(
        "worker starting: task_queue=%s workflows=%d activities=%d",
        task_queue,
        len(ALL_WORKFLOWS),
        len(ALL_ACTIVITIES),
    )
    worker = Worker(
        client,
        task_queue=task_queue,
        workflows=ALL_WORKFLOWS,
        activities=ALL_ACTIVITIES,
    )
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
