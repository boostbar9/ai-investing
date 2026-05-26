"""Per-decision audit log (§17, task 7).

For every LLM agent call we record the prompt, the raw response, whether
validation succeeded, and -- when the decision flows all the way through
risk -> execution -- the final orders the runner actually submitted.

Format
------
JSONL, one record per decision step. Fields:

    {
      "ts":              ISO timestamp,
      "decision_id":     UUID string (the trusted server-side id),
      "agent":           "research" | "strategy" | "risk" | "execution" | "discovery",
      "attempt":         1 | 2,
      "prompt":          str,
      "raw_response":    str | dict | None,   # whatever the router returned
      "validation_ok":   bool,
      "validation_error": str | None,
      "final_orders":    list[dict] | None,   # only present on terminal step
      "extra":           dict | None,         # free-form (e.g. recovered_via_repair=True)
    }

Why JSONL: human-greppable, append-safe, plays nicely with the existing
``data/paper_log/runs.jsonl`` style and tools like ``jq``. A SQL mirror
can be added later if we need joins; for now the audit log is a
read-rarely write-always artifact.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

AUDIT_LOG_PATH = Path(os.getenv("AGENT_AUDIT_LOG_PATH", "data/audit/decisions.jsonl"))

_LOCK = threading.Lock()
log = logging.getLogger(__name__)


def log_decision(
    *,
    decision_id: str,
    agent: str,
    prompt: str,
    raw_response: Any = None,
    validation_ok: bool = True,
    validation_error: str | None = None,
    attempt: int = 1,
    final_orders: list[dict[str, Any]] | None = None,
    extra: dict[str, Any] | None = None,
    path: Path = AUDIT_LOG_PATH,
) -> None:
    """Append one audit record. Best-effort: never raises on disk errors."""
    record = {
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "decision_id": str(decision_id),
        "agent": agent,
        "attempt": int(attempt),
        "prompt": prompt,
        "raw_response": raw_response,
        "validation_ok": bool(validation_ok),
        "validation_error": validation_error,
        "final_orders": final_orders,
        "extra": extra or {},
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, default=str) + "\n"
        with _LOCK, path.open("a", encoding="utf-8") as f:
            f.write(line)
    except OSError as e:  # pragma: no cover - I/O failure path
        log.warning("audit log write failed: %s", e)
