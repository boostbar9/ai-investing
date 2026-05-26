"""Per-cycle snapshot writer (§17, task 8).

After each paper-trade cycle the runner stamps an atomic JSON snapshot
into ``data/cockpit/snapshot.json``. The cockpit reads this on boot so
the dashboard has equity/positions/streak immediately, instead of
returning ``None`` until the next cycle fires.

Atomic write: ``tempfile.NamedTemporaryFile`` in the same directory,
flushed and replaced with ``os.replace`` (same pattern as
``packages/cockpit/state.py``). Crash mid-write leaves the previous
snapshot intact.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SNAPSHOT_PATH = Path(os.getenv("COCKPIT_SNAPSHOT_PATH", "data/cockpit/snapshot.json"))

log = logging.getLogger(__name__)


def write_snapshot(
    *,
    equity: float | None,
    buying_power: float | None = None,
    positions: list[dict[str, Any]] | None = None,
    target_weights: dict[str, float] | None = None,
    streak: dict[str, Any] | None = None,
    strategy: str | None = None,
    extras: dict[str, Any] | None = None,
    path: Path = SNAPSHOT_PATH,
) -> None:
    """Write an atomic snapshot of the latest paper cycle.

    Best-effort: any IO error is logged at WARNING and swallowed. A
    failed snapshot must never break the live trading loop.
    """
    payload: dict[str, Any] = {
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "equity": _coerce(equity),
        "buying_power": _coerce(buying_power),
        "positions": positions or [],
        "target_weights": target_weights or {},
        "streak": streak or {},
        "strategy": strategy,
    }
    if extras:
        payload.update(extras)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
        ) as f:
            json.dump(payload, f, indent=2, default=str)
            tmp_name = f.name
        os.replace(tmp_name, path)
    except OSError as e:  # pragma: no cover - I/O failure path
        log.warning("snapshot write failed: %s", e)


def load_snapshot(path: Path = SNAPSHOT_PATH) -> dict[str, Any] | None:
    """Return the last snapshot dict or None if absent/unreadable."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _coerce(x: Any) -> float | None:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None
