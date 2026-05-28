"""Greenlight flip detection + persistent event log.

When the shadow dashboard transitions from ``status="shadow"`` to
``status="ready"`` we want to:

* Record an immutable event so the cockpit can surface a one-time
  "shadow soak complete -- ready for live promotion" banner.
* Give the autopilot a hook to fire a desktop notification / Telegram
  message without the snapshot module owning that policy directly.

The detector itself is pure (no I/O); the persistence layer is a
single ``append_flip_event`` that does an atomic JSONL append. This
mirrors the rest of the data/ files (errors.jsonl, etc.) so the
healing snapshot can correlate flip events with error spikes.

Status file shape (``data/cockpit/shadow_flips.jsonl``)::

    {"ts": "2026-05-28T19:00:00+00:00", "from": "shadow", "to": "ready",
     "streak_days": 14, "reasons": [...]}
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "cockpit"
FLIPS_PATH = DATA_DIR / "shadow_flips.jsonl"

# Bound the file so a misbehaving caller can't blow up the cockpit's
# tail-read on /api/shadow/flip-events. 5000 is consistent with
# packages.healing.error_capture.MAX_ROWS.
MAX_FLIP_ROWS = 5000


@dataclass(frozen=True)
class FlipEvent:
    ts: str  # ISO-8601 UTC
    from_status: str
    to_status: str
    streak_days: int
    reasons: list[str] = field(default_factory=list)

    def to_row(self) -> dict[str, Any]:
        # Use "from" / "to" on disk (Python keyword-safe via dict literal)
        # but keep field names Pythonic for the dataclass.
        d = asdict(self)
        d["from"] = d.pop("from_status")
        d["to"] = d.pop("to_status")
        return d


def detect_flip(
    prev_payload: dict[str, Any] | None,
    new_status: str,
    streak_days: int,
    reasons: list[str] | None = None,
    *,
    now: datetime | None = None,
) -> FlipEvent | None:
    """Return a FlipEvent iff the verdict transitioned SHADOW -> READY.

    ``prev_payload`` is the JSON dict last written to shadow_status.json
    (or None on the very first evaluation). We only emit on the upward
    edge to "ready"; ready -> shadow regressions are interesting but not
    a promote-eligible event and the dashboard already shows them via
    the live streak.
    """
    prev = (prev_payload or {}).get("status", "shadow")
    if new_status != "ready":
        return None
    if prev == "ready":
        return None
    return FlipEvent(
        ts=(now or datetime.now(UTC)).isoformat(),
        from_status=prev,
        to_status="ready",
        streak_days=streak_days,
        reasons=list(reasons or []),
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _flips_path() -> Path:
    return Path(sys.modules[__name__].FLIPS_PATH)


def append_flip_event(event: FlipEvent) -> dict[str, Any]:
    """Atomically append a flip event to the JSONL log.

    Reads existing rows, appends, then rewrites via a tmp-then-rename
    so a partial write never corrupts the file. Old rows are trimmed
    to MAX_FLIP_ROWS from the tail.
    """
    target = _flips_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = read_flip_events()
    rows.append(event.to_row())
    if len(rows) > MAX_FLIP_ROWS:
        rows = rows[-MAX_FLIP_ROWS:]
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n",
        encoding="utf-8",
    )
    tmp.replace(target)
    return event.to_row()


def read_flip_events(limit: int | None = None) -> list[dict[str, Any]]:
    """Return flip events in chronological order; tolerates corrupt lines."""
    target = _flips_path()
    if not target.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # skip corrupt rows, keep going
    if limit is not None and limit > 0:
        rows = rows[-limit:]
    return rows
