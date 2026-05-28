"""Background poller that delivers new flip events through notifiers.

This is the Phase 8 cockpit-side glue. The snapshot writer appends an
event row to ``data/cockpit/shadow_flips.jsonl`` the instant the shadow
soak flips ``shadow -> ready``. We want the user to hear about it via a
desktop toast / webhook the moment it happens, even when they don't
have the cockpit open.

The loop is intentionally cursor-driven (not event-driven on the
snapshot path) so:

* The notification side-effect can't slow down or break the snapshot
  builder.
* Restarts are safe: we resume from the persisted cursor, never
  redeliver, and never lose events that arrived while the loop was
  down.
* Future sinks (PagerDuty, Slack, etc.) can be added without changing
  ``snapshot.py``.

State lives in ``data/cockpit/shadow_notify_cursor.json``::

    {"last_ts": "2026-05-28T19:00:00+00:00",
     "last_count": 7,
     "last_delivery": {...}}

We dedupe by event timestamp. Two events at the same ts are extremely
unlikely (status flips at most once per snapshot tick) and the soak's
upward-edge guard prevents repeated emits, but we still index by
(ts, from, to) tuples for safety.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from packages.shadow.notifiers import Notifier, build_default_notifiers, dispatch_flip_event
from packages.shadow.notify import read_flip_events

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
CURSOR_PATH = REPO_ROOT / "data" / "cockpit" / "shadow_notify_cursor.json"


def _cursor_path() -> Path:
    """Indirect through ``sys.modules`` so tests can monkeypatch."""
    return Path(sys.modules[__name__].CURSOR_PATH)


# ---------------------------------------------------------------------------
# Cursor persistence
# ---------------------------------------------------------------------------


def _event_key(row: dict[str, Any]) -> tuple[str, str, str]:
    """Stable identity for a flip-event row, robust to ordering changes."""
    return (
        str(row.get("ts") or ""),
        str(row.get("from") or ""),
        str(row.get("to") or ""),
    )


def read_cursor() -> dict[str, Any]:
    """Return the persisted cursor or an empty dict on first run / corruption."""
    p = _cursor_path()
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        log.warning("notify cursor corrupt; resetting: %s", p)
        return {}
    if not isinstance(data, dict):
        return {}
    # Cast known fields defensively.
    delivered = data.get("delivered_keys")
    if not isinstance(delivered, list):
        delivered = []
    # Re-tuple-ify for in-memory use.
    data["delivered_keys"] = [tuple(k) if isinstance(k, list) else (k,) for k in delivered]
    return data


def write_cursor(cursor: dict[str, Any]) -> Path:
    """Atomically persist the cursor (tmp + replace)."""
    p = _cursor_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    # Convert tuple keys back to lists for JSON serialization.
    serializable = dict(cursor)
    if "delivered_keys" in serializable:
        serializable["delivered_keys"] = [list(t) for t in serializable["delivered_keys"]]
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(serializable, sort_keys=True, indent=2), encoding="utf-8")
    tmp.replace(p)
    return p


# ---------------------------------------------------------------------------
# One-shot tick: compute undelivered events, dispatch, persist
# ---------------------------------------------------------------------------


# Keep at most this many delivered keys in the cursor file. Flip events
# are bounded to MAX_FLIP_ROWS=5000 in notify.py so 5000 here matches.
MAX_CURSOR_KEYS = 5000


def tick_once(
    notifiers: list[Notifier] | None = None,
    *,
    reader: Callable[[], list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Process any flip events not yet delivered. Returns a summary dict.

    Safe to call repeatedly; idempotent because we dedupe via the
    cursor's ``delivered_keys`` set.
    """
    read = reader or (lambda: read_flip_events())
    cursor = read_cursor()
    delivered: set[tuple[str, str, str]] = {
        tuple(k) for k in cursor.get("delivered_keys", []) if isinstance(k, tuple)
    }
    try:
        rows = read() or []
    except Exception as exc:
        log.warning("flip-events read failed: %s", exc)
        return {"ok": False, "delivered": 0, "error": str(exc)}

    new_rows = [r for r in rows if _event_key(r) not in delivered]
    if not new_rows:
        return {"ok": True, "delivered": 0, "scanned": len(rows)}

    sinks = notifiers if notifiers is not None else build_default_notifiers()

    summaries: list[dict[str, Any]] = []
    for row in new_rows:
        results = dispatch_flip_event(row, sinks)
        delivered.add(_event_key(row))
        summaries.append({"event": row, "results": results})

    # Cap delivered_keys to avoid unbounded growth.
    delivered_list = list(delivered)
    if len(delivered_list) > MAX_CURSOR_KEYS:
        # Keep the most recent N by ts ordering (lexicographic ISO sort).
        delivered_list = sorted(delivered_list)[-MAX_CURSOR_KEYS:]

    cursor["delivered_keys"] = delivered_list
    if summaries:
        last = summaries[-1]
        cursor["last_ts"] = last["event"].get("ts")
        cursor["last_event"] = last["event"]
        cursor["last_results"] = last["results"]
    cursor["last_count"] = len(rows)
    write_cursor(cursor)

    return {
        "ok": True,
        "delivered": len(summaries),
        "scanned": len(rows),
        "summaries": summaries,
    }


# ---------------------------------------------------------------------------
# Long-lived loop (lifespan task)
# ---------------------------------------------------------------------------


async def flip_notify_loop(  # pragma: no cover - long-lived task
    *,
    notifiers: list[Notifier] | None = None,
    poll_seconds: float = 60.0,
) -> None:
    """Poll for new flip events every ``poll_seconds`` and dispatch them.

    Mirrors :func:`packages.cockpit.automation.watchdog_loop` -- catches
    every non-cancellation exception, re-raises ``CancelledError`` so
    FastAPI shutdown stays clean.
    """
    sinks = notifiers if notifiers is not None else build_default_notifiers()
    while True:
        try:
            tick_once(sinks)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("flip_notify_loop tick failed: %s", exc)
        try:
            await asyncio.sleep(poll_seconds)
        except asyncio.CancelledError:
            raise


# Re-export for tests that want to short-circuit the loop sleep.
__all__ = [
    "CURSOR_PATH",
    "MAX_CURSOR_KEYS",
    "flip_notify_loop",
    "read_cursor",
    "tick_once",
    "write_cursor",
]


# Keep a defensive contextlib import so future use stays cheap.
_ = contextlib
