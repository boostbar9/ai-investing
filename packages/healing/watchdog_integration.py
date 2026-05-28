"""Glue between the cockpit watchdog and the healing subsystem.

When the watchdog records a halt (drawdown breach), the operator wants
to know what's *recently* broken -- not just the equity number. This
module produces a small ``HealingSnapshot`` that the cockpit/UI can
surface alongside the halt record.

We deliberately keep this read-only: the snapshot does NOT mutate halt
state or open PRs. That's a separate explicit operator action.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from packages.healing.classifier import classify, is_patchable
from packages.healing.error_capture import ErrorEvent, load_recent_errors


@dataclass(frozen=True)
class HealingSnapshot:
    total_errors: int
    categories: dict[str, int]
    patchable_count: int
    most_recent: list[dict[str, Any]] = field(default_factory=list)
    halt: dict[str, Any] | None = None


def _summarize(events: list[ErrorEvent], halt: dict[str, Any] | None) -> HealingSnapshot:
    cats: Counter[str] = Counter()
    patchable = 0
    for e in events:
        c = classify(e)
        cats[c.value] += 1
        if is_patchable(c):
            patchable += 1
    most_recent = [
        {
            "ts": e.ts,
            "where": e.where,
            "exc_type": e.exc_type,
            "exc_message": e.exc_message[:240],
            "category": classify(e).value,
        }
        for e in events[-10:][::-1]
    ]
    return HealingSnapshot(
        total_errors=len(events),
        categories=dict(cats),
        patchable_count=patchable,
        most_recent=most_recent,
        halt=halt,
    )


def snapshot_for_halt(
    *,
    halt: dict[str, Any] | None = None,
    limit: int = 100,
    events: list[ErrorEvent] | None = None,
) -> HealingSnapshot:
    """Build a snapshot for the cockpit/UI.

    ``events`` is an injection seam for tests; production callers pass
    nothing and we read from the JSONL store.
    """
    if events is None:
        events = load_recent_errors(limit=limit)
    return _summarize(events, halt)


def snapshot_from_watchdog() -> HealingSnapshot:
    """Read the active halt (if any) and return a snapshot.

    Imports ``packages.cockpit.watchdog`` lazily so importing
    ``packages.healing`` never drags the cockpit module in.
    """
    try:
        from packages.cockpit import watchdog  # local import
    except ImportError:
        return snapshot_for_halt(halt=None)
    halt = watchdog.read_halt() if hasattr(watchdog, "read_halt") else None
    return snapshot_for_halt(halt=halt)
