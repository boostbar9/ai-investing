"""Watchdog/healing snapshot tests."""

from __future__ import annotations

from packages.healing.error_capture import ErrorEvent
from packages.healing.watchdog_integration import (
    HealingSnapshot,
    snapshot_for_halt,
)


def _ev(exc_type: str, msg: str = "", tb: str = "", ts: str = "2026-05-28T00:00:00+00:00") -> ErrorEvent:
    return ErrorEvent(ts=ts, where="w", exc_type=exc_type, exc_message=msg, traceback=tb)


def test_snapshot_empty() -> None:
    snap = snapshot_for_halt(events=[])
    assert isinstance(snap, HealingSnapshot)
    assert snap.total_errors == 0
    assert snap.categories == {}
    assert snap.patchable_count == 0
    assert snap.most_recent == []
    assert snap.halt is None


def test_snapshot_counts_categories() -> None:
    events = [
        _ev("NotImplementedError", "foo"),
        _ev("NotImplementedError", "bar"),
        _ev("AttributeError", "'X' has no attribute 'y'", 'File "/packages/x/y.py" line 1'),
        _ev("TypeError", "bad"),
        _ev("RuntimeError", "completely random"),
    ]
    snap = snapshot_for_halt(events=events)
    assert snap.total_errors == 5
    assert snap.categories["missing_stub"] == 2
    assert snap.categories["attribute_error"] == 1
    assert snap.categories["type_error"] == 1
    assert snap.categories["unknown"] == 1
    # missing_stub (2) + attribute_error (1) = 3 patchable
    assert snap.patchable_count == 3


def test_snapshot_includes_halt_payload() -> None:
    halt = {"breach": True, "current_drawdown": 0.09}
    snap = snapshot_for_halt(events=[_ev("ValueError", "x")], halt=halt)
    assert snap.halt == halt


def test_snapshot_most_recent_reverse_ordered_and_capped() -> None:
    events = [_ev("ValueError", f"e{i}", ts=f"2026-05-28T00:00:{i:02d}+00:00") for i in range(15)]
    snap = snapshot_for_halt(events=events)
    assert len(snap.most_recent) == 10
    # newest first
    assert snap.most_recent[0]["exc_message"] == "e14"
    assert snap.most_recent[-1]["exc_message"] == "e5"
    # Each entry has a category
    assert all("category" in entry for entry in snap.most_recent)


def test_snapshot_truncates_long_messages() -> None:
    big = "x" * 1000
    snap = snapshot_for_halt(events=[_ev("ValueError", big)])
    assert len(snap.most_recent[0]["exc_message"]) == 240
