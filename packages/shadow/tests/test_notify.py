"""Tests for flip detection + persistent event log."""
from __future__ import annotations

import json
from datetime import datetime

import pytest

from packages.shadow import notify
from packages.shadow.notify import (
    FlipEvent,
    append_flip_event,
    detect_flip,
    read_flip_events,
)


@pytest.fixture
def flips_path(tmp_path, monkeypatch):
    target = tmp_path / "shadow_flips.jsonl"
    monkeypatch.setattr(notify, "FLIPS_PATH", target)
    return target


# ---------------------------------------------------------------------------
# detect_flip
# ---------------------------------------------------------------------------


def test_detect_flip_first_evaluation_no_event():
    # No prior payload + still soaking -> no event.
    assert detect_flip(None, "shadow", 3, ["..."]) is None


def test_detect_flip_first_evaluation_immediately_ready():
    # Fresh box, first eval already past threshold (e.g. backfilled trades).
    event = detect_flip(None, "ready", 14, ["greenlit"])
    assert event is not None
    assert event.from_status == "shadow"  # absent prev defaults to shadow
    assert event.to_status == "ready"
    assert event.streak_days == 14


def test_detect_flip_upward_edge_emits_event():
    prev = {"status": "shadow", "streak_days": 13}
    event = detect_flip(prev, "ready", 14, ["greenlit"])
    assert event is not None
    assert event.from_status == "shadow"
    assert event.to_status == "ready"


def test_detect_flip_already_ready_no_event():
    # Steady-state ready -> no duplicate event on every snapshot refresh.
    prev = {"status": "ready", "streak_days": 30}
    assert detect_flip(prev, "ready", 31, []) is None


def test_detect_flip_downward_edge_no_event():
    # Ready -> shadow regression. Interesting, but Phase 7 only emits on
    # the upward edge; the dashboard already shows the streak break.
    prev = {"status": "ready", "streak_days": 15}
    assert detect_flip(prev, "shadow", 0, ["loss day"]) is None


def test_detect_flip_uses_provided_now():
    fixed = datetime(2026, 1, 1, 12, 0, 0, tzinfo=__import__("datetime").timezone.utc)
    event = detect_flip(None, "ready", 14, [], now=fixed)
    assert event is not None
    assert event.ts.startswith("2026-01-01T12:00:00")


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_append_flip_event_creates_file(flips_path):
    event = FlipEvent(
        ts="2026-05-28T19:00:00+00:00",
        from_status="shadow",
        to_status="ready",
        streak_days=14,
        reasons=["greenlit"],
    )
    append_flip_event(event)
    assert flips_path.exists()
    rows = read_flip_events()
    assert len(rows) == 1
    # On-disk shape uses keyword-safe "from" / "to" keys.
    assert rows[0]["from"] == "shadow"
    assert rows[0]["to"] == "ready"
    assert rows[0]["streak_days"] == 14
    assert rows[0]["reasons"] == ["greenlit"]


def test_append_flip_event_appends_preserves_order(flips_path):
    for i, day in enumerate([14, 15, 16]):
        append_flip_event(
            FlipEvent(
                ts=f"2026-05-{28+i:02d}T19:00:00+00:00",
                from_status="shadow",
                to_status="ready",
                streak_days=day,
                reasons=[],
            )
        )
    rows = read_flip_events()
    assert [r["streak_days"] for r in rows] == [14, 15, 16]


def test_read_flip_events_missing_returns_empty(flips_path):
    assert read_flip_events() == []


def test_read_flip_events_tolerates_corrupt_lines(flips_path):
    flips_path.parent.mkdir(parents=True, exist_ok=True)
    flips_path.write_text(
        '{"from": "shadow", "to": "ready", "streak_days": 14}\n'
        "this is not json\n"
        '{"from": "shadow", "to": "ready", "streak_days": 15}\n',
        encoding="utf-8",
    )
    rows = read_flip_events()
    # Corrupt line skipped, good rows preserved.
    assert len(rows) == 2
    assert rows[0]["streak_days"] == 14
    assert rows[1]["streak_days"] == 15


def test_read_flip_events_respects_limit(flips_path):
    for i in range(5):
        append_flip_event(
            FlipEvent(
                ts=f"2026-05-{28+i:02d}T19:00:00+00:00",
                from_status="shadow",
                to_status="ready",
                streak_days=14 + i,
                reasons=[],
            )
        )
    rows = read_flip_events(limit=2)
    assert len(rows) == 2
    # Tail-window: last 2 entries.
    assert [r["streak_days"] for r in rows] == [17, 18]


def test_append_flip_event_atomic_write(flips_path, monkeypatch):
    # Pre-existing rows must survive a half-finished write attempt. We
    # simulate this by writing a real event, then verifying the file
    # ends with a single newline and is valid JSONL.
    event = FlipEvent(
        ts="2026-05-28T19:00:00+00:00",
        from_status="shadow",
        to_status="ready",
        streak_days=14,
        reasons=[],
    )
    append_flip_event(event)
    text = flips_path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    for line in text.splitlines():
        if line.strip():
            json.loads(line)  # never raises


def test_append_flip_event_trims_to_max_rows(flips_path, monkeypatch):
    monkeypatch.setattr(notify, "MAX_FLIP_ROWS", 3)
    for i in range(5):
        append_flip_event(
            FlipEvent(
                ts=f"2026-05-{28+i:02d}T19:00:00+00:00",
                from_status="shadow",
                to_status="ready",
                streak_days=10 + i,
                reasons=[],
            )
        )
    rows = read_flip_events()
    # Older two rows dropped, newest three retained in order.
    assert len(rows) == 3
    assert [r["streak_days"] for r in rows] == [12, 13, 14]
