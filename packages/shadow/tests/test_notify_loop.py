"""Tests for the flip-event poller (cursor + dispatch idempotency)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from packages.shadow import notify_loop as loop_mod
from packages.shadow.notify_loop import (
    MAX_CURSOR_KEYS,
    _event_key,
    read_cursor,
    tick_once,
    write_cursor,
)


@dataclass
class _Recorder:
    name: str = "recorder"
    calls: list[dict[str, Any]] = field(default_factory=list)

    def notify(self, event: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(event)
        return {"ok": True, "backend": self.name, "delivered": True}


def _make_event(ts: str, from_status: str = "shadow", to_status: str = "ready") -> dict[str, Any]:
    return {
        "ts": ts,
        "from": from_status,
        "to": to_status,
        "streak_days": 14,
        "reasons": [],
    }


@pytest.fixture(autouse=True)
def _isolate_cursor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the cursor file into tmp_path for every test."""
    cursor = tmp_path / "cursor.json"
    monkeypatch.setattr(loop_mod, "CURSOR_PATH", cursor)
    return cursor


# ---------------------------------------------------------------------------
# event_key
# ---------------------------------------------------------------------------


def test_event_key_uses_ts_from_to() -> None:
    row = _make_event("2026-05-28T19:00:00+00:00")
    assert _event_key(row) == ("2026-05-28T19:00:00+00:00", "shadow", "ready")


def test_event_key_handles_missing_fields() -> None:
    assert _event_key({}) == ("", "", "")


# ---------------------------------------------------------------------------
# cursor persistence
# ---------------------------------------------------------------------------


def test_read_cursor_missing_returns_empty(_isolate_cursor: Path) -> None:
    assert read_cursor() == {}


def test_read_cursor_corrupt_returns_empty(_isolate_cursor: Path) -> None:
    _isolate_cursor.parent.mkdir(parents=True, exist_ok=True)
    _isolate_cursor.write_text("{not json", encoding="utf-8")
    assert read_cursor() == {}


def test_write_then_read_cursor_roundtrip(_isolate_cursor: Path) -> None:
    keys = [("2026-05-28T19:00:00+00:00", "shadow", "ready")]
    write_cursor({"delivered_keys": keys, "last_ts": "x"})
    out = read_cursor()
    assert out["last_ts"] == "x"
    # Tuples are restored on read.
    assert out["delivered_keys"] == keys


def test_read_cursor_non_dict_payload(_isolate_cursor: Path) -> None:
    _isolate_cursor.parent.mkdir(parents=True, exist_ok=True)
    _isolate_cursor.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    assert read_cursor() == {}


def test_read_cursor_tolerates_missing_delivered_keys(_isolate_cursor: Path) -> None:
    _isolate_cursor.parent.mkdir(parents=True, exist_ok=True)
    _isolate_cursor.write_text(json.dumps({"last_ts": "z"}), encoding="utf-8")
    out = read_cursor()
    assert out["delivered_keys"] == []
    assert out["last_ts"] == "z"


def test_write_cursor_atomic_no_dangling_tmp(_isolate_cursor: Path) -> None:
    write_cursor({"delivered_keys": []})
    assert _isolate_cursor.exists()
    # Tmp file should be renamed away.
    tmp = _isolate_cursor.with_suffix(_isolate_cursor.suffix + ".tmp")
    assert not tmp.exists()


# ---------------------------------------------------------------------------
# tick_once
# ---------------------------------------------------------------------------


def test_tick_once_no_events_returns_zero(_isolate_cursor: Path) -> None:
    rec = _Recorder()
    result = tick_once([rec], reader=lambda: [])
    assert result == {"ok": True, "delivered": 0, "scanned": 0}
    assert rec.calls == []


def test_tick_once_dispatches_new_events(_isolate_cursor: Path) -> None:
    rec = _Recorder()
    events = [
        _make_event("2026-05-27T19:00:00+00:00"),
        _make_event("2026-05-28T19:00:00+00:00"),
    ]
    result = tick_once([rec], reader=lambda: events)
    assert result["delivered"] == 2
    assert rec.calls == events
    cursor = read_cursor()
    keys = set(cursor["delivered_keys"])
    assert keys == {_event_key(e) for e in events}
    assert cursor["last_ts"] == "2026-05-28T19:00:00+00:00"


def test_tick_once_idempotent_on_replay(_isolate_cursor: Path) -> None:
    rec = _Recorder()
    events = [_make_event("2026-05-28T19:00:00+00:00")]
    tick_once([rec], reader=lambda: events)
    tick_once([rec], reader=lambda: events)
    tick_once([rec], reader=lambda: events)
    assert rec.calls == events  # Only delivered once.


def test_tick_once_delivers_only_unseen_events(_isolate_cursor: Path) -> None:
    rec = _Recorder()
    first = _make_event("2026-05-27T19:00:00+00:00")
    second = _make_event("2026-05-28T19:00:00+00:00")
    tick_once([rec], reader=lambda: [first])
    rec.calls.clear()
    tick_once([rec], reader=lambda: [first, second])
    assert rec.calls == [second]


def test_tick_once_survives_corrupt_cursor(_isolate_cursor: Path) -> None:
    _isolate_cursor.parent.mkdir(parents=True, exist_ok=True)
    _isolate_cursor.write_text("garbage{", encoding="utf-8")
    rec = _Recorder()
    event = _make_event("2026-05-28T19:00:00+00:00")
    result = tick_once([rec], reader=lambda: [event])
    assert result["delivered"] == 1
    # Cursor file rewritten cleanly.
    cursor = read_cursor()
    assert cursor["delivered_keys"]


def test_tick_once_handles_reader_exception(_isolate_cursor: Path) -> None:
    def boom() -> list[dict[str, Any]]:
        raise OSError("disk gone")

    result = tick_once([_Recorder()], reader=boom)
    assert result["ok"] is False
    assert "disk gone" in result["error"]


def test_tick_once_persists_results_summary(_isolate_cursor: Path) -> None:
    rec = _Recorder()
    event = _make_event("2026-05-28T19:00:00+00:00")
    tick_once([rec], reader=lambda: [event])
    cursor = read_cursor()
    assert cursor["last_event"] == event
    assert cursor["last_results"][0]["backend"] == "recorder"
    assert cursor["last_count"] == 1


def test_tick_once_caps_delivered_keys(
    _isolate_cursor: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(loop_mod, "MAX_CURSOR_KEYS", 3)
    rec = _Recorder()
    events = [_make_event(f"2026-05-{20 + i:02d}T19:00:00+00:00") for i in range(5)]
    tick_once([rec], reader=lambda: events)
    cursor = read_cursor()
    # Cap honored.
    assert len(cursor["delivered_keys"]) == 3
    # The newest three are retained.
    retained = {tuple(k) for k in cursor["delivered_keys"]}
    expected = {_event_key(e) for e in events[-3:]}
    assert retained == expected


def test_max_cursor_keys_default_is_bounded() -> None:
    # Sanity guard: a regression that drops the bound would let the cursor
    # grow unboundedly on a long-running soak.
    assert MAX_CURSOR_KEYS >= 1000
    assert MAX_CURSOR_KEYS <= 100_000


# ---------------------------------------------------------------------------
# default reader integration
# ---------------------------------------------------------------------------


def test_tick_once_uses_read_flip_events_when_no_reader(
    _isolate_cursor: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = [_make_event("2026-05-28T19:00:00+00:00")]
    monkeypatch.setattr(loop_mod, "read_flip_events", lambda: events)
    rec = _Recorder()
    result = tick_once([rec])
    assert result["delivered"] == 1
    assert rec.calls == events
