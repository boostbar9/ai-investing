"""Tests for the cockpit error log module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from packages.cockpit import errors as err_log


@pytest.fixture
def isolated_log(monkeypatch, tmp_path: Path):
    """Redirect ERROR_LOG to a per-test temp file."""
    log_file = tmp_path / "errors.jsonl"
    monkeypatch.setattr(err_log, "ERROR_LOG", log_file)
    yield log_file


def test_record_error_writes_entry(isolated_log: Path) -> None:
    entry = err_log.record_error(
        source="paper_trade", message="thing went wrong"
    )
    assert entry["source"] == "paper_trade"
    assert entry["severity"] == "error"
    assert isolated_log.exists()
    lines = isolated_log.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["message"] == "thing went wrong"
    assert "ts" in parsed


def test_record_error_with_detail_and_context(isolated_log: Path) -> None:
    err_log.record_error(
        source="updater",
        message="git pull failed",
        severity="warning",
        detail="fatal: not a git repository",
        context={"cwd": "/tmp", "rc": 128},
    )
    entries = err_log.list_errors()
    assert len(entries) == 1
    e = entries[0]
    assert e["severity"] == "warning"
    assert e["detail"].startswith("fatal:")
    assert e["context"]["rc"] == 128


def test_record_exception_captures_traceback(isolated_log: Path) -> None:
    try:
        raise ValueError("bad value")
    except ValueError as exc:
        err_log.record_exception("test", exc, route="/api/bogus")
    entries = err_log.list_errors()
    assert len(entries) == 1
    e = entries[0]
    assert "ValueError" in e["message"]
    assert "bad value" in e["message"]
    assert "Traceback" in e["detail"]
    assert e["context"]["route"] == "/api/bogus"


def test_list_errors_newest_first(isolated_log: Path) -> None:
    err_log.record_error(source="a", message="first")
    err_log.record_error(source="b", message="second")
    err_log.record_error(source="c", message="third")
    entries = err_log.list_errors()
    assert [e["message"] for e in entries] == ["third", "second", "first"]


def test_list_errors_filter_by_severity(isolated_log: Path) -> None:
    err_log.record_error(source="x", message="warn1", severity="warning")
    err_log.record_error(source="x", message="err1", severity="error")
    err_log.record_error(source="x", message="info1", severity="info")
    warnings = err_log.list_errors(severity="warning")
    assert len(warnings) == 1
    assert warnings[0]["message"] == "warn1"


def test_list_errors_respects_limit(isolated_log: Path) -> None:
    for i in range(10):
        err_log.record_error(source="s", message=f"m{i}")
    out = err_log.list_errors(limit=3)
    assert len(out) == 3
    # newest first => m9, m8, m7
    assert [e["message"] for e in out] == ["m9", "m8", "m7"]


def test_count_unresolved(isolated_log: Path) -> None:
    err_log.record_error(source="s", message="a", severity="error")
    err_log.record_error(source="s", message="b", severity="error")
    err_log.record_error(source="s", message="c", severity="warning")
    err_log.record_error(source="s", message="d", severity="info")
    counts = err_log.count_unresolved()
    assert counts["error"] == 2
    assert counts["warning"] == 1
    assert counts["info"] == 1
    assert counts["total"] == 4


def test_clear_empties_log(isolated_log: Path) -> None:
    err_log.record_error(source="s", message="x")
    err_log.record_error(source="s", message="y")
    cleared = err_log.clear()
    assert cleared == 2
    assert err_log.list_errors() == []
    assert err_log.count_unresolved()["total"] == 0


def test_to_markdown_empty(isolated_log: Path) -> None:
    md = err_log.to_markdown()
    assert "No errors logged" in md


def test_to_markdown_includes_entry_fields(isolated_log: Path) -> None:
    err_log.record_error(
        source="broker",
        message="alpaca 404",
        severity="error",
        detail="GET /v2/v2/account returned 404",
        context={"url": "https://paper-api.alpaca.markets/v2"},
    )
    md = err_log.to_markdown()
    assert "alpaca 404" in md
    assert "broker" in md
    assert "/v2/v2/account" in md


def test_max_entries_enforced(isolated_log: Path, monkeypatch) -> None:
    """The log must self-prune to MAX_ENTRIES to avoid disk bloat."""
    monkeypatch.setattr(err_log, "MAX_ENTRIES", 5)
    for i in range(12):
        err_log.record_error(source="s", message=f"m{i}")
    entries = err_log.list_errors()
    assert len(entries) == 5
    # Oldest dropped: only m7..m11 should remain (newest first)
    assert [e["message"] for e in entries] == ["m11", "m10", "m9", "m8", "m7"]


def test_list_errors_returns_empty_when_no_file(isolated_log: Path) -> None:
    # File does not yet exist
    assert not isolated_log.exists()
    assert err_log.list_errors() == []
    assert err_log.count_unresolved()["total"] == 0


def test_corrupt_lines_are_skipped(isolated_log: Path) -> None:
    err_log.record_error(source="s", message="good1")
    # Inject a garbage line in the middle
    with isolated_log.open("a", encoding="utf-8") as f:
        f.write("{not valid json\n")
    err_log.record_error(source="s", message="good2")
    entries = err_log.list_errors()
    msgs = [e["message"] for e in entries]
    assert "good1" in msgs
    assert "good2" in msgs
