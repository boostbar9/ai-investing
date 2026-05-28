"""Tests for structured error capture.

Goals:
* Secrets are redacted in the context payload (key-based + value-based)
* Tracebacks are truncated to a sane upper bound
* JSONL store is appended atomically and survives prune
* ``capture`` context manager re-raises by default
* ``load_recent_errors`` is robust to malformed lines
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from packages.healing import error_capture as ec_mod
from packages.healing.error_capture import (
    ErrorEvent,
    capture,
    load_recent_errors,
    record_error,
    redact,
)


@pytest.fixture
def isolated_errors(monkeypatch, tmp_path) -> Path:
    p = tmp_path / "errors.jsonl"
    monkeypatch.setattr(ec_mod, "ERRORS_PATH", p)
    return p


def test_redact_key_based() -> None:
    payload = {
        "user": "devin",
        "api_key": "sk-12345",
        "TOKEN": "abc",
        "nested": {"password": "hunter2", "ok": "value"},
    }
    out = redact(payload)
    assert out["user"] == "devin"
    assert out["api_key"] == "<redacted>"
    assert out["TOKEN"] == "<redacted>"
    assert out["nested"]["password"] == "<redacted>"
    assert out["nested"]["ok"] == "value"
    # original untouched
    assert payload["api_key"] == "sk-12345"


def test_redact_value_based_inline() -> None:
    s = "Authorization: Bearer abc123def followed by stuff"
    out = redact({"note": s})
    assert "<redacted>" in out["note"]
    assert "abc123def" not in out["note"]


def test_redact_lists_and_tuples() -> None:
    data = [{"secret": "x"}, ("token=abc", "ok")]
    out = redact(data)
    assert out[0]["secret"] == "<redacted>"
    assert "<redacted>" in out[1][0]


def test_record_error_writes_jsonl(isolated_errors: Path) -> None:
    try:
        raise ValueError("boom")
    except ValueError as exc:
        ev = record_error("test.where", exc, context={"foo": "bar"})
    assert isinstance(ev, ErrorEvent)
    rows = isolated_errors.read_text().strip().splitlines()
    assert len(rows) == 1
    row = json.loads(rows[0])
    assert row["exc_type"] == "ValueError"
    assert row["exc_message"] == "boom"
    assert row["where"] == "test.where"
    assert row["context"] == {"foo": "bar"}
    assert "ValueError" in row["traceback"]


def test_record_error_redacts_context(isolated_errors: Path) -> None:
    try:
        raise RuntimeError("x")
    except RuntimeError as exc:
        record_error("w", exc, context={"api_key": "secret-xyz", "ok": 1})
    row = json.loads(isolated_errors.read_text().splitlines()[0])
    assert row["context"]["api_key"] == "<redacted>"
    assert row["context"]["ok"] == 1


def test_capture_reraises_by_default(isolated_errors: Path) -> None:
    with pytest.raises(ValueError), capture("blk"):
        raise ValueError("nope")
    assert isolated_errors.exists()
    assert len(isolated_errors.read_text().splitlines()) == 1


def test_capture_can_swallow(isolated_errors: Path) -> None:
    with capture("blk", reraise=False):
        raise ValueError("eaten")
    # error is still recorded even when swallowed
    assert len(isolated_errors.read_text().splitlines()) == 1


def test_capture_handles_clean_block(isolated_errors: Path) -> None:
    with capture("blk"):
        x = 1 + 1
    assert x == 2
    assert not isolated_errors.exists()


def test_load_recent_errors_returns_last_n(isolated_errors: Path) -> None:
    for i in range(5):
        try:
            raise ValueError(f"e{i}")
        except ValueError as exc:
            record_error("w", exc)
    out = load_recent_errors(limit=3)
    assert len(out) == 3
    assert [e.exc_message for e in out] == ["e2", "e3", "e4"]


def test_load_recent_errors_skips_malformed(isolated_errors: Path) -> None:
    isolated_errors.parent.mkdir(parents=True, exist_ok=True)
    isolated_errors.write_text("not-json\n{}\n{\"ts\":\"x\",\"where\":\"w\",\"exc_type\":\"V\",\"exc_message\":\"m\",\"traceback\":\"t\"}\n")
    out = load_recent_errors(limit=10)
    assert len(out) == 1
    assert out[0].exc_type == "V"


def test_load_recent_errors_empty_file(isolated_errors: Path) -> None:
    assert load_recent_errors(limit=10) == []


def test_traceback_truncated(isolated_errors: Path, monkeypatch) -> None:
    # Force the truncator path by shrinking the cap below a real tb size.
    monkeypatch.setattr(ec_mod, "MAX_TRACEBACK_LINES", 4)

    def a() -> None:
        b()

    def b() -> None:
        c()

    def c() -> None:
        raise RuntimeError("deep")

    try:
        a()
    except RuntimeError as exc:
        ev = record_error("w", exc)
    lines = ev.traceback.splitlines()
    assert len(lines) <= 4 + 1  # +1 for the elision marker
    assert any("elided" in line for line in lines)


def test_prune_keeps_last_max_rows(isolated_errors: Path, monkeypatch) -> None:
    monkeypatch.setattr(ec_mod, "MAX_ROWS", 3)
    for i in range(7):
        try:
            raise ValueError(f"e{i}")
        except ValueError as exc:
            record_error("w", exc)
    rows = isolated_errors.read_text().strip().splitlines()
    assert len(rows) == 3
    msgs = [json.loads(r)["exc_message"] for r in rows]
    assert msgs == ["e4", "e5", "e6"]
