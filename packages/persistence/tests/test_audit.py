"""Tests for the per-decision audit log (§17, task 7)."""

from __future__ import annotations

import json
from pathlib import Path

from packages.persistence.audit import log_decision


def _read_lines(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_log_decision_appends_record(tmp_path: Path) -> None:
    path = tmp_path / "decisions.jsonl"
    log_decision(
        decision_id="abc", agent="strategy", prompt="P", raw_response={"x": 1},
        validation_ok=True, path=path,
    )
    records = _read_lines(path)
    assert len(records) == 1
    r = records[0]
    assert r["decision_id"] == "abc"
    assert r["agent"] == "strategy"
    assert r["validation_ok"] is True
    assert r["raw_response"] == {"x": 1}
    assert r["attempt"] == 1


def test_log_decision_captures_validation_error(tmp_path: Path) -> None:
    path = tmp_path / "decisions.jsonl"
    log_decision(
        decision_id="abc", agent="risk", prompt="P", raw_response="bad",
        validation_ok=False, validation_error="missing approved", attempt=2,
        extra={"fallback": "validation_error"}, path=path,
    )
    r = _read_lines(path)[0]
    assert r["validation_ok"] is False
    assert r["validation_error"] == "missing approved"
    assert r["attempt"] == 2
    assert r["extra"]["fallback"] == "validation_error"


def test_log_decision_appends_multiple(tmp_path: Path) -> None:
    path = tmp_path / "decisions.jsonl"
    for i in range(3):
        log_decision(
            decision_id=f"d{i}", agent="research", prompt="P",
            raw_response={"i": i}, validation_ok=True, path=path,
        )
    assert len(_read_lines(path)) == 3
