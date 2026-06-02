"""Phase 33: tests for the AgentStatus narration sink.

The narration layer is pure observability \u2014 zero impact on trading
decisions \u2014 but it's the operator's primary window into what the
brain is doing. So the tests cover:

  * Schema stability (ACTORS list, AgentStatus shape)
  * Append-only write semantics with auto-timestamping
  * Latest-per-actor read semantics (history kept on disk, UI shows
    only the freshest row per lane)
  * Defensive degradation on missing / corrupt logs
  * Env-var override for test isolation
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from packages.agents.narration import (
    ACTORS,
    AgentStatus,
    emit,
    emit_many,
    read_latest,
)


# --- Schema pin --------------------------------------------------------------


def test_actors_list_is_stable() -> None:
    """If a future commit adds an agent, the cockpit panel column\n    order must be updated in lock-step. This test makes that explicit."""
    assert ACTORS == (
        "research",
        "strategy",
        "risk",
        "execution",
        "reflection",
        "curiosity",
        "discovery",
    )


def test_agent_status_frozen() -> None:
    """Frozen dataclass: mutation must raise. This prevents agents from\n    rewriting each other's status objects after emit."""
    s = AgentStatus(actor="research", working_on="x", waiting_on="y")
    with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
        s.working_on = "z"  # type: ignore[misc]


# --- Write semantics ---------------------------------------------------------


def test_emit_auto_timestamps(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AGENT_STATUS_PATH", str(tmp_path / "status.jsonl"))
    emit(AgentStatus(actor="research", working_on="x", waiting_on="y"))
    rows = (tmp_path / "status.jsonl").read_text().splitlines()
    assert len(rows) == 1
    parsed = json.loads(rows[0])
    assert parsed["ts"]  # auto-filled
    assert parsed["actor"] == "research"
    assert parsed["working_on"] == "x"


def test_emit_many_writes_one_row_per_status(tmp_path: Path, monkeypatch) -> None:
    """Batch variant must produce N lines for N statuses in stable order."""
    monkeypatch.setenv("AGENT_STATUS_PATH", str(tmp_path / "status.jsonl"))
    emit_many(
        [
            AgentStatus(actor="research", working_on="a", waiting_on=""),
            AgentStatus(actor="risk", working_on="b", waiting_on=""),
            AgentStatus(actor="execution", working_on="c", waiting_on=""),
        ]
    )
    rows = (tmp_path / "status.jsonl").read_text().splitlines()
    assert len(rows) == 3
    actors = [json.loads(r)["actor"] for r in rows]
    assert actors == ["research", "risk", "execution"]


def test_emit_many_empty_is_noop(tmp_path: Path, monkeypatch) -> None:
    """Calling emit_many([]) must not even create the file. Defensive\n    against degenerate sweeps that produced zero rows for some reason."""
    target = tmp_path / "status.jsonl"
    monkeypatch.setenv("AGENT_STATUS_PATH", str(target))
    emit_many([])
    assert not target.exists()


def test_emit_creates_parent_dir(tmp_path: Path, monkeypatch) -> None:
    """First call on a fresh deploy must mkdir its parent. Catches the\n    classic 'forgot to create data/cockpit/' bug on a clean checkout."""
    target = tmp_path / "deeply" / "nested" / "status.jsonl"
    monkeypatch.setenv("AGENT_STATUS_PATH", str(target))
    emit(AgentStatus(actor="research", working_on="x", waiting_on="y"))
    assert target.exists()


def test_unknown_actor_still_writes_but_warns(tmp_path: Path, monkeypatch, caplog) -> None:
    """If an agent typos its actor name we still want the row on disk\n    (so we can diagnose), but a warning must appear in the logs."""
    monkeypatch.setenv("AGENT_STATUS_PATH", str(tmp_path / "status.jsonl"))
    with caplog.at_level("WARNING"):
        emit(AgentStatus(actor="reserach", working_on="x", waiting_on=""))
    assert any("unknown actor" in rec.message for rec in caplog.records)
    assert (tmp_path / "status.jsonl").exists()


# --- Read semantics ----------------------------------------------------------


def test_read_latest_returns_one_per_actor(tmp_path: Path, monkeypatch) -> None:
    """Three sweeps for the same actor \u2014 read_latest must return only\n    the most recent. History is on disk but the UI is current-state only."""
    monkeypatch.setenv("AGENT_STATUS_PATH", str(tmp_path / "status.jsonl"))
    emit(AgentStatus(actor="research", working_on="sweep 1", waiting_on=""))
    emit(AgentStatus(actor="research", working_on="sweep 2", waiting_on=""))
    emit(AgentStatus(actor="research", working_on="sweep 3", waiting_on=""))
    emit(AgentStatus(actor="risk", working_on="approved 2", waiting_on=""))
    latest = read_latest()
    assert set(latest.keys()) == {"research", "risk"}
    assert latest["research"].working_on == "sweep 3"
    assert latest["risk"].working_on == "approved 2"


def test_read_latest_missing_log_returns_empty(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AGENT_STATUS_PATH", str(tmp_path / "never_existed.jsonl"))
    assert read_latest() == {}


def test_read_latest_skips_corrupt_lines(tmp_path: Path, monkeypatch) -> None:
    """One bad line in the middle of an otherwise-fine log must not\n    poison the whole read. The cockpit can't blank out because of one\n    truncated write."""
    target = tmp_path / "status.jsonl"
    monkeypatch.setenv("AGENT_STATUS_PATH", str(target))
    emit(AgentStatus(actor="research", working_on="ok", waiting_on=""))
    with target.open("a", encoding="utf-8") as fh:
        fh.write("{not valid json\n")
        fh.write("\n")  # empty line
    emit(AgentStatus(actor="risk", working_on="also ok", waiting_on=""))
    latest = read_latest()
    assert latest["research"].working_on == "ok"
    assert latest["risk"].working_on == "also ok"


def test_read_latest_preserves_hints_tuple(tmp_path: Path, monkeypatch) -> None:
    """Hints are read back as a tuple, not a list, to keep the contract\n    immutable end-to-end."""
    monkeypatch.setenv("AGENT_STATUS_PATH", str(tmp_path / "status.jsonl"))
    emit(
        AgentStatus(
            actor="research",
            working_on="x",
            waiting_on="y",
            hints=("loosen ATR floor", "wait for VWAP reclaim"),
        )
    )
    latest = read_latest()
    assert isinstance(latest["research"].hints, tuple)
    assert latest["research"].hints == ("loosen ATR floor", "wait for VWAP reclaim")

