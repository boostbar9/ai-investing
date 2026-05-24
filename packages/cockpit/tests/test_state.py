"""Tests for cockpit state load/save semantics."""

from __future__ import annotations

import json
from pathlib import Path

from packages.cockpit.state import CockpitState, load_state, record_action, save_state


def test_load_state_returns_defaults_when_file_missing(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    s = load_state(path)
    assert s.paused is False
    assert s.regime_override == "auto"
    assert s.paused_strategies == []
    assert s.last_action == ""


def test_save_then_load_roundtrips_all_fields(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    original = CockpitState(
        paused=True,
        regime_override="bear",
        paused_strategies=["trend-following"],
        last_action="pause from cockpit",
        last_action_at="2026-05-23T19:00:00+00:00",
    )
    save_state(original, path)
    loaded = load_state(path)
    assert loaded.paused is True
    assert loaded.regime_override == "bear"
    assert loaded.paused_strategies == ["trend-following"]
    assert loaded.last_action == "pause from cockpit"
    assert loaded.last_action_at == "2026-05-23T19:00:00+00:00"


def test_load_state_tolerates_corrupt_json(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    path.write_text("{not valid json")
    s = load_state(path)
    assert s.paused is False
    assert s.regime_override == "auto"


def test_save_state_is_atomic_no_partial_file(tmp_path: Path) -> None:
    """save_state writes to a temp file then renames - the destination is never
    left half-written even if we read between writes."""
    path = tmp_path / "state.json"
    save_state(CockpitState(paused=True), path)
    # No leftover .tmp files in the directory.
    leftover = [p for p in tmp_path.iterdir() if p.suffix == ".tmp"]
    assert leftover == []
    # Destination is valid JSON.
    json.loads(path.read_text())


def test_record_action_sets_timestamp(tmp_path: Path) -> None:
    s = CockpitState()
    s = record_action(s, "force-flatten requested")
    assert s.last_action == "force-flatten requested"
    assert s.last_action_at != ""
    # ISO 8601 with timezone suffix
    assert "T" in s.last_action_at


def test_trading_mode_defaults_to_paper(tmp_path: Path) -> None:
    """Brand-new state must default to safe paper mode."""
    s = load_state(tmp_path / "missing.json")
    assert s.trading_mode == "paper"


def test_trading_mode_roundtrips_live(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    save_state(CockpitState(trading_mode="live"), path)
    s = load_state(path)
    assert s.trading_mode == "live"


def test_trading_mode_invalid_value_falls_back_to_paper(tmp_path: Path) -> None:
    """A corrupt or unknown mode in the file must not let live slip through."""
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"trading_mode": "yolo"}))
    s = load_state(path)
    assert s.trading_mode == "paper"


def test_trading_mode_missing_field_in_old_state_file(tmp_path: Path) -> None:
    """State files written before the mode field existed must still load."""
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"paused": True, "regime_override": "bull"}))
    s = load_state(path)
    assert s.trading_mode == "paper"
    assert s.paused is True
    assert s.regime_override == "bull"
