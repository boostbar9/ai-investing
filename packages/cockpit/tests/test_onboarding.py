"""Tests for the first-boot onboarding state module.

Mirrors the contract of ``test_state.py``: defaults are sane, persistence
is atomic, corruption falls back to defaults, and individual mutators do
exactly what they advertise.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from packages.cockpit.onboarding import (
    DEFAULT_FLOAT_CAP_USD,
    OnboardingState,
    accept_disclaimer,
    load_onboarding,
    mark_completed,
    mark_started,
    reset,
    save_onboarding,
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_default_state_is_uncompleted() -> None:
    s = OnboardingState()
    assert s.completed is False
    assert s.robinhood_status == "unknown"
    assert s.rh_mode == "shadow"
    assert s.live_float_cap_usd == DEFAULT_FLOAT_CAP_USD
    assert s.accepted_disclaimer_at == ""
    assert s.wizard_started_at == ""
    assert s.wizard_completed_at == ""
    assert s.display_name == ""


def test_default_float_cap_is_300() -> None:
    """User's stated comfort floor is $300; if this ever changes we want
    to know about it via test failure rather than silent drift."""
    assert DEFAULT_FLOAT_CAP_USD == 300.0


# ---------------------------------------------------------------------------
# load: missing / corrupt files
# ---------------------------------------------------------------------------


def test_load_missing_file_returns_defaults(tmp_path: Path) -> None:
    s = load_onboarding(tmp_path / "absent.json")
    assert s == OnboardingState()


def test_load_corrupt_json_returns_defaults(tmp_path: Path) -> None:
    target = tmp_path / "onboarding.json"
    target.write_text("{not valid json", encoding="utf-8")
    s = load_onboarding(target)
    assert s == OnboardingState()


def test_load_invalid_status_falls_back_to_unknown(tmp_path: Path) -> None:
    target = tmp_path / "onboarding.json"
    target.write_text(
        json.dumps({"robinhood_status": "gibberish"}), encoding="utf-8"
    )
    s = load_onboarding(target)
    assert s.robinhood_status == "unknown"


def test_load_invalid_rh_mode_falls_back_to_shadow(tmp_path: Path) -> None:
    target = tmp_path / "onboarding.json"
    target.write_text(json.dumps({"rh_mode": "yolo"}), encoding="utf-8")
    s = load_onboarding(target)
    assert s.rh_mode == "shadow"


def test_load_negative_float_cap_falls_back_to_default(tmp_path: Path) -> None:
    target = tmp_path / "onboarding.json"
    target.write_text(
        json.dumps({"live_float_cap_usd": -42.0}), encoding="utf-8"
    )
    s = load_onboarding(target)
    assert s.live_float_cap_usd == DEFAULT_FLOAT_CAP_USD


def test_load_non_numeric_float_cap_falls_back(tmp_path: Path) -> None:
    target = tmp_path / "onboarding.json"
    target.write_text(
        json.dumps({"live_float_cap_usd": "not_a_number"}), encoding="utf-8"
    )
    s = load_onboarding(target)
    assert s.live_float_cap_usd == DEFAULT_FLOAT_CAP_USD


# ---------------------------------------------------------------------------
# save: round-trip + atomicity
# ---------------------------------------------------------------------------


def test_save_then_load_round_trips_all_fields(tmp_path: Path) -> None:
    target = tmp_path / "onboarding.json"
    s = OnboardingState(
        completed=True,
        robinhood_status="granted",
        live_float_cap_usd=500.0,
        accepted_disclaimer_at="2026-05-28T14:00:00+00:00",
        rh_mode="live",
        wizard_started_at="2026-05-28T13:55:00+00:00",
        wizard_completed_at="2026-05-28T14:01:00+00:00",
        display_name="Devin",
    )
    save_onboarding(s, target)
    loaded = load_onboarding(target)
    assert loaded == s


def test_save_creates_parent_dirs(tmp_path: Path) -> None:
    target = tmp_path / "deeply" / "nested" / "onboarding.json"
    save_onboarding(OnboardingState(), target)
    assert target.exists()


def test_save_is_atomic_no_partial_file(tmp_path: Path) -> None:
    """No ``.tmp`` artefact should remain after a clean write."""
    target = tmp_path / "onboarding.json"
    save_onboarding(OnboardingState(completed=True), target)
    # No leftover tempfiles in the dir.
    leftovers = [p for p in tmp_path.iterdir() if p.suffix == ".tmp"]
    assert leftovers == []
    assert target.exists()


# ---------------------------------------------------------------------------
# Mutators
# ---------------------------------------------------------------------------


def test_mark_started_sets_timestamp_when_unset() -> None:
    s = OnboardingState()
    mark_started(s)
    # Parses cleanly as ISO 8601.
    parsed = datetime.fromisoformat(s.wizard_started_at)
    # Within last minute.
    delta = (datetime.now(UTC) - parsed).total_seconds()
    assert 0 <= delta < 60


def test_mark_started_is_idempotent() -> None:
    """A user re-opening the wizard should not lose their original start
    timestamp -- it's analytics data."""
    s = OnboardingState(wizard_started_at="2026-01-01T00:00:00+00:00")
    mark_started(s)
    assert s.wizard_started_at == "2026-01-01T00:00:00+00:00"


def test_mark_completed_sets_flag_and_timestamp() -> None:
    s = OnboardingState()
    mark_completed(s)
    assert s.completed is True
    parsed = datetime.fromisoformat(s.wizard_completed_at)
    delta = (datetime.now(UTC) - parsed).total_seconds()
    assert 0 <= delta < 60


def test_mark_completed_updates_timestamp_on_repeat() -> None:
    """Unlike mark_started, completing again (e.g. after a re-run) bumps
    the timestamp so we see the most-recent completion."""
    s = OnboardingState(wizard_completed_at="2026-01-01T00:00:00+00:00")
    mark_completed(s)
    assert s.wizard_completed_at != "2026-01-01T00:00:00+00:00"


def test_accept_disclaimer_stamps_acceptance() -> None:
    s = OnboardingState()
    accept_disclaimer(s)
    parsed = datetime.fromisoformat(s.accepted_disclaimer_at)
    delta = (datetime.now(UTC) - parsed).total_seconds()
    assert 0 <= delta < 60


# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------


def test_reset_deletes_file(tmp_path: Path) -> None:
    target = tmp_path / "onboarding.json"
    save_onboarding(OnboardingState(completed=True), target)
    assert target.exists()
    reset(target)
    assert not target.exists()


def test_reset_is_noop_when_file_missing(tmp_path: Path) -> None:
    """No exception when there's nothing to delete."""
    reset(tmp_path / "absent.json")  # should not raise


def test_reset_then_load_returns_fresh_defaults(tmp_path: Path) -> None:
    target = tmp_path / "onboarding.json"
    save_onboarding(
        OnboardingState(completed=True, display_name="Devin"), target
    )
    reset(target)
    fresh = load_onboarding(target)
    assert fresh == OnboardingState()


# ---------------------------------------------------------------------------
# to_dict
# ---------------------------------------------------------------------------


def test_to_dict_contains_all_fields() -> None:
    s = OnboardingState()
    d = s.to_dict()
    expected_keys = {
        "completed",
        "robinhood_status",
        "live_float_cap_usd",
        "accepted_disclaimer_at",
        "rh_mode",
        "broker_backend",
        "rh_account_number",
        "wizard_started_at",
        "wizard_completed_at",
        "display_name",
    }
    assert set(d.keys()) == expected_keys


# ---------------------------------------------------------------------------
# Parametrized validity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", ["unknown", "waitlist", "granted", "declined"])
def test_each_valid_status_round_trips(tmp_path: Path, status: str) -> None:
    target = tmp_path / "onboarding.json"
    save_onboarding(OnboardingState(robinhood_status=status), target)  # type: ignore[arg-type]
    assert load_onboarding(target).robinhood_status == status


@pytest.mark.parametrize("mode", ["shadow", "live"])
def test_each_valid_rh_mode_round_trips(tmp_path: Path, mode: str) -> None:
    target = tmp_path / "onboarding.json"
    save_onboarding(OnboardingState(rh_mode=mode), target)  # type: ignore[arg-type]
    assert load_onboarding(target).rh_mode == mode
