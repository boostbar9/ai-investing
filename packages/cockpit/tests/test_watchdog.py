"""Tests for the drawdown watchdog.

The watchdog enforces the §16 8% drawdown halt -- a hard safety rail.
Tests must cover both the math (does the verdict fire at the right
moment?) and the persistence semantics (does a restart preserve an
active halt; does only the operator clear it?).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from packages.cockpit import watchdog as wd

# ---------------------------------------------------------------------------
# evaluate_curve
# ---------------------------------------------------------------------------


def _curve(*equities: float) -> list[dict]:
    return [{"t": f"2026-05-{i:02d}", "equity": e} for i, e in enumerate(equities, start=1)]


def test_evaluate_empty_curve_is_no_breach() -> None:
    v = wd.evaluate_curve([])
    assert v.breach is False
    assert v.current_drawdown == 0.0


def test_evaluate_curve_with_no_equities_is_no_breach() -> None:
    v = wd.evaluate_curve([{"t": "x", "equity": None}])
    assert v.breach is False


def test_evaluate_rising_curve_no_drawdown() -> None:
    v = wd.evaluate_curve(_curve(100_000, 101_000, 102_500))
    assert v.breach is False
    assert v.current_drawdown == 0.0
    assert v.peak_equity == 102_500
    assert v.current_equity == 102_500


def test_evaluate_modest_drawdown_no_breach() -> None:
    # 100k -> 97k = 3% DD, below the 8% threshold.
    v = wd.evaluate_curve(_curve(100_000, 100_000, 97_000))
    assert v.breach is False
    assert v.current_drawdown == pytest.approx(0.03, abs=1e-6)


def test_evaluate_at_threshold_breach() -> None:
    # 100k -> 92k = exactly 8% -> breach.
    v = wd.evaluate_curve(_curve(100_000, 92_000))
    assert v.breach is True
    assert v.current_drawdown == pytest.approx(0.08, abs=1e-6)
    assert "drawdown" in v.message.lower()


def test_evaluate_deep_drawdown_breach() -> None:
    v = wd.evaluate_curve(_curve(100_000, 105_000, 90_000))
    assert v.breach is True
    # Peak is 105k, current 90k -> 14.3%.
    assert v.current_drawdown > 0.08
    assert v.peak_equity == 105_000


def test_evaluate_custom_threshold() -> None:
    v = wd.evaluate_curve(_curve(100_000, 96_000), threshold=0.03)
    assert v.breach is True
    assert v.threshold == 0.03


def test_evaluate_peak_zero_returns_neutral_verdict() -> None:
    v = wd.evaluate_curve(_curve(0.0, 0.0))
    assert v.breach is False
    assert "zero" in v.message.lower() or "negative" in v.message.lower()


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_halt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the halt file into a per-test temp directory."""
    monkeypatch.setattr(wd, "DATA_DIR", tmp_path)
    monkeypatch.setattr(wd, "HALT_FILE", tmp_path / "halt.json")
    return tmp_path / "halt.json"


def test_read_halt_missing_returns_none(tmp_halt: Path) -> None:
    assert wd.read_halt() is None
    assert wd.is_halt_active() is False


def test_write_halt_persists_breach(tmp_halt: Path) -> None:
    v = wd.evaluate_curve(_curve(100_000, 90_000))
    rec = wd.write_halt(v)
    assert rec["active"] is True
    assert "since" in rec
    assert rec["drawdown"] == pytest.approx(0.10, abs=1e-6)
    assert tmp_halt.exists()
    assert wd.is_halt_active() is True


def test_write_halt_idempotent_preserves_since(tmp_halt: Path) -> None:
    """A second breach record must NOT overwrite 'since' -- the audit trail
    needs the moment the halt started, not the latest tick."""
    v1 = wd.evaluate_curve(_curve(100_000, 90_000))
    wd.write_halt(v1)
    original = wd.read_halt()
    assert original is not None
    original_since = original["since"]

    v2 = wd.evaluate_curve(_curve(100_000, 85_000))
    wd.write_halt(v2)
    updated = wd.read_halt()
    assert updated is not None
    assert updated["since"] == original_since
    # But the latest snapshot should reflect the deeper DD.
    assert updated["drawdown"] == pytest.approx(0.15, abs=1e-6)


def test_clear_halt_records_release(tmp_halt: Path) -> None:
    v = wd.evaluate_curve(_curve(100_000, 90_000))
    wd.write_halt(v)
    cleared = wd.clear_halt(acknowledged_by="devin")
    assert cleared["active"] is False
    assert cleared["released_by"] == "devin"
    assert cleared["prior_reason"]  # the original reason is preserved
    assert wd.is_halt_active() is False


def test_clear_halt_on_no_prior_halt_still_writes_record(tmp_halt: Path) -> None:
    """Clearing 'nothing' is allowed -- the audit trail records the
    explicit ack and that's useful even if the file was missing."""
    cleared = wd.clear_halt(acknowledged_by="devin")
    assert cleared["active"] is False
    assert wd.read_halt() is not None


def test_read_halt_handles_corrupt_file(tmp_halt: Path) -> None:
    tmp_halt.parent.mkdir(parents=True, exist_ok=True)
    tmp_halt.write_text("not json", encoding="utf-8")
    assert wd.read_halt() is None
    assert wd.is_halt_active() is False


def test_read_halt_handles_non_dict_json(tmp_halt: Path) -> None:
    tmp_halt.parent.mkdir(parents=True, exist_ok=True)
    tmp_halt.write_text("[1,2,3]", encoding="utf-8")
    assert wd.read_halt() is None


# ---------------------------------------------------------------------------
# Combined evaluate_and_persist
# ---------------------------------------------------------------------------


def test_evaluate_and_persist_breach_writes_halt(tmp_halt: Path) -> None:
    v = wd.evaluate_and_persist(_curve(100_000, 90_000))
    assert v.breach is True
    assert wd.is_halt_active() is True


def test_evaluate_and_persist_no_breach_does_not_clear(tmp_halt: Path) -> None:
    """A non-breach must NOT clear an active halt -- only the operator does."""
    wd.write_halt(wd.evaluate_curve(_curve(100_000, 90_000)))
    assert wd.is_halt_active() is True
    # Curve recovers to within 8%.
    wd.evaluate_and_persist(_curve(100_000, 95_000))
    # Halt still active. Recovery is not auto-release.
    assert wd.is_halt_active() is True
