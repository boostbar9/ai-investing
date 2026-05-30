"""Tests for the shared atomic-write + stale-tmp cleanup helpers."""
from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from packages.shared.atomic_io import (
    DEFAULT_STALE_AGE_S,
    cleanup_stale_tmp_files,
    write_json_atomic,
)

# ---------------------------------------------------------------------------
# write_json_atomic
# ---------------------------------------------------------------------------


def test_write_json_atomic_writes_payload(tmp_path):
    target = tmp_path / "snap.json"
    write_json_atomic(target, {"peak": 99_757.15, "n": 3})
    assert json.loads(target.read_text()) == {"peak": 99_757.15, "n": 3}


def test_write_json_atomic_creates_parent_dirs(tmp_path):
    target = tmp_path / "nested" / "deep" / "snap.json"
    write_json_atomic(target, {"ok": True})
    assert target.exists()
    assert json.loads(target.read_text()) == {"ok": True}


def test_write_json_atomic_overwrites_existing(tmp_path):
    target = tmp_path / "snap.json"
    target.write_text('{"old": 1}')
    write_json_atomic(target, {"new": 2})
    assert json.loads(target.read_text()) == {"new": 2}


def test_write_json_atomic_accepts_string_path(tmp_path):
    target = tmp_path / "snap.json"
    write_json_atomic(str(target), {"ok": True})
    assert target.exists()


def test_write_json_atomic_leaves_no_tmp_files_on_success(tmp_path):
    target = tmp_path / "snap.json"
    for _ in range(5):
        write_json_atomic(target, {"i": 1})
    leftover = list(tmp_path.glob("tmp*.tmp"))
    assert leftover == [], f"stale tmp files leaked: {leftover}"


def test_write_json_atomic_retries_on_permission_error(tmp_path):
    """Simulate Windows AV holding the destination open briefly.

    First two ``os.replace`` calls raise PermissionError, third succeeds.
    The retry loop should swallow the first two and succeed on the third.
    """
    target = tmp_path / "snap.json"
    real_replace = __import__("os").replace
    call_count = {"n": 0}

    def flaky_replace(src, dst):
        call_count["n"] += 1
        if call_count["n"] <= 2:
            raise PermissionError("WinError 5: simulated AV lock")
        return real_replace(src, dst)

    with patch("packages.shared.atomic_io.os.replace", side_effect=flaky_replace):
        write_json_atomic(target, {"ok": True})

    assert call_count["n"] >= 3
    assert json.loads(target.read_text()) == {"ok": True}


def test_write_json_atomic_falls_back_after_outer_attempts(tmp_path, caplog):
    """If all retries fail, the helper writes directly + logs a warning."""
    target = tmp_path / "snap.json"

    def always_fail(src, dst):
        raise PermissionError("WinError 5: simulated stuck AV lock")

    with (
        patch("packages.shared.atomic_io.os.replace", side_effect=always_fail),
        caplog.at_level("WARNING", logger="packages.shared.atomic_io"),
    ):
        # Should NOT raise -- falls back to direct write.
        write_json_atomic(target, {"fallback": True})

    assert target.exists()
    assert json.loads(target.read_text()) == {"fallback": True}
    assert any("exhausted retries" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# cleanup_stale_tmp_files
# ---------------------------------------------------------------------------


def _make_tmp(p: Path, *, age_s: float) -> Path:
    p.write_text("orphan")
    mtime = time.time() - age_s
    import os as _os

    _os.utime(p, (mtime, mtime))
    return p


def test_cleanup_removes_old_tmp_files(tmp_path):
    old = _make_tmp(tmp_path / "tmpabc.tmp", age_s=DEFAULT_STALE_AGE_S + 60)
    removed = cleanup_stale_tmp_files(tmp_path)
    assert old in removed
    assert not old.exists()


def test_cleanup_keeps_fresh_tmp_files(tmp_path):
    fresh = _make_tmp(tmp_path / "tmpxyz.tmp", age_s=5)
    removed = cleanup_stale_tmp_files(tmp_path)
    assert fresh.exists()
    assert removed == []


def test_cleanup_ignores_non_tmp_files(tmp_path):
    keep = tmp_path / "state.json"
    keep.write_text("{}")
    import os as _os

    old_mtime = time.time() - DEFAULT_STALE_AGE_S - 60
    _os.utime(keep, (old_mtime, old_mtime))
    removed = cleanup_stale_tmp_files(tmp_path)
    assert keep.exists()
    assert removed == []


def test_cleanup_missing_directory(tmp_path):
    missing = tmp_path / "does-not-exist"
    assert cleanup_stale_tmp_files(missing) == []


def test_cleanup_respects_custom_age(tmp_path):
    medium = _make_tmp(tmp_path / "tmpmed.tmp", age_s=120)
    # 60s threshold should sweep it.
    removed = cleanup_stale_tmp_files(tmp_path, max_age_s=60)
    assert medium in removed


def test_cleanup_swallows_unlink_errors(tmp_path):
    old = _make_tmp(tmp_path / "tmpabc.tmp", age_s=DEFAULT_STALE_AGE_S + 60)

    def boom(self):
        raise PermissionError("locked by another process")

    with patch.object(Path, "unlink", boom):
        # Should not raise.
        removed = cleanup_stale_tmp_files(tmp_path)

    # File still there (unlink failed) but no exception bubbled up.
    assert old.exists()
    assert removed == []


# ---------------------------------------------------------------------------
# Integration smoke: real callers use the helper correctly
# ---------------------------------------------------------------------------


def test_update_session_peak_uses_atomic_write(tmp_path, monkeypatch):
    """The Phase 15 DD taper reads session_peak.json -- it must survive
    crash-mid-write. Verify update_session_peak now goes through the
    atomic helper (no half-written file possible)."""
    from tools import paper_trade

    monkeypatch.setattr(paper_trade, "PAPER_LOG_DIR", tmp_path)
    monkeypatch.setattr(paper_trade, "EQUITY_PEAK_FILE", tmp_path / "session_peak.json")

    peak = paper_trade.update_session_peak(100_000.0)
    assert peak == 100_000.0
    payload = json.loads((tmp_path / "session_peak.json").read_text())
    assert payload["peak"] == 100_000.0
    # No orphan tmp left behind.
    assert list(tmp_path.glob("tmp*.tmp")) == []


def test_update_session_peak_keeps_high_water_mark(tmp_path, monkeypatch):
    """Drawdown taper depends on the peak being monotonic non-decreasing."""
    from tools import paper_trade

    monkeypatch.setattr(paper_trade, "PAPER_LOG_DIR", tmp_path)
    monkeypatch.setattr(paper_trade, "EQUITY_PEAK_FILE", tmp_path / "session_peak.json")

    paper_trade.update_session_peak(100_000.0)
    paper_trade.update_session_peak(99_200.0)  # dip
    final = paper_trade.update_session_peak(99_800.0)  # partial recover
    assert final == 100_000.0


def test_calibrator_save_is_atomic(tmp_path):
    """A crashed calibration save must not corrupt the live model."""
    from packages.agents.calibration import IsotonicCalibrator

    target = tmp_path / "policy_isotonic.json"
    cal = IsotonicCalibrator()
    cal.fit([(0.1, 0), (0.3, 0), (0.7, 1), (0.9, 1)])
    cal.save(target)
    assert target.exists()
    # Re-load round-trip works.
    reloaded = IsotonicCalibrator.load(target)
    assert reloaded.x_breakpoints == cal.x_breakpoints
    # No orphan tmps.
    assert list(tmp_path.glob("tmp*.tmp")) == []


def test_calibrator_save_survives_replace_failure(tmp_path):
    """Even when os.replace fails permanently, save() should not raise --
    the fallback path writes directly and the loader still recovers."""
    from packages.agents.calibration import IsotonicCalibrator

    target = tmp_path / "policy_isotonic.json"
    cal = IsotonicCalibrator()
    cal.fit([(0.1, 0), (0.3, 0), (0.7, 1), (0.9, 1)])

    def always_fail(src, dst):
        raise PermissionError("simulated AV lock")

    with patch("packages.shared.atomic_io.os.replace", side_effect=always_fail):
        # Should NOT raise.
        cal.save(target)

    assert target.exists()
    reloaded = IsotonicCalibrator.load(target)
    assert reloaded.x_breakpoints == cal.x_breakpoints


@pytest.mark.parametrize("payload", [{}, [], {"a": 1}, [1, 2, 3], "scalar", 42, None])
def test_write_json_atomic_handles_json_payload_shapes(tmp_path, payload):
    target = tmp_path / "snap.json"
    write_json_atomic(target, payload)
    assert json.loads(target.read_text()) == payload
