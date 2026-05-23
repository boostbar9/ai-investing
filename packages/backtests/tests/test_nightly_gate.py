"""Tests for the nightly Sharpe-drop gate script."""

from __future__ import annotations

import json
from pathlib import Path

from packages.backtests.nightly_gate import compare, main


def _write(dir_: Path, key: str, sharpe: float) -> None:
    dir_.mkdir(parents=True, exist_ok=True)
    (dir_ / f"{key}.json").write_text(json.dumps({"strategy": key, "sharpe": sharpe}))


def test_compare_blocks_on_big_drop():
    cur = {"trend-bull": {"sharpe": 0.5}}
    base = {"trend-bull": {"sharpe": 1.5}}
    block, reasons = compare(cur, base, max_drop=0.10)
    assert block is True
    assert "66" in reasons[0] or "67" in reasons[0]


def test_compare_passes_on_small_drop():
    cur = {"trend-bull": {"sharpe": 1.40}}
    base = {"trend-bull": {"sharpe": 1.50}}
    block, _ = compare(cur, base, max_drop=0.10)
    assert block is False  # ~6.7% drop, under threshold


def test_compare_handles_zero_baseline():
    cur = {"trend-bull": {"sharpe": -0.5}}
    base = {"trend-bull": {"sharpe": 0.0}}
    block, reasons = compare(cur, base, max_drop=0.10)
    assert block is True
    assert "absolute drop" in reasons[0]


def test_compare_skips_unknown_baseline():
    """Newly added strategy → no baseline → must not block."""
    cur = {"new-strat-bull": {"sharpe": -1.0}}
    base = {"old-strat-bull": {"sharpe": 1.0}}
    block, _ = compare(cur, base, max_drop=0.10)
    assert block is False


def test_main_exit_codes(tmp_path: Path):
    cur_dir = tmp_path / "cur"
    base_dir = tmp_path / "base"
    _write(cur_dir, "trend-bull", 1.4)
    _write(base_dir, "trend-bull", 1.5)
    assert main(["--current", str(cur_dir), "--baseline", str(base_dir)]) == 0

    _write(cur_dir, "trend-bull", 0.5)  # huge drop
    assert main(["--current", str(cur_dir), "--baseline", str(base_dir)]) == 1


def test_main_no_artifacts_is_noop(tmp_path: Path):
    assert main(["--current", str(tmp_path / "empty"), "--baseline", str(tmp_path / "empty")]) == 0
