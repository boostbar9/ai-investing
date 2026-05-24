"""Tests for the §16 paper-day streak counter."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from packages.paper.streak import (
    DAILY_DD_LIMIT,
    GATE_TARGET_DAYS,
    PaperDayStats,
    compute_paper_streak,
    summarise_paper_days,
)


def _run(day_offset: int, equity: float, *, halted: bool = False, errors: list[str] | None = None) -> dict:
    base = datetime(2026, 1, 1, tzinfo=UTC)
    ts = (base + timedelta(days=day_offset, hours=23, minutes=59)).isoformat()
    return {
        "ts": ts,
        "strategy": "test",
        "dry_run": True,
        "halted": halted,
        "account_equity": equity,
        "errors": errors or [],
    }


def test_empty_log_returns_zero_streak():
    summary = compute_paper_streak(runs=[])
    assert summary.current_streak == 0
    assert summary.longest_streak == 0
    assert summary.total_days == 0
    assert summary.gate_passed is False
    assert summary.days_remaining == GATE_TARGET_DAYS


def test_single_clean_day_starts_streak_of_one():
    summary = compute_paper_streak(runs=[_run(0, 100_000)])
    assert summary.current_streak == 1
    assert summary.longest_streak == 1
    assert summary.peak_equity == 100_000
    assert summary.current_drawdown == 0.0
    assert summary.last_clean is True


def test_three_clean_days_streak_grows():
    runs = [_run(0, 100_000), _run(1, 100_500), _run(2, 101_000)]
    summary = compute_paper_streak(runs=runs)
    assert summary.current_streak == 3
    assert summary.longest_streak == 3
    assert summary.clean_days == 3
    assert summary.total_days == 3


def test_halt_breaks_streak():
    runs = [_run(0, 100_000), _run(1, 100_500), _run(2, 100_400, halted=True), _run(3, 100_600)]
    summary = compute_paper_streak(runs=runs)
    # Last run was clean — streak rebuilds to 1, longest captures the pre-halt 2.
    assert summary.current_streak == 1
    assert summary.longest_streak == 2
    assert summary.clean_days == 3


def test_drawdown_breach_breaks_streak():
    # Peak 100k → drop to 91k (-9%) breaches the 8% intraday DD limit.
    runs = [_run(0, 100_000), _run(1, 100_000), _run(2, 91_000), _run(3, 95_000)]
    summary = compute_paper_streak(runs=runs)
    assert summary.current_streak == 1  # only day 3 is clean (DD recovered)
    assert summary.longest_streak == 2
    assert summary.last_break_reason is None  # last day clean


def test_errors_break_streak():
    runs = [_run(0, 100_000), _run(1, 100_500, errors=["broker timeout"])]
    summary = compute_paper_streak(runs=runs)
    assert summary.current_streak == 0
    assert "errors" in (summary.last_break_reason or "")


def test_multiple_runs_same_day_use_last_equity():
    base = datetime(2026, 2, 1, tzinfo=UTC)
    runs = [
        {"ts": (base + timedelta(hours=1)).isoformat(), "account_equity": 100_000, "halted": False, "errors": []},
        {"ts": (base + timedelta(hours=20)).isoformat(), "account_equity": 99_900, "halted": False, "errors": []},
    ]
    stats = summarise_paper_days(runs)
    assert len(stats) == 1
    assert stats[0].end_equity == 99_900
    assert stats[0].runs == 2


def test_multiple_runs_same_day_halt_propagates():
    base = datetime(2026, 2, 1, tzinfo=UTC)
    runs = [
        {"ts": (base + timedelta(hours=1)).isoformat(), "account_equity": 100_000, "halted": True, "errors": []},
        {"ts": (base + timedelta(hours=20)).isoformat(), "account_equity": 100_000, "halted": False, "errors": []},
    ]
    stats = summarise_paper_days(runs)
    assert stats[0].halted is True
    assert stats[0].clean is False


def test_gate_passed_after_target_days():
    runs = [_run(i, 100_000 + i * 100) for i in range(GATE_TARGET_DAYS)]
    summary = compute_paper_streak(runs=runs)
    assert summary.current_streak == GATE_TARGET_DAYS
    assert summary.gate_passed is True
    assert summary.days_remaining == 0


def test_streak_below_threshold_dd_does_not_break():
    # 5% DD < 8% limit — should still be clean.
    runs = [_run(0, 100_000), _run(1, 95_000)]
    stats = summarise_paper_days(runs)
    assert stats[1].drawdown == -0.05
    assert stats[1].drawdown >= -DAILY_DD_LIMIT
    assert stats[1].clean is True


def test_compute_from_file(tmp_path: Path):
    log = tmp_path / "runs.jsonl"
    log.write_text("\n".join(json.dumps(_run(i, 100_000 + i * 10)) for i in range(3)) + "\n")
    summary = compute_paper_streak(log_path=log)
    assert summary.current_streak == 3
    assert summary.total_days == 3


def test_paperdaystat_to_dict_includes_clean_flag():
    stat = PaperDayStats(
        day=datetime(2026, 3, 1).date(),
        end_equity=100_000,
        peak_to_day=100_000,
        drawdown=0.0,
        halted=False,
        error_count=0,
        runs=1,
    )
    d = stat.to_dict()
    assert d["clean"] is True
    assert d["day"] == "2026-03-01"
