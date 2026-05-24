"""Paper-trading streak counter (§16 60-day live-promotion gate).

Aggregates `data/paper_log/runs.jsonl` into per-trading-day equity series and
counts the longest current streak of "clean" paper days. A day is clean iff:

  * end-of-day equity is non-negative (account_equity > 0)
  * the rolling drawdown from the running peak stays above -8% (spec §16)
  * no run on that day reported `halted=True`
  * no run on that day reported errors

The streak is reset by any failing day. The output also exposes the all-time
longest streak, the running peak equity, and the live drawdown.

This module is deliberately stdlib-only so it can run in the nightly paper
runner without dragging in pandas. A pandas helper is offered for callers
(dashboards) who already use it.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path

DEFAULT_LOG_PATH = Path("data/paper_log/runs.jsonl")
DEFAULT_PEAK_PATH = Path("data/paper_log/session_peak.json")

# §16 v1.0 gate: 60 clean paper days with intraday DD ≤ 8% before live promotion.
GATE_TARGET_DAYS = 60
DAILY_DD_LIMIT = 0.08  # 8%


@dataclass(frozen=True)
class PaperDayStats:
    """Aggregated stats for a single paper-trading day."""

    day: date
    end_equity: float
    peak_to_day: float
    drawdown: float  # negative number, e.g. -0.043 for -4.3%
    halted: bool
    error_count: int
    runs: int

    @property
    def clean(self) -> bool:
        if self.end_equity <= 0:
            return False
        if self.halted:
            return False
        if self.error_count > 0:
            return False
        return not self.drawdown < -DAILY_DD_LIMIT

    def to_dict(self) -> dict:
        d = asdict(self)
        d["day"] = self.day.isoformat()
        d["clean"] = self.clean
        return d


@dataclass
class StreakSummary:
    """Current + historical streak stats for the §16 promotion gate."""

    current_streak: int = 0
    longest_streak: int = 0
    total_days: int = 0
    clean_days: int = 0
    last_day: date | None = None
    last_clean: bool = False
    last_break_reason: str | None = None
    peak_equity: float = 0.0
    current_drawdown: float = 0.0
    gate_target_days: int = GATE_TARGET_DAYS
    days_remaining: int = GATE_TARGET_DAYS
    gate_passed: bool = False
    per_day: list[PaperDayStats] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "current_streak": self.current_streak,
            "longest_streak": self.longest_streak,
            "total_days": self.total_days,
            "clean_days": self.clean_days,
            "last_day": self.last_day.isoformat() if self.last_day else None,
            "last_clean": self.last_clean,
            "last_break_reason": self.last_break_reason,
            "peak_equity": self.peak_equity,
            "current_drawdown": self.current_drawdown,
            "gate_target_days": self.gate_target_days,
            "days_remaining": self.days_remaining,
            "gate_passed": self.gate_passed,
            "per_day": [d.to_dict() for d in self.per_day],
        }


def iter_paper_runs(path: Path = DEFAULT_LOG_PATH) -> Iterator[dict]:
    """Yield run dicts from `runs.jsonl`. Missing file → empty iterator."""
    if not path.exists():
        return iter(())

    def _gen() -> Iterator[dict]:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    # Tolerate partial writes from concurrent runners.
                    continue

    return _gen()


def _parse_ts(ts: str) -> datetime:
    # Handles "...Z" or "...+00:00" variants.
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    return datetime.fromisoformat(ts).astimezone(UTC)


def summarise_paper_days(runs: Iterable[dict]) -> list[PaperDayStats]:
    """Aggregate a stream of run dicts into one record per trading day.

    Multiple runs on the same UTC date are folded together:
      * end_equity = equity of the LAST run that day
      * halted, errors = OR / SUM across runs
      * peak_to_day, drawdown computed against running peak across all days
    """
    by_day: dict[date, list[dict]] = {}
    for run in runs:
        ts = run.get("ts")
        if not ts:
            continue
        try:
            dt = _parse_ts(ts)
        except (ValueError, TypeError):
            continue
        by_day.setdefault(dt.date(), []).append(run)

    stats: list[PaperDayStats] = []
    peak = 0.0
    for day in sorted(by_day):
        runs_today = by_day[day]
        # End-of-day equity = last run we observed.
        end_equity = float(runs_today[-1].get("account_equity") or 0.0)
        halted = any(bool(r.get("halted")) for r in runs_today)
        error_count = sum(len(r.get("errors") or []) for r in runs_today)
        peak = max(peak, end_equity)
        drawdown = (end_equity - peak) / peak if peak > 0 else 0.0
        stats.append(
            PaperDayStats(
                day=day,
                end_equity=end_equity,
                peak_to_day=peak,
                drawdown=drawdown,
                halted=halted,
                error_count=error_count,
                runs=len(runs_today),
            )
        )
    return stats


def _break_reason(stat: PaperDayStats) -> str | None:
    if stat.clean:
        return None
    if stat.end_equity <= 0:
        return "equity-non-positive"
    if stat.halted:
        return "kill-switch-halted"
    if stat.error_count > 0:
        return f"errors={stat.error_count}"
    if stat.drawdown < -DAILY_DD_LIMIT:
        return f"drawdown={stat.drawdown:.1%}"
    return "unknown"


def compute_paper_streak(
    log_path: Path = DEFAULT_LOG_PATH,
    *,
    runs: Iterable[dict] | None = None,
    gate_target_days: int = GATE_TARGET_DAYS,
) -> StreakSummary:
    """Build a `StreakSummary` from `data/paper_log/runs.jsonl`.

    Pass `runs=` to bypass file I/O (used in tests).
    """
    source = runs if runs is not None else iter_paper_runs(Path(log_path))
    per_day = summarise_paper_days(source)
    summary = StreakSummary(gate_target_days=gate_target_days, days_remaining=gate_target_days)
    summary.per_day = per_day
    summary.total_days = len(per_day)
    if not per_day:
        return summary

    longest = 0
    current = 0
    clean_days = 0
    last_break: str | None = None
    for stat in per_day:
        if stat.clean:
            current += 1
            clean_days += 1
            longest = max(longest, current)
        else:
            current = 0
            last_break = _break_reason(stat)

    last = per_day[-1]
    summary.current_streak = current
    summary.longest_streak = longest
    summary.clean_days = clean_days
    summary.last_day = last.day
    summary.last_clean = last.clean
    summary.last_break_reason = last_break if not last.clean else None
    summary.peak_equity = last.peak_to_day
    summary.current_drawdown = last.drawdown
    summary.days_remaining = max(0, gate_target_days - current)
    summary.gate_passed = current >= gate_target_days
    return summary
