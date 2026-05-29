"""Per-cycle decision instrumentation for the paper-trade loop.

Phase 11: every cycle of ``tools/paper_trade.py`` now writes a single
JSONL row capturing the full decision pipeline:

  research sweep candidates considered
  -> corroboration gate
  -> agent-approved symbols
  -> target weights
  -> planned orders
  -> submitted (or skipped, with reason)
  -> halt reasons (kill switches / cockpit pause / agent halt)

The point of this is *visibility before money*. Even when ``orders_submitted == 0``
(which has been the case for every cycle so far), we now know exactly
why: did the sweep find anything? did the agent veto? did the
rebalance threshold absorb the signal? did a kill switch fire?

The schema is intentionally additive -- never break old readers. JSONL
so we can ``tail -f`` and ``grep``. Each row is self-contained; the
``/shadow`` page renders the last N rows and a per-stage funnel.
"""
from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Path knobs (env-overridable for tests). The cockpit reads this file
# from /api/shadow/decisions on every dashboard refresh.
DEFAULT_DECISIONS_PATH = Path(
    os.getenv("PAPER_DECISIONS_PATH", "data/paper_log/decisions.jsonl")
)


@dataclass(frozen=True)
class PipelineStage:
    """One stage of the candidate funnel."""

    name: str
    count: int
    # First few symbols at this stage. Capped so we never blow up the
    # JSONL line size; the user can pull the full set from runs.jsonl.
    sample_symbols: list[str] = field(default_factory=list)


@dataclass
class DecisionRecord:
    """A single cycle's decision trace.

    Either ``submitted_count > 0`` (real orders went out) or one of the
    halt reasons fired or the rebalance threshold absorbed everything.
    The dashboard groups cycles by the first non-empty bucket.
    """

    ts: str
    strategy: str
    dry_run: bool
    halted: bool
    halt_reasons: list[str]
    # Funnel: each stage explains how many symbols survived. The
    # dashboard renders this as a vertical bar chart.
    pipeline: list[PipelineStage]
    # Per-symbol planned/submitted detail. Skipped symbols (delta below
    # MIN_REBALANCE_BPS) are NOT in planned -- they show up in the
    # ``rebalance_absorbed`` stage of the pipeline instead.
    planned_count: int
    submitted_count: int
    error_count: int
    # Surface a few headline metrics so the cockpit table doesn't have
    # to do its own JSON munging.
    account_equity: float
    regime: str
    decision_id: str

    def to_row(self) -> dict[str, Any]:
        out = asdict(self)
        # asdict() handles nested dataclasses -- pipeline is already a
        # list of dicts. Just ensure JSON-serialisable types.
        return out


def _stage(name: str, symbols: Iterable[str], *, sample_n: int = 8) -> PipelineStage:
    """Make a PipelineStage from a symbol iterable. Dedup + cap sample.

    Order is preserved (first-seen) so the dashboard's sample shows the
    user the *top* candidates, not random ones.
    """
    seen: list[str] = []
    seen_set: set[str] = set()
    for s in symbols:
        s = str(s).upper().strip()
        if not s or s in seen_set:
            continue
        seen.append(s)
        seen_set.add(s)
    return PipelineStage(name=name, count=len(seen), sample_symbols=seen[:sample_n])


def build_record(
    *,
    ts: str,
    strategy: str,
    dry_run: bool,
    halted: bool,
    halt_reasons: list[str],
    sweep_candidates: list[dict[str, Any]] | None,
    agent_approved_symbols: Iterable[str],
    target_weights: dict[str, float],
    planned_orders: list[dict[str, Any]],
    submitted_orders: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    account_equity: float,
    regime: str,
    decision_id: str,
) -> DecisionRecord:
    """Compose a DecisionRecord from the paper_trade loop's locals.

    Pure: no I/O. All inputs are best-effort; missing data degrades
    gracefully (zero-count stages, empty samples) rather than raising.
    """
    sweep_candidates = sweep_candidates or []

    # Stage 1: sweep produced these candidates (raw output of research_sweep).
    sweep_syms = [str(c.get("symbol", "")).upper() for c in sweep_candidates if c.get("symbol")]

    # Stage 2: corroborated subset of the sweep (Phase 3 gate).
    corroborated_syms = [
        str(c.get("symbol", "")).upper()
        for c in sweep_candidates
        if c.get("symbol") and bool(c.get("corroborated", False))
    ]

    # Stage 3: agent (LangGraph risk pass) approved these.
    approved_list = [str(s).upper() for s in agent_approved_symbols if s]

    # Stage 4: target weights with a non-trivial allocation.
    target_syms = [s.upper() for s, w in (target_weights or {}).items() if abs(float(w)) >= 1e-6]

    # Stage 5: planned orders (delta >= MIN_REBALANCE_BPS).
    planned_syms = [str(p.get("symbol", "")).upper() for p in planned_orders if p.get("symbol")]

    # Stage 6: actually submitted (live in non-dry-run mode).
    submitted_syms = [
        str(o.get("symbol", "")).upper() for o in submitted_orders if o.get("symbol")
    ]

    pipeline = [
        _stage("sweep_candidates", sweep_syms),
        _stage("corroborated", corroborated_syms),
        _stage("agent_approved", approved_list),
        _stage("target_weighted", target_syms),
        _stage("orders_planned", planned_syms),
        _stage("orders_submitted", submitted_syms),
    ]

    return DecisionRecord(
        ts=ts,
        strategy=strategy,
        dry_run=bool(dry_run),
        halted=bool(halted),
        halt_reasons=list(halt_reasons or []),
        pipeline=pipeline,
        planned_count=len(planned_syms),
        submitted_count=len(submitted_syms),
        error_count=len(errors or []),
        account_equity=float(account_equity or 0.0),
        regime=str(regime or "unknown"),
        decision_id=str(decision_id or ""),
    )


def append_decision(
    record: DecisionRecord, path: Path | None = None
) -> None:
    """Append one record to the decisions JSONL log.

    Best-effort: a failure here must never break the paper loop. We
    write a single ``\\n``-terminated line so concurrent readers (the
    cockpit poller) always see complete records.
    """
    import sys

    target = path if path is not None else sys.modules[__name__].DEFAULT_DECISIONS_PATH
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record.to_row(), separators=(",", ":"), default=str) + "\n"
        with open(target, "a", encoding="utf-8") as f:
            f.write(line)
    except OSError as exc:
        # Logging only -- the paper loop continues. Decisions logging
        # is a observability feature, not a correctness gate.
        log.warning("could not append decision record: %s", exc)


def iter_decisions(path: Path | None = None) -> Iterator[dict[str, Any]]:
    """Yield decision rows oldest-first. Skips malformed lines."""
    import sys

    target = path if path is not None else sys.modules[__name__].DEFAULT_DECISIONS_PATH
    if not target.exists():
        return
    for raw in target.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def load_recent(limit: int = 50, path: Path | None = None) -> list[dict[str, Any]]:
    """Return the last ``limit`` decision records, newest-first.

    The cockpit /api/shadow/decisions endpoint uses this directly; the
    page renders the table in this order so the freshest cycle is on top.
    """
    rows = list(iter_decisions(path=path))
    if limit <= 0:
        return []
    return rows[-limit:][::-1]


def latest_pipeline(path: Path | None = None) -> dict[str, Any]:
    """Aggregate the funnel across the last 24h of decisions.

    Used by /api/shadow/pipeline to render the candidate funnel without
    forcing the page to do a sum over N rows in JS. Returns an empty
    skeleton if no records exist yet so the dashboard can render a
    sensible placeholder.
    """
    rows = list(iter_decisions(path=path))
    if not rows:
        return {
            "stages": [],
            "n_cycles": 0,
            "window_hours": 24,
            "halts": {},
        }

    cutoff = datetime.now(UTC).timestamp() - 24 * 3600
    recent: list[dict[str, Any]] = []
    for row in rows:
        ts = row.get("ts")
        try:
            t = datetime.fromisoformat(str(ts).replace("Z", "+00:00")).timestamp()
        except (ValueError, AttributeError):
            continue
        if t >= cutoff:
            recent.append(row)
    if not recent:
        # No cycles in the last 24h -- fall back to the most recent record
        # so the dashboard isn't blank during off-hours.
        recent = rows[-1:]

    # Sum stage counts across the window. The dashboard divides by
    # n_cycles for the average; raw totals are also useful so we send both.
    stage_totals: dict[str, int] = {}
    stage_samples: dict[str, list[str]] = {}
    for row in recent:
        for stage in row.get("pipeline", []) or []:
            name = stage.get("name", "")
            if not name:
                continue
            stage_totals[name] = stage_totals.get(name, 0) + int(stage.get("count", 0))
            # Keep a representative sample (just take the most recent).
            stage_samples[name] = list(stage.get("sample_symbols", []) or [])

    # Tally halt reasons so the user can spot dominant blockers.
    halts: dict[str, int] = {}
    for row in recent:
        for reason in row.get("halt_reasons", []) or []:
            key = str(reason).split(":", 1)[0].strip() or "unknown"
            halts[key] = halts.get(key, 0) + 1

    # Preserve the canonical stage order even if some stages are missing
    # from the window (e.g. nothing ever made it past 'corroborated').
    canonical = [
        "sweep_candidates",
        "corroborated",
        "agent_approved",
        "target_weighted",
        "orders_planned",
        "orders_submitted",
    ]
    stages = [
        {
            "name": name,
            "total": stage_totals.get(name, 0),
            "avg_per_cycle": round(stage_totals.get(name, 0) / max(1, len(recent)), 2),
            "sample_symbols": stage_samples.get(name, []),
        }
        for name in canonical
    ]
    return {
        "stages": stages,
        "n_cycles": len(recent),
        "window_hours": 24,
        "halts": halts,
    }


def window_status(
    *,
    target_days: int = 14,
    path: Path | None = None,
) -> dict[str, Any]:
    """Compute shadow-window progress.

    Returns ``start_day``, ``today``, ``days_elapsed``, ``days_remaining``,
    ``days_with_activity``, and a per-day grid the dashboard renders as
    a calendar. Days are local-date (UTC) keyed.
    """
    rows = list(iter_decisions(path=path))
    today = datetime.now(UTC).date()

    if not rows:
        return {
            "start_day": today.isoformat(),
            "today": today.isoformat(),
            "days_elapsed": 0,
            "days_remaining": target_days,
            "days_with_activity": 0,
            "target_days": target_days,
            "grid": [],
        }

    # Group counts per UTC date.
    per_day: dict[date, dict[str, int]] = {}
    for row in rows:
        ts = row.get("ts")
        try:
            d = datetime.fromisoformat(str(ts).replace("Z", "+00:00")).date()
        except (ValueError, AttributeError):
            continue
        cell = per_day.setdefault(
            d, {"cycles": 0, "submitted": 0, "halted": 0, "errors": 0}
        )
        cell["cycles"] += 1
        cell["submitted"] += int(row.get("submitted_count", 0))
        if row.get("halted"):
            cell["halted"] += 1
        cell["errors"] += int(row.get("error_count", 0))

    start_day = min(per_day.keys())
    days_elapsed = (today - start_day).days + 1
    days_remaining = max(0, target_days - days_elapsed)
    days_with_activity = len(per_day)

    # Dense grid from start to max(today, start+target-1) so the calendar
    # always shows the full window.
    from datetime import timedelta

    end = max(today, start_day + timedelta(days=target_days - 1))
    grid: list[dict[str, Any]] = []
    cursor = start_day
    while cursor <= end:
        cell = per_day.get(cursor, {"cycles": 0, "submitted": 0, "halted": 0, "errors": 0})
        grid.append(
            {
                "day": cursor.isoformat(),
                "cycles": int(cell["cycles"]),
                "submitted": int(cell["submitted"]),
                "halted": int(cell["halted"]),
                "errors": int(cell["errors"]),
                "is_future": cursor > today,
            }
        )
        cursor += timedelta(days=1)

    return {
        "start_day": start_day.isoformat(),
        "today": today.isoformat(),
        "days_elapsed": days_elapsed,
        "days_remaining": days_remaining,
        "days_with_activity": days_with_activity,
        "target_days": target_days,
        "grid": grid,
    }


__all__ = [
    "DEFAULT_DECISIONS_PATH",
    "DecisionRecord",
    "PipelineStage",
    "append_decision",
    "build_record",
    "iter_decisions",
    "latest_pipeline",
    "load_recent",
    "window_status",
]
