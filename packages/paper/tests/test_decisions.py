"""Phase 11 — tests for packages.paper.decisions.

Per-cycle decision record writer, reader, pipeline aggregator, and
14-day window status.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from packages.paper import decisions as dec_mod
from packages.paper.decisions import (
    DecisionRecord,
    PipelineStage,
    append_decision,
    build_record,
    iter_decisions,
    latest_pipeline,
    load_recent,
    window_status,
)


@pytest.fixture
def isolated_log(monkeypatch, tmp_path) -> Path:
    """Point DEFAULT_DECISIONS_PATH at tmp_path so writes never escape."""
    log = tmp_path / "decisions.jsonl"
    monkeypatch.setattr(dec_mod, "DEFAULT_DECISIONS_PATH", log)
    return log


def _record(
    *,
    ts: str = "2026-05-29T20:00:00+00:00",
    halted: bool = False,
    halt_reasons: list[str] | None = None,
    sweep: list[dict] | None = None,
    corroborated: list[dict] | None = None,
    approved: list[str] | None = None,
    targets: dict[str, float] | None = None,
    planned: list[dict] | None = None,
    submitted: list[dict] | None = None,
    errors: list[dict] | None = None,
    equity: float = 100_000.0,
    regime: str = "chop",
    decision_id: str = "abc123",
) -> DecisionRecord:
    sweep_default = [
        {"symbol": "SPY", "corroborated": True},
        {"symbol": "QQQ", "corroborated": False},
    ]
    return build_record(
        ts=ts,
        strategy="ensemble",
        dry_run=True,
        halted=halted,
        halt_reasons=halt_reasons or [],
        sweep_candidates=sweep if sweep is not None else sweep_default,
        agent_approved_symbols=approved if approved is not None else ["SPY"],
        target_weights=targets if targets is not None else {"SPY": 0.5},
        planned_orders=planned if planned is not None else [{"symbol": "SPY"}],
        submitted_orders=submitted if submitted is not None else [],
        errors=errors or [],
        account_equity=equity,
        regime=regime,
        decision_id=decision_id,
    )


# ---------------------------------------------------------------------------
# build_record
# ---------------------------------------------------------------------------


def test_build_record_produces_six_canonical_stages():
    rec = _record()
    assert isinstance(rec, DecisionRecord)
    names = [s.name for s in rec.pipeline]
    assert names == [
        "sweep_candidates",
        "corroborated",
        "agent_approved",
        "target_weighted",
        "orders_planned",
        "orders_submitted",
    ]


def test_build_record_pipeline_counts():
    rec = _record(
        sweep=[
            {"symbol": "SPY", "corroborated": True},
            {"symbol": "QQQ", "corroborated": True},
            {"symbol": "IWM", "corroborated": False},
        ],
        approved=["SPY", "QQQ"],
        targets={"SPY": 0.4, "QQQ": 0.4},
        planned=[{"symbol": "SPY"}, {"symbol": "QQQ"}],
        submitted=[{"symbol": "SPY"}],
    )
    counts = {s.name: s.count for s in rec.pipeline}
    assert counts["sweep_candidates"] == 3
    assert counts["corroborated"] == 2
    assert counts["agent_approved"] == 2
    assert counts["target_weighted"] == 2
    assert counts["orders_planned"] == 2
    assert counts["orders_submitted"] == 1
    assert rec.planned_count == 2
    assert rec.submitted_count == 1


def test_build_record_dedups_and_uppercases_symbols():
    rec = _record(
        sweep=[{"symbol": "spy"}, {"symbol": "SPY"}, {"symbol": "qqq"}],
        approved=["spy", "spy"],
    )
    sweep_stage = next(s for s in rec.pipeline if s.name == "sweep_candidates")
    assert sweep_stage.count == 2
    assert "SPY" in sweep_stage.sample_symbols
    assert "QQQ" in sweep_stage.sample_symbols
    approved_stage = next(s for s in rec.pipeline if s.name == "agent_approved")
    assert approved_stage.count == 1
    assert approved_stage.sample_symbols == ["SPY"]


def test_build_record_sample_capped_at_eight():
    sweep = [{"symbol": f"S{i:02d}", "corroborated": True} for i in range(20)]
    rec = _record(sweep=sweep)
    sweep_stage = next(s for s in rec.pipeline if s.name == "sweep_candidates")
    assert sweep_stage.count == 20
    assert len(sweep_stage.sample_symbols) == 8


def test_build_record_filters_tiny_target_weights():
    rec = _record(targets={"SPY": 0.5, "DUST": 1e-9})
    target_stage = next(s for s in rec.pipeline if s.name == "target_weighted")
    assert target_stage.count == 1
    assert "SPY" in target_stage.sample_symbols


def test_build_record_handles_missing_sweep():
    rec = _record(sweep=[])
    assert rec.pipeline[0].count == 0
    assert rec.pipeline[1].count == 0


def test_build_record_records_halt_reasons():
    rec = _record(halted=True, halt_reasons=["kill_switch:daily_dd", "cockpit_pause"])
    assert rec.halted is True
    assert rec.halt_reasons == ["kill_switch:daily_dd", "cockpit_pause"]


def test_build_record_error_count_from_errors_list():
    rec = _record(errors=[{"symbol": "SPY", "error": "broker_timeout"}])
    assert rec.error_count == 1


def test_build_record_to_row_is_json_safe():
    rec = _record()
    row = rec.to_row()
    serialised = json.dumps(row, default=str)
    assert "pipeline" in serialised
    assert "ensemble" in serialised


def test_pipeline_stage_is_frozen_dataclass():
    from dataclasses import FrozenInstanceError

    s = PipelineStage(name="x", count=1, sample_symbols=["A"])
    with pytest.raises(FrozenInstanceError):
        s.count = 2  # type: ignore[misc]


# ---------------------------------------------------------------------------
# append_decision + iter_decisions
# ---------------------------------------------------------------------------


def test_append_creates_parent_and_writes_one_line(isolated_log: Path):
    append_decision(_record())
    assert isolated_log.exists()
    lines = isolated_log.read_text().splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["strategy"] == "ensemble"


def test_append_explicit_path_overrides_default(tmp_path: Path):
    custom = tmp_path / "custom" / "log.jsonl"
    append_decision(_record(), path=custom)
    assert custom.exists()
    assert len(custom.read_text().splitlines()) == 1


def test_iter_decisions_returns_empty_for_missing_file(tmp_path: Path):
    missing = tmp_path / "nope.jsonl"
    rows = list(iter_decisions(path=missing))
    assert rows == []


def test_iter_decisions_skips_malformed_lines(tmp_path: Path):
    p = tmp_path / "log.jsonl"
    p.write_text(
        json.dumps({"ts": "x", "strategy": "s"}) + "\n"
        + "not-json\n"
        + json.dumps({"ts": "y", "strategy": "t"}) + "\n"
    )
    rows = list(iter_decisions(path=p))
    assert len(rows) == 2
    assert rows[0]["strategy"] == "s"
    assert rows[1]["strategy"] == "t"


def test_load_recent_returns_newest_first(isolated_log: Path):
    for i in range(5):
        ts = (datetime(2026, 5, 29, 12, 0, tzinfo=UTC) + timedelta(minutes=i)).isoformat()
        append_decision(_record(ts=ts, decision_id=f"id{i}"))
    rows = load_recent(limit=3)
    assert len(rows) == 3
    assert [r["decision_id"] for r in rows] == ["id4", "id3", "id2"]


def test_load_recent_zero_limit_returns_empty(isolated_log: Path):
    append_decision(_record())
    assert load_recent(limit=0) == []


def test_append_failure_swallowed(monkeypatch, isolated_log: Path):
    """A write error must NOT raise — instrumentation can't break the loop."""

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("builtins.open", boom)
    # Must not raise:
    append_decision(_record())


# ---------------------------------------------------------------------------
# latest_pipeline
# ---------------------------------------------------------------------------


def test_latest_pipeline_empty_skeleton(isolated_log: Path):
    out = latest_pipeline()
    assert out == {"stages": [], "n_cycles": 0, "window_hours": 24, "halts": {}}


def test_latest_pipeline_aggregates_recent_window(isolated_log: Path):
    now = datetime.now(UTC)
    # Two cycles within the last 24h.
    for i in range(2):
        append_decision(_record(ts=(now - timedelta(hours=i)).isoformat()))
    out = latest_pipeline()
    assert out["n_cycles"] == 2
    # Default sweep_candidates is 2 symbols per cycle (SPY + QQQ).
    sweep = next(s for s in out["stages"] if s["name"] == "sweep_candidates")
    assert sweep["total"] == 4
    assert sweep["avg_per_cycle"] == 2.0


def test_latest_pipeline_tallies_halt_reasons(isolated_log: Path):
    now = datetime.now(UTC)
    append_decision(
        _record(
            ts=now.isoformat(),
            halted=True,
            halt_reasons=["kill_switch:daily_dd", "cockpit_pause"],
        )
    )
    append_decision(
        _record(
            ts=(now - timedelta(minutes=5)).isoformat(),
            halted=True,
            halt_reasons=["kill_switch:weekly_dd"],
        )
    )
    out = latest_pipeline()
    assert out["halts"].get("kill_switch") == 2
    assert out["halts"].get("cockpit_pause") == 1


def test_latest_pipeline_falls_back_to_last_when_no_recent(isolated_log: Path):
    # Only an ancient record exists -- should still render something.
    old_ts = (datetime.now(UTC) - timedelta(days=7)).isoformat()
    append_decision(_record(ts=old_ts))
    out = latest_pipeline()
    assert out["n_cycles"] == 1


# ---------------------------------------------------------------------------
# window_status
# ---------------------------------------------------------------------------


def test_window_status_empty_returns_full_target(isolated_log: Path):
    out = window_status(target_days=14)
    assert out["days_elapsed"] == 0
    assert out["days_remaining"] == 14
    assert out["days_with_activity"] == 0
    assert out["grid"] == []


def test_window_status_dense_grid_spans_full_target(isolated_log: Path):
    start = datetime.now(UTC) - timedelta(days=3)
    append_decision(_record(ts=start.isoformat()))
    out = window_status(target_days=14)
    assert out["days_with_activity"] == 1
    # Grid spans at least from start to start+13 days.
    assert len(out["grid"]) >= 14
    # First grid day is the start date of the first record.
    assert out["grid"][0]["day"] == start.date().isoformat()


def test_window_status_aggregates_per_day(isolated_log: Path):
    today = datetime.now(UTC).replace(hour=12, minute=0, second=0, microsecond=0)
    append_decision(_record(ts=today.isoformat(), submitted=[{"symbol": "SPY"}]))
    append_decision(
        _record(
            ts=(today + timedelta(hours=1)).isoformat(),
            halted=True,
            halt_reasons=["kill_switch:dd"],
            submitted=[],
        )
    )
    out = window_status(target_days=14)
    today_cell = next(c for c in out["grid"] if c["day"] == today.date().isoformat())
    assert today_cell["cycles"] == 2
    assert today_cell["submitted"] == 1
    assert today_cell["halted"] == 1


def test_window_status_future_cells_flagged(isolated_log: Path):
    today = datetime.now(UTC)
    append_decision(_record(ts=today.isoformat()))
    out = window_status(target_days=14)
    futures = [c for c in out["grid"] if c["is_future"]]
    # All future cells must have zero cycle counts.
    assert all(c["cycles"] == 0 for c in futures)
