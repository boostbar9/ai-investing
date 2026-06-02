"""Phase 28 tests — outcome labeler.

Locks in:
  * Pure functions: signed_return, label_correct, make_pick_id.
  * Entry/exit resolution handles weekends, missing horizons, empty bars.
  * is_pick_settled is conservative (waits past longest horizon + buffer).
  * label_pick produces a complete Outcome with correct returns.
  * Backfill is idempotent: re-running doesn't duplicate.
  * Agent-vote join uses status=="ok".
  * Per-agent scores: win rate, avg returns aggregate correctly.
  * Summary stats: by-regime breakdown.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from packages.data.adapters.base import Bar
from packages.learning.outcome_labeler import (
    DEFAULT_HORIZONS,
    BackfillReport,
    Outcome,
    Pick,
    append_outcome,
    backfill_outcomes,
    is_pick_settled,
    iter_picks_from_predictions,
    label_correct,
    label_pick,
    load_agent_votes,
    load_existing_pick_ids,
    load_outcomes,
    make_pick_id,
    per_agent_scores,
    resolve_entry_and_exits,
    signed_return,
    summary_stats,
)


# --- Helpers ---------------------------------------------------------------


def _mk_bars(start: date, prices: Sequence[float], symbol: str = "AAPL") -> list[Bar]:
    """Build a list of daily Bars starting on ``start`` (skipping weekends)."""
    bars: list[Bar] = []
    cur = start
    for px in prices:
        # Skip Sat/Sun to look more realistic.
        while cur.weekday() >= 5:
            cur = cur + timedelta(days=1)
        ts = datetime(cur.year, cur.month, cur.day, 16, 0, tzinfo=UTC)
        bars.append(
            Bar(
                symbol=symbol,
                ts=ts,
                open=px,
                high=px,
                low=px,
                close=px,
                volume=1_000_000,
            )
        )
        cur = cur + timedelta(days=1)
    return bars


def _mk_pick(
    *,
    ts: datetime,
    symbol: str = "AAPL",
    target_weight: float = 0.5,
    decision_id: str = "dec-1",
    regime: str = "chop",
) -> Pick:
    return Pick(
        pick_id=make_pick_id(decision_id, symbol),
        decision_id=decision_id,
        ts=ts,
        symbol=symbol,
        target_weight=target_weight,
        predicted_pnl=100.0,
        strategy="mean-reversion",
        regime=regime,
    )


@dataclass
class _FakeAdapter:
    """Has the minimum surface label_pick uses: get_daily_bars."""
    bars_by_symbol: dict[str, list[Bar]]
    calls: list[str] = None

    def __post_init__(self):
        if self.calls is None:
            self.calls = []

    async def get_daily_bars(self, symbol: str, range_: str = "5y"):
        self.calls.append(symbol)
        return list(self.bars_by_symbol.get(symbol.upper(), []))


# --- Pure functions --------------------------------------------------------


def test_signed_return_basic():
    assert signed_return(100.0, 110.0) == pytest.approx(0.1)
    assert signed_return(100.0, 90.0) == pytest.approx(-0.1)
    # Defensive: 0 or negative entry → 0.0 (avoid div-by-zero).
    assert signed_return(0.0, 100.0) == 0.0
    assert signed_return(-5.0, 100.0) == 0.0


def test_label_correct_long_and_short_and_flat():
    assert label_correct(0.5, 0.02) is True
    assert label_correct(0.5, -0.02) is False
    assert label_correct(-0.5, -0.02) is True
    assert label_correct(-0.5, 0.02) is False
    assert label_correct(0.0, 0.02) is None
    assert label_correct(0.5, None) is None


def test_make_pick_id_is_deterministic_and_case_insensitive():
    a = make_pick_id("dec-1", "AAPL")
    b = make_pick_id("dec-1", "aapl")
    c = make_pick_id("dec-2", "AAPL")
    assert a == b
    assert a != c
    assert len(a) == 16


# --- Settlement & resolution ----------------------------------------------


def test_is_pick_settled_waits_past_longest_horizon():
    now = datetime(2026, 6, 1, tzinfo=UTC)
    # Picked yesterday — not settled.
    assert not is_pick_settled(_mk_pick(ts=now - timedelta(days=1)), now=now)
    # Picked 40 days ago — past the 20*1.5 + 4 = 34 day buffer.
    assert is_pick_settled(_mk_pick(ts=now - timedelta(days=40)), now=now)


def test_is_pick_settled_handles_naive_now():
    # A naive ``now`` shouldn't crash the comparison.
    now = datetime(2026, 6, 1)
    assert is_pick_settled(
        _mk_pick(ts=datetime(2026, 4, 1, tzinfo=UTC)), now=now
    )


def test_resolve_entry_and_exits_skips_weekends():
    """Pick on a Saturday → entry is the next Monday's close."""
    bars = _mk_bars(date(2026, 5, 1), [100, 101, 102, 103, 104, 105])
    # Pick on a non-trading day (Saturday 2026-05-02):
    entry, exits = resolve_entry_and_exits(bars, date(2026, 5, 2),
                                            horizons=(1, 2))
    assert entry is not None
    # Entry should be Monday's bar (still index 1 in our generator).
    assert entry.close == 101  # Bar 1 (Mon 2026-05-04 — first bar after Sat)
    # Exit horizons are entry_idx+N.
    assert exits[1].close == 102
    assert exits[2].close == 103


def test_resolve_entry_and_exits_missing_long_horizon():
    bars = _mk_bars(date(2026, 5, 1), [100, 101, 102])  # only 3 bars
    entry, exits = resolve_entry_and_exits(bars, date(2026, 5, 1),
                                            horizons=(1, 5))
    assert entry is not None
    assert exits[1] is not None
    assert exits[5] is None  # too short


def test_resolve_entry_and_exits_empty_bars():
    entry, exits = resolve_entry_and_exits([], date(2026, 5, 1))
    assert entry is None
    assert all(v is None for v in exits.values())


# --- label_pick (full path) -----------------------------------------------


@pytest.mark.asyncio
async def test_label_pick_produces_complete_outcome():
    # 25 trading days of rising prices: entry=100, +1d=101, +5d=105, +20d=120
    prices = [100.0 + i for i in range(25)]
    bars = _mk_bars(date(2026, 5, 1), prices)
    adapter = _FakeAdapter(bars_by_symbol={"AAPL": bars})

    pick = _mk_pick(
        ts=datetime(2026, 5, 1, 13, 0, tzinfo=UTC),
        target_weight=0.5,
    )
    outcome = await label_pick(
        pick, adapter, agents_voted=("research", "strategy"),
        now=datetime(2026, 7, 1, tzinfo=UTC),
    )
    assert outcome is not None
    assert outcome.symbol == "AAPL"
    assert outcome.entry_price == 100.0
    assert outcome.exit_price_1d == 101.0
    assert outcome.exit_price_5d == 105.0
    assert outcome.exit_price_20d == 120.0
    assert outcome.return_1d == pytest.approx(0.01)
    assert outcome.return_5d == pytest.approx(0.05)
    assert outcome.return_20d == pytest.approx(0.20)
    assert outcome.correct is True  # long pick, +5d > 0
    assert outcome.agents_voted == ("research", "strategy")  # sorted
    assert outcome.regime_at_pick == "chop"
    # Sanity: confidence is |target_weight|.
    assert outcome.confidence == 0.5


@pytest.mark.asyncio
async def test_label_pick_short_pick_correct_when_price_falls():
    prices = [100.0 - i * 0.5 for i in range(25)]  # downward
    bars = _mk_bars(date(2026, 5, 1), prices)
    adapter = _FakeAdapter(bars_by_symbol={"AAPL": bars})
    pick = _mk_pick(
        ts=datetime(2026, 5, 1, 13, 0, tzinfo=UTC),
        target_weight=-0.5,  # short
    )
    outcome = await label_pick(
        pick, adapter, now=datetime(2026, 7, 1, tzinfo=UTC),
    )
    assert outcome is not None
    assert outcome.return_5d < 0
    # Short pick + falling price = correct
    assert outcome.correct is True


@pytest.mark.asyncio
async def test_label_pick_returns_none_when_horizon_not_settled():
    """Only 10 bars of data — 20-day horizon can't resolve, return None."""
    bars = _mk_bars(date(2026, 5, 1), [100 + i for i in range(10)])
    adapter = _FakeAdapter(bars_by_symbol={"AAPL": bars})
    pick = _mk_pick(ts=datetime(2026, 5, 1, tzinfo=UTC))
    outcome = await label_pick(pick, adapter)
    assert outcome is None


@pytest.mark.asyncio
async def test_label_pick_returns_none_when_no_bars():
    adapter = _FakeAdapter(bars_by_symbol={})
    pick = _mk_pick(ts=datetime(2026, 5, 1, tzinfo=UTC))
    outcome = await label_pick(pick, adapter)
    assert outcome is None


# --- Agent-vote join -------------------------------------------------------


def test_load_agent_votes_filters_to_status_ok(tmp_path: Path):
    log = tmp_path / "agents_log.jsonl"
    rows = [
        {
            "decision_id": "d1",
            "agents": {
                "research": {"status": "ok"},
                "strategy": {"status": "ok"},
                "discovery": {"status": "idle"},   # excluded
                "risk": {"status": "halted"},      # excluded
            },
        },
        {
            "decision_id": "d2",
            "agents": {"execution": {"status": "ok"}},
        },
    ]
    log.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    votes = load_agent_votes(log)
    assert set(votes["d1"]) == {"research", "strategy"}
    assert votes["d2"] == ["execution"]


def test_load_agent_votes_missing_file_returns_empty(tmp_path: Path):
    assert load_agent_votes(tmp_path / "no_such.jsonl") == {}


def test_iter_picks_from_predictions_skips_malformed(tmp_path: Path):
    preds = tmp_path / "predictions.jsonl"
    rows = [
        # valid
        {
            "ts": "2026-05-01T12:00:00+00:00",
            "symbol": "AAPL",
            "decision_id": "d1",
            "target_weight": 0.5,
            "predicted_pnl": 12.0,
            "strategy": "mean-reversion",
            "regime": "chop",
        },
        # missing decision_id
        {"ts": "2026-05-02T12:00:00+00:00", "symbol": "MSFT", "target_weight": 0.3},
        # invalid ts → skipped silently
        {"ts": "not-a-date", "symbol": "TLT", "decision_id": "d3"},
    ]
    preds.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n" + "not json\n",
        encoding="utf-8",
    )
    out = list(iter_picks_from_predictions(preds))
    assert len(out) == 1
    assert out[0].symbol == "AAPL"
    assert out[0].decision_id == "d1"


# --- Persistence + idempotency --------------------------------------------


@pytest.mark.asyncio
async def test_backfill_is_idempotent(tmp_path: Path):
    # Build inputs: one settled pick, one fresh pick.
    settled_ts = datetime(2026, 5, 1, 12, 0, tzinfo=UTC)
    fresh_ts = datetime(2026, 7, 15, 12, 0, tzinfo=UTC)
    now = datetime(2026, 7, 20, tzinfo=UTC)

    preds = tmp_path / "predictions.jsonl"
    preds.write_text(
        "\n".join(
            json.dumps(r) for r in [
                {
                    "ts": settled_ts.isoformat(),
                    "symbol": "AAPL",
                    "decision_id": "d-settled",
                    "target_weight": 0.5,
                    "predicted_pnl": 10.0,
                    "strategy": "mean-reversion",
                    "regime": "chop",
                },
                {
                    "ts": fresh_ts.isoformat(),
                    "symbol": "MSFT",
                    "decision_id": "d-fresh",
                    "target_weight": 0.5,
                    "predicted_pnl": 5.0,
                    "strategy": "mean-reversion",
                    "regime": "chop",
                },
            ]
        ) + "\n",
        encoding="utf-8",
    )
    agents_log = tmp_path / "agents_log.jsonl"
    agents_log.write_text(
        json.dumps({
            "decision_id": "d-settled",
            "agents": {"research": {"status": "ok"}, "strategy": {"status": "ok"}},
        }) + "\n",
        encoding="utf-8",
    )
    outcomes_path = tmp_path / "outcomes.jsonl"

    bars = _mk_bars(date(2026, 5, 1), [100.0 + i for i in range(25)])
    adapter = _FakeAdapter(bars_by_symbol={"AAPL": bars})

    # First run.
    report = await backfill_outcomes(
        adapter,
        predictions_path=preds,
        agents_log_path=agents_log,
        outcomes_path=outcomes_path,
        now=now,
    )
    assert isinstance(report, BackfillReport)
    assert report.labeled == 1
    assert report.skipped_unsettled == 1
    assert outcomes_path.exists()
    rows = load_outcomes(outcomes_path)
    assert len(rows) == 1
    assert rows[0]["symbol"] == "AAPL"
    assert set(rows[0]["agents_voted"]) == {"research", "strategy"}
    assert rows[0]["correct"] is True

    # Second run with same inputs: must not duplicate.
    report2 = await backfill_outcomes(
        adapter,
        predictions_path=preds,
        agents_log_path=agents_log,
        outcomes_path=outcomes_path,
        now=now,
    )
    assert report2.labeled == 0
    assert report2.skipped_already_labeled == 1
    rows2 = load_outcomes(outcomes_path)
    assert len(rows2) == 1


def test_load_existing_pick_ids_handles_missing_and_malformed(tmp_path: Path):
    # Missing file.
    assert load_existing_pick_ids(tmp_path / "missing.jsonl") == set()
    # File with one valid and one malformed line.
    p = tmp_path / "outcomes.jsonl"
    p.write_text(
        json.dumps({"pick_id": "abc"}) + "\nnot-json\n" + json.dumps({"pick_id": "xyz"}) + "\n",
        encoding="utf-8",
    )
    assert load_existing_pick_ids(p) == {"abc", "xyz"}


# --- Aggregation -----------------------------------------------------------


def test_per_agent_scores_basic():
    outcomes = [
        {
            "agents_voted": ["research", "strategy"],
            "return_5d": 0.05, "return_20d": 0.15,
            "correct": True,
        },
        {
            "agents_voted": ["research"],
            "return_5d": -0.02, "return_20d": -0.05,
            "correct": False,
        },
        {
            "agents_voted": ["strategy"],
            "return_5d": 0.03, "return_20d": 0.04,
            "correct": True,
        },
    ]
    scores = per_agent_scores(outcomes)
    by_name = {s.agent: s for s in scores}
    # strategy: 2 picks, 2 wins → win_rate 1.0
    assert by_name["strategy"].picks == 2
    assert by_name["strategy"].wins == 2
    assert by_name["strategy"].win_rate == 1.0
    # research: 2 picks (1 win, 1 loss) → 0.5
    assert by_name["research"].picks == 2
    assert by_name["research"].win_rate == 0.5
    # Sort order: strategy first (higher win_rate).
    assert scores[0].agent == "strategy"


def test_summary_stats_breaks_down_by_regime():
    outcomes = [
        {"regime_at_pick": "risk_on", "return_5d": 0.04, "return_20d": 0.10, "correct": True},
        {"regime_at_pick": "risk_on", "return_5d": -0.01, "return_20d": -0.02, "correct": False},
        {"regime_at_pick": "chop", "return_5d": 0.01, "return_20d": 0.02, "correct": True},
    ]
    s = summary_stats(outcomes)
    assert s["total_picks"] == 3
    assert s["decided_picks"] == 3
    # 2 wins out of 3 decided.
    assert s["win_rate"] == pytest.approx(2 / 3, rel=1e-4)
    assert s["avg_return_5d"] == pytest.approx((0.04 - 0.01 + 0.01) / 3, rel=1e-4)
    # By-regime: risk_on has 2 picks (1 win), chop has 1 pick (1 win).
    assert s["by_regime"]["risk_on"]["picks"] == 2
    assert s["by_regime"]["risk_on"]["win_rate"] == 0.5
    assert s["by_regime"]["chop"]["picks"] == 1
    assert s["by_regime"]["chop"]["win_rate"] == 1.0


def test_summary_stats_empty():
    s = summary_stats([])
    assert s["total_picks"] == 0
    assert s["decided_picks"] == 0
    assert s["win_rate"] == 0.0
    assert s["by_regime"] == {}


def test_append_and_load_outcome_roundtrip(tmp_path: Path):
    path = tmp_path / "outcomes.jsonl"
    o = Outcome(
        pick_id="p1",
        decision_id="d1",
        ts="2026-05-01T12:00:00+00:00",
        symbol="AAPL",
        confidence=0.5,
        regime_at_pick="chop",
        agents_voted=("research",),
        strategy="mean-reversion",
        entry_price=100.0,
        entry_date="2026-05-01",
        exit_price_1d=101.0,
        exit_price_5d=105.0,
        exit_price_20d=120.0,
        return_1d=0.01,
        return_5d=0.05,
        return_20d=0.20,
        correct=True,
        labeled_at="2026-07-01T00:00:00+00:00",
    )
    append_outcome(o, path)
    rows = load_outcomes(path)
    assert len(rows) == 1
    assert rows[0]["pick_id"] == "p1"
    assert rows[0]["agents_voted"] == ["research"]
