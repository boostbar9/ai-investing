"""Tests for the outcome attribution module (data foundation of self-improvement).

These tests pin three things that downstream code depends on:
  * sign convention (long vs short)
  * look-ahead safety (no attributing runs that haven't matured)
  * idempotency via decision_id

Plus the ScorecardSummary rollup math used by both the dashboard and the
prompt self-reflection injection.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from packages.agents.attribution import (
    DEFAULT_HORIZONS_DAYS,
    RunScorecard,
    SignalAttribution,
    _signed_return_bps,
    attribute_run,
    compute_scorecard,
    run_attribution,
    summarize_scorecard,
    write_scorecard,
)

# ---------------------------------------------------------------------------
# _signed_return_bps — the core sign convention every downstream metric uses
# ---------------------------------------------------------------------------


def test_signed_return_buy_long_up_is_positive() -> None:
    """A 'buy' that goes up should record positive bps."""
    bps = _signed_return_bps("buy", entry=100.0, exit_=103.0)
    assert bps == pytest.approx(300.0)


def test_signed_return_buy_long_down_is_negative() -> None:
    bps = _signed_return_bps("buy", entry=100.0, exit_=97.0)
    assert bps == pytest.approx(-300.0)


def test_signed_return_sell_short_down_is_positive() -> None:
    """A 'sell' (short) that goes DOWN should record POSITIVE bps —
    the short paid off."""
    bps = _signed_return_bps("sell", entry=100.0, exit_=97.0)
    assert bps == pytest.approx(300.0)


def test_signed_return_sell_short_up_is_negative() -> None:
    bps = _signed_return_bps("sell", entry=100.0, exit_=103.0)
    assert bps == pytest.approx(-300.0)


def test_signed_return_zero_entry_is_zero() -> None:
    """Guard against divide-by-zero in pathological data."""
    assert _signed_return_bps("buy", entry=0.0, exit_=10.0) == 0.0
    assert _signed_return_bps("sell", entry=-1.0, exit_=10.0) == 0.0


# ---------------------------------------------------------------------------
# attribute_run — look-ahead safety + horizon maturation
# ---------------------------------------------------------------------------


def _make_row(
    ts: datetime,
    signals: list[dict],
    *,
    decision_id: str = "dec-1",
    regime: str = "bull",
    used_llm: bool = True,
) -> dict:
    return {
        "decision_id": decision_id,
        "ts": ts.isoformat().replace("+00:00", "Z"),
        "regime": regime,
        "used_llm": used_llm,
        "agents": {"strategy": {"signals": signals}},
    }


def test_attribute_run_skips_if_shortest_horizon_not_matured() -> None:
    """If the run is younger than horizons[0] days, we MUST return None —
    attributing a position we couldn't have closed yet would leak the
    future into the score."""
    now = datetime(2026, 5, 24, tzinfo=UTC)
    fresh = now - timedelta(hours=12)  # younger than 1 day
    row = _make_row(fresh, [{"symbol": "SPY", "side": "buy", "strength": 0.5}])

    out = attribute_run(row, lambda s, t: 100.0, now=now)
    assert out is None


def test_attribute_run_returns_card_when_all_horizons_matured() -> None:
    """When the run is older than the longest horizon, every horizon
    should populate."""
    now = datetime(2026, 5, 24, tzinfo=UTC)
    ts = now - timedelta(days=60)  # well beyond 30d horizon

    # entry=100, exit at +1d=101, +5d=105, +30d=110 → buy long ascending
    def closes(symbol: str, t: datetime) -> float:
        days_from_ts = (t - ts).total_seconds() / 86400
        if days_from_ts < 0.5:
            return 100.0
        if days_from_ts < 1.5:
            return 101.0
        if days_from_ts < 5.5:
            return 105.0
        return 110.0

    row = _make_row(ts, [{"symbol": "SPY", "side": "buy", "strength": 0.4}])
    card = attribute_run(row, closes, now=now)

    assert card is not None
    assert card.decision_id == "dec-1"
    assert card.regime == "bull"
    assert len(card.signals) == 1
    sig = card.signals[0]
    assert sig.symbol == "SPY"
    assert sig.side == "buy"
    assert sig.strength == 0.4
    assert sig.entry_price == 100.0
    # All three horizons should be populated and positive (long, price up).
    assert sig.horizon_returns_bps[1] == pytest.approx(100.0)
    assert sig.horizon_returns_bps[5] == pytest.approx(500.0)
    assert sig.horizon_returns_bps[30] == pytest.approx(1000.0)
    assert sig.hit_5d is True


def test_attribute_run_partial_horizons_omits_unmatured() -> None:
    """A run aged 3 days should have h=1 populated but NOT h=5 or h=30."""
    now = datetime(2026, 5, 24, tzinfo=UTC)
    ts = now - timedelta(days=3)

    def closes(symbol: str, t: datetime) -> float:
        return 100.0 if (t - ts).total_seconds() < 86400 / 2 else 102.0

    row = _make_row(ts, [{"symbol": "SPY", "side": "buy", "strength": 0.5}])
    card = attribute_run(row, closes, now=now)
    assert card is not None
    sig = card.signals[0]
    assert 1 in sig.horizon_returns_bps
    assert 5 not in sig.horizon_returns_bps
    assert 30 not in sig.horizon_returns_bps


def test_attribute_run_no_signals_returns_none() -> None:
    """A run with no strategy signals can't be scored."""
    now = datetime(2026, 5, 24, tzinfo=UTC)
    ts = now - timedelta(days=10)
    row = _make_row(ts, [])
    assert attribute_run(row, lambda s, t: 100.0, now=now) is None


def test_attribute_run_skips_bad_side() -> None:
    """Side must be 'buy' or 'sell'; anything else is a bug-bait edge case."""
    now = datetime(2026, 5, 24, tzinfo=UTC)
    ts = now - timedelta(days=10)
    row = _make_row(ts, [{"symbol": "SPY", "side": "hold", "strength": 0.5}])
    assert attribute_run(row, lambda s, t: 100.0, now=now) is None


def test_attribute_run_missing_decision_id_returns_none() -> None:
    """No decision_id means we can't make it idempotent — refuse to score."""
    now = datetime(2026, 5, 24, tzinfo=UTC)
    ts = now - timedelta(days=10)
    row = _make_row(ts, [{"symbol": "SPY", "side": "buy", "strength": 0.5}])
    row.pop("decision_id")
    assert attribute_run(row, lambda s, t: 100.0, now=now) is None


def test_attribute_run_default_horizons_constant_is_stable() -> None:
    """If someone changes DEFAULT_HORIZONS_DAYS, downstream cron + UI
    assume (1, 5, 30). This test pins that contract."""
    assert DEFAULT_HORIZONS_DAYS == (1, 5, 30)


# ---------------------------------------------------------------------------
# compute_scorecard / run_attribution — idempotency via decision_id
# ---------------------------------------------------------------------------


def test_run_attribution_is_idempotent(tmp_path: Path) -> None:
    """Running attribution twice over the same log should append once."""
    now = datetime(2026, 5, 24, tzinfo=UTC)
    ts = now - timedelta(days=40)
    agent_log = tmp_path / "agents_log.jsonl"
    scorecard = tmp_path / "scorecard.jsonl"

    row = _make_row(ts, [{"symbol": "SPY", "side": "buy", "strength": 0.5}])
    agent_log.write_text(json.dumps(row) + "\n", encoding="utf-8")

    def closes(symbol: str, t: datetime) -> float:
        return 100.0 if (t - ts).total_seconds() < 86400 / 2 else 105.0

    n1 = run_attribution(agent_log, scorecard, closes, now=now)
    n2 = run_attribution(agent_log, scorecard, closes, now=now)
    assert n1 == 1
    assert n2 == 0  # second pass must NOT duplicate

    lines = scorecard.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1


def test_compute_scorecard_respects_skip_decision_ids(tmp_path: Path) -> None:
    """Explicit skip_decision_ids should pre-filter rows."""
    now = datetime(2026, 5, 24, tzinfo=UTC)
    ts = now - timedelta(days=40)
    agent_log = tmp_path / "agents_log.jsonl"

    rows = [
        _make_row(ts, [{"symbol": "SPY", "side": "buy", "strength": 0.5}], decision_id="A"),
        _make_row(ts, [{"symbol": "QQQ", "side": "buy", "strength": 0.6}], decision_id="B"),
    ]
    agent_log.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    cards = compute_scorecard(
        agent_log,
        lambda s, t: 100.0 if (t - ts).total_seconds() < 86400 / 2 else 104.0,
        now=now,
        skip_decision_ids={"A"},
    )
    assert len(cards) == 1
    assert cards[0].decision_id == "B"


def test_write_scorecard_round_trips(tmp_path: Path) -> None:
    """write_scorecard JSON shape must be readable by summarize_scorecard."""
    path = tmp_path / "scorecard.jsonl"
    card = RunScorecard(
        decision_id="X",
        ts="2026-05-01T00:00:00+00:00",
        regime="bull",
        used_llm=True,
        signals=[
            SignalAttribution(
                symbol="SPY",
                side="buy",
                strength=0.5,
                entry_price=100.0,
                horizon_returns_bps={1: 50.0, 5: 250.0},
                horizon_exit_prices={1: 100.5, 5: 102.5},
            )
        ],
    )
    n = write_scorecard([card], path)
    assert n == 1
    text = path.read_text(encoding="utf-8")
    parsed = json.loads(text.strip())
    # JSON keys for horizons must be stringified so jsonl is well-formed.
    assert parsed["signals"][0]["horizon_returns_bps"]["5"] == 250.0


# ---------------------------------------------------------------------------
# summarize_scorecard — feeds the dashboard panel AND the prompt injection
# ---------------------------------------------------------------------------


def test_summarize_empty_scorecard(tmp_path: Path) -> None:
    """A missing file must NOT raise — it just yields a zeroed summary."""
    s = summarize_scorecard(tmp_path / "nope.jsonl")
    assert s.n_runs == 0
    assert s.n_signals == 0
    assert s.hit_rate_5d is None
    assert s.avg_pnl_bps_5d is None
    assert s.regime_bias == {}


def test_summarize_hit_rate_math(tmp_path: Path) -> None:
    """Hit rate = (positive 5d signals) / (scored 5d signals)."""
    path = tmp_path / "scorecard.jsonl"
    rows = [
        # run 1: 2 signals, both up at 5d
        {
            "decision_id": "r1",
            "ts": "2026-05-01T00:00:00+00:00",
            "regime": "bull",
            "signals": [
                {"symbol": "SPY", "side": "buy", "strength": 0.5,
                 "horizon_returns_bps": {"1": 50.0, "5": 250.0}},
                {"symbol": "QQQ", "side": "buy", "strength": 0.5,
                 "horizon_returns_bps": {"1": 30.0, "5": 100.0}},
            ],
        },
        # run 2: 2 signals, one down
        {
            "decision_id": "r2",
            "ts": "2026-05-02T00:00:00+00:00",
            "regime": "chop",
            "signals": [
                {"symbol": "IWM", "side": "buy", "strength": 0.4,
                 "horizon_returns_bps": {"1": -20.0, "5": -150.0}},
                {"symbol": "TLT", "side": "buy", "strength": 0.4,
                 "horizon_returns_bps": {"1": 10.0, "5": 50.0}},
            ],
        },
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    s = summarize_scorecard(path)
    assert s.n_runs == 2
    assert s.n_signals == 4
    # 3 of 4 signals positive at 5d → 0.75
    assert s.hit_rate_5d == pytest.approx(0.75)
    # avg 5d bps = (250 + 100 - 150 + 50) / 4 = 62.5
    assert s.avg_pnl_bps_5d == pytest.approx(62.5)
    assert s.regime_bias == {"bull": 2, "chop": 2}
    assert s.last_run_ts == "2026-05-02T00:00:00+00:00"


def test_summarize_last_n_runs_window(tmp_path: Path) -> None:
    """summarize_scorecard must only look at the last_n_runs window."""
    path = tmp_path / "scorecard.jsonl"
    # 5 old losers + 1 new winner
    rows = [
        {
            "decision_id": f"r{i}",
            "ts": f"2026-04-0{i+1}T00:00:00+00:00",
            "regime": "bull",
            "signals": [{"symbol": "SPY", "side": "buy", "strength": 0.5,
                         "horizon_returns_bps": {"5": -100.0}}],
        }
        for i in range(5)
    ] + [
        {
            "decision_id": "rwin",
            "ts": "2026-05-01T00:00:00+00:00",
            "regime": "bull",
            "signals": [{"symbol": "SPY", "side": "buy", "strength": 0.5,
                         "horizon_returns_bps": {"5": 200.0}}],
        }
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    # Last 1 run only → should see only the winner.
    s = summarize_scorecard(path, last_n_runs=1)
    assert s.n_runs == 1
    assert s.hit_rate_5d == 1.0
    assert s.avg_pnl_bps_5d == pytest.approx(200.0)


def test_summarize_jsonable_serializes_cleanly(tmp_path: Path) -> None:
    """to_jsonable must round-trip through json.dumps — the cockpit
    endpoint relies on this."""
    path = tmp_path / "scorecard.jsonl"
    s = summarize_scorecard(path)  # empty
    payload = s.to_jsonable()
    blob = json.dumps(payload)  # must not raise
    assert "n_runs" in blob
