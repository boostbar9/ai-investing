"""Phase 11 — tests for packages.paper.sim_pnl.

Derive synthetic round-trip stream from runs.jsonl + merge with real
shadow trades + build a daily equity curve.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from packages.paper import sim_pnl as sim_mod
from packages.paper.sim_pnl import (
    daily_equity_curve,
    merge_real_and_synth,
    synth_trades_from_runs,
)
from packages.shadow.pairing import pair_round_trips


@pytest.fixture
def isolated_runs(monkeypatch, tmp_path) -> Path:
    p = tmp_path / "runs.jsonl"
    monkeypatch.setattr(sim_mod, "DEFAULT_RUNS_PATH", p)
    return p


def _write_runs(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


# ---------------------------------------------------------------------------
# synth_trades_from_runs
# ---------------------------------------------------------------------------


def test_synth_empty_when_no_runs(isolated_runs: Path):
    assert synth_trades_from_runs() == []


def test_synth_extracts_planned_orders(isolated_runs: Path):
    _write_runs(
        isolated_runs,
        [
            {
                "ts": "2026-05-01T10:00:00+00:00",
                "strategy": "ensemble",
                "halted": False,
                "dry_run": True,
                "orders_planned": [
                    {"symbol": "SPY", "side": "buy", "qty": 10, "last_price": 100.0},
                    {"symbol": "QQQ", "side": "buy", "qty": 5, "last_price": 200.0},
                ],
            }
        ],
    )
    synth = synth_trades_from_runs()
    assert len(synth) == 2
    spy = next(s for s in synth if s["symbol"] == "SPY")
    assert spy["side"] == "buy"
    assert spy["qty"] == 10.0
    assert spy["limit_price"] == 100.0
    assert spy["synthetic"] is True
    assert spy["strategy"] == "ensemble"


def test_synth_skips_halted_cycles(isolated_runs: Path):
    _write_runs(
        isolated_runs,
        [
            {
                "ts": "2026-05-01T10:00:00+00:00",
                "halted": True,
                "orders_planned": [
                    {"symbol": "SPY", "side": "buy", "qty": 10, "last_price": 100.0},
                ],
            }
        ],
    )
    assert synth_trades_from_runs() == []


def test_synth_dry_run_excluded_when_flag_off(isolated_runs: Path):
    _write_runs(
        isolated_runs,
        [
            {
                "ts": "2026-05-01T10:00:00+00:00",
                "dry_run": True,
                "halted": False,
                "orders_planned": [
                    {"symbol": "SPY", "side": "buy", "qty": 10, "last_price": 100.0},
                ],
            }
        ],
    )
    assert synth_trades_from_runs(include_dry_run=False) == []
    assert len(synth_trades_from_runs(include_dry_run=True)) == 1


def test_synth_skips_missing_price(isolated_runs: Path):
    _write_runs(
        isolated_runs,
        [
            {
                "ts": "2026-05-01T10:00:00+00:00",
                "halted": False,
                "orders_planned": [
                    {"symbol": "SPY", "side": "buy", "qty": 10},  # no price
                    {"symbol": "QQQ", "side": "buy", "qty": 5, "last_price": 200.0},
                ],
            }
        ],
    )
    synth = synth_trades_from_runs()
    assert [s["symbol"] for s in synth] == ["QQQ"]


def test_synth_skips_unknown_side(isolated_runs: Path):
    _write_runs(
        isolated_runs,
        [
            {
                "ts": "2026-05-01T10:00:00+00:00",
                "halted": False,
                "orders_planned": [
                    {"symbol": "SPY", "side": "hold", "qty": 10, "last_price": 100.0},
                    {"symbol": "QQQ", "side": "sell", "qty": 5, "last_price": 200.0},
                ],
            }
        ],
    )
    synth = synth_trades_from_runs()
    assert [s["symbol"] for s in synth] == ["QQQ"]


def test_synth_skips_zero_or_negative_qty(isolated_runs: Path):
    _write_runs(
        isolated_runs,
        [
            {
                "ts": "2026-05-01T10:00:00+00:00",
                "halted": False,
                "orders_planned": [
                    {"symbol": "SPY", "side": "buy", "qty": 0, "last_price": 100.0},
                    {"symbol": "QQQ", "side": "buy", "qty": -3, "last_price": 200.0},
                    {"symbol": "IWM", "side": "buy", "qty": 1, "last_price": 50.0},
                ],
            }
        ],
    )
    synth = synth_trades_from_runs()
    assert [s["symbol"] for s in synth] == ["IWM"]


def test_synth_skips_runs_without_ts(isolated_runs: Path):
    _write_runs(
        isolated_runs,
        [
            {
                "halted": False,
                "orders_planned": [
                    {"symbol": "SPY", "side": "buy", "qty": 1, "last_price": 100.0},
                ],
            }
        ],
    )
    assert synth_trades_from_runs() == []


def test_synth_explicit_path_overrides_default(tmp_path: Path):
    p = tmp_path / "custom.jsonl"
    _write_runs(
        p,
        [
            {
                "ts": "2026-05-01T10:00:00+00:00",
                "halted": False,
                "orders_planned": [
                    {"symbol": "SPY", "side": "buy", "qty": 1, "last_price": 100.0}
                ],
            }
        ],
    )
    assert len(synth_trades_from_runs(p)) == 1


def test_synth_handles_malformed_lines(isolated_runs: Path):
    isolated_runs.parent.mkdir(parents=True, exist_ok=True)
    isolated_runs.write_text(
        json.dumps(
            {
                "ts": "2026-05-01T10:00:00+00:00",
                "halted": False,
                "orders_planned": [
                    {"symbol": "SPY", "side": "buy", "qty": 1, "last_price": 100.0}
                ],
            }
        )
        + "\nnot-json\n"
    )
    assert len(synth_trades_from_runs()) == 1


# ---------------------------------------------------------------------------
# merge_real_and_synth
# ---------------------------------------------------------------------------


def test_merge_real_only():
    real = [{"symbol": "SPY", "ts": "x", "side": "buy"}]
    out = merge_real_and_synth(real, [])
    assert out == real


def test_merge_synth_only():
    synth = [{"symbol": "SPY", "ts": "x", "side": "buy", "synthetic": True}]
    out = merge_real_and_synth([], synth)
    assert out == synth


def test_merge_real_wins_on_duplicate_key():
    real = [{"symbol": "SPY", "ts": "x", "side": "buy", "limit_price": 100.0}]
    synth = [
        {
            "symbol": "SPY",
            "ts": "x",
            "side": "buy",
            "limit_price": 999.0,
            "synthetic": True,
        }
    ]
    out = merge_real_and_synth(real, synth)
    assert len(out) == 1
    assert out[0]["limit_price"] == 100.0


def test_merge_concatenates_non_duplicates():
    real = [{"symbol": "SPY", "ts": "x", "side": "buy"}]
    synth = [{"symbol": "QQQ", "ts": "y", "side": "buy", "synthetic": True}]
    out = merge_real_and_synth(real, synth)
    assert len(out) == 2


def test_merge_dedup_is_case_insensitive_on_symbol():
    real = [{"symbol": "spy", "ts": "x", "side": "buy"}]
    synth = [{"symbol": "SPY", "ts": "x", "side": "buy", "synthetic": True}]
    out = merge_real_and_synth(real, synth)
    assert len(out) == 1
    assert out[0].get("synthetic") is not True


def test_merge_none_inputs_safe():
    assert merge_real_and_synth(None, None) == []  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# daily_equity_curve
# ---------------------------------------------------------------------------


def _round_trip(buy_day: str, sell_day: str, pnl: float, qty: float = 1.0, symbol: str = "SPY"):
    """Build a pair of trades that pair_round_trips will pair with the
    given PnL on the sell day."""
    buy_px = 100.0
    sell_px = 100.0 + pnl / qty
    return [
        {
            "ts": f"{buy_day}T10:00:00+00:00",
            "symbol": symbol,
            "side": "buy",
            "qty": qty,
            "limit_price": buy_px,
        },
        {
            "ts": f"{sell_day}T15:00:00+00:00",
            "symbol": symbol,
            "side": "sell",
            "qty": qty,
            "limit_price": sell_px,
        },
    ]


def test_equity_curve_empty_returns_starting_only():
    out = daily_equity_curve([], starting_equity=100_000)
    assert out == [{"day": None, "equity": 100_000.0}]


def test_equity_curve_cumulates_pnl():
    trades = _round_trip("2026-05-01", "2026-05-02", pnl=10.0) + _round_trip(
        "2026-05-03", "2026-05-04", pnl=-3.0
    )
    paired = pair_round_trips(trades)
    out = daily_equity_curve(paired, starting_equity=100_000)
    # Daily aggregator fills gaps -> may include zero-PnL middle day.
    equities = [row["equity"] for row in out]
    assert equities[-1] == pytest.approx(100_000 + 10 - 3)
    assert min(equities) >= 100_000 - 3


def test_equity_curve_uses_start_equity():
    trades = _round_trip("2026-05-01", "2026-05-02", pnl=5.0)
    paired = pair_round_trips(trades)
    out = daily_equity_curve(paired, starting_equity=50_000)
    assert out[-1]["equity"] == pytest.approx(50_005.0)
