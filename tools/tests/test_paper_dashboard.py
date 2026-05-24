"""Tests for the paper-trading dashboard generator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.paper_dashboard import (
    build_chart_data,
    compute_summary,
    load_runs,
    render,
)


def _runs() -> list[dict]:
    return [
        {
            "ts": "2026-05-20T20:00:00+00:00",
            "strategy": "mean-reversion",
            "dry_run": True,
            "halted": False,
            "account_equity": 100000.0,
            "orders_submitted": [],
        },
        {
            "ts": "2026-05-21T20:00:00+00:00",
            "strategy": "mean-reversion",
            "dry_run": False,
            "halted": False,
            "account_equity": 101000.0,
            "orders_submitted": [{"symbol": "SPY", "qty": 1}],
        },
        {
            "ts": "2026-05-22T20:00:00+00:00",
            "strategy": "trend-following",
            "dry_run": False,
            "halted": True,
            "account_equity": 95000.0,
            "orders_submitted": [],
        },
    ]


def test_load_runs_handles_missing_file(tmp_path: Path) -> None:
    assert load_runs(tmp_path / "missing.jsonl") == []


def test_load_runs_skips_malformed_lines(tmp_path: Path) -> None:
    p = tmp_path / "runs.jsonl"
    p.write_text(
        json.dumps({"ts": "2026-05-20T20:00:00+00:00", "account_equity": 1.0}) + "\n"
        + "this is not json\n"
        + json.dumps({"ts": "2026-05-21T20:00:00+00:00", "account_equity": 2.0}) + "\n"
    )
    runs = load_runs(p)
    assert len(runs) == 2
    assert runs[0]["ts"] < runs[1]["ts"]


def test_compute_summary_empty() -> None:
    s = compute_summary([])
    assert s["total_runs"] == 0
    assert s["max_dd_pct"] == 0.0
    assert s["strategies"] == {}


def test_compute_summary_drawdown() -> None:
    s = compute_summary(_runs())
    assert s["total_runs"] == 3
    assert s["halted_runs"] == 1
    assert s["current_equity"] == 95000.0
    assert s["peak_equity"] == 101000.0
    assert s["starting_equity"] == 100000.0
    # peak 101000 -> trough 95000 = ~5.94% dd
    assert s["max_dd_pct"] == pytest.approx(5.940594, rel=1e-3)
    assert s["total_pnl"] == pytest.approx(-5000.0)
    assert s["trading_days"] == 3
    assert s["strategies"]["mean-reversion"]["runs"] == 2
    assert s["strategies"]["mean-reversion"]["orders"] == 1
    assert s["strategies"]["trend-following"]["halts"] == 1


def test_build_chart_data_aligned() -> None:
    c = build_chart_data(_runs())
    assert len(c["labels"]) == 3
    assert c["equity"] == [100000.0, 101000.0, 95000.0]
    assert c["peak"] == [100000.0, 101000.0, 101000.0]
    # final drawdown ~ -5.94%
    assert c["drawdown_pct"][-1] == pytest.approx(-5.940594, rel=1e-3)
    # first pnl is 0 (no prev), then +1000, then -6000
    assert c["pnl_per_run"][0] == 0.0
    assert c["pnl_per_run"][1] == pytest.approx(1000.0)
    assert c["pnl_per_run"][2] == pytest.approx(-6000.0)


def test_render_produces_valid_html_with_runs() -> None:
    html = render(_runs())
    assert "<!doctype html>" in html
    assert "Paper Trading Dashboard" in html
    assert "mean-reversion" in html
    assert "trend-following" in html
    # halt counter shows up
    assert ">1<" in html  # halted_runs cell
    # chart.js loaded
    assert "chart.js" in html.lower()


def test_render_handles_empty_runs() -> None:
    html = render([])
    assert "<!doctype html>" in html
    assert "No runs yet" in html or "no runs" in html.lower() or "0" in html
