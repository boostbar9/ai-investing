"""Tests for the kill-switch logic in tools/paper_trade.py.

The kill-switch checks live in a tool script, not a package module, but they
are pure functions so we can import + exercise them in unit tests by adding
the repo root to sys.path. This guards the most safety-critical bit of the
paper trading runner: the thing that decides whether ANY orders go out.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from tools.paper_trade import (  # noqa: E402
    KillSwitchResult,
    check_kill_switches,
    update_session_peak,
)


@pytest.fixture(autouse=True)
def _isolate_peak_file(tmp_path, monkeypatch):
    """Each test gets a fresh on-disk peak file."""
    fake = tmp_path / "session_peak.json"
    import tools.paper_trade as pt
    monkeypatch.setattr(pt, "EQUITY_PEAK_FILE", fake)
    monkeypatch.setattr(pt, "PAPER_LOG_DIR", tmp_path)
    yield


def test_halt_when_enable_flag_missing(monkeypatch):
    monkeypatch.delenv("ENABLE_PAPER_TRADING", raising=False)
    out = check_kill_switches({"status": "ACTIVE", "equity": 100000, "last_equity": 100000,
                                "buying_power": 200000, "long_market_value": 0})
    assert out.halt is True
    assert any("ENABLE_PAPER_TRADING" in r for r in out.reasons)


def test_halt_when_account_not_active(monkeypatch):
    monkeypatch.setenv("ENABLE_PAPER_TRADING", "true")
    out = check_kill_switches({"status": "SUSPENDED", "equity": 100000, "last_equity": 100000,
                                "buying_power": 200000, "long_market_value": 0})
    assert out.halt is True
    assert any("status" in r for r in out.reasons)


def test_halt_on_drawdown_threshold(monkeypatch, tmp_path):
    monkeypatch.setenv("ENABLE_PAPER_TRADING", "true")
    # Seed a high session peak, then come in with much lower equity.
    import tools.paper_trade as pt
    pt.EQUITY_PEAK_FILE.write_text(json.dumps({"peak": 100_000.0, "updated_at": "now"}))
    out = check_kill_switches({
        "status": "ACTIVE",
        "equity": 90_000,        # 10% drawdown from 100k peak
        "last_equity": 95_000,
        "buying_power": 100_000,
        "long_market_value": 0,
    })
    assert out.halt is True
    assert any("DD" in r for r in out.reasons)


def test_halt_on_margin_utilization(monkeypatch):
    monkeypatch.setenv("ENABLE_PAPER_TRADING", "true")
    out = check_kill_switches({
        "status": "ACTIVE",
        "equity": 100_000,
        "last_equity": 100_000,
        "buying_power": 1_000,       # almost out of room
        "long_market_value": 199_000,
    })
    assert out.halt is True
    assert any("margin" in r for r in out.reasons)


def test_no_halt_when_healthy(monkeypatch):
    monkeypatch.setenv("ENABLE_PAPER_TRADING", "true")
    out = check_kill_switches({
        "status": "ACTIVE",
        "equity": 100_000,
        "last_equity": 100_000,
        "buying_power": 200_000,
        "long_market_value": 0,
    })
    assert isinstance(out, KillSwitchResult)
    assert out.halt is False
    assert out.reasons == []


def test_session_peak_only_increases(monkeypatch):
    """Peak is sticky: a dip then rebound preserves the highest seen."""
    p1 = update_session_peak(100_000)
    assert p1 == 100_000
    p2 = update_session_peak(90_000)
    assert p2 == 100_000  # still the old peak
    p3 = update_session_peak(105_000)
    assert p3 == 105_000  # new peak overrides
    p4 = update_session_peak(95_000)
    assert p4 == 105_000
