"""Tests for the Trading Controls guardrail (budget, confidence gate,
pending queue) and its API.

Covers:
  * evaluate_trade_against_controls — every gate, clamping, budget
    accumulation across a pass.
  * load/update_controls — clamping, fail-safe, preset<->min_confidence
    linkage, and that budget routes through the onboarding cap.
  * pending_trades store — upsert/dedupe/resolve.
  * GET/POST /api/trading-controls + /api/trading-controls/pending.
"""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from packages.cockpit import onboarding as ob
from packages.cockpit import pending_trades as pq
from packages.cockpit import trading_controls as tc
from packages.cockpit.web import server as srv


# ---------------------------------------------------------------------------
# Fixtures — isolate all on-disk state to tmp paths.
# ---------------------------------------------------------------------------
@pytest.fixture
def isolated(monkeypatch, tmp_path):
    ob_path = tmp_path / "onboarding.json"
    tc_path = tmp_path / "trading_controls.json"
    pq_path = tmp_path / "pending_trades.jsonl"
    monkeypatch.setattr(ob, "ONBOARDING_PATH", ob_path)
    monkeypatch.setattr(tc, "TRADING_CONTROLS_PATH", tc_path)
    monkeypatch.setattr(pq, "PENDING_TRADES_PATH", pq_path)
    return {"ob": ob_path, "tc": tc_path, "pq": pq_path}


@pytest.fixture
def client():
    return TestClient(srv.app)


def _controls(**kw):
    base = dict(
        total_budget_usd=300.0,
        max_per_trade_usd=50.0,
        max_trades_per_day=5,
        max_open_positions=3,
        min_confidence=0.60,
        risk_preset="balanced",
        pending_mode="auto_when_qualified",
    )
    base.update(kw)
    return tc.TradingControls(**base)


def _cand(**kw):
    base = dict(symbol="NVDA", side="buy", confidence=0.8, notional=40.0)
    base.update(kw)
    return tc.TradeCandidate(**base)


def _state(**kw):
    base = dict(used_budget_usd=0.0, open_positions=0, trades_today=0)
    base.update(kw)
    return tc.PortfolioState(**base)


# ---------------------------------------------------------------------------
# evaluate_trade_against_controls — gates
# ---------------------------------------------------------------------------
def test_qualifies_when_all_gates_pass():
    v = tc.evaluate_trade_against_controls(_cand(), _controls(), _state())
    assert v.qualifies is True
    assert v.reasons == []
    assert v.clamped_notional == pytest.approx(40.0)


def test_confidence_gate_blocks_low_confidence():
    v = tc.evaluate_trade_against_controls(
        _cand(confidence=0.52), _controls(min_confidence=0.60), _state()
    )
    assert v.qualifies is False
    assert any("52%" in r and "60%" in r for r in v.reasons)


def test_per_trade_cap_clamps_notional():
    v = tc.evaluate_trade_against_controls(
        _cand(notional=500.0), _controls(max_per_trade_usd=50.0), _state()
    )
    # Clamped down to the per-trade cap; still qualifies (size reduced).
    assert v.clamped_notional == pytest.approx(50.0)
    assert v.qualifies is True


def test_budget_gate_blocks_when_fully_allocated():
    v = tc.evaluate_trade_against_controls(
        _cand(notional=40.0),
        _controls(total_budget_usd=300.0),
        _state(used_budget_usd=300.0),
    )
    assert v.qualifies is False
    assert any("budget" in r.lower() for r in v.reasons)
    assert v.clamped_notional == pytest.approx(0.0)


def test_budget_remaining_clamps_notional():
    v = tc.evaluate_trade_against_controls(
        _cand(notional=50.0),
        _controls(total_budget_usd=300.0, max_per_trade_usd=50.0),
        _state(used_budget_usd=280.0),  # only $20 left
    )
    assert v.qualifies is True
    assert v.clamped_notional == pytest.approx(20.0)


def test_open_positions_gate():
    v = tc.evaluate_trade_against_controls(
        _cand(), _controls(max_open_positions=3), _state(open_positions=3)
    )
    assert v.qualifies is False
    assert any("open positions" in r for r in v.reasons)


def test_trades_today_gate():
    v = tc.evaluate_trade_against_controls(
        _cand(), _controls(max_trades_per_day=5), _state(trades_today=5)
    )
    assert v.qualifies is False
    assert any("today's limit" in r for r in v.reasons)


def test_unspecified_notional_sizes_to_per_trade_cap():
    v = tc.evaluate_trade_against_controls(
        _cand(notional=0.0), _controls(max_per_trade_usd=50.0), _state()
    )
    assert v.qualifies is True
    assert v.clamped_notional == pytest.approx(50.0)


def test_multiple_gates_accumulate_reasons():
    v = tc.evaluate_trade_against_controls(
        _cand(confidence=0.1),
        _controls(min_confidence=0.6, max_open_positions=1),
        _state(open_positions=2),
    )
    assert v.qualifies is False
    assert len(v.reasons) >= 2


# ---------------------------------------------------------------------------
# process_candidates — budget accumulation across one pass
# ---------------------------------------------------------------------------
def test_process_candidates_accumulates_budget():
    # Budget $120, $50/trade. Three good candidates -> 2 execute ($100),
    # the third is held because only $20 remains... actually it clamps to
    # $20 and still executes. So budget never blocks until fully used.
    controls = _controls(total_budget_usd=120.0, max_per_trade_usd=50.0)
    cands = [_cand(symbol=s, notional=50.0) for s in ("AAA", "BBB", "CCC")]
    executed = []

    async def executor(c, notional):
        executed.append((c.symbol, notional))

    summary = asyncio.run(
        tc.process_candidates(
            cands, controls, _state(), executor=executor
        )
    )
    # 120 / 50 -> 50 + 50 + 20 = 120; all three execute (last clamped).
    assert summary["executed"] == 3
    assert executed[-1][1] == pytest.approx(20.0)


def test_process_candidates_holds_until_budget_exhausted():
    controls = _controls(total_budget_usd=50.0, max_per_trade_usd=50.0)
    cands = [_cand(symbol=s, notional=50.0) for s in ("AAA", "BBB")]
    executed = []
    held = []

    async def executor(c, notional):
        executed.append(c.symbol)

    def record(c, reasons):
        held.append((c.symbol, reasons))

    summary = asyncio.run(
        tc.process_candidates(
            cands, controls, _state(), executor=executor, record_pending=record
        )
    )
    assert summary["executed"] == 1
    assert summary["held"] == 1
    assert held[0][0] == "BBB"


# ---------------------------------------------------------------------------
# load / update controls — clamping + preset linkage
# ---------------------------------------------------------------------------
def test_defaults_when_unset(isolated):
    c = tc.load_controls()
    assert c.total_budget_usd == pytest.approx(300.0)
    assert c.max_per_trade_usd == pytest.approx(50.0)
    assert c.max_trades_per_day == 5
    assert c.max_open_positions == 3


def test_budget_clamped_to_ceiling(isolated):
    c = tc.update_controls({"total_budget_usd": 99_999.0})
    assert c.total_budget_usd == pytest.approx(10_000.0)
    # And it routed through the onboarding cap (single source of truth).
    assert ob.load_onboarding().live_float_cap_usd == pytest.approx(10_000.0)


def test_negative_and_garbage_fail_safe(isolated):
    c = tc.update_controls(
        {"max_trades_per_day": -3, "max_open_positions": 9999, "min_confidence": 5.0}
    )
    assert c.max_trades_per_day == 0  # clamped to floor
    assert c.max_open_positions == 50  # clamped to ceiling
    assert c.min_confidence == pytest.approx(1.0)  # clamped into [0,1]


def test_preset_sets_min_confidence(isolated):
    c = tc.update_controls({"risk_preset": "conservative"})
    assert c.risk_preset == "conservative"
    assert c.min_confidence == pytest.approx(tc.PRESET_CONFIDENCE["conservative"])


def test_direct_min_confidence_switches_to_custom(isolated):
    tc.update_controls({"risk_preset": "balanced"})
    c = tc.update_controls({"min_confidence": 0.33})
    assert c.risk_preset == "custom"
    assert c.min_confidence == pytest.approx(0.33)


def test_per_trade_clamped_to_budget(isolated):
    c = tc.update_controls({"total_budget_usd": 100.0, "max_per_trade_usd": 500.0})
    assert c.max_per_trade_usd == pytest.approx(100.0)


def test_persistence_round_trip(isolated):
    tc.update_controls({"max_trades_per_day": 9, "risk_preset": "aggressive"})
    c = tc.load_controls()
    assert c.max_trades_per_day == 9
    assert c.risk_preset == "aggressive"
    assert c.min_confidence == pytest.approx(tc.PRESET_CONFIDENCE["aggressive"])


# ---------------------------------------------------------------------------
# pending_trades store
# ---------------------------------------------------------------------------
def test_pending_upsert_dedupes(isolated):
    c = _cand(symbol="NVDA", confidence=0.5, notional=40.0)
    pq.upsert_pending(c, ["Confidence 50% is below your 60% minimum"])
    pq.upsert_pending(c, ["Confidence 50% is below your 60% minimum"])
    waiting = pq.load_waiting()
    assert len(waiting) == 1
    assert waiting[0]["symbol"] == "NVDA"


def test_pending_mark_executed_removes_from_waiting(isolated):
    pq.upsert_pending(_cand(symbol="NVDA"), ["held"])
    pq.mark_executed("NVDA", "buy")
    assert pq.load_waiting() == []
    # but it's still in the full log as executed_shadow
    allrows = pq.load_pending()
    assert allrows[0]["status"] == "executed_shadow"


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
def test_api_get_defaults(isolated, client):
    r = client.get("/api/trading-controls")
    assert r.status_code == 200
    j = r.json()
    assert j["total_budget_usd"] == pytest.approx(300.0)
    assert "remaining_budget_usd" in j
    assert "preset_thresholds" in j


def test_api_post_clamps_budget(isolated, client):
    r = client.post("/api/trading-controls", json={"total_budget_usd": 50000})
    assert r.status_code == 200
    assert r.json()["total_budget_usd"] == pytest.approx(10_000.0)


def test_api_post_preset_sets_confidence(isolated, client):
    r = client.post("/api/trading-controls", json={"risk_preset": "conservative"})
    j = r.json()
    assert j["risk_preset"] == "conservative"
    assert j["min_confidence"] == pytest.approx(tc.PRESET_CONFIDENCE["conservative"])


def test_api_post_min_confidence_custom(isolated, client):
    client.post("/api/trading-controls", json={"risk_preset": "balanced"})
    r = client.post("/api/trading-controls", json={"min_confidence": 0.42})
    j = r.json()
    assert j["risk_preset"] == "custom"
    assert j["min_confidence"] == pytest.approx(0.42)


def test_api_budget_shares_onboarding_cap(isolated, client):
    client.post("/api/trading-controls", json={"total_budget_usd": 750})
    # The legacy cap endpoint reflects the same value.
    cap = client.get("/api/onboarding/robinhood/cap").json()
    assert cap["cap_usd"] == pytest.approx(750.0)


def test_api_pending_endpoint(isolated, client):
    pq.upsert_pending(_cand(symbol="TSLA", confidence=0.5), ["held"])
    r = client.get("/api/trading-controls/pending")
    assert r.status_code == 200
    j = r.json()
    assert j["count"] == 1
    assert j["pending"][0]["symbol"] == "TSLA"


def test_page_renders(isolated, client):
    r = client.get("/trading-controls")
    assert r.status_code == 200
    assert "How picky should the AI be?" in r.text
    assert 'data-path="/trading-controls"' in r.text  # nav highlight hook
