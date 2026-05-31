"""Tests for the Always-On Brain (Phase 20).

The autonomy module owns three responsibilities:

  1. Pick which symbols Curiosity should focus on next, given a fresh
     sweep result.
  2. Run one full tick \u2014 sweep + curiosity + chatter writes \u2014 without
     ever raising, even when the sweep itself blows up.
  3. Expose a snapshot the API + dashboard can render.

We exercise each independently and confirm the HTTP surface stays
healthy. The actual long-lived asyncio loop is *not* exercised here
(it's a thin wrapper around ``run_one_tick`` and the FastAPI startup
hook is guarded by ``AUTONOMY_DISABLED=1`` in CI when needed).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from fastapi.testclient import TestClient

from packages.cockpit.web import autonomy, chatter
from packages.cockpit.web.server import app


@pytest.fixture(autouse=True)
def _reset_brain() -> None:
    """Start every test from a clean autonomy + chatter state.

    Self-improvement is disabled by default so these Phase 20 tests
    don't exercise (or write files for) Phase 21 internals. Phase 21
    behaviour has its own dedicated test module.
    """
    autonomy.reset_for_tests()
    chatter.clear()
    autonomy.configure(self_improve_enabled=False)
    yield
    autonomy.reset_for_tests()
    chatter.clear()


# ---------------------------------------------------------------------------
# Curiosity scoring
# ---------------------------------------------------------------------------


def _cand(**kw: Any) -> dict[str, Any]:
    """Tiny helper to build a sweep candidate dict."""
    base = {
        "symbol": "SPY",
        "signal_kind": "long",
        "thesis": "stub",
        "confidence": 0.5,
        "sentiment_score": 0.0,
        "mentions": 0,
        "sources": [],
        "sample_headlines": [],
        "created_at": "",
        "reddit_trust": 0.0,
        "corroborated": False,
        "news_headlines": 0,
        "corroboration_score": 0.0,
        "corroboration_reason": "",
        "analyst_mean_rating": 0.0,
        "analyst_num": 0,
        "analyst_target_mean": 0.0,
        "analyst_recent_action": "",
        "analyst_recent_firm": "",
        "insider_net_shares": 0.0,
        "insider_buy_count": 0,
        "insider_sell_count": 0,
        "insider_form4_30d": 0,
        "stocktwits_trending": False,
        "stocktwits_watchlist": 0,
        "yahoo_news_count": 0,
    }
    base.update(kw)
    return base


def test_score_picks_up_corroboration_signal() -> None:
    plain = _cand(symbol="AAA", confidence=0.5)
    corro = _cand(
        symbol="BBB", confidence=0.5, corroborated=True, corroboration_score=0.7
    )
    s_plain, _r_plain, _f_plain = autonomy._score_candidate(plain)
    s_corro, r_corro, f_corro = autonomy._score_candidate(corro)
    assert s_corro > s_plain
    assert any("corroborat" in r for r in r_corro)
    assert "corroborated" in f_corro


def test_score_rewards_insider_activity_and_analyst_signals() -> None:
    bullish = _cand(
        symbol="AAPL",
        confidence=0.4,
        analyst_mean_rating=2.1,
        analyst_num=10,
        analyst_recent_action="upgrade",
        insider_form4_30d=4,
        insider_net_shares=10_000.0,
    )
    bare = _cand(symbol="OTHER", confidence=0.4)
    s_bull, reasons, feats = autonomy._score_candidate(bullish)
    s_bare, _, _ = autonomy._score_candidate(bare)
    assert s_bull > s_bare + 0.15
    assert any("analyst" in r for r in reasons)
    assert any("insider" in r for r in reasons)
    assert "insider" in feats and "analyst_action" in feats


def test_score_resilient_to_malformed_input() -> None:
    s, r, f = autonomy._score_candidate({})
    assert s == 0.0
    assert r == []
    assert f == []
    # Non-dict input also doesn't raise.
    s2, r2, f2 = autonomy._score_candidate("not a dict")  # type: ignore[arg-type]
    assert s2 == 0.0
    assert r2 == []
    assert f2 == []


def test_pick_focus_ranks_and_caps_results() -> None:
    cands = [
        _cand(symbol="LOW", confidence=0.2),
        _cand(symbol="HIGH", confidence=0.8, corroborated=True),
        _cand(symbol="MID", confidence=0.5, stocktwits_trending=True),
        _cand(symbol="ALSO", confidence=0.6, corroboration_score=0.6),
    ]
    syms, details = autonomy.pick_focus(cands, top_n=8, focus_count=2)
    assert syms == ["HIGH", "ALSO"]
    assert len(details) == 2
    assert details[0]["symbol"] == "HIGH"
    assert details[0]["score"] >= details[1]["score"]


def test_pick_focus_empty_candidates_returns_empty() -> None:
    assert autonomy.pick_focus([]) == ([], [])


def test_pick_focus_deduplicates_repeated_symbols() -> None:
    cands = [
        _cand(symbol="AAA", confidence=0.7, corroborated=True),
        _cand(symbol="AAA", confidence=0.65),  # same symbol, second mention
        _cand(symbol="BBB", confidence=0.6),
    ]
    syms, _ = autonomy.pick_focus(cands, focus_count=3)
    assert syms == ["AAA", "BBB"]


# ---------------------------------------------------------------------------
# run_one_tick
# ---------------------------------------------------------------------------


def _fake_sweep_payload(symbols: list[str]) -> dict[str, Any]:
    return {
        "status": "ok",
        "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "duration_s": 1.2,
        "portfolio_symbols": [],
        "candidates": [
            _cand(symbol=s, confidence=0.4 + 0.1 * i, corroborated=(i == 0))
            for i, s in enumerate(symbols)
        ],
        "error": "",
        "sources_meta": {},
    }


@pytest.mark.asyncio
async def test_run_one_tick_pushes_research_and_curiosity_chatter() -> None:
    async def fake_sweep() -> dict[str, Any]:
        return _fake_sweep_payload(["AAPL", "MSFT", "GOOG"])

    out = await autonomy.run_one_tick(
        sweep_runner=fake_sweep,
        pause_check=lambda: False,
    )
    assert out["ok"] is True
    assert out["candidates"] == 3
    assert "AAPL" in out["focus"]
    # Two chatter entries: research summary + curiosity decision.
    snap = chatter.snapshot()
    agents = [e["agent"] for e in snap]
    assert "research" in agents
    assert "curiosity" in agents


@pytest.mark.asyncio
async def test_run_one_tick_invokes_focus_hook() -> None:
    captured: dict[str, Any] = {}

    def hook(symbols: list[str], reason: str) -> None:
        captured["symbols"] = symbols
        captured["reason"] = reason

    autonomy.configure(
        on_curiosity_focus=hook,
        portfolio_symbols_getter=lambda: ["SPY"],
    )

    async def fake_sweep() -> dict[str, Any]:
        return _fake_sweep_payload(["AAPL", "MSFT"])

    await autonomy.run_one_tick(sweep_runner=fake_sweep, pause_check=lambda: False)
    assert "SPY" in captured["symbols"]
    # The portfolio symbol is kept first so we never drop the user's
    # actual holdings.
    assert captured["symbols"][0] == "SPY"
    assert any(sym in captured["symbols"] for sym in ("AAPL", "MSFT"))


@pytest.mark.asyncio
async def test_run_one_tick_respects_pause() -> None:
    called = {"n": 0}

    async def fake_sweep() -> dict[str, Any]:
        called["n"] += 1
        return _fake_sweep_payload(["X"])

    out = await autonomy.run_one_tick(
        sweep_runner=fake_sweep, pause_check=lambda: True
    )
    assert out == {"skipped": True, "reason": "paused"}
    assert called["n"] == 0
    # A chatter line is still written so the user sees why nothing
    # happened.
    msgs = [e["message"] for e in chatter.snapshot()]
    assert any("paused" in m.lower() for m in msgs)


@pytest.mark.asyncio
async def test_run_one_tick_captures_sweep_failure_without_raising() -> None:
    async def boom() -> Any:
        raise RuntimeError("network unreachable")

    out = await autonomy.run_one_tick(
        sweep_runner=boom, pause_check=lambda: False
    )
    assert out["ok"] is False
    assert "network unreachable" in out["error"]
    assert "network unreachable" in autonomy.STATE.last_error
    # The user sees the failure in the chatter feed.
    msgs = [e["message"] for e in chatter.snapshot()]
    assert any("Sweep failed" in m for m in msgs)


@pytest.mark.asyncio
async def test_run_one_tick_handles_dataclass_result() -> None:
    """run_sweep returns a dataclass with to_dict() in production."""

    class _Result:
        def to_dict(self) -> dict[str, Any]:
            return _fake_sweep_payload(["ZZZ"])

    async def fake_sweep() -> _Result:
        return _Result()

    out = await autonomy.run_one_tick(
        sweep_runner=fake_sweep, pause_check=lambda: False
    )
    assert out["ok"] is True
    assert "ZZZ" in out["focus"]


# ---------------------------------------------------------------------------
# Snapshot + HTTP
# ---------------------------------------------------------------------------


def test_snapshot_shape() -> None:
    snap = autonomy.snapshot()
    assert {
        "enabled",
        "running",
        "market_open",
        "current_interval_s",
        "last_sweep_started_at",
        "last_sweep_finished_at",
        "last_sweep_status",
        "last_sweep_candidates",
        "last_curiosity_at",
        "last_curiosity_focus",
        "last_curiosity_reason",
        "last_error",
        "config",
    }.issubset(snap.keys())
    cfg = snap["config"]
    assert cfg["sweep_market_seconds"] > 0
    assert cfg["sweep_off_seconds"] > 0


def test_api_autonomy_returns_snapshot() -> None:
    client = TestClient(app)
    r = client.get("/api/autonomy")
    assert r.status_code == 200
    body = r.json()
    assert "enabled" in body
    assert "running" in body
    assert "config" in body


def test_api_autonomy_disable_then_enable_round_trip() -> None:
    client = TestClient(app)
    r = client.post("/api/autonomy/disable")
    assert r.status_code == 200
    assert r.json()["enabled"] is False
    r = client.post("/api/autonomy/enable")
    assert r.status_code == 200
    body = r.json()
    assert body["enabled"] is True
