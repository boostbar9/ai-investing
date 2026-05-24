"""Smoke tests for the cockpit FastAPI server.

These tests use FastAPI's TestClient to drive the endpoints in-process.
The server reads from ``data/paper_log/runs.jsonl``; we monkeypatch the
module-level path to a temp file so tests are hermetic and don't require
the user's real logs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from packages.cockpit.web import server as srv


@pytest.fixture
def fake_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the server at a temp runs.jsonl seeded with two runs."""
    log = tmp_path / "runs.jsonl"
    runs = [
        {
            "ts": "2026-05-21T20:00:00+00:00",
            "strategy": "ensemble",
            "halted": False,
            "account_equity": 100000.0,
            "account_buying_power": 200000.0,
            "target_weights": {"SPY": 0.6, "TLT": 0.4},
            "orders_submitted": [{"symbol": "SPY", "side": "buy", "qty": 50}],
        },
        {
            "ts": "2026-05-22T20:00:00+00:00",
            "strategy": "ensemble",
            "halted": False,
            "account_equity": 100250.0,
            "account_buying_power": 200500.0,
            "target_weights": {"SPY": 0.5, "TLT": 0.5},
            "orders_submitted": [{"symbol": "TLT", "side": "buy", "qty": 10}],
        },
    ]
    log.write_text("\n".join(json.dumps(r) for r in runs) + "\n")
    monkeypatch.setattr(srv, "PAPER_LOG", log)
    return log


@pytest.fixture
def fake_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect cockpit state writes/reads to a temp file.

    ``load_state`` and ``save_state`` take ``path`` as a default argument,
    which Python binds at definition time - patching ``state.STATE_PATH``
    alone does not redirect them. We also rebind the default tuples so the
    test never touches the real state file on disk.
    """
    from packages.cockpit import state as st

    path = tmp_path / "state.json"
    monkeypatch.setattr(st, "STATE_PATH", path)
    # Default args are baked into the function object - rebind them too.
    monkeypatch.setattr(st.load_state, "__defaults__", (path,))
    monkeypatch.setattr(st.save_state, "__defaults__", (path,))
    return path


@pytest.fixture
def fake_agent_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the agent run log to a tmp file so /api/agents/run is hermetic."""
    log = tmp_path / "agents_log.jsonl"
    monkeypatch.setattr(srv, "AGENT_LOG", log)
    return log


@pytest.fixture
def fake_discovery_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the advisory discovery log to a tmp file so tests are hermetic."""
    log = tmp_path / "discoveries_log.jsonl"
    monkeypatch.setattr(srv, "DISCOVERY_LOG", log)
    return log


@pytest.fixture
def fake_scorecard_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the agent scorecard log to a tmp file."""
    log = tmp_path / "agent_scorecard.jsonl"
    monkeypatch.setattr(srv, "SCORECARD_LOG", log)
    return log


@pytest.fixture
def fake_promotion_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the promotion candidates log to a tmp file."""
    log = tmp_path / "promotion_candidates.jsonl"
    monkeypatch.setattr(srv, "SCORECARD_PROMOTION_LOG", log)
    return log


@pytest.fixture
def client(fake_log: Path, fake_state: Path, fake_agent_log: Path) -> TestClient:
    return TestClient(srv.app)


def test_index_serves_html(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert "ai-investing cockpit" in r.text.lower() or "cockpit" in r.text.lower()


def test_health_endpoint_shape(client: TestClient) -> None:
    """/api/health must return the contract the global topbar expects."""
    r = client.get("/api/health")
    assert r.status_code == 200
    j = r.json()
    for key in (
        "status",
        "now",
        "mode",
        "paused",
        "last_paper_run",
        "last_paper_halted",
        "errors",
        "jobs",
        "commit",
    ):
        assert key in j, f"missing key {key!r}"
    assert j["status"] in {"ok", "warn", "idle", "down"}
    assert isinstance(j["errors"], dict)
    assert isinstance(j["jobs"], dict)


def test_static_assets_mounted(client: TestClient) -> None:
    """Shared CSS/JS must be served at /static so every page can load them."""
    for path in ("/static/cockpit.css", "/static/cockpit.js"):
        r = client.get(path)
        assert r.status_code == 200, f"{path} -> {r.status_code}"
        assert len(r.text) > 100


def test_state_endpoint_returns_snapshot(client: TestClient) -> None:
    r = client.get("/api/state")
    assert r.status_code == 200
    data = r.json()
    # Top-level shape
    for key in ("now", "control", "account", "regime", "streak", "positions", "trades", "equity_curve"):
        assert key in data, f"missing key {key!r}"
    # Account reflects the latest run
    assert data["account"]["equity"] == 100250.0
    assert data["account"]["strategy"] == "ensemble"


def test_positions_endpoint_returns_target_weights(client: TestClient) -> None:
    r = client.get("/api/positions")
    assert r.status_code == 200
    positions = r.json()
    symbols = {p["symbol"] for p in positions}
    assert symbols == {"SPY", "TLT"}
    for p in positions:
        assert p["target_weight"] in (0.5,)  # latest run has 0.5/0.5
        assert p["approx_value"] is not None


def test_trades_endpoint_returns_submitted_orders(client: TestClient) -> None:
    r = client.get("/api/trades")
    assert r.status_code == 200
    trades = r.json()
    # Newest first
    assert trades[0]["symbol"] == "TLT"
    assert trades[1]["symbol"] == "SPY"
    for t in trades:
        assert t["strategy"] == "ensemble"


def test_pause_then_resume_flips_state(client: TestClient) -> None:
    r1 = client.post("/api/pause")
    assert r1.status_code == 200
    assert r1.json()["paused"] is True

    snap = client.get("/api/state").json()
    assert snap["control"]["paused"] is True

    r2 = client.post("/api/resume")
    assert r2.status_code == 200
    assert r2.json()["paused"] is False


def test_override_regime_rejects_invalid_value(client: TestClient) -> None:
    r = client.post("/api/override-regime", json={"regime": "moon"})
    assert r.status_code == 400


def test_override_regime_accepts_valid_value(client: TestClient) -> None:
    r = client.post("/api/override-regime", json={"regime": "bear"})
    assert r.status_code == 200
    assert r.json()["regime_override"] == "bear"

    regime = client.get("/api/regime").json()
    assert regime["override"] == "bear"


def test_equity_curve_returns_chronological_points(client: TestClient) -> None:
    r = client.get("/api/equity-curve")
    assert r.status_code == 200
    curve = r.json()
    assert len(curve) == 2
    # Chronological (older first)
    assert curve[0]["equity"] == 100000.0
    assert curve[1]["equity"] == 100250.0


# --------------------------------------------------------------------------
# /api/mode (paper vs live)
# --------------------------------------------------------------------------


def test_mode_defaults_to_paper(client: TestClient, monkeypatch) -> None:
    monkeypatch.delenv("ALPACA_LIVE_KEY_ID", raising=False)
    monkeypatch.delenv("ALPACA_LIVE_SECRET", raising=False)
    r = client.get("/api/mode")
    assert r.status_code == 200
    j = r.json()
    assert j["mode"] == "paper"
    assert j["live_keys_present"] is False


def test_mode_switch_to_paper_works(client: TestClient) -> None:
    r = client.post("/api/mode", json={"mode": "paper"})
    assert r.status_code == 200
    assert r.json()["mode"] == "paper"


def test_mode_switch_to_live_requires_confirm(client: TestClient, monkeypatch) -> None:
    monkeypatch.setenv("ALPACA_LIVE_KEY_ID", "AK_LIVE")
    monkeypatch.setenv("ALPACA_LIVE_SECRET", "sek")
    r = client.post("/api/mode", json={"mode": "live"})
    assert r.status_code == 400
    assert "confirm_live" in r.json()["detail"]


def test_mode_switch_to_live_requires_keys(client: TestClient, monkeypatch) -> None:
    monkeypatch.delenv("ALPACA_LIVE_KEY_ID", raising=False)
    monkeypatch.delenv("ALPACA_LIVE_SECRET", raising=False)
    r = client.post("/api/mode", json={"mode": "live", "confirm_live": True})
    assert r.status_code == 400
    assert "live" in r.json()["detail"].lower()


def test_mode_switch_to_live_succeeds_with_keys_and_confirm(
    client: TestClient, monkeypatch
) -> None:
    monkeypatch.setenv("ALPACA_LIVE_KEY_ID", "AK_LIVE")
    monkeypatch.setenv("ALPACA_LIVE_SECRET", "sek")
    r = client.post("/api/mode", json={"mode": "live", "confirm_live": True})
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "live"
    # Switching mode must auto-pause the bot for safety.
    assert body["paused"] is True


def test_mode_rejects_invalid_value(client: TestClient) -> None:
    r = client.post("/api/mode", json={"mode": "yolo"})
    assert r.status_code == 400


# --------------------------------------------------------------------------
# /api/errors
# --------------------------------------------------------------------------


@pytest.fixture
def isolated_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the error log to a tmp file for endpoint tests."""
    from packages.cockpit import errors as err_log

    log = tmp_path / "errors.jsonl"
    monkeypatch.setattr(err_log, "ERROR_LOG", log)
    return log


def test_errors_endpoint_empty(client: TestClient, isolated_errors: Path) -> None:
    r = client.get("/api/errors")
    assert r.status_code == 200
    body = r.json()
    assert body["entries"] == []
    assert body["counts"]["total"] == 0


def test_errors_endpoint_lists_entries(client: TestClient, isolated_errors: Path) -> None:
    from packages.cockpit import errors as err_log

    err_log.record_error(source="unit", message="hello", severity="warning")
    err_log.record_error(source="unit", message="world", severity="error")
    r = client.get("/api/errors")
    body = r.json()
    assert body["counts"]["total"] == 2
    assert body["counts"]["warning"] == 1
    assert body["counts"]["error"] == 1
    # newest first
    assert body["entries"][0]["message"] == "world"


def test_errors_endpoint_filters_by_severity(
    client: TestClient, isolated_errors: Path
) -> None:
    from packages.cockpit import errors as err_log

    err_log.record_error(source="u", message="a", severity="error")
    err_log.record_error(source="u", message="b", severity="warning")
    r = client.get("/api/errors?severity=warning")
    body = r.json()
    msgs = [e["message"] for e in body["entries"]]
    assert msgs == ["b"]


def test_errors_markdown_endpoint(client: TestClient, isolated_errors: Path) -> None:
    from packages.cockpit import errors as err_log

    err_log.record_error(source="broker", message="alpaca 404")
    r = client.get("/api/errors/markdown")
    body = r.json()
    assert "alpaca 404" in body["markdown"]
    assert "broker" in body["markdown"]


def test_errors_clear_endpoint(client: TestClient, isolated_errors: Path) -> None:
    from packages.cockpit import errors as err_log

    err_log.record_error(source="u", message="a")
    err_log.record_error(source="u", message="b")
    r = client.post("/api/errors/clear")
    assert r.status_code == 200
    assert r.json()["cleared"] == 2
    r2 = client.get("/api/errors")
    assert r2.json()["counts"]["total"] == 0


def test_errors_page_renders(client: TestClient) -> None:
    r = client.get("/errors")
    assert r.status_code == 200
    assert "Error console" in r.text or "errors" in r.text.lower()


def test_dashboard_shows_agent_strip(client: TestClient) -> None:
    """Dashboard must expose the agent status strip so users see state at a glance."""
    r = client.get("/")
    assert r.status_code == 200
    assert "agent-lights" in r.text
    assert "/api/agents/last" in r.text


def test_agents_page_renders(client: TestClient) -> None:
    r = client.get("/agents")
    assert r.status_code == 200
    body = r.text.lower()
    assert "langgraph" in body or "agent" in body
    # Pipeline cards should be present so the dashboard JS can paint them.
    for name in ("card-research", "card-strategy", "card-risk", "card-execution"):
        assert name in r.text


def test_agents_last_idle_before_first_run(client: TestClient) -> None:
    """/api/agents/last must respond with idle defaults before any run."""
    # Reset the module-level cache so this test is order-independent.
    srv._LAST_AGENT_RUN.clear()
    srv._LAST_AGENT_RUN.update(
        {
            "ran_at": None,
            "decision_id": None,
            "halted": False,
            "halt_reason": None,
            "used_llm": False,
            "agents": {
                "research": {"status": "idle", "detail": ""},
                "strategy": {"status": "idle", "detail": ""},
                "risk": {"status": "idle", "detail": ""},
                "execution": {"status": "idle", "detail": ""},
                "discovery": {"status": "idle", "detail": ""},
            },
            "audit": [],
        }
    )
    r = client.get("/api/agents/last")
    assert r.status_code == 200
    j = r.json()
    assert j["ran_at"] is None
    assert set(j["agents"].keys()) == {"research", "strategy", "risk", "execution", "discovery"}
    for a in j["agents"].values():
        assert a["status"] == "idle"


def test_agents_run_stub_returns_full_pipeline(client: TestClient) -> None:
    """POST /api/agents/run with stub backend must walk all four agents."""
    r = client.post(
        "/api/agents/run",
        json={"symbols": ["SPY", "QQQ"], "regime": "chop", "use_llm": False},
    )
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["used_llm"] is False
    assert j["regime"] == "chop"
    assert j["decision_id"]
    # Stub research is neutral by default -> sentiment ~0 -> status ok.
    assert j["agents"]["research"]["status"] in {"ok", "warn"}
    # With weights for SPY+QQQ the strategy stub produces signals.
    assert j["agents"]["strategy"]["status"] in {"ok", "warn"}
    assert isinstance(j["agents"]["strategy"]["signals"], list)
    assert isinstance(j["audit"], list) and len(j["audit"]) >= 3
    # /last must now reflect the run.
    r2 = client.get("/api/agents/last").json()
    assert r2["decision_id"] == j["decision_id"]


def test_agents_run_crisis_regime_halts_strategy(client: TestClient) -> None:
    """Crisis regime must short-circuit the strategy gate per spec §5."""
    r = client.post(
        "/api/agents/run",
        json={"symbols": ["SPY"], "regime": "crisis", "use_llm": False},
    )
    assert r.status_code == 200
    j = r.json()
    assert j["agents"]["strategy"]["status"] == "halt"


def test_agents_crisis_regime_kills_chain_in_stub(client: TestClient) -> None:
    """Spec §5: crisis regime → zero signals from strategy AND zero approved by risk."""
    r = client.post(
        "/api/agents/run",
        json={"symbols": ["SPY", "QQQ"], "regime": "crisis", "use_llm": False},
    )
    assert r.status_code == 200
    j = r.json()
    strat = j["agents"]["strategy"]
    assert strat["status"] == "halt"
    assert strat.get("signals") == []
    risk = j["agents"]["risk"]
    assert (risk.get("approved") or []) == []


def test_agents_run_persists_to_log(client: TestClient, fake_agent_log: Path) -> None:
    """Each /api/agents/run must append one row to data/agents_log.jsonl and be readable via /history."""
    assert not fake_agent_log.exists()
    r = client.post(
        "/api/agents/run",
        json={"symbols": ["SPY"], "regime": "chop", "use_llm": False},
    )
    assert r.status_code == 200
    assert fake_agent_log.exists()
    lines = [line for line in fake_agent_log.read_text().splitlines() if line.strip()]
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["decision_id"]
    assert row["regime"] == "chop"
    assert "agents" in row
    # And /history surfaces it.
    h = client.get("/api/agents/history?limit=10").json()
    assert h["total"] == 1
    assert h["runs"][0]["decision_id"] == row["decision_id"]


def test_agents_schedule_get_default(client: TestClient) -> None:
    """Scheduler must start disabled with sane defaults."""
    # Reset module-level state so this test is order-independent.
    srv._AGENT_SCHED["enabled"] = False
    srv._AGENT_SCHED["interval_seconds"] = 1800
    srv._AGENT_SCHED["use_llm"] = False
    srv._AGENT_SCHED["symbols"] = ["SPY", "QQQ", "TLT"]
    srv._AGENT_SCHED["last_run_at"] = None
    srv._AGENT_SCHED["last_run_status"] = None
    srv._AGENT_SCHED["last_error"] = None
    srv._AGENT_SCHED["_task"] = None
    r = client.get("/api/agents/schedule")
    assert r.status_code == 200
    j = r.json()
    assert j["enabled"] is False
    assert j["running"] is False
    assert j["interval_seconds"] == 1800
    assert j["use_llm"] is False
    assert isinstance(j["symbols"], list)


def test_agents_schedule_set_validates_interval(client: TestClient) -> None:
    """Intervals below 60s must be rejected with 400."""
    r = client.post(
        "/api/agents/schedule",
        json={"enabled": False, "interval_seconds": 10},
    )
    assert r.status_code == 400


def test_agents_schedule_tick_runs_pipeline(
    client: TestClient, fake_agent_log: Path
) -> None:
    """POST /api/agents/schedule/tick must run one pipeline pass and log it."""
    # Make sure cockpit isn't paused.
    from packages.cockpit import state as st

    s = st.load_state()
    s.paused = False
    st.save_state(s)
    # Ensure stub backend with default symbols.
    srv._AGENT_SCHED["use_llm"] = False
    srv._AGENT_SCHED["symbols"] = ["SPY", "QQQ"]

    r = client.post("/api/agents/schedule/tick", json={})
    assert r.status_code == 200, r.text
    j = r.json()
    assert not j.get("skipped")
    assert j.get("decision_id")
    # Run was persisted to log.
    assert fake_agent_log.exists()
    # Schedule reflects the run.
    sched = client.get("/api/agents/schedule").json()
    assert sched["last_run_status"] in {"ok", "halted"}


def test_agents_schedule_tick_skips_when_paused(
    client: TestClient, fake_agent_log: Path
) -> None:
    """When cockpit is paused, schedule tick must skip without running."""
    from packages.cockpit import state as st

    s = st.load_state()
    s.paused = True
    st.save_state(s)
    # Clear any prior log.
    if fake_agent_log.exists():
        fake_agent_log.unlink()

    r = client.post("/api/agents/schedule/tick", json={})
    assert r.status_code == 200
    j = r.json()
    assert j.get("skipped") is True
    assert "paus" in (j.get("reason") or "").lower()
    # No log row should have been written.
    assert not fake_agent_log.exists() or fake_agent_log.read_text().strip() == ""
    # Restore for other tests.
    s = st.load_state()
    s.paused = False
    st.save_state(s)


# ---------------------------------------------------------------------------
# Advisory Discovery agent (NOT in the order path, must never gate trading).
# ---------------------------------------------------------------------------


def test_agents_run_includes_discovery_block(
    client: TestClient, fake_discovery_log: Path
) -> None:
    """/api/agents/run must always surface a discovery block (advisory_only)."""
    r = client.post(
        "/api/agents/run",
        json={"symbols": ["SPY", "QQQ"], "regime": "chop", "use_llm": False},
    )
    assert r.status_code == 200, r.text
    j = r.json()
    disc = j["agents"]["discovery"]
    # Contract: status + advisory_only flag + patterns list always present.
    assert disc["status"] in {"ok", "idle", "warn"}
    assert disc["advisory_only"] is True
    assert isinstance(disc["patterns"], list)
    # Core invariant: discovery never gates execution.
    assert j["agents"]["execution"]["status"] != "halt" or j["halted"]


def test_discovery_stub_silent_in_crisis(
    client: TestClient, fake_discovery_log: Path
) -> None:
    """Spec §5: crisis regime → discovery emits zero patterns even if
    other signals would have triggered one."""
    r = client.post(
        "/api/agents/run",
        json={"symbols": ["SPY", "QQQ"], "regime": "crisis", "use_llm": False},
    )
    assert r.status_code == 200
    j = r.json()
    disc = j["agents"]["discovery"]
    assert disc["advisory_only"] is True
    assert disc["patterns"] == []
    # And nothing should have been written to the discovery log.
    assert not fake_discovery_log.exists() or fake_discovery_log.read_text().strip() == ""


def test_discovery_stub_emits_on_strong_sentiment(
    client: TestClient, fake_discovery_log: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bull regime with |sentiment| >= 0.3 must produce exactly one pattern.

    We wrap ``paper_bridge.advise`` to bump the research sentiment so the
    stub's deterministic emit rule fires. This proves the wiring without
    needing Ollama.
    """
    import dataclasses

    from packages.agents import paper_bridge
    from packages.shared.schemas import ResearchOutput

    original_advise = paper_bridge.advise

    async def _advise_with_strong_sentiment(**kwargs):  # type: ignore[no-untyped-def]
        result = await original_advise(**kwargs)
        # Replace research output so the discovery stub's |s| >= 0.3 path fires.
        boosted = ResearchOutput(
            decision_id=result.research.decision_id,
            thesis="strong positive sentiment for testing",
            sentiment=0.5,
            citations=result.research.citations,
        )
        return dataclasses.replace(result, research=boosted)

    monkeypatch.setattr(paper_bridge, "advise", _advise_with_strong_sentiment)

    r = client.post(
        "/api/agents/run",
        json={"symbols": ["SPY", "QQQ", "TLT"], "regime": "bull", "use_llm": False},
    )
    assert r.status_code == 200, r.text
    j = r.json()
    disc = j["agents"]["discovery"]
    assert disc["advisory_only"] is True
    assert len(disc["patterns"]) >= 1
    p = disc["patterns"][0]
    # Stub contract: name encodes regime + side; symbols come from the curated
    # universe (uppercase); confidence is bounded; horizon is short.
    assert p["name"].startswith("bull-")
    assert all(s == s.upper() for s in p["symbols"])
    assert 0.0 <= p["confidence"] <= 1.0
    assert 1 <= p["horizon_days"] <= 60
    assert "sentiment" in p["feature_keys"]


def test_api_agents_discoveries_history(
    client: TestClient, fake_discovery_log: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After a run that produced patterns, GET /api/agents/discoveries must
    return them newest-first with the advisory_only flag."""
    import dataclasses

    from packages.agents import paper_bridge
    from packages.shared.schemas import ResearchOutput

    original_advise = paper_bridge.advise

    async def _advise_with_strong_sentiment(**kwargs):  # type: ignore[no-untyped-def]
        result = await original_advise(**kwargs)
        boosted = ResearchOutput(
            decision_id=result.research.decision_id,
            thesis="strong positive sentiment for history test",
            sentiment=0.45,
            citations=result.research.citations,
        )
        return dataclasses.replace(result, research=boosted)

    monkeypatch.setattr(paper_bridge, "advise", _advise_with_strong_sentiment)

    # Empty before any run.
    r0 = client.get("/api/agents/discoveries")
    assert r0.status_code == 200
    assert r0.json() == {"discoveries": [], "total": 0, "advisory_only": True}

    # One run that should produce a pattern.
    r1 = client.post(
        "/api/agents/run",
        json={"symbols": ["SPY", "QQQ"], "regime": "bull", "use_llm": False},
    )
    assert r1.status_code == 200

    r2 = client.get("/api/agents/discoveries?limit=10")
    assert r2.status_code == 200
    j = r2.json()
    assert j["advisory_only"] is True
    assert j["total"] >= 1
    row = j["discoveries"][0]
    assert row["regime"] == "bull"
    assert row["used_llm"] is False
    assert isinstance(row["patterns"], list) and len(row["patterns"]) >= 1


def test_state_endpoint_handles_empty_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No runs.jsonl yet -> still returns 200 with sensible empty values."""
    empty = tmp_path / "runs.jsonl"
    monkeypatch.setattr(srv, "PAPER_LOG", empty)
    from packages.cockpit import state as st
    monkeypatch.setattr(st, "STATE_PATH", tmp_path / "state.json")

    with TestClient(srv.app) as c:
        r = c.get("/api/state")
        assert r.status_code == 200
        data = r.json()
        assert data["account"]["equity"] is None
        assert data["positions"] == []
        assert data["trades"] == []
        assert data["equity_curve"] == []


# --------------------------------------------------------------------------
# Self-improvement endpoints
# --------------------------------------------------------------------------


def test_scorecard_endpoint_empty_log(
    client: TestClient,
    fake_scorecard_log: Path,
) -> None:
    """No scorecard file yet -> 200 with empty runs and zeroed summary."""
    r = client.get("/api/agents/scorecard")
    assert r.status_code == 200
    data = r.json()
    assert data["runs"] == []
    assert data["total"] == 0
    summary = data["summary"]
    assert summary["n_runs"] == 0
    assert summary["n_signals"] == 0
    assert summary["hit_rate_5d"] is None


def test_scorecard_endpoint_returns_rows(
    client: TestClient,
    fake_scorecard_log: Path,
) -> None:
    """Two scorecard rows -> endpoint returns them newest-first with a
    correctly computed summary."""
    row1 = {
        "decision_id": "r1",
        "ts": "2026-05-22T20:00:00+00:00",
        "regime": "bull",
        "used_llm": True,
        "signals": [
            {"symbol": "SPY", "side": "buy", "strength": 0.5,
             "horizon_returns_bps": {"1": 30.0, "5": 150.0}},
        ],
    }
    row2 = {
        "decision_id": "r2",
        "ts": "2026-05-23T20:00:00+00:00",
        "regime": "chop",
        "used_llm": True,
        "signals": [
            {"symbol": "QQQ", "side": "buy", "strength": 0.4,
             "horizon_returns_bps": {"1": -10.0, "5": -50.0}},
        ],
    }
    with fake_scorecard_log.open("w", encoding="utf-8") as f:
        f.write(json.dumps(row1) + "\n")
        f.write(json.dumps(row2) + "\n")

    r = client.get("/api/agents/scorecard")
    assert r.status_code == 200
    data = r.json()
    assert data["total"] == 2
    assert data["runs"][0]["decision_id"] == "r2"
    assert data["runs"][1]["decision_id"] == "r1"
    summary = data["summary"]
    assert summary["n_runs"] == 2
    assert summary["n_signals"] == 2
    assert summary["hit_rate_5d"] == 0.5
    assert summary["regime_bias"] == {"bull": 1, "chop": 1}


def test_attribute_endpoint_no_price_fetcher_appends_zero(
    client: TestClient,
    fake_agent_log: Path,
    fake_scorecard_log: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without Alpaca credentials the endpoint must still 200 with 0 rows
    appended (price fetcher returns None for every symbol)."""
    monkeypatch.delenv("ALPACA_KEY_ID", raising=False)
    monkeypatch.delenv("ALPACA_SECRET", raising=False)

    matured_ts = "2024-01-15T00:00:00+00:00"
    row = {
        "decision_id": "old-1",
        "ts": matured_ts,
        "regime": "bull",
        "used_llm": True,
        "agents": {"strategy": {"signals": [
            {"symbol": "SPY", "side": "buy", "strength": 0.5},
        ]}},
    }
    fake_agent_log.write_text(json.dumps(row) + "\n", encoding="utf-8")

    r = client.post("/api/agents/attribute")
    assert r.status_code == 200
    data = r.json()
    assert data["appended"] == 0
    assert "scorecard_path" in data


def test_promotion_candidates_endpoint_empty_log(
    client: TestClient,
    fake_promotion_log: Path,
) -> None:
    """No promotion file -> 200 with empty candidates list."""
    r = client.get("/api/agents/promotion_candidates")
    assert r.status_code == 200
    data = r.json()
    assert data["candidates"] == []
    assert data["total"] == 0


def test_promotion_candidates_endpoint_returns_rows(
    client: TestClient,
    fake_promotion_log: Path,
) -> None:
    """Two promotion rows -> endpoint returns them newest-first."""
    rows = [
        {
            "ts": "2026-05-22T20:00:00+00:00",
            "decision_id": "d1",
            "regime": "bull",
            "verdict": {
                "pattern_name": "tech-momentum",
                "symbols": ["QQQ"],
                "horizon_days": 5,
                "confidence": 0.6,
                "sharpe": 1.3,
                "max_dd": 0.05,
                "cagr": 0.18,
                "n_bars": 80,
                "passed": True,
                "reasons": ["passed floor"],
            },
            "human_status": "pending",
        },
        {
            "ts": "2026-05-23T20:00:00+00:00",
            "decision_id": "d2",
            "regime": "chop",
            "verdict": {
                "pattern_name": "yield-curve-rotation",
                "symbols": ["TLT", "GLD"],
                "horizon_days": 10,
                "confidence": 0.5,
                "sharpe": 1.1,
                "max_dd": 0.07,
                "cagr": 0.12,
                "n_bars": 90,
                "passed": True,
                "reasons": ["passed floor"],
            },
            "human_status": "pending",
        },
    ]
    with fake_promotion_log.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    r = client.get("/api/agents/promotion_candidates")
    assert r.status_code == 200
    data = r.json()
    assert data["total"] == 2
    assert data["candidates"][0]["decision_id"] == "d2"
    assert data["candidates"][0]["verdict"]["pattern_name"] == "yield-curve-rotation"
    assert data["candidates"][1]["decision_id"] == "d1"


def test_scorecard_limit_clamps_returned_rows(
    client: TestClient,
    fake_scorecard_log: Path,
) -> None:
    """The limit query param must truncate the rows list (newest kept)."""
    rows = [
        {
            "decision_id": f"r{i}",
            "ts": f"2026-05-{20+i:02d}T20:00:00+00:00",
            "regime": "bull",
            "used_llm": True,
            "signals": [{"symbol": "SPY", "side": "buy", "strength": 0.5,
                         "horizon_returns_bps": {"5": 50.0}}],
        }
        for i in range(5)
    ]
    with fake_scorecard_log.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    r = client.get("/api/agents/scorecard?limit=2")
    assert r.status_code == 200
    data = r.json()
    assert len(data["runs"]) == 2
    assert data["runs"][0]["decision_id"] == "r4"
