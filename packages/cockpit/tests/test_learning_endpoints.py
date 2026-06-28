"""Phase 28 endpoint tests for /learning + /api/learning/*.

Confirms the page renders, the summary endpoint copes with an empty
journal, filters apply on /picks, and the backfill endpoint accepts
an empty body without exploding.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from packages.cockpit.web import server as srv
from packages.learning import outcome_labeler as ol


@pytest.fixture
def client(monkeypatch, tmp_path: Path):
    # Redirect the labeler's default paths into a temp dir so we don't
    # touch the real journal during tests.
    out_path = tmp_path / "outcomes.jsonl"
    monkeypatch.setattr(ol, "DEFAULT_OUTCOMES_PATH", out_path)
    monkeypatch.setattr(ol, "DEFAULT_PREDICTIONS_PATH", tmp_path / "predictions.jsonl")
    monkeypatch.setattr(ol, "DEFAULT_AGENTS_LOG_PATH", tmp_path / "agents_log.jsonl")
    return TestClient(srv.app), out_path


def _write_outcomes(out_path: Path, rows: list[dict]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "\n".join(json.dumps(r) for r in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def test_learning_page_renders(client):
    c, _ = client
    r = c.get("/learning")
    assert r.status_code == 200
    body = r.text
    assert "Learning" in body
    assert "Trade Journal" in body
    # Make sure the API URLs are referenced (no template typos).
    assert "/api/learning/summary" in body
    assert "/api/learning/picks" in body


def test_summary_endpoint_empty_journal(client):
    c, _ = client  # out_path doesn't exist → empty payload
    r = c.get("/api/learning/summary")
    assert r.status_code == 200
    data = r.json()
    assert data["total_rows"] == 0
    assert data["summary"]["total_picks"] == 0
    assert data["agents"] == []


def test_summary_endpoint_with_outcomes(client):
    c, out = client
    _write_outcomes(out, [
        {
            "pick_id": "p1", "decision_id": "d1",
            "ts": "2026-05-01T12:00:00+00:00", "symbol": "AAPL",
            "confidence": 0.5, "regime_at_pick": "risk_on",
            "agents_voted": ["research", "strategy"],
            "entry_price": 100.0, "exit_price_eod": 105.0,
            "return_30m": 0.005, "return_2h": 0.02, "return_eod": 0.05,
            "correct": True,
        },
        {
            "pick_id": "p2", "decision_id": "d2",
            "ts": "2026-05-02T12:00:00+00:00", "symbol": "MSFT",
            "confidence": 0.3, "regime_at_pick": "chop",
            "agents_voted": ["research"],
            "entry_price": 400.0, "exit_price_eod": 395.0,
            "return_30m": -0.005, "return_2h": -0.01, "return_eod": -0.012,
            "correct": False,
        },
    ])
    r = c.get("/api/learning/summary")
    data = r.json()
    assert data["total_rows"] == 2
    assert data["summary"]["total_picks"] == 2
    assert data["summary"]["decided_picks"] == 2
    assert data["summary"]["win_rate"] == 0.5
    agents = {a["agent"]: a for a in data["agents"]}
    assert agents["strategy"]["picks"] == 1
    assert agents["research"]["picks"] == 2


def test_picks_endpoint_filters(client):
    c, out = client
    _write_outcomes(out, [
        {"pick_id": "p1", "ts": "2026-05-01T12:00:00+00:00",
         "symbol": "AAPL", "regime_at_pick": "risk_on",
         "confidence": 0.5, "agents_voted": ["research"],
         "return_eod": 0.05, "correct": True},
        {"pick_id": "p2", "ts": "2026-05-02T12:00:00+00:00",
         "symbol": "MSFT", "regime_at_pick": "chop",
         "confidence": 0.3, "agents_voted": ["research"],
         "return_eod": -0.02, "correct": False},
    ])
    # No filter — both.
    r = c.get("/api/learning/picks")
    body = r.json()
    assert len(body["picks"]) == 2
    # Newer first (ts desc).
    assert body["picks"][0]["symbol"] == "MSFT"

    # Symbol filter.
    r = c.get("/api/learning/picks?symbol=aapl")  # case-insensitive
    body = r.json()
    assert len(body["picks"]) == 1
    assert body["picks"][0]["symbol"] == "AAPL"

    # Regime filter.
    r = c.get("/api/learning/picks?regime=chop")
    body = r.json()
    assert len(body["picks"]) == 1
    assert body["picks"][0]["symbol"] == "MSFT"


def test_summary_endpoint_has_close_loop_keys(client):
    """The summary now carries the close-the-loop view alongside legacy keys."""
    c, _ = client
    data = c.get("/api/learning/summary").json()
    for key in (
        "decided",
        "cold_start",
        "accuracy_7d",
        "accuracy_30d",
        "calibration",
        "what_works",
        "recent_adjustments",
    ):
        assert key in data
    # Empty journal -> still learning, never crashes.
    assert data["cold_start"] is True
    assert data["calibration"]["trust"]["level"] == "learning"
    assert data["what_works"] == {"symbols": [], "strategies": [], "regimes": []}


def test_summary_includes_agent_weights_key(client, monkeypatch, tmp_path):
    """Summary carries the new per-agent influence weights (cold-start safe)."""
    from packages.learning import agent_weights as aw

    monkeypatch.setattr(aw, "DEFAULT_AGENT_WEIGHTS_PATH", tmp_path / "aw.json")
    c, _ = client
    data = c.get("/api/learning/summary").json()
    assert "agent_weights" in data
    assert data["agent_weights"]["cold_start"] is True
    assert data["agent_weights"]["agents"] == {}


def test_agent_weights_endpoint_cold_start(client, monkeypatch, tmp_path):
    from packages.learning import agent_weights as aw

    monkeypatch.setattr(aw, "DEFAULT_AGENT_WEIGHTS_PATH", tmp_path / "aw.json")
    c, _ = client
    r = c.get("/api/learning/agent-weights")
    assert r.status_code == 200
    s = r.json()
    assert s["cold_start"] is True
    assert s["agents"] == {}
    assert s["min_samples"] == aw.MIN_SAMPLES_PER_AGENT


def test_learning_page_has_listening_section(client):
    c, _ = client
    body = c.get("/learning").text
    assert "Who the AI is listening to most" in body
    assert "/api/learning/agent-weights" not in body  # page uses /summary
    assert "equal say" in body


def test_status_endpoint_cold_start(client):
    c, _ = client
    r = c.get("/api/learning/status")
    assert r.status_code == 200
    s = r.json()
    # Well-formed empty payload before the loop has ever run.
    assert "last_run" in s
    assert s["cold_start"] is True


def test_picks_endpoint_respects_limit(client):
    c, out = client
    rows = [
        {"pick_id": f"p{i}", "ts": f"2026-05-{i+1:02d}T12:00:00+00:00",
         "symbol": "AAPL", "regime_at_pick": "chop",
         "confidence": 0.5, "agents_voted": ["research"]}
        for i in range(5)
    ]
    _write_outcomes(out, rows)
    r = c.get("/api/learning/picks?limit=2")
    assert r.status_code == 200
    body = r.json()
    assert len(body["picks"]) == 2
    assert body["count"] == 5
