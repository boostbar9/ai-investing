"""Live-progress endpoint for the agents Run button.

These tests verify the /api/agents/in_flight endpoint that powers the
per-agent ticker (e.g. ``strategy ... 12s``) shown while the Run button
on /agents waits for /api/agents/run to return.

We exercise four scenarios:
  * idle baseline before any run
  * stub-backend run completes and reports completed agents
  * preflight failure when Ollama is unreachable (use_llm=true)
  * shape of the in-flight payload is stable
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from packages.cockpit.web import server as srv


@pytest.fixture(autouse=True)
def _hermetic_logs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect side-effect log files so the run handler is hermetic."""
    monkeypatch.setattr(srv, "AGENT_LOG", tmp_path / "agents_log.jsonl")
    monkeypatch.setattr(srv, "SCORECARD_LOG", tmp_path / "scorecard.jsonl")
    monkeypatch.setattr(srv, "PAPER_LOG", tmp_path / "paper.jsonl")
    monkeypatch.setattr(srv, "DISCOVERY_LOG", tmp_path / "discovery.jsonl")
    yield


@pytest.fixture()
def client() -> TestClient:
    return TestClient(srv.app)


def test_in_flight_idle_before_any_run(client: TestClient) -> None:
    """Baseline payload must always be well-formed even with no run history."""
    # Reset progress so prior tests don't leak state.
    srv._AGENT_PROGRESS.clear()
    srv._AGENT_PROGRESS.update({
        "active": False, "started_at": None, "current_agent": None,
        "agent_started_at": None, "completed": [], "backend": None, "error": None,
    })
    r = client.get("/api/agents/in_flight")
    assert r.status_code == 200
    j = r.json()
    assert j["active"] is False
    assert j["current_agent"] is None
    assert j["completed"] == []
    assert isinstance(j["all_agents"], list) and "research" in j["all_agents"]
    assert j["elapsed_s"] == 0.0
    assert j["current_agent_elapsed_s"] == 0.0


def test_stub_run_records_completed_agents(client: TestClient) -> None:
    """After a stub run, in_flight.completed must list all four core agents."""
    r = client.post(
        "/api/agents/run",
        json={"symbols": ["SPY"], "regime": "chop", "use_llm": False},
    )
    assert r.status_code == 200, r.text
    snap = client.get("/api/agents/in_flight").json()
    # Run is finished, so active is False but completed history is preserved
    # until the next run resets it.
    assert snap["active"] is False
    completed_names = [c["agent"] for c in snap["completed"]]
    assert "research" in completed_names
    assert "strategy" in completed_names
    assert "risk" in completed_names
    assert "execution" in completed_names
    # All stub agents should report ok status.
    for c in snap["completed"]:
        assert c["status"] == "ok"
        assert c["elapsed_s"] >= 0.0


def test_llm_preflight_fails_fast_when_ollama_down(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When use_llm=true and Ollama is unreachable, return 503 immediately
    rather than letting the user sit through a 90s cold-start timeout per
    agent. The in_flight payload should record the preflight failure."""

    class _BoomClient:
        def __init__(self, *_a, **_kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *_a): return False
        async def get(self, *_a, **_kw):
            raise ConnectionError("Ollama refused connection")

    monkeypatch.setattr("httpx.AsyncClient", _BoomClient)

    r = client.post(
        "/api/agents/run",
        json={"symbols": ["SPY"], "regime": "chop", "use_llm": True},
    )
    assert r.status_code == 503
    assert "Ollama" in r.json()["detail"]
    # The progress dict should remember the preflight failure.
    snap = client.get("/api/agents/in_flight").json()
    assert snap["active"] is False
    completed_names = [c["agent"] for c in snap["completed"]]
    assert "preflight" in completed_names
    preflight = next(c for c in snap["completed"] if c["agent"] == "preflight")
    assert preflight["status"] == "failed"
    assert snap["error"] is not None


def test_in_flight_payload_shape_is_stable(client: TestClient) -> None:
    """The contract the agents.html ticker relies on."""
    j = client.get("/api/agents/in_flight").json()
    required = {
        "active", "current_agent", "backend", "completed", "error",
        "all_agents", "elapsed_s", "current_agent_elapsed_s",
    }
    assert required.issubset(j.keys())


def test_in_flight_is_in_quiet_path_prefixes() -> None:
    """The endpoint is polled every 1s during a run -- it MUST be in the
    uvicorn access-log quiet list or the terminal will flood."""
    assert "/api/agents/in_flight" in srv._QUIET_PATH_PREFIXES
