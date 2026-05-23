from fastapi.testclient import TestClient

from apps.api.main import app


def test_health():
    r = TestClient(app).get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_version():
    r = TestClient(app).get("/version")
    assert r.json()["spec"] == "v3.1"


def test_seed_and_decide_approval():
    c = TestClient(app)
    seeded = c.post("/_dev/seed-approval").json()
    did = seeded["decision_id"]
    assert c.get("/approvals/pending").json()["pending"]
    r = c.post(f"/approvals/{did}", json={"approve": True, "note": "ok"})
    assert r.status_code == 200 and r.json()["approved"] is True
    # Now pending should be empty for this id; audit should have the event.
    assert all(item["decision_id"] != did for item in c.get("/approvals/pending").json()["pending"])
    trace = c.get(f"/audit/{did}").json()
    assert any(e["event_type"] == "approval" for e in trace["events"])


def test_agents_status_shape():
    r = TestClient(app).get("/agents/status").json()
    assert set(r.keys()) == {"research", "strategy", "risk", "execution"}


def test_strategies_catalogue():
    r = TestClient(app).get("/strategies").json()
    names = {s["name"] for s in r["strategies"]}
    assert names == {
        "trend-following",
        "sector-rotation",
        "mean-reversion",
        "sentiment-overlay",
    }


def test_activity_feed_after_seed():
    c = TestClient(app)
    c.post("/_dev/seed-approval")
    events = c.get("/activity").json()["events"]
    assert any(e["event_type"] == "seed" for e in events)


def test_health_detail_shape():
    r = TestClient(app).get("/health/detail").json()
    assert {"api", "broker", "llm", "regime", "db", "cache"} <= r.keys()
    assert all(v["ok"] is True for v in r.values())


def test_live_promotion_empty_curves_fails_closed(tmp_path, monkeypatch):
    # No env vars set → empty curves → gate fails closed.
    monkeypatch.delenv("PAPER_EQUITY_PATH", raising=False)
    monkeypatch.delenv("LIVE_EQUITY_PATH", raising=False)
    monkeypatch.delenv("ENABLE_LIVE_TRADING", raising=False)
    r = TestClient(app).get("/live/promotion").json()
    assert r["live_enabled"] is False
    assert r["capital_fraction"] == 0.0
    assert r["readiness"]["ready"] is False
    assert r["canary"] is None


def test_live_promotion_ready_returns_canary(tmp_path, monkeypatch):
    import json

    import numpy as np

    rng = np.random.default_rng(42)
    eq = [100.0]
    for r in rng.normal(0.002, 0.005, 60):
        eq.append(eq[-1] * (1 + r))
    p = tmp_path / "paper.json"
    p.write_text(json.dumps({"equity": eq[1:]}))
    monkeypatch.setenv("PAPER_EQUITY_PATH", str(p))
    monkeypatch.setenv("ENABLE_LIVE_TRADING", "true")

    r = TestClient(app).get("/live/promotion").json()
    assert r["live_enabled"] is True
    # Empty live curve → tier 0 → 5%
    assert r["capital_fraction"] == 0.05
    assert r["canary"]["tier_index"] == 0
