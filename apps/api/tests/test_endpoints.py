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
    assert trace["events"][0]["event_type"] == "approval"


def test_agents_status_shape():
    r = TestClient(app).get("/agents/status").json()
    assert set(r.keys()) == {"research", "strategy", "risk", "execution"}
