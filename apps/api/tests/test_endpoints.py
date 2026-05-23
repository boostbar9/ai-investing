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


def test_security_rotation_reminder_records_audit():
    c = TestClient(app)
    payload = {
        "ts": "2026-04-01T14:00:00Z",
        "scope": "broker, market_data",
        "runbook": "https://example.test/runbook",
        "channel": "n8n-quarterly",
    }
    r = c.post("/security/rotation-reminder", json=payload).json()
    assert r["ok"] is True and r["audit_id"]
    listed = c.get("/security/audit").json()["events"]
    assert any(e["audit_id"] == r["audit_id"] for e in listed)


def test_passkey_full_round_trip(monkeypatch):
    import base64
    import json

    monkeypatch.setenv("WEBAUTHN_RP_ID", "localhost")
    monkeypatch.setenv("WEBAUTHN_ORIGIN", "http://localhost:3000")
    # Reset the in-process store so test order doesn't matter
    from apps.api import main as api_main
    from packages.shared.passkeys import PasskeyStore
    api_main._PASSKEY_STORE = PasskeyStore()

    c = TestClient(app)

    def _b64url(b: bytes) -> str:
        return base64.urlsafe_b64encode(b).rstrip(b"=").decode("ascii")

    def _client_data(ctype: str, challenge: str) -> str:
        return _b64url(json.dumps({
            "type": ctype, "challenge": challenge,
            "origin": "http://localhost:3000",
        }).encode())

    # 1. registration options
    opts = c.post("/auth/passkey/register/options", json={}).json()
    assert opts["rp"]["id"] == "localhost"
    assert opts["challenge"]

    # 2. registration verify
    reg = c.post("/auth/passkey/register/verify", json={
        "response": {
            "id": "cred-test",
            "rawId": "cred-test",
            "type": "public-key",
            "response": {
                "clientDataJSON": _client_data("webauthn.create", opts["challenge"]),
                "publicKey": "pk-blob",
                "transports": ["internal"],
            },
        },
        "label": "Test device",
    }).json()
    assert reg["ok"] is True and reg["credential_id"] == "cred-test"

    # 3. authentication options
    auth_opts = c.post("/auth/passkey/authenticate/options", json={}).json()
    assert any(cred["id"] == "cred-test" for cred in auth_opts["allowCredentials"])

    # 4. authentication verify
    signed = c.post("/auth/passkey/authenticate/verify", json={
        "response": {
            "id": "cred-test",
            "rawId": "cred-test",
            "type": "public-key",
            "response": {
                "clientDataJSON": _client_data("webauthn.get", auth_opts["challenge"]),
            },
        },
    }).json()
    assert signed["ok"] is True
    assert signed["session_hint"]


def test_passkey_authenticate_rejects_unknown_credential(monkeypatch):
    import base64
    import json

    monkeypatch.setenv("WEBAUTHN_RP_ID", "localhost")
    monkeypatch.setenv("WEBAUTHN_ORIGIN", "http://localhost:3000")
    from apps.api import main as api_main
    from packages.shared.passkeys import PasskeyStore
    api_main._PASSKEY_STORE = PasskeyStore()

    c = TestClient(app)
    opts = c.post("/auth/passkey/authenticate/options", json={}).json()
    body = {
        "response": {
            "id": "cred-does-not-exist",
            "rawId": "cred-does-not-exist",
            "type": "public-key",
            "response": {
                "clientDataJSON": base64.urlsafe_b64encode(
                    json.dumps({
                        "type": "webauthn.get",
                        "challenge": opts["challenge"],
                        "origin": "http://localhost:3000",
                    }).encode()
                ).rstrip(b"=").decode("ascii"),
            },
        },
    }
    r = c.post("/auth/passkey/authenticate/verify", json=body)
    assert r.status_code == 401
