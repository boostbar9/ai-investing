"""Tests for one-click arm-live workflow + audit log + HTTP endpoints."""
from __future__ import annotations

import json
import os

import pytest
from fastapi.testclient import TestClient

from packages.cockpit.web import arm_live as arm_mod
from packages.cockpit.web import server as srv
from packages.shared import secrets as secrets_mod


@pytest.fixture
def isolated_arm(monkeypatch, tmp_path):
    """Isolate .env, audit log, and ENABLE_LIVE_TRADING env var."""
    env_path = tmp_path / ".env"
    audit_path = tmp_path / "arm_live_audit.jsonl"

    # secrets._env_path is called via _write_env_file; patch the underlying
    # path-resolver to point at tmp_path.
    monkeypatch.setattr(secrets_mod, "_env_path", lambda: env_path)
    monkeypatch.setattr(arm_mod, "ARM_AUDIT_PATH", audit_path)

    # Always start tests with the flag unset; the test that wants armed
    # state will set it explicitly via os.environ.
    monkeypatch.delenv("ENABLE_LIVE_TRADING", raising=False)
    return {"env_path": env_path, "audit_path": audit_path}


# ---------------------------------------------------------------------------
# Pure logic (no HTTP)
# ---------------------------------------------------------------------------


def _all_clear() -> dict:
    return {
        "live_enabled": True,
        "capital_fraction": 0.05,
        "readiness": {"ready": True, "reasons": [], "metrics": {}},
        "progress": {"enable_live_flag": False},
    }


def _blocked(reasons: list[str]) -> dict:
    return {
        "live_enabled": False,
        "capital_fraction": 0.0,
        "readiness": {"ready": False, "reasons": reasons, "metrics": {}},
        "progress": {"enable_live_flag": False},
    }


def test_arm_live_writes_env_when_gate_clear(isolated_arm):
    r = arm_mod.arm_live(actor="op", gate_evaluator=_all_clear)
    assert r.ok is True
    assert r.action == "armed"
    assert os.environ.get("ENABLE_LIVE_TRADING") == "true"
    env_text = isolated_arm["env_path"].read_text(encoding="utf-8")
    assert "ENABLE_LIVE_TRADING=true" in env_text


def test_arm_live_refuses_when_gate_blocked(isolated_arm):
    reasons = ["Shadow soak not complete (5/14)"]
    r = arm_mod.arm_live(
        actor="op", gate_evaluator=lambda: _blocked(reasons)
    )
    assert r.ok is False
    assert r.action == "blocked"
    assert "Shadow soak" in r.reasons[0]
    # No .env mutation when blocked.
    assert "ENABLE_LIVE_TRADING" not in os.environ
    if isolated_arm["env_path"].exists():
        assert "ENABLE_LIVE_TRADING" not in isolated_arm["env_path"].read_text(
            encoding="utf-8"
        )


def test_arm_live_blocked_still_audits_attempt(isolated_arm):
    """Even refused arm attempts get a row -- audit completeness matters."""
    arm_mod.arm_live(
        actor="op", gate_evaluator=lambda: _blocked(["paper days 12/60"])
    )
    rows = arm_mod.read_audit()
    assert len(rows) == 1
    assert rows[0]["action"] == "blocked"
    assert rows[0]["actor"] == "op"


def test_arm_live_noop_when_already_armed(isolated_arm):
    os.environ["ENABLE_LIVE_TRADING"] = "true"
    r = arm_mod.arm_live(actor="op", gate_evaluator=_all_clear)
    assert r.ok is True
    assert r.action == "noop"
    # No audit row -- this is intentional. Repeated armed-button clicks
    # shouldn't pollute the log; only state transitions get recorded.
    assert arm_mod.read_audit() == []


def test_arm_live_appends_audit_with_note(isolated_arm):
    r = arm_mod.arm_live(
        actor="op", gate_evaluator=_all_clear, note="14-day soak cleared"
    )
    assert r.ok and r.action == "armed"
    rows = arm_mod.read_audit()
    assert len(rows) == 1
    assert rows[0]["action"] == "armed"
    assert rows[0]["note"] == "14-day soak cleared"
    assert rows[0]["capital_fraction"] == 0.05


def test_disarm_requires_reason(isolated_arm):
    os.environ["ENABLE_LIVE_TRADING"] = "true"
    r = arm_mod.disarm_live(actor="op", reason="")
    assert r.ok is False
    # No audit row, no env mutation.
    assert arm_mod.read_audit() == []
    assert os.environ.get("ENABLE_LIVE_TRADING") == "true"


def test_disarm_flips_env_and_writes_audit(isolated_arm):
    os.environ["ENABLE_LIVE_TRADING"] = "true"
    # Pre-seed .env so we can verify the line gets removed.
    isolated_arm["env_path"].write_text(
        "ENABLE_LIVE_TRADING=true\nOTHER_KEY=keep_me\n", encoding="utf-8"
    )
    r = arm_mod.disarm_live(actor="op", reason="weekend pause")
    assert r.ok is True
    assert r.action == "disarmed"
    assert "ENABLE_LIVE_TRADING" not in os.environ
    text = isolated_arm["env_path"].read_text(encoding="utf-8")
    assert "ENABLE_LIVE_TRADING" not in text
    assert "OTHER_KEY=keep_me" in text  # unrelated keys preserved
    rows = arm_mod.read_audit()
    assert len(rows) == 1
    assert rows[0]["action"] == "disarmed"
    assert rows[0]["reasons"] == ["weekend pause"]


def test_disarm_noop_when_already_off(isolated_arm):
    r = arm_mod.disarm_live(actor="op", reason="just in case")
    assert r.ok is True
    assert r.action == "noop"
    assert arm_mod.read_audit() == []


def test_audit_log_atomic_write_jsonl_valid(isolated_arm):
    arm_mod.arm_live(actor="op", gate_evaluator=_all_clear)
    arm_mod.disarm_live(actor="op", reason="drill")
    text = isolated_arm["audit_path"].read_text(encoding="utf-8")
    assert text.endswith("\n")
    for line in text.splitlines():
        if line.strip():
            json.loads(line)  # parseable JSONL


def test_audit_trims_to_max_rows(isolated_arm, monkeypatch):
    monkeypatch.setattr(arm_mod, "MAX_AUDIT_ROWS", 3)
    for i in range(5):
        arm_mod._append_audit_row(
            {"ts": f"2026-05-{20+i:02d}T00:00:00+00:00", "action": "armed"}
        )
    rows = arm_mod.read_audit()
    assert len(rows) == 3
    # Oldest two dropped.
    assert rows[0]["ts"] == "2026-05-22T00:00:00+00:00"


def test_audit_tolerates_corrupt_line(isolated_arm):
    p = isolated_arm["audit_path"]
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        '{"action": "armed", "ts": "a"}\n'
        "not json\n"
        '{"action": "disarmed", "ts": "b"}\n',
        encoding="utf-8",
    )
    rows = arm_mod.read_audit()
    assert len(rows) == 2
    assert rows[0]["action"] == "armed"


# ---------------------------------------------------------------------------
# HTTP endpoints
# ---------------------------------------------------------------------------


def test_api_arm_live_blocked_returns_reasons(isolated_arm, monkeypatch):
    monkeypatch.setattr(
        srv, "_compute_promote_payload", lambda: _blocked(["shadow soak 5/14"])
    )
    client = TestClient(srv.app)
    r = client.post("/api/arm-live", json={"note": "trying"})
    assert r.status_code == 200
    j = r.json()
    assert j["ok"] is False
    assert j["action"] == "blocked"
    assert any("shadow" in s.lower() for s in j["reasons"])


def test_api_arm_live_armed_writes_env(isolated_arm, monkeypatch):
    monkeypatch.setattr(srv, "_compute_promote_payload", _all_clear)
    client = TestClient(srv.app)
    r = client.post("/api/arm-live", json={"note": "soak cleared"})
    assert r.status_code == 200
    j = r.json()
    assert j["ok"] is True
    assert j["action"] == "armed"
    assert os.environ.get("ENABLE_LIVE_TRADING") == "true"
    # Audit row visible via the audit endpoint.
    audit = client.get("/api/arm-live/audit").json()
    assert audit["count"] == 1
    assert audit["events"][0]["action"] == "armed"
    assert audit["events"][0]["note"] == "soak cleared"


def test_api_disarm_live_requires_reason(isolated_arm, monkeypatch):
    os.environ["ENABLE_LIVE_TRADING"] = "true"
    client = TestClient(srv.app)
    r = client.post("/api/disarm-live", json={"reason": ""})
    j = r.json()
    assert j["ok"] is False
    assert "reason is required" in " ".join(j["reasons"]).lower()


def test_api_disarm_live_flips_env(isolated_arm):
    os.environ["ENABLE_LIVE_TRADING"] = "true"
    client = TestClient(srv.app)
    r = client.post("/api/disarm-live", json={"reason": "drill"})
    j = r.json()
    assert j["ok"] is True
    assert j["action"] == "disarmed"
    assert "ENABLE_LIVE_TRADING" not in os.environ


def test_api_arm_live_audit_empty(isolated_arm):
    client = TestClient(srv.app)
    r = client.get("/api/arm-live/audit")
    assert r.status_code == 200
    assert r.json() == {"events": [], "count": 0}


def test_api_arm_live_audit_limit_clamped(isolated_arm):
    # Limit must clamp to [1, 500] -- guard against pathological inputs.
    client = TestClient(srv.app)
    r1 = client.get("/api/arm-live/audit?limit=0")
    r2 = client.get("/api/arm-live/audit?limit=10000")
    assert r1.status_code == 200
    assert r2.status_code == 200
