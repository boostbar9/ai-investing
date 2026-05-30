"""Tests for the Phase 15a one-click sizing-preset workflow.

Mirrors test_arm_live.py: fixture isolates ``.env`` + audit log + the
POLICY_* environment so the test can assert both file and runtime
state cleanly.
"""
from __future__ import annotations

import json
import os

import pytest
from fastapi.testclient import TestClient

from packages.cockpit.web import server as srv
from packages.cockpit.web import sizing_control as sc
from packages.shared import secrets as secrets_mod


@pytest.fixture
def isolated_sizing(monkeypatch, tmp_path):
    """Point .env, audit log, and POLICY_* env vars at a clean fixture."""
    env_path = tmp_path / ".env"
    audit_path = tmp_path / "sizing_audit.jsonl"

    monkeypatch.setattr(secrets_mod, "_env_path", lambda: env_path)
    monkeypatch.setattr(sc, "SIZING_AUDIT_PATH", audit_path)

    # Strip any inherited POLICY_* env so tests start from a known state.
    for key in sc.SIZING_KEYS:
        monkeypatch.delenv(key, raising=False)
    return {"env_path": env_path, "audit_path": audit_path}


# ---------------------------------------------------------------------------
# Pure logic (no HTTP)
# ---------------------------------------------------------------------------


def test_presets_cover_all_four_modes():
    assert set(sc.PRESETS.keys()) == {"off", "conservative", "balanced", "aggressive"}
    for name, preset in sc.PRESETS.items():
        # Every preset must touch every key so apply() is deterministic.
        assert set(preset.keys()) == set(sc.SIZING_KEYS), name


def test_off_preset_deletes_all_keys(isolated_sizing):
    # Seed .env with a balanced config first.
    sc.configure(preset="balanced")
    assert os.environ.get("POLICY_SIZING_MODE") == "fractional_kelly"

    # Now flip Off.
    r = sc.configure(preset="off")
    assert r.ok is True
    assert r.action == "applied"
    for key in sc.SIZING_KEYS:
        assert key not in os.environ, key
    env_text = isolated_sizing["env_path"].read_text(encoding="utf-8")
    for key in sc.SIZING_KEYS:
        # Empty values must be removed from .env, not left as KEY=
        for line in env_text.splitlines():
            assert not line.startswith(f"{key}="), f"{key} should be deleted, got: {line}"


def test_balanced_preset_writes_expected_env(isolated_sizing):
    r = sc.configure(preset="balanced")
    assert r.ok is True
    assert os.environ["POLICY_SIZING_MODE"] == "fractional_kelly"
    assert os.environ["POLICY_KELLY_FRACTION"] == "0.25"
    env_text = isolated_sizing["env_path"].read_text(encoding="utf-8")
    assert "POLICY_SIZING_MODE=fractional_kelly" in env_text
    assert "POLICY_KELLY_FRACTION=0.25" in env_text


def test_aggressive_preset_writes_higher_kelly(isolated_sizing):
    r = sc.configure(preset="aggressive")
    assert r.ok is True
    assert os.environ["POLICY_KELLY_FRACTION"] == "0.40"
    assert float(os.environ["POLICY_MAX_POSITION_WEIGHT"]) > 0.20


def test_conservative_preset_writes_confidence_proportional(isolated_sizing):
    r = sc.configure(preset="conservative")
    assert r.ok is True
    assert os.environ["POLICY_SIZING_MODE"] == "confidence_proportional"
    assert float(os.environ["POLICY_KELLY_FRACTION"]) < 0.25


def test_unknown_preset_blocked(isolated_sizing):
    r = sc.configure(preset="yolo")
    assert r.ok is False
    assert r.action == "blocked"
    assert any("unknown preset" in reason for reason in r.reasons)


def test_missing_preset_and_overrides_blocked(isolated_sizing):
    r = sc.configure()
    assert r.ok is False
    assert r.action == "blocked"


def test_overrides_layer_on_top_of_preset(isolated_sizing):
    r = sc.configure(preset="balanced", overrides={"POLICY_KELLY_FRACTION": "0.10"})
    assert r.ok is True
    assert os.environ["POLICY_KELLY_FRACTION"] == "0.10"
    # Other balanced values still applied.
    assert os.environ["POLICY_SIZING_MODE"] == "fractional_kelly"


def test_override_validation_rejects_bad_mode(isolated_sizing):
    r = sc.configure(overrides={"POLICY_SIZING_MODE": "moon_mode"})
    assert r.ok is False
    assert any("POLICY_SIZING_MODE" in reason for reason in r.reasons)


def test_override_validation_rejects_non_numeric(isolated_sizing):
    r = sc.configure(overrides={"POLICY_KELLY_FRACTION": "high"})
    assert r.ok is False
    assert any("must be numeric" in reason for reason in r.reasons)


def test_override_validation_rejects_out_of_range(isolated_sizing):
    r = sc.configure(overrides={"POLICY_KELLY_FRACTION": "2.5"})
    assert r.ok is False
    assert any("must be in (0, 1]" in reason for reason in r.reasons)


def test_override_validation_rejects_unknown_key(isolated_sizing):
    r = sc.configure(overrides={"POLICY_HAYWIRE": "1.0"})
    assert r.ok is False
    assert any("unknown key" in reason for reason in r.reasons)


def test_audit_row_appended(isolated_sizing):
    sc.configure(preset="balanced", note="day-2 soak start")
    sc.configure(preset="off", note="rollback")
    rows = sc.read_audit(limit=10)
    assert len(rows) == 2
    # Newest first.
    assert rows[0]["preset"] == "off"
    assert rows[1]["preset"] == "balanced"
    assert rows[1]["note"] == "day-2 soak start"


def test_audit_row_is_valid_jsonl(isolated_sizing):
    sc.configure(preset="balanced")
    text = isolated_sizing["audit_path"].read_text(encoding="utf-8")
    for line in text.splitlines():
        if not line.strip():
            continue
        json.loads(line)  # must parse


def test_current_config_matches_active_preset(isolated_sizing):
    sc.configure(preset="balanced")
    cfg = sc.current_config()
    assert cfg["matched_preset"] == "balanced"
    assert cfg["effective_mode"] == "fractional_kelly"
    assert "off" in cfg["presets"]
    assert cfg["presets"]["balanced"]["description"]


def test_current_config_off_when_unset(isolated_sizing):
    cfg = sc.current_config()
    assert cfg["matched_preset"] == "off"
    assert cfg["effective_mode"] == "equal_weight"


def test_current_config_custom_when_overrides_present(isolated_sizing):
    sc.configure(preset="balanced", overrides={"POLICY_KELLY_FRACTION": "0.33"})
    cfg = sc.current_config()
    assert cfg["matched_preset"] is None  # diverges from any preset
    assert cfg["active"]["POLICY_KELLY_FRACTION"] == "0.33"


# ---------------------------------------------------------------------------
# HTTP endpoints
# ---------------------------------------------------------------------------


def test_get_sizing_config_endpoint(isolated_sizing):
    client = TestClient(srv.app)
    resp = client.get("/api/sizing/config")
    assert resp.status_code == 200
    body = resp.json()
    assert "active" in body
    assert "presets" in body
    assert {"off", "conservative", "balanced", "aggressive"}.issubset(body["presets"].keys())


def test_post_sizing_configure_endpoint(isolated_sizing):
    client = TestClient(srv.app)
    resp = client.post("/api/sizing/configure", json={"preset": "balanced"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["action"] == "applied"
    assert body["preset"] == "balanced"
    assert os.environ["POLICY_SIZING_MODE"] == "fractional_kelly"


def test_post_sizing_configure_with_overrides(isolated_sizing):
    client = TestClient(srv.app)
    resp = client.post(
        "/api/sizing/configure",
        json={"preset": "balanced", "overrides": {"POLICY_KELLY_FRACTION": "0.30"}},
    )
    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    assert os.environ["POLICY_KELLY_FRACTION"] == "0.30"


def test_post_sizing_configure_rejects_bad_preset(isolated_sizing):
    client = TestClient(srv.app)
    resp = client.post("/api/sizing/configure", json={"preset": "moon"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is False
    assert body["action"] == "blocked"


def test_post_sizing_configure_no_body(isolated_sizing):
    client = TestClient(srv.app)
    resp = client.post("/api/sizing/configure", json={})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is False


def test_sizing_audit_endpoint(isolated_sizing):
    client = TestClient(srv.app)
    client.post("/api/sizing/configure", json={"preset": "balanced"})
    client.post("/api/sizing/configure", json={"preset": "off"})
    resp = client.get("/api/sizing/audit?limit=10")
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 2
    # Newest first.
    assert body["events"][0]["preset"] == "off"


def test_settings_page_references_sizing_card():
    """The /settings HTML must mention the new card so the JS hooks fire."""
    client = TestClient(srv.app)
    resp = client.get("/settings")
    assert resp.status_code == 200
    html = resp.text
    assert "Risk-Adaptive Sizing" in html
    assert "/api/sizing/config" in html
    assert "/api/sizing/configure" in html
    assert "sizing-presets" in html
