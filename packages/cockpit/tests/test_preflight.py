"""Phase 16: tests for the pre-flight aggregator + endpoint + page."""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from packages.cockpit.web import preflight as pf
from packages.cockpit.web import server as srv

# ---------------------------------------------------------------------------
# Fixture: isolate every persistence path the checks read
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_data_dir(monkeypatch, tmp_path):
    """Point the preflight module at a clean tmp data tree."""
    data = tmp_path / "data"
    (data / "paper_log").mkdir(parents=True)
    (data / "calibration").mkdir(parents=True)
    monkeypatch.setattr(pf, "DATA_DIR", data)

    # Strip noisy env so checks read clean defaults.
    for env_key in (
        "APCA_API_KEY_ID",
        "APCA_API_SECRET_KEY",
        "TELEGRAM_BOT_TOKEN",
        "TELEGRAM_CHAT_ID",
        "POLICY_SIZING_MODE",
        "POLICY_KELLY_FRACTION",
        "POLICY_DD_TAPER_START",
        "POLICY_DD_HARD_LIMIT",
        "POLICY_MAX_POSITION_WEIGHT",
        "POLICY_TARGET_VOL_ANNUAL",
    ):
        monkeypatch.delenv(env_key, raising=False)
    return data


# ---------------------------------------------------------------------------
# CheckResult dataclass
# ---------------------------------------------------------------------------


def test_check_result_to_dict_round_trip():
    r = pf.CheckResult(key="x", name="X", status="ok", message="hi", details={"a": 1})
    d = r.to_dict()
    assert d == {"key": "x", "name": "X", "status": "ok", "message": "hi", "details": {"a": 1}}


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def test_session_peak_missing(fake_data_dir):
    r = pf._check_session_peak()
    assert r.status == "info"
    assert "no peak file" in r.message


def test_session_peak_ok(fake_data_dir):
    (fake_data_dir / "paper_log" / "session_peak.json").write_text(
        json.dumps({"peak": 100_000.0, "updated_at": "2026-05-30T00:00:00Z"})
    )
    r = pf._check_session_peak()
    assert r.status == "ok"
    assert "100,000.00" in r.message


def test_session_peak_corrupt(fake_data_dir):
    (fake_data_dir / "paper_log" / "session_peak.json").write_text("not json")
    r = pf._check_session_peak()
    assert r.status == "fail"
    assert "corrupt" in r.message


def test_session_peak_zero(fake_data_dir):
    (fake_data_dir / "paper_log" / "session_peak.json").write_text(
        json.dumps({"peak": 0.0})
    )
    r = pf._check_session_peak()
    assert r.status == "warn"


def test_calibration_missing(fake_data_dir):
    r = pf._check_calibration()
    assert r.status == "warn"
    assert "identity" in r.message


def test_calibration_under_fit(fake_data_dir):
    (fake_data_dir / "calibration" / "policy_isotonic.json").write_text(
        json.dumps({"n_samples_fit": 12, "raw_ece": 0.1, "calibrated_ece": 0.05})
    )
    r = pf._check_calibration()
    assert r.status == "warn"
    assert "12 samples" in r.message


def test_calibration_ok(fake_data_dir):
    (fake_data_dir / "calibration" / "policy_isotonic.json").write_text(
        json.dumps({"n_samples_fit": 200, "raw_ece": 0.1, "calibrated_ece": 0.04})
    )
    r = pf._check_calibration()
    assert r.status == "ok"
    assert "200 samples" in r.message


def test_calibration_corrupt(fake_data_dir):
    (fake_data_dir / "calibration" / "policy_isotonic.json").write_text("xxx")
    r = pf._check_calibration()
    assert r.status == "fail"


def test_decision_log_missing(fake_data_dir):
    r = pf._check_decision_log()
    assert r.status == "warn"


def test_decision_log_ok(fake_data_dir):
    (fake_data_dir / "paper_log" / "decisions.jsonl").write_text('{"a":1}\n')
    r = pf._check_decision_log()
    assert r.status == "ok"


def test_decision_log_stale(fake_data_dir):
    import os
    import time

    path = fake_data_dir / "paper_log" / "decisions.jsonl"
    path.write_text('{"a":1}\n')
    old = time.time() - 72 * 3600
    os.utime(path, (old, old))
    r = pf._check_decision_log()
    assert r.status == "warn"
    assert "last write" in r.message


def test_disk_space_ok(fake_data_dir):
    r = pf._check_disk_space()
    assert r.status in {"ok", "warn"}
    assert "free" in r.message


def test_alpaca_keys_missing(fake_data_dir):
    r = pf._check_alpaca_keys()
    assert r.status == "fail"


def test_alpaca_keys_ok(fake_data_dir, monkeypatch):
    monkeypatch.setenv("APCA_API_KEY_ID", "PK1234567890ABCDEF")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "secret-xyz")
    r = pf._check_alpaca_keys()
    assert r.status == "ok"
    assert "CDEF" in r.message


def test_telegram_missing(fake_data_dir):
    r = pf._check_telegram()
    assert r.status == "warn"


def test_telegram_configured(fake_data_dir, monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "x")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "123456")
    r = pf._check_telegram()
    assert r.status == "ok"


def test_sizing_off_is_info(fake_data_dir):
    # No POLICY_SIZING_MODE env -> equal_weight -> info
    r = pf._check_sizing_preset()
    assert r.status == "info"
    assert "OFF" in r.message or "equal_weight" in r.message


def test_sizing_balanced_is_ok(fake_data_dir, monkeypatch):
    monkeypatch.setenv("POLICY_SIZING_MODE", "fractional_kelly")
    monkeypatch.setenv("POLICY_KELLY_FRACTION", "0.25")
    monkeypatch.setenv("POLICY_DD_TAPER_START", "0.03")
    monkeypatch.setenv("POLICY_DD_HARD_LIMIT", "0.08")
    monkeypatch.setenv("POLICY_MAX_POSITION_WEIGHT", "0.20")
    monkeypatch.setenv("POLICY_TARGET_VOL_ANNUAL", "0.18")
    r = pf._check_sizing_preset()
    assert r.status == "ok"
    assert "fractional_kelly" in r.message


# ---------------------------------------------------------------------------
# Promote payload integration
# ---------------------------------------------------------------------------


def _green_promote():
    return {
        "live_enabled": True,
        "capital_fraction": 0.10,
        "readiness": {"ready": True, "reasons": [], "metrics": {"paper_days": 30}},
        "requirements": {"paper_min_days": 14, "paper_max_dd": 0.10, "paper_min_sharpe": 0.5},
        "progress": {
            "paper_days": 30,
            "days_remaining": 0,
            "telegram_connected": True,
            "enable_live_flag": False,
            "shadow_status": "ready",
            "shadow_streak_days": 14,
            "shadow_days_required": 14,
            "shadow_ready": True,
        },
        "canary": None,
    }


def _blocked_promote():
    return {
        "live_enabled": False,
        "capital_fraction": 0.0,
        "readiness": {"ready": False, "reasons": ["not enough days"], "metrics": {}},
        "requirements": {"paper_min_days": 14},
        "progress": {
            "paper_days": 1,
            "days_remaining": 13,
            "shadow_streak_days": 0,
            "shadow_days_required": 14,
            "shadow_ready": False,
        },
        "canary": None,
    }


def test_check_paper_days_ok():
    r = pf._check_paper_days(_green_promote())
    assert r.status == "ok"
    assert "30" in r.message and "14" in r.message


def test_check_paper_days_fail():
    r = pf._check_paper_days(_blocked_promote())
    assert r.status == "fail"
    assert "13 more" in r.message


def test_check_shadow_streak_ok():
    r = pf._check_shadow_streak(_green_promote())
    assert r.status == "ok"


def test_check_shadow_streak_fail():
    r = pf._check_shadow_streak(_blocked_promote())
    assert r.status == "fail"


def test_check_live_gate_ok():
    r = pf._check_live_gate(_green_promote())
    assert r.status == "ok"
    assert "10%" in r.message


def test_check_live_gate_blocked():
    r = pf._check_live_gate(_blocked_promote())
    assert r.status == "fail"
    assert "not enough days" in r.message


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


def test_compute_preflight_blocked(fake_data_dir, monkeypatch):
    """No keys, no peak, no calibrator, blocked promote -> not ready."""
    monkeypatch.setattr(
        "packages.cockpit.web.server._compute_promote_payload",
        _blocked_promote,
    )
    summary = pf.compute_preflight()
    d = summary.to_dict()
    assert d["ready"] is False
    assert d["counts"]["fail"] >= 1  # alpaca keys + live gate at minimum
    assert isinstance(d["sections"], list)
    assert len(d["sections"]) == 4
    keys = [s["key"] for s in d["sections"]]
    assert keys == ["persistence", "connectivity", "policy", "gate"]
    assert any("live" in r.lower() or "key" in r.lower() for r in d["blocking_reasons"])


def test_compute_preflight_green_path(fake_data_dir, monkeypatch):
    """Every check green -> ready=True."""
    # Seed persistence files.
    (fake_data_dir / "paper_log" / "session_peak.json").write_text(
        json.dumps({"peak": 100_000.0})
    )
    (fake_data_dir / "calibration" / "policy_isotonic.json").write_text(
        json.dumps({"n_samples_fit": 200})
    )
    (fake_data_dir / "paper_log" / "decisions.jsonl").write_text('{"a":1}\n')

    monkeypatch.setenv("APCA_API_KEY_ID", "PKABCDEFGH123456")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "secret")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "x")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
    monkeypatch.setenv("POLICY_SIZING_MODE", "fractional_kelly")
    monkeypatch.setenv("POLICY_KELLY_FRACTION", "0.25")
    monkeypatch.setenv("POLICY_DD_TAPER_START", "0.03")
    monkeypatch.setenv("POLICY_DD_HARD_LIMIT", "0.08")
    monkeypatch.setenv("POLICY_MAX_POSITION_WEIGHT", "0.20")
    monkeypatch.setenv("POLICY_TARGET_VOL_ANNUAL", "0.18")

    monkeypatch.setattr(
        "packages.cockpit.web.server._compute_promote_payload",
        _green_promote,
    )
    # Pretend the account snapshot is healthy.
    monkeypatch.setattr(
        "packages.cockpit.web.server.latest_account_snapshot",
        lambda: {"equity": "100000.00", "status": "ACTIVE"},
    )

    summary = pf.compute_preflight()
    d = summary.to_dict()
    assert d["ready"] is True
    assert d["counts"]["fail"] == 0
    assert d["blocking_reasons"] == []


def test_safe_run_traps_exception():
    def boom():
        raise RuntimeError("nope")

    r = pf._safe_run("x", "X", boom)
    assert r.status == "fail"
    assert "RuntimeError" in r.message


# ---------------------------------------------------------------------------
# HTTP endpoint + page render
# ---------------------------------------------------------------------------


def test_api_preflight_endpoint(fake_data_dir, monkeypatch):
    monkeypatch.setattr(
        "packages.cockpit.web.server._compute_promote_payload", _blocked_promote
    )
    client = TestClient(srv.app)
    r = client.get("/api/preflight")
    assert r.status_code == 200
    body = r.json()
    assert "ready" in body
    assert "counts" in body
    assert "sections" in body


def test_preflight_page_renders():
    client = TestClient(srv.app)
    r = client.get("/preflight")
    assert r.status_code == 200
    html = r.text
    assert "Pre-flight Checklist" in html
    assert "/api/preflight" in html
    assert "Arm Live" in html


def test_preflight_link_in_nav():
    """The new nav link must appear on every page that already had Promote."""
    client = TestClient(srv.app)
    for path in ("/", "/settings", "/shadow", "/trading", "/health"):
        r = client.get(path)
        assert r.status_code == 200, path
        assert "/preflight" in r.text, f"{path} missing preflight nav link"
