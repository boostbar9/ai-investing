"""Phase 36c — remote-control bridge tests.

Covers the security gate, every route in the ``/api/remote/*`` surface,
and the fail-closed default. The cockpit's job manager is stubbed so we
don't spawn real subprocesses; state mutations are persisted to a
per-test temp file via fixture monkeypatches.
"""

from __future__ import annotations

import types
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from packages.cockpit import proc as cockpit_proc
from packages.cockpit import state as st
from packages.cockpit.web import remote as remote_mod
from packages.cockpit.web import server as srv

GOOD_TOKEN = "x" * 32  # 32 chars, easily exceeds 16-char floor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "state.json"
    monkeypatch.setattr(st, "STATE_PATH", path)
    monkeypatch.setattr(st.load_state, "__defaults__", (path,))
    monkeypatch.setattr(st.save_state, "__defaults__", (path,))
    return path


@pytest.fixture
def token_env(monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv(remote_mod.ENV_TOKEN, GOOD_TOKEN)
    return GOOD_TOKEN


@pytest.fixture
def no_token_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(remote_mod.ENV_TOKEN, raising=False)


@pytest.fixture
def stub_jobs(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Stub the proc.start / proc.stop / proc.status / proc.tail_log calls."""
    calls: dict[str, Any] = {"start": [], "stop": [], "status": [], "tail": []}

    def _start(kind: str, cmd: list[str], cwd: Path | None = None):
        calls["start"].append((kind, list(cmd)))
        return types.SimpleNamespace(to_dict=lambda: {"running": True, "pid": 999})

    def _stop(kind: str):
        calls["stop"].append(kind)
        return types.SimpleNamespace(to_dict=lambda: {"running": False})

    def _status(kind: str):
        calls["status"].append(kind)
        return types.SimpleNamespace(
            to_dict=lambda: {"running": False, "kind": kind},
            is_running=lambda: False,
        )

    def _tail(kind: str, max_bytes: int = 64 * 1024) -> str:
        calls["tail"].append(kind)
        return "[stub] last 3 lines of paper_loop log\nline 2\nline 3\n"

    monkeypatch.setattr(cockpit_proc, "start", _start)
    monkeypatch.setattr(cockpit_proc, "stop", _stop)
    monkeypatch.setattr(cockpit_proc, "status", _status)
    monkeypatch.setattr(cockpit_proc, "tail_log", _tail)
    return calls


@pytest.fixture
def client() -> TestClient:
    return TestClient(srv.app)


def auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# Fail-closed default
# ---------------------------------------------------------------------------


def test_health_works_without_auth(
    client: TestClient, no_token_env: None
) -> None:
    """/health is always reachable so operators can confirm the surface."""
    r = client.get("/api/remote/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["enabled"] is False


def test_health_reports_enabled_when_token_set(
    client: TestClient, token_env: str
) -> None:
    r = client.get("/api/remote/health")
    assert r.status_code == 200
    assert r.json()["enabled"] is True


@pytest.mark.parametrize(
    "method,path,body",
    [
        ("GET", "/api/remote/whoami", None),
        ("GET", "/api/remote/snapshot", None),
        ("GET", "/api/remote/log", None),
        ("POST", "/api/remote/pause", None),
        ("POST", "/api/remote/resume", None),
        ("POST", "/api/remote/loop/start", {"strategy": "ensemble", "dry_run": False}),
        ("POST", "/api/remote/loop/stop", None),
        ("POST", "/api/remote/liquidate", {"confirm": "LIQUIDATE"}),
    ],
)
def test_surface_returns_503_when_disabled(
    client: TestClient, no_token_env: None, method: str, path: str, body
) -> None:
    """With COCKPIT_REMOTE_TOKEN unset, EVERY mutating route is 503."""
    r = client.request(method, path, json=body, headers=auth("anything"))
    assert r.status_code == 503, f"{method} {path} should be disabled"
    assert "disabled" in r.json()["detail"]


def test_short_token_treated_as_disabled(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 12-char token isn't long enough to count as configured."""
    monkeypatch.setenv(remote_mod.ENV_TOKEN, "tooshort1234")
    r = client.get("/api/remote/whoami", headers=auth("tooshort1234"))
    assert r.status_code == 503


# ---------------------------------------------------------------------------
# Auth header parsing
# ---------------------------------------------------------------------------


def test_missing_token_returns_401(client: TestClient, token_env: str) -> None:
    r = client.get("/api/remote/whoami")
    assert r.status_code == 401


def test_wrong_token_returns_403(client: TestClient, token_env: str) -> None:
    r = client.get("/api/remote/whoami", headers=auth("y" * 32))
    assert r.status_code == 403


def test_bearer_header_accepted(client: TestClient, token_env: str) -> None:
    r = client.get("/api/remote/whoami", headers={"Authorization": f"Bearer {token_env}"})
    assert r.status_code == 200


def test_bare_authorization_accepted(client: TestClient, token_env: str) -> None:
    """Some clients drop the 'Bearer ' prefix \u2014 we still accept it."""
    r = client.get("/api/remote/whoami", headers={"Authorization": token_env})
    assert r.status_code == 200


def test_x_cockpit_token_header_accepted(
    client: TestClient, token_env: str
) -> None:
    r = client.get("/api/remote/whoami", headers={"X-Cockpit-Token": token_env})
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# Read endpoints
# ---------------------------------------------------------------------------


def test_snapshot_returns_full_payload(
    client: TestClient, token_env: str, fake_state: Path, stub_jobs: dict[str, Any]
) -> None:
    r = client.get("/api/remote/snapshot", headers=auth(token_env))
    assert r.status_code == 200
    body = r.json()
    assert "state" in body
    assert "paper_loop" in body
    assert "log_tail" in body
    assert body["log_tail"].startswith("[stub]")
    assert body["errors"] == {}


def test_log_tail_endpoint(
    client: TestClient, token_env: str, stub_jobs: dict[str, Any]
) -> None:
    r = client.get("/api/remote/log", headers=auth(token_env))
    assert r.status_code == 200
    body = r.json()
    assert body["kind"] == "paper_loop"
    assert "[stub]" in body["tail"]


def test_log_download_returns_plaintext(
    client: TestClient,
    token_env: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """?download=1 returns plain text content from the on-disk log."""
    fake_log = tmp_path / "paper_loop.log"
    fake_log.write_text("first cycle\nsecond cycle\n", encoding="utf-8")
    monkeypatch.setattr(cockpit_proc, "log_path", lambda kind: fake_log)
    r = client.get("/api/remote/log?download=1", headers=auth(token_env))
    assert r.status_code == 200
    assert "text/plain" in r.headers["content-type"]
    assert "second cycle" in r.text


def test_log_download_missing_file_returns_placeholder(
    client: TestClient,
    token_env: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cockpit_proc, "log_path", lambda kind: tmp_path / "missing.log"
    )
    r = client.get("/api/remote/log?download=1", headers=auth(token_env))
    assert r.status_code == 200
    assert "no log" in r.text.lower()


# ---------------------------------------------------------------------------
# Mutating endpoints
# ---------------------------------------------------------------------------


def test_pause_persists_state(
    client: TestClient, token_env: str, fake_state: Path
) -> None:
    r = client.post("/api/remote/pause", headers=auth(token_env))
    assert r.status_code == 200
    assert r.json()["paused"] is True
    assert st.load_state().paused is True


def test_resume_clears_pause(
    client: TestClient, token_env: str, fake_state: Path
) -> None:
    # Pre-set paused to True so we can verify resume flips it.
    s = st.load_state()
    s.paused = True
    st.save_state(s)
    r = client.post("/api/remote/resume", headers=auth(token_env))
    assert r.status_code == 200
    assert r.json()["paused"] is False
    assert st.load_state().paused is False


def test_loop_start_spawns_and_records_intent(
    client: TestClient,
    token_env: str,
    fake_state: Path,
    stub_jobs: dict[str, Any],
) -> None:
    r = client.post(
        "/api/remote/loop/start",
        headers=auth(token_env),
        json={"strategy": "ensemble", "dry_run": False},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["state"]["paper_loop_intended"] is True
    assert body["state"]["paper_loop_strategy"] == "ensemble"
    assert body["state"]["paper_loop_user_touched"] is True
    assert body["job"]["running"] is True
    # Verify proc.start was called with --strategy ensemble --loop and no --dry-run.
    assert len(stub_jobs["start"]) == 1
    kind, cmd = stub_jobs["start"][0]
    assert kind == "paper_loop"
    assert "--strategy" in cmd and "ensemble" in cmd and "--loop" in cmd
    assert "--dry-run" not in cmd


def test_loop_start_dry_run_passes_flag(
    client: TestClient,
    token_env: str,
    fake_state: Path,
    stub_jobs: dict[str, Any],
) -> None:
    r = client.post(
        "/api/remote/loop/start",
        headers=auth(token_env),
        json={"strategy": "ensemble", "dry_run": True},
    )
    assert r.status_code == 200
    _, cmd = stub_jobs["start"][0]
    assert "--dry-run" in cmd
    assert r.json()["state"]["paper_loop_dry_run"] is True


def test_loop_stop_clears_intent_and_calls_proc(
    client: TestClient,
    token_env: str,
    fake_state: Path,
    stub_jobs: dict[str, Any],
) -> None:
    # Pre-set intent so we verify it's cleared.
    s = st.load_state()
    s.paper_loop_intended = True
    st.save_state(s)
    r = client.post("/api/remote/loop/stop", headers=auth(token_env))
    assert r.status_code == 200
    body = r.json()
    assert body["state"]["paper_loop_intended"] is False
    assert body["state"]["paper_loop_user_touched"] is True
    assert stub_jobs["stop"] == ["paper_loop"]


def test_liquidate_requires_confirm_token(
    client: TestClient,
    token_env: str,
    fake_state: Path,
    stub_jobs: dict[str, Any],
) -> None:
    r = client.post(
        "/api/remote/liquidate",
        headers=auth(token_env),
        json={"confirm": "yes please"},
    )
    assert r.status_code == 400
    assert "LIQUIDATE" in r.json()["detail"]


def test_liquidate_with_confirm_pauses_and_clears_intent(
    client: TestClient,
    token_env: str,
    fake_state: Path,
    stub_jobs: dict[str, Any],
) -> None:
    s = st.load_state()
    s.paper_loop_intended = True
    s.paused = False
    st.save_state(s)
    r = client.post(
        "/api/remote/liquidate",
        headers=auth(token_env),
        json={"confirm": "LIQUIDATE"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["state"]["paused"] is True
    assert body["state"]["paper_loop_intended"] is False
    assert body["state"]["paper_loop_user_touched"] is True
    assert "broker" in body["note"].lower()
    # The loop must be told to stop so it doesn't try to keep trading.
    assert "paper_loop" in stub_jobs["stop"]
