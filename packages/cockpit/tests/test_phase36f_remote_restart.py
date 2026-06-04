"""Phase 36f -- /api/remote/restart endpoint tests.

The endpoint spawns a detached helper process via subprocess.Popen with
Windows-specific flags. We stub Popen so no real process spawns, and we
patch sys.platform to "win32" so the platform gate doesn't 501 us on
the Linux CI runner.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from packages.cockpit import state as st
from packages.cockpit.web import remote as remote_mod
from packages.cockpit.web import server as srv

GOOD_TOKEN = "x" * 32


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
def stub_popen_and_win32(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> dict[str, Any]:
    """Pretend we are on Windows and capture the spawn args.

    Also stub Path.exists so the helper-script-missing check passes
    without us having to set up a real tools/ tree at the repo root the
    endpoint resolves.
    """
    import subprocess as sp_mod
    import sys as sys_mod
    from pathlib import Path as PathCls

    captured: dict[str, Any] = {"popen_calls": []}

    class FakePopen:
        def __init__(self, args: list[str], **kwargs: Any) -> None:
            captured["popen_calls"].append({"args": list(args), "kwargs": dict(kwargs)})
            self.pid = 99999

    monkeypatch.setattr(sp_mod, "Popen", FakePopen)
    monkeypatch.setattr(sys_mod, "platform", "win32")

    # The endpoint checks `helper.exists()`. Force True so we don't have
    # to manufacture a tools/cockpit_restart_helper.ps1 inside the
    # actual workspace path the endpoint resolves to.
    orig_exists = PathCls.exists

    def _exists(self: PathCls) -> bool:
        if self.name == "cockpit_restart_helper.ps1":
            return True
        return orig_exists(self)

    monkeypatch.setattr(PathCls, "exists", _exists)
    return captured


@pytest.fixture
def client() -> TestClient:
    return TestClient(srv.app)


def auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# Auth gate
# ---------------------------------------------------------------------------


def test_restart_requires_token(client: TestClient, token_env: str) -> None:
    r = client.post("/api/remote/restart", json={})
    assert r.status_code == 401


def test_restart_disabled_without_token_env(
    client: TestClient, no_token_env: None
) -> None:
    r = client.post(
        "/api/remote/restart",
        json={},
        headers={"Authorization": "Bearer whatever"},
    )
    assert r.status_code == 503


# ---------------------------------------------------------------------------
# Platform gate -- on real Linux CI we expect 501 (no patch applied).
# ---------------------------------------------------------------------------


def test_restart_non_windows_returns_501(
    client: TestClient, token_env: str, fake_state: Path
) -> None:
    """Without the win32 patch, the endpoint must 501 on Linux."""
    r = client.post("/api/remote/restart", json={}, headers=auth(token_env))
    assert r.status_code == 501
    assert "Windows-only" in r.json()["detail"]


# ---------------------------------------------------------------------------
# Happy path -- helper spawned with the right args
# ---------------------------------------------------------------------------


def test_restart_spawns_helper_with_defaults(
    client: TestClient,
    token_env: str,
    fake_state: Path,
    stub_popen_and_win32: dict[str, Any],
) -> None:
    r = client.post("/api/remote/restart", json={}, headers=auth(token_env))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["helper"] == "spawned"
    assert body["pull"] is True
    assert body["delay_sec"] == 2
    assert body["pid_to_kill"] > 0
    assert "version" in body["note"].lower() or "/version" in body["note"]

    calls = stub_popen_and_win32["popen_calls"]
    assert len(calls) == 1
    args = calls[0]["args"]
    assert args[0] == "powershell.exe"
    assert "-File" in args
    assert "-UvicornPid" in args
    assert "-RepoRoot" in args
    assert "-VenvPython" in args
    assert "-Token" in args
    assert "-DelaySec" in args
    # Default pull=True -> -NoPull must NOT appear.
    assert "-NoPull" not in args

    # State should record the action.
    assert "Restart helper spawned" in body["state"]["last_action"]


def test_restart_no_pull_flag_propagates(
    client: TestClient,
    token_env: str,
    fake_state: Path,
    stub_popen_and_win32: dict[str, Any],
) -> None:
    r = client.post(
        "/api/remote/restart",
        json={"pull": False},
        headers=auth(token_env),
    )
    assert r.status_code == 200
    args = stub_popen_and_win32["popen_calls"][0]["args"]
    assert "-NoPull" in args
    assert r.json()["pull"] is False


def test_restart_delay_is_bounded(
    client: TestClient,
    token_env: str,
    fake_state: Path,
    stub_popen_and_win32: dict[str, Any],
) -> None:
    # Try to request 999s -- should clamp to 10.
    r = client.post(
        "/api/remote/restart",
        json={"delay_sec": 999},
        headers=auth(token_env),
    )
    assert r.status_code == 200
    assert r.json()["delay_sec"] == 10
    args = stub_popen_and_win32["popen_calls"][0]["args"]
    delay_idx = args.index("-DelaySec")
    assert args[delay_idx + 1] == "10"


def test_restart_delay_below_minimum_clamps_up(
    client: TestClient,
    token_env: str,
    fake_state: Path,
    stub_popen_and_win32: dict[str, Any],
) -> None:
    r = client.post(
        "/api/remote/restart",
        json={"delay_sec": 0},
        headers=auth(token_env),
    )
    assert r.status_code == 200
    assert r.json()["delay_sec"] == 1


def test_restart_detached_flags_present(
    client: TestClient,
    token_env: str,
    fake_state: Path,
    stub_popen_and_win32: dict[str, Any],
) -> None:
    """The Popen kwargs must include creationflags for true detachment."""
    client.post("/api/remote/restart", json={}, headers=auth(token_env))
    kwargs = stub_popen_and_win32["popen_calls"][0]["kwargs"]
    # DETACHED_PROCESS (0x8) | CREATE_NEW_PROCESS_GROUP (0x200) = 0x208 = 520
    assert kwargs.get("creationflags") == 520
    assert kwargs.get("close_fds") is True
    # stdin/stdout/stderr must be DEVNULL so we don't hold pipes to a
    # dying parent.
    import subprocess as sp_mod
    assert kwargs.get("stdin") == sp_mod.DEVNULL
    assert kwargs.get("stdout") == sp_mod.DEVNULL
    assert kwargs.get("stderr") == sp_mod.DEVNULL
