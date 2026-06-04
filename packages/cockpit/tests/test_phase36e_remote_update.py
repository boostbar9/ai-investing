"""Phase 36e — remote self-update endpoint tests.

Covers ``/api/remote/version``, ``/api/remote/update/check``, and
``/api/remote/update/apply``. The updater module is stubbed so no real
git or pip calls leave the test process; ``proc`` is stubbed the same
way the 36c suite does it.
"""

from __future__ import annotations

import types
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from packages.cockpit import proc as cockpit_proc
from packages.cockpit import state as st
from packages.cockpit import updater as cockpit_updater
from packages.cockpit.web import remote as remote_mod
from packages.cockpit.web import server as srv

GOOD_TOKEN = "x" * 32


# ---------------------------------------------------------------------------
# Fixtures (mirror the 36c suite so behavior is consistent across tests)
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
    """Stub proc.start / proc.stop so no real subprocesses spawn."""
    calls: dict[str, Any] = {"start": [], "stop": []}

    def _start(kind: str, cmd: list[str], cwd: Path | None = None):
        calls["start"].append((kind, list(cmd)))
        return types.SimpleNamespace(
            to_dict=lambda: {"running": True, "pid": 12345, "kind": kind}
        )

    def _stop(kind: str):
        calls["stop"].append(kind)
        return types.SimpleNamespace(to_dict=lambda: {"running": False})

    monkeypatch.setattr(cockpit_proc, "start", _start)
    monkeypatch.setattr(cockpit_proc, "stop", _stop)
    return calls


@pytest.fixture
def stub_updater_happy(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Stub updater so check + apply both succeed."""
    calls: dict[str, int] = {"current": 0, "check": 0, "apply": 0}

    def _current_commit() -> dict[str, str]:
        calls["current"] += 1
        return {
            "sha": "deadbeef" * 5,
            "short": "deadbee",
            "subject": "Phase 36e: test commit",
            "date": "2026-06-04 03:35:00 +0000",
            "branch": "main",
        }

    def _check_updates() -> dict[str, Any]:
        calls["check"] += 1
        return {
            "ok": True,
            "current": _current_commit(),
            "behind": 2,
            "commits": [
                {"short": "abc1234", "subject": "fix A", "author": "x", "date": "d"},
                {"short": "def5678", "subject": "fix B", "author": "x", "date": "d"},
            ],
            "upstream": "origin/main",
        }

    def _apply_update() -> dict[str, Any]:
        calls["apply"] += 1
        return {
            "ok": True,
            "log": "$ git pull --ff-only origin HEAD\nAlready up to date.\n$ pip install -e . (rc=0)",
            "current": _current_commit(),
        }

    monkeypatch.setattr(cockpit_updater, "current_commit", _current_commit)
    monkeypatch.setattr(cockpit_updater, "check_updates", _check_updates)
    monkeypatch.setattr(cockpit_updater, "apply_update", _apply_update)
    return calls


@pytest.fixture
def stub_updater_pull_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub updater so apply fails at the pull step."""

    def _current_commit() -> dict[str, str]:
        return {"sha": "a" * 40, "short": "aaaaaaa", "subject": "x", "date": "d", "branch": "main"}

    def _apply_update() -> dict[str, Any]:
        return {
            "ok": False,
            "step": "git pull",
            "log": "fatal: Not possible to fast-forward, aborting.",
        }

    monkeypatch.setattr(cockpit_updater, "current_commit", _current_commit)
    monkeypatch.setattr(cockpit_updater, "apply_update", _apply_update)


@pytest.fixture
def client() -> TestClient:
    return TestClient(srv.app)


def auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# /version
# ---------------------------------------------------------------------------


def test_version_requires_token(client: TestClient, token_env: str) -> None:
    r = client.get("/api/remote/version")
    assert r.status_code == 401


def test_version_returns_current_commit(
    client: TestClient, token_env: str, stub_updater_happy: dict[str, int]
) -> None:
    r = client.get("/api/remote/version", headers=auth(token_env))
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["current"]["short"] == "deadbee"
    assert body["current"]["branch"] == "main"
    assert stub_updater_happy["current"] >= 1


def test_version_disabled_without_token_env(
    client: TestClient, no_token_env: None
) -> None:
    r = client.get("/api/remote/version", headers={"Authorization": "Bearer whatever"})
    assert r.status_code == 503


# ---------------------------------------------------------------------------
# /update/check
# ---------------------------------------------------------------------------


def test_update_check_requires_token(client: TestClient, token_env: str) -> None:
    r = client.get("/api/remote/update/check")
    assert r.status_code == 401


def test_update_check_reports_behind(
    client: TestClient, token_env: str, stub_updater_happy: dict[str, int]
) -> None:
    r = client.get("/api/remote/update/check", headers=auth(token_env))
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["behind"] == 2
    assert len(body["commits"]) == 2
    assert body["commits"][0]["subject"] == "fix A"
    assert stub_updater_happy["check"] == 1


# ---------------------------------------------------------------------------
# /update/apply
# ---------------------------------------------------------------------------


def test_update_apply_requires_token(client: TestClient, token_env: str) -> None:
    r = client.post("/api/remote/update/apply", json={})
    assert r.status_code == 401


def test_update_apply_happy_path_restarts_loop(
    client: TestClient,
    token_env: str,
    fake_state: Path,
    stub_jobs: dict[str, Any],
    stub_updater_happy: dict[str, int],
) -> None:
    """Default behavior: pull, pip, stop old loop, spawn new loop."""
    r = client.post(
        "/api/remote/update/apply",
        json={},  # all defaults: restart_loop=True, dry_run=False, strategy=ensemble
        headers=auth(token_env),
    )
    assert r.status_code == 200, r.text
    body = r.json()

    # Updater ran and reported success.
    assert body["update"]["ok"] is True
    assert "pip install" in body["update"]["log"]
    assert stub_updater_happy["apply"] == 1

    # Loop was stopped then started.
    assert stub_jobs["stop"] == ["paper_loop"]
    assert len(stub_jobs["start"]) == 1
    kind, cmd = stub_jobs["start"][0]
    assert kind == "paper_loop"
    assert "tools/paper_trade.py" in cmd
    assert "--strategy" in cmd and "ensemble" in cmd
    assert "--loop" in cmd
    assert "--dry-run" not in cmd  # default dry_run=False

    # State reflects the new intent.
    assert body["state"]["paper_loop_intended"] is True
    assert body["state"]["paper_loop_strategy"] == "ensemble"
    assert body["state"]["paper_loop_dry_run"] is False
    assert "Loop restarted via remote update" in body["state"]["last_action"]

    # Job info is from the spawn, not skipped.
    assert body["job"].get("running") is True
    assert body["job"].get("pid") == 12345


def test_update_apply_dry_run_flag_propagates(
    client: TestClient,
    token_env: str,
    fake_state: Path,
    stub_jobs: dict[str, Any],
    stub_updater_happy: dict[str, int],
) -> None:
    r = client.post(
        "/api/remote/update/apply",
        json={"restart_loop": True, "dry_run": True, "strategy": "ensemble"},
        headers=auth(token_env),
    )
    assert r.status_code == 200
    _, cmd = stub_jobs["start"][0]
    assert "--dry-run" in cmd
    assert r.json()["state"]["paper_loop_dry_run"] is True


def test_update_apply_skips_restart_when_requested(
    client: TestClient,
    token_env: str,
    fake_state: Path,
    stub_jobs: dict[str, Any],
    stub_updater_happy: dict[str, int],
) -> None:
    r = client.post(
        "/api/remote/update/apply",
        json={"restart_loop": False},
        headers=auth(token_env),
    )
    assert r.status_code == 200
    body = r.json()
    assert body["update"]["ok"] is True
    assert body["job"] == {"skipped": True, "reason": "restart_loop=False"}
    # Neither stop nor start should fire when restart is suppressed.
    assert stub_jobs["stop"] == []
    assert stub_jobs["start"] == []


def test_update_apply_pull_failure_leaves_loop_alone(
    client: TestClient,
    token_env: str,
    fake_state: Path,
    stub_jobs: dict[str, Any],
    stub_updater_pull_fail: None,
) -> None:
    """If git pull fails, we must NOT restart the loop into a half state."""
    r = client.post(
        "/api/remote/update/apply",
        json={"restart_loop": True},
        headers=auth(token_env),
    )
    assert r.status_code == 200
    body = r.json()
    assert body["update"]["ok"] is False
    assert body["update"]["step"] == "git pull"
    assert "fast-forward" in body["update"]["log"]
    # Critical: no restart happened.
    assert body["job"] == {"skipped": True, "reason": "update failed"}
    assert stub_jobs["stop"] == []
    assert stub_jobs["start"] == []


def test_update_apply_updater_exception_is_caught(
    client: TestClient,
    token_env: str,
    fake_state: Path,
    stub_jobs: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If updater.apply_update raises, we return ok=False not a 500."""

    def _boom() -> dict[str, Any]:
        raise RuntimeError("simulated git crash")

    monkeypatch.setattr(cockpit_updater, "apply_update", _boom)
    r = client.post(
        "/api/remote/update/apply",
        json={},
        headers=auth(token_env),
    )
    assert r.status_code == 200
    body = r.json()
    assert body["update"]["ok"] is False
    assert body["update"]["step"] == "exception"
    assert "simulated git crash" in body["update"]["log"]
    assert stub_jobs["start"] == []
