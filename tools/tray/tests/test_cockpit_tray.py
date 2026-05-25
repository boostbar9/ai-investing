"""Tests for the cockpit-tray pure helpers.

The GUI loop (``main``) is excluded -- it requires pystray and a real
display, both of which we don't want CI to depend on. Everything else
(lifecycle, lock, health translation, menu actions) is fair game.
"""

from __future__ import annotations

import os
import socket
import subprocess
import threading
from pathlib import Path

import pytest

from tools.tray import cockpit_tray as tray

# ---------------------------------------------------------------------------
# Single-instance lock
# ---------------------------------------------------------------------------


def test_acquire_lock_first_time(tmp_path: Path) -> None:
    lock = tmp_path / "tray.lock"
    assert tray.acquire_lock(lock) is True
    assert lock.read_text(encoding="utf-8").strip() == str(os.getpid())


def test_acquire_lock_blocks_second_instance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock = tmp_path / "tray.lock"
    # Pretend PID 12345 owns the lock and is alive.
    lock.write_text("12345", encoding="utf-8")
    monkeypatch.setattr(tray, "_pid_alive", lambda pid: pid == 12345)
    assert tray.acquire_lock(lock) is False


def test_acquire_lock_clears_stale_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stale lock (PID no longer exists) must not block startup forever."""
    lock = tmp_path / "tray.lock"
    lock.write_text("99999", encoding="utf-8")
    monkeypatch.setattr(tray, "_pid_alive", lambda _pid: False)
    assert tray.acquire_lock(lock) is True
    assert lock.read_text(encoding="utf-8").strip() == str(os.getpid())


def test_acquire_lock_handles_garbage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A corrupt lock file is treated as stale, not as a fatal error."""
    lock = tmp_path / "tray.lock"
    lock.write_text("not a number", encoding="utf-8")
    monkeypatch.setattr(tray, "_pid_alive", lambda _pid: False)
    assert tray.acquire_lock(lock) is True


def test_release_lock_only_removes_own_lock(tmp_path: Path) -> None:
    lock = tmp_path / "tray.lock"
    lock.write_text("12345", encoding="utf-8")
    tray.release_lock(lock)
    # We didn't write our own PID, so we must not have deleted somebody else's.
    assert lock.exists()


def test_release_lock_removes_self(tmp_path: Path) -> None:
    lock = tmp_path / "tray.lock"
    lock.write_text(str(os.getpid()), encoding="utf-8")
    tray.release_lock(lock)
    assert not lock.exists()


# ---------------------------------------------------------------------------
# Port check
# ---------------------------------------------------------------------------


def test_is_port_open_false_for_unused_port() -> None:
    # Bind a socket, read the assigned port, close it -- that port is almost
    # certainly free by the time we check.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    assert tray.is_port_open(port=port, timeout=0.2) is False


def test_is_port_open_true_for_listening_port() -> None:
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    try:
        port = srv.getsockname()[1]
        assert tray.is_port_open(port=port, timeout=0.5) is True
    finally:
        srv.close()


# ---------------------------------------------------------------------------
# Health -> status mapping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "snap,expected",
    [
        (None, "error"),
        ({"status": "ok"}, "ok"),
        ({"status": "warn"}, "warn"),
        ({"status": "error"}, "error"),
        # 'info' is treated as warn so the operator notices a check that
        # couldn't reach a definite verdict.
        ({"status": "info"}, "warn"),
        # Unknown payload -> warn (better than silently green).
        ({}, "warn"),
        ({"status": "weird"}, "warn"),
    ],
)
def test_status_from_health(snap: dict | None, expected: str) -> None:
    assert tray.status_from_health(snap) == expected


def test_poll_health_returns_none_when_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_a, **_kw):
        raise tray.httpx.HTTPError("connection refused")

    monkeypatch.setattr(tray.httpx, "get", _boom)
    assert tray.poll_health() is None


def test_poll_health_returns_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Resp:
        status_code = 200

        def json(self) -> dict:
            return {"status": "ok", "counts": {"ok": 7}}

    monkeypatch.setattr(tray.httpx, "get", lambda _url, timeout=2.0: _Resp())
    snap = tray.poll_health()
    assert snap is not None
    assert snap["status"] == "ok"
    assert snap["counts"]["ok"] == 7


def test_poll_health_returns_none_on_non_200(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Resp:
        status_code = 500

        def json(self) -> dict:
            return {}

    monkeypatch.setattr(tray.httpx, "get", lambda _url, timeout=2.0: _Resp())
    assert tray.poll_health() is None


# ---------------------------------------------------------------------------
# Cockpit lifecycle (with subprocess stubbed)
# ---------------------------------------------------------------------------


class _FakeProc:
    """Stand-in for subprocess.Popen good enough for lifecycle tests."""

    def __init__(self, pid: int = 4242, alive: bool = True) -> None:
        self.pid = pid
        self._alive = alive
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        return None if self._alive else 0

    def terminate(self) -> None:
        self.terminated = True
        self._alive = False

    def kill(self) -> None:
        self.killed = True
        self._alive = False

    def wait(self, timeout: float | None = None) -> int:
        self._alive = False
        return 0


def test_start_cockpit_spawns_when_port_free(monkeypatch: pytest.MonkeyPatch) -> None:
    spawned: dict[str, object] = {}

    def _fake_popen(cmd, **kwargs):
        spawned["cmd"] = cmd
        spawned["env"] = kwargs.get("env")
        return _FakeProc()

    monkeypatch.setattr(tray, "is_port_open", lambda **_kw: False)
    monkeypatch.setattr(tray.subprocess, "Popen", _fake_popen)
    state = tray.TrayState()
    assert tray.start_cockpit(state) is True
    assert state.proc is not None
    assert state.status == "starting"
    cmd = spawned["cmd"]
    assert isinstance(cmd, list)
    assert "uvicorn" in cmd
    assert "packages.cockpit.web.server:app" in cmd
    assert "--port" in cmd and "8765" in cmd


def test_start_cockpit_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Calling start twice in a row must not spawn two cockpits."""
    state = tray.TrayState(proc=_FakeProc())
    called: list[bool] = []
    monkeypatch.setattr(
        tray.subprocess,
        "Popen",
        lambda *a, **kw: called.append(True) or _FakeProc(),
    )
    assert tray.start_cockpit(state) is True
    assert called == []  # already-running proc -> no spawn


def test_start_cockpit_adopts_external_cockpit(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the port is already in use (user launched manually), don't fight."""
    monkeypatch.setattr(tray, "is_port_open", lambda **_kw: True)
    called: list[bool] = []
    monkeypatch.setattr(
        tray.subprocess,
        "Popen",
        lambda *a, **kw: called.append(True) or _FakeProc(),
    )
    state = tray.TrayState()
    assert tray.start_cockpit(state) is True
    assert state.proc is None  # we didn't spawn -- we adopted
    assert state.status == "starting"
    assert called == []


def test_start_cockpit_surfaces_spawn_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tray, "is_port_open", lambda **_kw: False)

    def _boom(*_a, **_kw):
        raise OSError("denied")

    monkeypatch.setattr(tray.subprocess, "Popen", _boom)
    state = tray.TrayState()
    assert tray.start_cockpit(state) is False
    assert state.status == "error"
    assert state.last_error and "denied" in state.last_error


def test_stop_cockpit_terminates_alive_proc() -> None:
    proc = _FakeProc()
    state = tray.TrayState(proc=proc)
    assert tray.stop_cockpit(state) is True
    assert proc.terminated is True
    assert state.proc is None
    assert state.status == "stopped"


def test_stop_cockpit_kills_after_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """A child that ignores terminate() must be killed, not left running."""

    class _Stubborn(_FakeProc):
        def __init__(self) -> None:
            super().__init__()
            self._wait_calls = 0

        def terminate(self) -> None:
            # Refuse to die on the first request.
            self.terminated = True

        def wait(self, timeout: float | None = None) -> int:
            self._wait_calls += 1
            if self._wait_calls == 1:
                raise subprocess.TimeoutExpired(cmd="x", timeout=timeout or 0)
            self._alive = False
            return 0

    proc = _Stubborn()
    state = tray.TrayState(proc=proc)
    assert tray.stop_cockpit(state, timeout=0.01) is True
    assert proc.killed is True
    assert state.proc is None


def test_stop_cockpit_no_proc_is_noop() -> None:
    state = tray.TrayState()
    assert tray.stop_cockpit(state) is True
    assert state.status == "stopped"


def test_stop_cockpit_handles_already_dead_proc() -> None:
    proc = _FakeProc(alive=False)
    state = tray.TrayState(proc=proc)
    assert tray.stop_cockpit(state) is True
    assert state.proc is None
    assert state.status == "stopped"


def test_restart_cockpit_calls_stop_then_start(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        tray,
        "stop_cockpit",
        lambda s, timeout=5.0: calls.append("stop") or True,
    )
    monkeypatch.setattr(
        tray, "start_cockpit", lambda s: calls.append("start") or True
    )
    monkeypatch.setattr(tray.time, "sleep", lambda _s: None)
    assert tray.restart_cockpit(tray.TrayState()) is True
    assert calls == ["stop", "start"]


# ---------------------------------------------------------------------------
# Poll loop (one iteration, then stop)
# ---------------------------------------------------------------------------


def test_poll_loop_updates_state_and_icon(monkeypatch: pytest.MonkeyPatch) -> None:
    """One full pass through the poll loop drives icon + state from a snapshot."""
    state = tray.TrayState()
    stop = threading.Event()

    monkeypatch.setattr(
        tray,
        "poll_health",
        lambda timeout=2.0: {"status": "warn", "counts": {"warn": 2}},
    )
    # Sleep just long enough to let one iteration run, then signal stop.
    monkeypatch.setattr(tray.time, "sleep", lambda _s: stop.set())

    seen: list[str] = []
    tray._poll_loop(state, seen.append, stop)
    assert state.status == "warn"
    assert state.last_health == {"status": "warn", "counts": {"warn": 2}}
    assert seen == ["warn"]


def test_poll_loop_marks_dead_proc_as_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the spawned cockpit exits, the tray must flip red, not stay green."""
    dead = _FakeProc(alive=False)
    state = tray.TrayState(proc=dead, status="ok")
    stop = threading.Event()

    monkeypatch.setattr(tray, "poll_health", lambda timeout=2.0: None)
    monkeypatch.setattr(tray.time, "sleep", lambda _s: stop.set())

    seen: list[str] = []
    tray._poll_loop(state, seen.append, stop)
    assert state.proc is None
    assert state.status == "error"
    assert state.last_error and "exited" in state.last_error


# ---------------------------------------------------------------------------
# Menu helpers
# ---------------------------------------------------------------------------


def test_open_dashboard_uses_cockpit_url(monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[str] = []
    monkeypatch.setattr(tray.webbrowser, "open", opened.append)
    tray.open_dashboard()
    assert opened == [tray.COCKPIT_URL]


def test_open_health_targets_health_page(monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[str] = []
    monkeypatch.setattr(tray.webbrowser, "open", opened.append)
    tray.open_health()
    assert opened == [f"{tray.COCKPIT_URL}/health"]


def test_icon_tooltip_reflects_status() -> None:
    s = tray.TrayState()
    assert "stopped" in tray._icon_tooltip(s)
    s.status = "starting"
    assert "starting" in tray._icon_tooltip(s)
    s.status = "ok"
    s.last_health = {"counts": {"ok": 5, "warn": 0, "error": 0}}
    tooltip = tray._icon_tooltip(s)
    assert "ok" in tooltip and "5 ok" in tooltip


def test_spawn_env_strips_pythonhome(monkeypatch: pytest.MonkeyPatch) -> None:
    """Defense against STATUS_DLL_INIT_FAILED on Windows."""
    monkeypatch.setenv("PYTHONHOME", "/wrong/path")
    env = tray._spawn_env()
    assert "PYTHONHOME" not in env
    assert env.get("PYTHONUNBUFFERED") == "1"
