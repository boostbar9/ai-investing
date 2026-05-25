"""System-tray launcher for the cockpit daily-driver.

Unlike :mod:`tools.tray.launcher` (which drives the Docker stack), this
tray app owns the **native cockpit process** -- uvicorn running on
http://127.0.0.1:8765 -- and is what Devin clicks to start his day.

Goals (from the daily-driver scope):

* One double-click (or boot autostart) brings the cockpit up.
* The tray icon's colour mirrors ``/api/health/full`` -- green when
  every diagnostic is OK, yellow on a warning, red on an error.
* Right-click menu exposes Start / Stop / Restart / Open dashboard /
  Open Health / View logs / Quit. No PowerShell required.
* Single-instance: a second invocation refuses to start so the operator
  never accidentally ends up with two cockpits fighting for port 8765.
* All process spawning reuses the same env sanitisation and
  CREATE_BREAKAWAY_FROM_JOB pattern that powers
  :mod:`packages.cockpit.proc`, so the tray-launched cockpit doesn't
  inherit the STATUS_DLL_INIT_FAILED footgun we fixed earlier.

Run (after ``pip install -e \".[tray]\"``)::

    python -m tools.tray.cockpit_tray

This module is import-safe: pystray and Pillow are imported lazily
inside :func:`main` so the pure helpers below remain testable without
the GUI deps installed.
"""

from __future__ import annotations

import logging
import os
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import httpx

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
COCKPIT_HOST = "127.0.0.1"
COCKPIT_PORT = 8765
COCKPIT_URL = f"http://{COCKPIT_HOST}:{COCKPIT_PORT}"
HEALTH_FULL_URL = f"{COCKPIT_URL}/api/health/full"
LOG_DIR = REPO_ROOT / "data" / "cockpit"
TRAY_LOG = LOG_DIR / "tray.log"
COCKPIT_LOG = LOG_DIR / "cockpit_tray.log"
LOCK_FILE = LOG_DIR / "tray.lock"

# How often the background poller asks the cockpit for its health status.
POLL_INTERVAL_S = 10.0

# How long we wait for the cockpit to start answering after we spawn it.
STARTUP_TIMEOUT_S = 30.0

log = logging.getLogger("cockpit_tray")


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclass
class TrayState:
    """Mutable state shared between the GUI thread and the poller.

    Kept as a plain dataclass (not a singleton/global) so unit tests can
    instantiate one per test and the poller helpers can be exercised
    without a running GUI loop.
    """

    proc: subprocess.Popen | None = None
    status: str = "stopped"  # one of: stopped, starting, ok, warn, error
    last_health: dict | None = None
    last_error: str | None = None


# ---------------------------------------------------------------------------
# Single-instance lock
# ---------------------------------------------------------------------------


def acquire_lock(lock_path: Path = LOCK_FILE) -> bool:
    """Best-effort single-instance guard.

    Writes the current PID to ``lock_path``. If the file already exists
    and the PID inside is still alive, refuses to start. Stale locks
    (process gone) are cleaned up automatically so a crashed tray
    doesn't permanently block restarts.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if lock_path.exists():
        try:
            existing = int(lock_path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            existing = -1
        if existing > 0 and _pid_alive(existing):
            return False
        # Stale lock -- safe to overwrite.
    lock_path.write_text(str(os.getpid()), encoding="utf-8")
    return True


def release_lock(lock_path: Path = LOCK_FILE) -> None:
    try:
        if lock_path.exists():
            content = lock_path.read_text(encoding="utf-8").strip()
            if content == str(os.getpid()):
                lock_path.unlink()
    except OSError:
        pass  # nothing we can do; tray is exiting anyway


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        # tasklist returns 0 even if PID not found; check stdout
        try:
            out = subprocess.check_output(
                ["tasklist", "/FI", f"PID eq {pid}"],
                stderr=subprocess.DEVNULL,
                timeout=3,
            ).decode("utf-8", errors="replace")
            return f" {pid} " in out or f"{pid}\t" in out
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            return False
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Cockpit lifecycle
# ---------------------------------------------------------------------------


def is_port_open(host: str = COCKPIT_HOST, port: int = COCKPIT_PORT, timeout: float = 0.5) -> bool:
    """Return True iff *something* is listening on the cockpit's port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        try:
            s.connect((host, port))
            return True
        except OSError:
            return False


def _spawn_env() -> dict[str, str]:
    """Build the child env for spawning the cockpit.

    We piggy-back on :func:`packages.cockpit.proc._child_env` if it's
    importable -- that's the single source of truth for the env
    sanitisation rules that prevent STATUS_DLL_INIT_FAILED on Windows.
    If the cockpit package isn't importable for some reason (e.g. the
    user broke their checkout), we fall back to a minimal-but-safe env
    so the tray can at least try to start.
    """
    try:
        from packages.cockpit.proc import _child_env  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - defensive fallback
        env = os.environ.copy()
        env.pop("PYTHONHOME", None)
        env.setdefault("PYTHONPATH", ".")
        env["PYTHONUNBUFFERED"] = "1"
        return env
    return _child_env()


def _popen_kwargs() -> dict:
    """OS-specific Popen kwargs that detach the cockpit from the tray.

    On Windows we set CREATE_NEW_PROCESS_GROUP + DETACHED_PROCESS so a
    Ctrl+C in the tray doesn't propagate, and CREATE_BREAKAWAY_FROM_JOB
    so the cockpit isn't killed if Task Scheduler tears down the tray's
    job object. On POSIX we use ``start_new_session=True`` for the same
    detachment effect.
    """
    if os.name == "nt":
        flags = (
            getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            | getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
            | getattr(subprocess, "CREATE_BREAKAWAY_FROM_JOB", 0x01000000)
        )
        return {"creationflags": flags, "close_fds": False}
    return {"start_new_session": True, "close_fds": True}


def start_cockpit(state: TrayState) -> bool:
    """Spawn ``uvicorn packages.cockpit.web.server:app`` in the background.

    Idempotent: if a cockpit is already alive on the port, we adopt it
    (the tray will reflect its health) rather than fail loudly. This
    matches the "I just want it working" promise of the daily driver.
    """
    if state.proc and state.proc.poll() is None:
        log.info("start_cockpit: already running (pid=%s)", state.proc.pid)
        return True
    if is_port_open():
        log.info("start_cockpit: port %d already in use; adopting", COCKPIT_PORT)
        state.status = "starting"  # health poll will upgrade us
        return True

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    state.status = "starting"

    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "packages.cockpit.web.server:app",
        "--host",
        COCKPIT_HOST,
        "--port",
        str(COCKPIT_PORT),
        "--log-level",
        "warning",
    ]

    log_file = COCKPIT_LOG.open("ab")
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            env=_spawn_env(),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            **_popen_kwargs(),
        )
    except OSError as exc:
        log.error("start_cockpit: spawn failed: %s", exc)
        state.status = "error"
        state.last_error = f"spawn failed: {exc}"
        log_file.close()
        return False

    state.proc = proc
    log.info("start_cockpit: pid=%s", proc.pid)
    return True


def stop_cockpit(state: TrayState, timeout: float = 5.0) -> bool:
    """Politely terminate the cockpit.

    Tries SIGTERM (or terminate() on Windows) first, falls back to kill
    after ``timeout`` seconds. Either way the tray state is reset so the
    next Start cycle works from a clean slate.
    """
    proc = state.proc
    if proc is None:
        # Nothing we spawned -- but somebody else might be on the port.
        # Don't try to kill them; surface it to the operator instead.
        state.status = "stopped"
        return True
    if proc.poll() is not None:
        state.proc = None
        state.status = "stopped"
        return True
    try:
        proc.terminate()
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            log.warning("stop_cockpit: terminate timed out, killing pid=%s", proc.pid)
            proc.kill()
            proc.wait(timeout=timeout)
    except OSError as exc:
        log.error("stop_cockpit: %s", exc)
        state.last_error = f"stop failed: {exc}"
        return False
    state.proc = None
    state.status = "stopped"
    return True


def restart_cockpit(state: TrayState) -> bool:
    stop_cockpit(state)
    # Give the OS a moment to release the port before re-binding.
    time.sleep(0.5)
    return start_cockpit(state)


# ---------------------------------------------------------------------------
# Health polling
# ---------------------------------------------------------------------------


def poll_health(timeout: float = 2.0) -> dict | None:
    """One-shot health probe. Returns the JSON dict or None on failure.

    Pulled out so tests can call it directly without spinning a thread.
    """
    try:
        r = httpx.get(HEALTH_FULL_URL, timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except (httpx.HTTPError, OSError):
        return None
    return None


def status_from_health(snap: dict | None) -> str:
    """Translate a health snapshot into a tray icon colour key.

    The cockpit may legitimately be ``starting`` while uvicorn boots;
    callers preserve that state by passing ``snap=None`` only when they
    haven't tried yet, and reading the returned status only after they
    do.
    """
    if snap is None:
        return "error"  # unreachable -> red
    overall = snap.get("status")
    if overall in {"ok", "warn", "error", "info"}:
        # 'info' is treated as warn for the icon (yellow) because if every
        # check is purely informational the operator probably wants to look.
        return "warn" if overall == "info" else overall  # type: ignore[return-value]
    return "warn"


def _poll_loop(state: TrayState, set_icon: Callable[[str], None], stop_event: threading.Event) -> None:
    """Background poller that keeps the tray icon in sync with /health.

    Doesn't own the cockpit lifecycle -- that's start/stop's job. It
    just reads and reports. Sleeps in small chunks so a Quit from the
    GUI thread is responsive.
    """
    while not stop_event.is_set():
        # If we spawned a cockpit that's now dead, reflect that.
        if state.proc is not None and state.proc.poll() is not None:
            state.proc = None
            state.status = "error"
            state.last_error = "cockpit exited unexpectedly (see cockpit_tray.log)"
        snap = poll_health()
        if snap is not None:
            state.last_health = snap
            state.status = status_from_health(snap)
        elif state.status not in {"stopped", "starting"}:
            state.status = "error"
        set_icon(state.status)
        # Wake up promptly on quit so we don't make the user wait the
        # full POLL_INTERVAL_S to close the app.
        for _ in range(int(POLL_INTERVAL_S * 10)):
            if stop_event.is_set():
                return
            time.sleep(0.1)


# ---------------------------------------------------------------------------
# Menu actions (UI-free helpers so they can be tested)
# ---------------------------------------------------------------------------


def open_dashboard() -> None:
    webbrowser.open(COCKPIT_URL)


def open_health() -> None:
    webbrowser.open(f"{COCKPIT_URL}/health")


def open_log_dir() -> None:
    """Open the cockpit log directory in the OS file manager."""
    path = str(LOG_DIR)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        os.startfile(path)  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


# ---------------------------------------------------------------------------
# GUI loop
# ---------------------------------------------------------------------------


_ICON_COLOURS: dict[str, tuple[int, int, int]] = {
    "ok": (0x4C, 0xAF, 0x50),       # green
    "warn": (0xFF, 0xC1, 0x07),     # amber
    "error": (0xF4, 0x43, 0x36),    # red
    "starting": (0x21, 0x96, 0xF3), # blue
    "stopped": (0x9E, 0x9E, 0x9E),  # grey
}


def _make_icon_image(status: str):  # pragma: no cover - GUI helper
    from PIL import Image, ImageDraw  # lazy import

    rgb = _ICON_COLOURS.get(status, _ICON_COLOURS["warn"])
    img = Image.new("RGB", (64, 64), color=(30, 30, 30))
    draw = ImageDraw.Draw(img)
    draw.ellipse((8, 8, 56, 56), fill=rgb)
    return img


def main() -> int:  # pragma: no cover - GUI loop
    logging.basicConfig(
        filename=str(TRAY_LOG),
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    log.info("cockpit_tray starting (repo=%s)", REPO_ROOT)

    if not acquire_lock():
        log.warning("another cockpit tray is already running; exiting")
        return 1

    try:
        import pystray
    except ImportError:
        sys.stderr.write(
            "pystray is not installed. Run:  pip install -e \".[tray]\"\n"
        )
        release_lock()
        return 2

    state = TrayState()
    stop_event = threading.Event()
    icon_holder: dict[str, object] = {}

    def _set_icon(status: str) -> None:
        icon = icon_holder.get("icon")
        if icon is None:
            return
        icon.icon = _make_icon_image(status)  # type: ignore[attr-defined]
        icon.title = _icon_tooltip(state)     # type: ignore[attr-defined]

    def _on_start(icon, item) -> None:
        start_cockpit(state)

    def _on_stop(icon, item) -> None:
        stop_cockpit(state)

    def _on_restart(icon, item) -> None:
        restart_cockpit(state)

    def _on_open(icon, item) -> None:
        open_dashboard()

    def _on_health(icon, item) -> None:
        open_health()

    def _on_logs(icon, item) -> None:
        open_log_dir()

    def _on_quit(icon, item) -> None:
        log.info("quit requested")
        stop_event.set()
        stop_cockpit(state)
        icon.stop()

    menu = pystray.Menu(
        pystray.MenuItem("Open dashboard", _on_open, default=True),
        pystray.MenuItem("Open Health page", _on_health),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Start cockpit", _on_start),
        pystray.MenuItem("Stop cockpit", _on_stop),
        pystray.MenuItem("Restart cockpit", _on_restart),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Open logs folder", _on_logs),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Quit", _on_quit),
    )
    icon = pystray.Icon(
        "ai_investing_cockpit",
        _make_icon_image("starting"),
        "AI Investing Cockpit (starting...)",
        menu,
    )
    icon_holder["icon"] = icon

    # Spawn the cockpit then start the poller -- the poller will repaint
    # the icon to green/yellow/red as soon as /api/health/full answers.
    start_cockpit(state)
    threading.Thread(
        target=_poll_loop, args=(state, _set_icon, stop_event), daemon=True
    ).start()

    try:
        icon.run()
    finally:
        stop_event.set()
        release_lock()
    return 0


def _icon_tooltip(state: TrayState) -> str:
    if state.status == "stopped":
        return "AI Investing Cockpit (stopped)"
    if state.status == "starting":
        return "AI Investing Cockpit (starting...)"
    if state.last_health is None:
        return f"AI Investing Cockpit ({state.status})"
    counts = state.last_health.get("counts", {})
    return (
        f"AI Investing Cockpit ({state.status}) -- "
        f"{counts.get('ok', 0)} ok, {counts.get('warn', 0)} warn, "
        f"{counts.get('error', 0)} err"
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
