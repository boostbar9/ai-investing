"""Managed-process registry for the cockpit.

The cockpit needs to start, watch, and stop long-running subprocesses
(``paper_trade``, ``pretrain``, ``retune``, ``git pull`` + reinstall, etc.)
without blocking HTTP requests or losing logs across page reloads.

This module gives each *job kind* a single slot. Starting a new job of the
same kind while one is running returns the existing PID (idempotent). Logs
are buffered to disk under ``data/cockpit/logs/<kind>.log`` so the UI can
tail them via Server-Sent Events.

Design constraints:

* Pure stdlib (no async deps) so it works inside FastAPI's threadpool.
* Stop is best-effort: ``terminate`` first, ``kill`` after a short grace.
* Survives cockpit restarts only weakly: PIDs are re-read at import time
  but we don't try to reattach to the child's stdout if we lost it.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
LOG_DIR: Final[Path] = REPO_ROOT / "data" / "cockpit" / "logs"
STATE_FILE: Final[Path] = REPO_ROOT / "data" / "cockpit" / "procs.json"

LOG_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class JobInfo:
    """Snapshot of a managed job."""

    kind: str
    pid: int | None = None
    started_at: str | None = None
    finished_at: str | None = None
    exit_code: int | None = None
    command: list[str] = field(default_factory=list)
    log_file: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "pid": self.pid,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "exit_code": self.exit_code,
            "command": self.command,
            "log_file": self.log_file,
            "running": self.is_running(),
        }

    def is_running(self) -> bool:
        if self.pid is None:
            return False
        try:
            # Signal 0 just checks for existence on POSIX; Windows raises if dead.
            os.kill(self.pid, 0)
            return True
        except OSError:
            return False


_lock = threading.Lock()
_jobs: dict[str, JobInfo] = {}
_procs: dict[str, subprocess.Popen] = {}


def _persist() -> None:
    with contextlib.suppress(OSError):
        STATE_FILE.write_text(
            json.dumps({k: v.to_dict() for k, v in _jobs.items()}, indent=2),
            encoding="utf-8",
        )


def _load() -> None:
    if not STATE_FILE.exists():
        return
    try:
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    for kind, d in data.items():
        info = JobInfo(
            kind=kind,
            pid=d.get("pid"),
            started_at=d.get("started_at"),
            finished_at=d.get("finished_at"),
            exit_code=d.get("exit_code"),
            command=list(d.get("command") or []),
            log_file=d.get("log_file", ""),
        )
        # We can't reattach to the live process's stdout after a cockpit restart,
        # but we can keep the PID + log file so the UI shows accurate state.
        _jobs[kind] = info


_load()


# Directories Windows needs in PATH so a freshly-spawned Python can load its
# own DLLs (VC++ runtime, ucrtbase, etc.). We add these defensively because
# some shells strip System32 from PATH or replace it with a sanitized one,
# which causes child Python processes to die instantly with STATUS_DLL_INIT_FAILED
# (Windows exit code 3221225794) before they can print a traceback.
_WINDOWS_SYSTEM_PATHS = (
    r"C:\Windows\System32",
    r"C:\Windows",
    r"C:\Windows\System32\Wbem",
    r"C:\Windows\System32\WindowsPowerShell\v1.0",
)

# Environment variables that, if leaked from the parent shell, will break
# child Python interpreters by pointing them at the wrong install.
_PYTHON_ENV_LANDMINES = ("PYTHONHOME",)


def _child_env() -> dict[str, str]:
    """Build a sanitized environment for spawning Python child jobs.

    Three jobs:

    * Inherit the cockpit's env (so user-configured API keys flow through).
    * Strip variables that are known to break child Python processes when
      they leak from an enclosing shell (currently ``PYTHONHOME`` — if it
      points at a different install, the child crashes at startup with
      ``STATUS_DLL_INIT_FAILED``).
    * On Windows, ensure ``PATH`` contains the system directories that
      Python needs to find its runtime DLLs. We append rather than prepend
      so user-customised PATH still wins for everything else.
    """
    env = os.environ.copy()
    for var in _PYTHON_ENV_LANDMINES:
        env.pop(var, None)
    env.setdefault("PYTHONPATH", ".")
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    if os.name == "nt":
        path = env.get("PATH", "")
        parts = [p for p in path.split(os.pathsep) if p]
        existing = {p.lower() for p in parts}
        for sys_dir in _WINDOWS_SYSTEM_PATHS:
            if sys_dir.lower() not in existing:
                parts.append(sys_dir)
        env["PATH"] = os.pathsep.join(parts)
    return env


def status(kind: str) -> JobInfo:
    with _lock:
        return _jobs.get(kind, JobInfo(kind=kind))


def all_status() -> list[dict[str, object]]:
    with _lock:
        return [info.to_dict() for info in _jobs.values()]


def start(kind: str, command: list[str], cwd: Path | None = None) -> JobInfo:
    """Start a job. If one of this kind is already running, return it as-is."""
    with _lock:
        existing = _jobs.get(kind)
        if existing and existing.is_running():
            return existing

        log_path = LOG_DIR / f"{kind}.log"
        # Truncate prior log so the UI tail starts clean.
        log_path.write_text("", encoding="utf-8")

        env = _child_env()

        # Open the log file twice: one handle for the subprocess, one for tailers.
        f_out = log_path.open("a", encoding="utf-8", buffering=1)

        try:
            proc = subprocess.Popen(
                command,
                cwd=str(cwd or REPO_ROOT),
                env=env,
                stdout=f_out,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
        except Exception as e:
            f_out.close()
            raise RuntimeError(f"failed to launch {kind}: {e}") from e

        info = JobInfo(
            kind=kind,
            pid=proc.pid,
            started_at=datetime.now(UTC).isoformat(timespec="seconds"),
            finished_at=None,
            exit_code=None,
            command=command,
            log_file=str(log_path),
        )
        _jobs[kind] = info
        _procs[kind] = proc
        _persist()

        # Spawn a watcher so we capture exit_code without polling from the UI.
        threading.Thread(target=_watch, args=(kind,), daemon=True).start()
        return info


# Windows NT status codes we care about. The list is intentionally short — we
# only translate codes the operator is likely to actually see when a helper
# script crashes on startup (DLL load fail, AV/permission, Ctrl+C, stack
# corruption). Everything else falls through to the raw integer so we never
# hide an unknown failure.
_WINDOWS_NT_STATUS: dict[int, tuple[str, str]] = {
    3221225477: (
        "STATUS_ACCESS_VIOLATION",
        "the process tried to read or write protected memory — usually a native crash or a corrupted install",
    ),
    3221225725: (
        "STATUS_STACK_OVERFLOW",
        "the process ran out of stack space — usually unbounded recursion in a helper script",
    ),
    3221225786: (
        "STATUS_CONTROL_C_EXIT",
        "the process was interrupted with Ctrl+C from the console",
    ),
    3221225794: (
        "STATUS_DLL_INIT_FAILED",
        "a required DLL failed to initialize — usually a missing VC++ runtime, a broken Ollama install, "
        "or antivirus quarantine of a binary; try reinstalling Ollama or whitelisting it in your AV",
    ),
    3221226505: (
        "STATUS_STACK_BUFFER_OVERRUN",
        "the process tripped Windows stack-protection — usually a native bug or a corrupted binary",
    ),
    3221225547: (
        "STATUS_DLL_NOT_FOUND",
        "a required DLL is missing — usually a missing VC++ runtime or a broken install",
    ),
}


def exit_hint(rc: int | None) -> dict[str, str]:
    """Return a hint dict for ``rc`` suitable for an error log ``context``.

    Empty when we have no special interpretation, which lets callers merge
    it into other context without conditional logic.
    """
    if rc is None or rc not in _WINDOWS_NT_STATUS:
        return {}
    name, explanation = _WINDOWS_NT_STATUS[rc]
    return {"exit_status_name": name, "exit_status_hint": explanation}


def describe_exit(kind: str, rc: int | None) -> str:
    """Friendly one-line description of a job exit code.

    Falls back to the bare ``exited with code N`` message if we don't know
    the code, so operators always see *something* useful.
    """
    if rc in _WINDOWS_NT_STATUS:
        name, explanation = _WINDOWS_NT_STATUS[rc]
        return f"{kind} exited with code {rc} ({name}: {explanation})"
    return f"{kind} exited with code {rc}"


def _watch(kind: str) -> None:
    proc = _procs.get(kind)
    if proc is None:
        return
    try:
        rc = proc.wait()
    except Exception:
        rc = -1
    with _lock:
        info = _jobs.get(kind)
        if info is not None:
            info.exit_code = rc
            info.finished_at = datetime.now(UTC).isoformat(timespec="seconds")
        _persist()
    # Surface failures to the cockpit error log. Import lazily to avoid a
    # circular import at module load (errors module is tiny but independent).
    if rc not in (0, None):
        try:
            from packages.cockpit import errors as err_log

            log_file = str(LOG_DIR / f"{kind}.log")
            tail = ""
            with contextlib.suppress(OSError):
                tail = (LOG_DIR / f"{kind}.log").read_text(encoding="utf-8", errors="replace")[-2000:]
            err_log.record_error(
                source=f"job.{kind}",
                message=describe_exit(kind, rc),
                severity="error",
                detail=tail or None,
                context={"exit_code": rc, "log_file": log_file, **exit_hint(rc)},
            )
        except Exception:
            pass


def stop(kind: str, timeout: float = 5.0) -> JobInfo:
    """Politely terminate a job; escalate to kill if it doesn't exit."""
    with _lock:
        info = _jobs.get(kind)
        proc = _procs.get(kind)

    if info is None or not info.is_running() or proc is None:
        return info or JobInfo(kind=kind)

    try:
        if sys.platform.startswith("win"):
            proc.terminate()
        else:
            proc.send_signal(signal.SIGTERM)
    except OSError:
        pass

    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            break
        time.sleep(0.1)
    else:
        with contextlib.suppress(OSError):
            proc.kill()

    with _lock:
        info = _jobs.get(kind, JobInfo(kind=kind))
        info.finished_at = datetime.now(UTC).isoformat(timespec="seconds")
        info.exit_code = proc.returncode if proc.returncode is not None else -15
        _persist()
        return info


def tail_log(kind: str, max_bytes: int = 64 * 1024) -> str:
    """Return up to ``max_bytes`` of the tail of the log file."""
    log_path = LOG_DIR / f"{kind}.log"
    if not log_path.exists():
        return ""
    try:
        size = log_path.stat().st_size
        with log_path.open("rb") as f:
            if size > max_bytes:
                f.seek(size - max_bytes)
            data = f.read()
        return data.decode("utf-8", errors="replace")
    except OSError:
        return ""


def log_path(kind: str) -> Path:
    return LOG_DIR / f"{kind}.log"
