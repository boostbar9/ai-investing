"""Self-diagnosis and auto-heal for the cockpit.

This module is the single source of truth for *"is my local cockpit
healthy?"*. It powers two surfaces:

* the ``/health`` UI page (one row per check, traffic-light icon, one-click
  fix-it button for the green-path fixes)
* the system-tray launcher (``tools/tray/launcher.py``) which polls this
  module to set its icon colour

Each check is a tiny pure function returning a :class:`Check` dataclass.
That makes the layer trivially testable: no monkeypatching of HTTP frameworks,
no FastAPI startup, just call the function and assert on the result.

Design rules:

* **Never block**. Every check has a short timeout. The Health page must
  render in well under a second even when half the world is on fire.
* **Auto-heal only the boring stuff**. We kill our own orphan processes
  and restart Ollama if it crashed. We never touch the user's ``.env``,
  never run ``git pull``, never download models without asking.
* **Surface a plain-English fix command** for every red row. The operator
  can copy-paste it into PowerShell -- no Python knowledge required.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
from contextlib import closing
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

import httpx

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "cockpit"
PAPER_LOG = DATA_DIR / "paper_loop.jsonl"
PRETRAIN_STATE = DATA_DIR / "pretrain_state.json"
ENV_FILE = REPO_ROOT / ".env"

Status = Literal["ok", "warn", "error", "info"]


@dataclass
class Check:
    """A single health check result.

    Attributes
    ----------
    name:
        Stable identifier (``snake_case``) used as the route fragment for
        ``POST /api/health/fix/{name}``.
    title:
        Short label shown in the UI ("Ollama running").
    status:
        ``ok`` (green), ``warn`` (yellow, usually auto-fixable),
        ``error`` (red, requires operator action), or ``info`` (grey,
        purely informational).
    message:
        One-sentence description of *why* this status. Human-readable.
    fix_command:
        A copy-pasteable shell command the operator can run to fix this
        if ``auto_fixable`` is False. ``None`` when no manual fix exists.
    auto_fixable:
        True when the cockpit can fix this itself via
        :func:`auto_heal`. The UI shows a "Fix it" button in this case.
    detail:
        Optional extra context (PIDs killed, paths, etc.). Surfaced as a
        hover/expandable section in the UI.
    """

    name: str
    title: str
    status: Status
    message: str
    fix_command: str | None = None
    auto_fixable: bool = False
    detail: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def check_venv() -> Check:
    """Is the Python virtual environment present and usable?"""
    venv_python = REPO_ROOT / ".venv" / ("Scripts" if os.name == "nt" else "bin") / (
        "python.exe" if os.name == "nt" else "python"
    )
    if not venv_python.exists():
        return Check(
            name="venv",
            title="Python virtualenv",
            status="error",
            message="The .venv directory is missing. The cockpit cannot run without it.",
            fix_command=r"py -3.12 -m venv .venv ; .\.venv\Scripts\Activate.ps1 ; pip install -e .",
        )
    return Check(
        name="venv",
        title="Python virtualenv",
        status="ok",
        message=f"Using {venv_python}",
        detail={"path": str(venv_python)},
    )


def check_env_file() -> Check:
    """Does .env exist and have the bare-minimum Alpaca paper keys?"""
    if not ENV_FILE.exists():
        return Check(
            name="env_file",
            title="Configuration (.env)",
            status="error",
            message=".env is missing. Copy .env.example to .env and fill in your keys.",
            fix_command="Copy-Item .env.example .env",
        )
    text = ENV_FILE.read_text(encoding="utf-8", errors="replace")
    has_key = "ALPACA_PAPER_KEY_ID=" in text and not _looks_blank(text, "ALPACA_PAPER_KEY_ID")
    has_secret = "ALPACA_PAPER_SECRET=" in text and not _looks_blank(text, "ALPACA_PAPER_SECRET")
    if not (has_key and has_secret):
        return Check(
            name="env_file",
            title="Configuration (.env)",
            status="warn",
            message=(
                "Alpaca paper keys not set. The cockpit can still display data, but no "
                "paper trades will execute until you add ALPACA_PAPER_KEY_ID and "
                "ALPACA_PAPER_SECRET to .env (free at app.alpaca.markets)."
            ),
            fix_command="notepad .env",
        )
    return Check(
        name="env_file",
        title="Configuration (.env)",
        status="ok",
        message="Alpaca paper keys present.",
    )


def _looks_blank(env_text: str, key: str) -> bool:
    """Return True when ``KEY=`` appears in the env file with no value."""
    for line in env_text.splitlines():
        line = line.strip()
        if line.startswith(f"{key}="):
            _, _, value = line.partition("=")
            return value.strip() in {"", "change_me", "your_key_here", "..."}
    return True


def check_port(port: int = 8765) -> Check:
    """Is the cockpit port either free, or held by *our own* cockpit?"""
    if _our_cockpit_pid_on_port(port):
        return Check(
            name="port_8765_clear",
            title=f"Port {port}",
            status="ok",
            message=f"Cockpit is bound to port {port}.",
        )
    holder_pid = _foreign_pid_on_port(port)
    if holder_pid is None:
        return Check(
            name="port_8765_clear",
            title=f"Port {port}",
            status="ok",
            message=f"Port {port} is free.",
        )
    return Check(
        name="port_8765_clear",
        title=f"Port {port}",
        status="warn",
        message=(
            f"Port {port} is held by another process (PID {holder_pid}). "
            "Click 'Fix it' to free the port, or stop the other app manually."
        ),
        auto_fixable=True,
        detail={"holder_pid": holder_pid},
    )


def check_orphan_pythons() -> Check:
    """Stray python.exe processes running our code without a live parent.

    These accumulate when the operator closes a PowerShell window while the
    cockpit is running -- the cockpit gets orphaned and keeps holding the
    port, but spawning new children from it fails on Windows with
    STATUS_DLL_INIT_FAILED. Listing them up front means the user knows.
    """
    orphans = _list_orphan_repo_pythons()
    if not orphans:
        return Check(
            name="orphan_pythons",
            title="Orphan processes",
            status="ok",
            message="No stranded Python processes from the repo.",
        )
    pids = ", ".join(str(p["pid"]) for p in orphans)
    return Check(
        name="orphan_pythons",
        title="Orphan processes",
        status="warn",
        message=(
            f"{len(orphans)} stranded python.exe process(es) from this repo "
            f"(PID {pids}). They can block startup -- click 'Fix it' to "
            "terminate them."
        ),
        auto_fixable=True,
        detail={"orphans": orphans},
    )


def check_ollama_installed() -> Check:
    """Is the Ollama CLI on PATH?"""
    if shutil.which("ollama"):
        return Check(
            name="ollama_installed",
            title="Ollama installed",
            status="ok",
            message="ollama CLI is on PATH.",
        )
    return Check(
        name="ollama_installed",
        title="Ollama installed",
        status="error",
        message=(
            "Ollama is not installed. The agents (research, strategy, risk, "
            "discovery) need it to run. Install from ollama.com/download."
        ),
        fix_command="Start-Process https://ollama.com/download/windows",
    )


def check_ollama_running(timeout: float = 1.5) -> Check:
    """Does ``localhost:11434`` answer an HTTP probe?"""
    if not shutil.which("ollama"):
        return Check(
            name="ollama_running",
            title="Ollama running",
            status="info",
            message="Skipping -- Ollama is not installed.",
        )
    try:
        r = httpx.get("http://127.0.0.1:11434/api/tags", timeout=timeout)
        if r.status_code == 200:
            models = [m.get("name", "?") for m in r.json().get("models", [])]
            return Check(
                name="ollama_running",
                title="Ollama running",
                status="ok",
                message=(
                    f"Ollama responding ({len(models)} model(s) cached)."
                    if models
                    else "Ollama responding (no models pulled yet)."
                ),
                detail={"models": models},
            )
    except (httpx.HTTPError, OSError):
        pass
    return Check(
        name="ollama_running",
        title="Ollama running",
        status="warn",
        message=(
            "Ollama is installed but the daemon is not responding on port 11434. "
            "Click 'Fix it' to launch it in the background."
        ),
        auto_fixable=True,
    )


def check_models_pulled() -> Check:
    """Are the active hardware profile's models actually pulled?

    Symptom this catches: Ollama responds 200 on ``/api/tags`` (so
    :func:`check_ollama_running` is green) but every ``/api/generate``
    call returns 404 because the agent chain asks for a model that has
    never been pulled. The log fills with hundreds of ``404 Not Found``
    lines and the operator has no idea why the agents are silently
    producing nothing.

    The fix is to kick off ``tools/check_ollama.py --auto`` which pulls
    every model the active profile needs. We reuse the existing
    cockpit job so progress streams to the Ollama panel for free.
    """
    if not shutil.which("ollama"):
        return Check(
            name="models_pulled",
            title="Agent models pulled",
            status="info",
            message="Skipping -- Ollama is not installed.",
        )
    try:
        from tools.check_ollama import status_snapshot
    except ImportError as exc:  # pragma: no cover - defensive
        return Check(
            name="models_pulled",
            title="Agent models pulled",
            status="warn",
            message=f"Could not load model inventory: {exc!r}",
        )
    try:
        snap = status_snapshot()
    except Exception as exc:  # pragma: no cover - defensive
        return Check(
            name="models_pulled",
            title="Agent models pulled",
            status="warn",
            message=f"Model inventory failed: {exc!r}",
        )
    if not snap.get("daemon_alive", False):
        return Check(
            name="models_pulled",
            title="Agent models pulled",
            status="info",
            message="Skipping -- Ollama daemon is not running yet.",
        )
    missing = list(snap.get("missing", []))
    required = list(snap.get("required", []))
    installed = list(snap.get("installed", []))
    profile_name = (snap.get("profile") or {}).get("name", "?")
    if not missing:
        return Check(
            name="models_pulled",
            title="Agent models pulled",
            status="ok",
            message=(
                f"All {len(required)} model(s) for profile '{profile_name}' "
                "are pulled."
            ),
            detail={"profile": profile_name, "installed": installed},
        )
    sample = ", ".join(missing[:3])
    more = f" (+{len(missing) - 3} more)" if len(missing) > 3 else ""
    return Check(
        name="models_pulled",
        title="Agent models pulled",
        status="error",
        message=(
            f"{len(missing)} agent model(s) are not pulled for profile "
            f"'{profile_name}': {sample}{more}. Agents will fail with 404 "
            "until these are pulled. Click 'Fix it' to pull them now."
        ),
        auto_fixable=True,
        detail={
            "profile": profile_name,
            "missing": missing,
            "required": required,
            "installed": installed,
        },
    )


def check_last_pretrain() -> Check:
    """When was pretrain last successful?"""
    if not PRETRAIN_STATE.exists():
        return Check(
            name="last_pretrain",
            title="Pretrain freshness",
            status="warn",
            message=(
                "Pretrain has never run on this checkout. Click 'Fix it' to run "
                "it now (takes about 30 seconds)."
            ),
            auto_fixable=True,
        )
    try:
        obj = json.loads(PRETRAIN_STATE.read_text(encoding="utf-8"))
        finished = obj.get("finished_at")
        if not finished:
            raise ValueError("no finished_at in pretrain state")
        ts = datetime.fromisoformat(finished.replace("Z", "+00:00"))
    except (OSError, json.JSONDecodeError, ValueError, KeyError):
        return Check(
            name="last_pretrain",
            title="Pretrain freshness",
            status="warn",
            message="Pretrain state file is unreadable -- click 'Fix it' to rebuild it.",
            auto_fixable=True,
        )
    age = datetime.now(UTC) - ts
    if age > timedelta(hours=36):
        hours = int(age.total_seconds() // 3600)
        return Check(
            name="last_pretrain",
            title="Pretrain freshness",
            status="warn",
            message=(
                f"Last pretrain was {hours}h ago. Click 'Fix it' to refresh data now."
            ),
            auto_fixable=True,
            detail={"finished_at": finished, "age_hours": hours},
        )
    return Check(
        name="last_pretrain",
        title="Pretrain freshness",
        status="ok",
        message=f"Last pretrain succeeded at {finished}.",
        detail={"finished_at": finished},
    )


# ---------------------------------------------------------------------------
# Auto-heal actions
# ---------------------------------------------------------------------------


def auto_heal(check_name: str) -> dict[str, object]:
    """Run the auto-heal action for ``check_name``.

    Returns a dict with at least ``ok`` (bool) and ``message`` (str) so the
    UI can show success/failure without parsing free-form text.
    """
    fixers = {
        "port_8765_clear": _heal_port,
        "orphan_pythons": _heal_orphan_pythons,
        "ollama_running": _heal_ollama_running,
        "models_pulled": _heal_models_pulled,
        "last_pretrain": _heal_run_pretrain,
    }
    fn = fixers.get(check_name)
    if fn is None:
        return {"ok": False, "message": f"No auto-fix is available for '{check_name}'."}
    try:
        return fn()
    except Exception as exc:  # pragma: no cover - defensive
        return {"ok": False, "message": f"Auto-fix raised an error: {exc!r}"}


def _heal_port() -> dict[str, object]:
    pid = _foreign_pid_on_port(8765)
    if pid is None:
        return {"ok": True, "message": "Port 8765 is already free."}
    # Only kill if it's a python.exe we recognise from our repo -- never kill
    # a stranger's process from a cockpit auto-fix.
    if not _is_our_repo_python(pid):
        return {
            "ok": False,
            "message": (
                f"Port 8765 is held by PID {pid}, but it doesn't look like our "
                "cockpit. Refusing to kill it from here -- stop it manually."
            ),
        }
    if _kill_pid(pid):
        return {"ok": True, "message": f"Stopped stranded cockpit (PID {pid}).", "pid": pid}
    return {"ok": False, "message": f"Could not stop PID {pid}. Try 'taskkill /F /PID {pid}'."}


def _heal_orphan_pythons() -> dict[str, object]:
    orphans = _list_orphan_repo_pythons()
    killed: list[int] = []
    for entry in orphans:
        pid = int(entry["pid"])
        if _kill_pid(pid):
            killed.append(pid)
    if not orphans:
        return {"ok": True, "message": "No orphan processes were found."}
    if killed:
        return {
            "ok": True,
            "message": f"Stopped {len(killed)} orphan process(es): PID {killed}.",
            "killed": killed,
        }
    return {"ok": False, "message": "Could not stop any orphan processes."}


def _heal_ollama_running() -> dict[str, object]:
    if not shutil.which("ollama"):
        return {"ok": False, "message": "Ollama is not installed -- nothing to start."}
    # ``ollama serve`` is the daemon; spawn it detached so it survives this
    # request. On Windows we use CREATE_NEW_PROCESS_GROUP + DETACHED_PROCESS;
    # on POSIX, a double-fork via start_new_session is enough.
    creationflags = 0
    if os.name == "nt":
        creationflags = (
            subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
            | getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
            | getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
        )
    try:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            start_new_session=(os.name != "nt"),
            creationflags=creationflags,
        )
    except OSError as exc:
        return {"ok": False, "message": f"Could not launch ollama serve: {exc!r}"}
    # Poll for up to 8s to confirm it came up.
    for _ in range(16):
        try:
            r = httpx.get("http://127.0.0.1:11434/api/tags", timeout=0.5)
            if r.status_code == 200:
                return {"ok": True, "message": "Ollama started."}
        except (httpx.HTTPError, OSError):
            time.sleep(0.5)
    return {
        "ok": False,
        "message": (
            "Started ollama serve but it didn't respond within 8 seconds. "
            "Check the Ollama tray icon or run 'ollama serve' manually."
        ),
    }


def _heal_models_pulled() -> dict[str, object]:
    """Pull the active profile's missing models via the existing setup job.

    Reusing ``tools/check_ollama.py --auto`` (the same driver the
    Ollama panel uses) means progress is already streamable through
    ``/api/jobs/ollama_setup/stream`` -- the operator doesn't need a
    separate channel for Fix-It pulls.
    """
    try:
        from packages.cockpit import proc as job_mgr
    except ImportError as exc:
        return {"ok": False, "message": f"Job manager unavailable: {exc!r}"}
    cmd = [sys.executable, "tools/check_ollama.py", "--auto"]
    info = job_mgr.start("ollama_setup", cmd)
    return {
        "ok": True,
        "message": (
            f"Pulling missing models in the background (PID {info.pid}). "
            "Watch the Ollama panel or /api/jobs/ollama_setup/stream for "
            "live progress -- this can take a while for large models."
        ),
        "pid": info.pid,
    }


def _heal_run_pretrain() -> dict[str, object]:
    """Kick off a pretrain in-process via the existing job manager.

    This re-uses the same launcher (env sanitization, breakaway flags,
    diagnostic log header) that powers every other job, so a successful
    fix here proves the launcher works.
    """
    try:
        from packages.cockpit import proc as job_mgr
    except ImportError as exc:
        return {"ok": False, "message": f"Job manager unavailable: {exc!r}"}
    venv_python = sys.executable
    info = job_mgr.start(
        "pretrain",
        [venv_python, "-m", "packages.data.pretrain"],
    )
    return {
        "ok": True,
        "message": f"Pretrain started (PID {info.pid}). Watch /api/jobs/pretrain/stream for output.",
        "pid": info.pid,
    }


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def run_all() -> list[Check]:
    """Run every check in order and return the results.

    Order matters for the UI: most-fundamental first (venv, env) then
    runtime (port, processes) then optional services (Ollama) then
    freshness (pretrain). A red row near the top usually explains red
    rows below it.
    """
    return [
        check_venv(),
        check_env_file(),
        check_port(),
        check_orphan_pythons(),
        check_ollama_installed(),
        check_ollama_running(),
        check_models_pulled(),
        check_last_pretrain(),
    ]


def summary(checks: list[Check] | None = None) -> dict[str, object]:
    """Top-line rollup: ``status`` is the worst severity across checks.

    The tray launcher uses this for icon colour. ``error`` -> red,
    ``warn`` -> yellow, ``ok`` -> green, ``info`` -> grey.
    """
    checks = checks if checks is not None else run_all()
    statuses = {c.status for c in checks}
    if "error" in statuses:
        overall: Status = "error"
    elif "warn" in statuses:
        overall = "warn"
    elif "ok" in statuses:
        overall = "ok"
    else:
        overall = "info"
    return {
        "status": overall,
        "counts": {
            "ok": sum(1 for c in checks if c.status == "ok"),
            "warn": sum(1 for c in checks if c.status == "warn"),
            "error": sum(1 for c in checks if c.status == "error"),
            "info": sum(1 for c in checks if c.status == "info"),
        },
        "checks": [c.to_dict() for c in checks],
        "now": datetime.now(UTC).isoformat(timespec="seconds"),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _port_in_use(port: int) -> bool:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.settimeout(0.2)
        try:
            s.connect(("127.0.0.1", port))
            return True
        except OSError:
            return False


def _foreign_pid_on_port(port: int) -> int | None:
    """Return the PID listening on ``port``, or None if nothing/we can't tell."""
    if not _port_in_use(port):
        return None
    return _pid_listening_on(port)


def _our_cockpit_pid_on_port(port: int) -> int | None:
    pid = _pid_listening_on(port)
    if pid is None:
        return None
    return pid if _is_our_repo_python(pid) else None


def _pid_listening_on(port: int) -> int | None:
    """Cross-platform: which PID owns the TCP listen on ``port``?

    On Windows we shell out to ``netstat -ano``. On POSIX we try ``lsof``.
    Either way we cache nothing -- this is one network-stack lookup, fast.
    """
    if os.name == "nt":
        try:
            out = subprocess.check_output(
                ["netstat", "-ano", "-p", "tcp"],
                stderr=subprocess.DEVNULL,
                timeout=3,
            ).decode("utf-8", errors="replace")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            return None
        for line in out.splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            if not parts[1].endswith(f":{port}"):
                continue
            if parts[3].upper() != "LISTENING":
                continue
            try:
                return int(parts[4])
            except ValueError:
                continue
        return None
    # POSIX
    try:
        out = subprocess.check_output(
            ["lsof", "-iTCP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
            stderr=subprocess.DEVNULL,
            timeout=3,
        ).decode("utf-8", errors="replace")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return None
    for line in out.splitlines():
        try:
            return int(line.strip())
        except ValueError:
            continue
    return None


def _list_repo_python_pids() -> list[dict[str, object]]:
    """Every python.exe whose command line points at our repo."""
    if os.name != "nt":
        # On POSIX we don't bother -- this is a Windows-orphan problem.
        return []
    try:
        out = subprocess.check_output(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                "Select-Object ProcessId, ParentProcessId, ExecutablePath, CommandLine | "
                "ConvertTo-Json -Compress",
            ],
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode("utf-8", errors="replace")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return []
    if not out.strip():
        return []
    try:
        raw = json.loads(out)
    except json.JSONDecodeError:
        return []
    rows = [raw] if isinstance(raw, dict) else raw
    repo_str = str(REPO_ROOT).lower()
    result: list[dict[str, object]] = []
    for row in rows:
        path = (row.get("ExecutablePath") or "").lower()
        cmd = (row.get("CommandLine") or "").lower()
        if repo_str in path or repo_str in cmd:
            result.append(
                {
                    "pid": int(row.get("ProcessId", 0)),
                    "parent_pid": int(row.get("ParentProcessId", 0)),
                    "path": row.get("ExecutablePath"),
                    "cmd": row.get("CommandLine"),
                }
            )
    return result


def _list_orphan_repo_pythons() -> list[dict[str, object]]:
    """python.exe processes from our repo whose parent process is gone."""
    pythons = _list_repo_python_pids()
    if not pythons:
        return []
    live_pids = _live_pids()
    return [p for p in pythons if int(p["parent_pid"]) not in live_pids]


def _live_pids() -> set[int]:
    if os.name != "nt":
        return set()
    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", "(Get-Process).Id"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode("utf-8", errors="replace")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return set()
    return {int(line) for line in out.splitlines() if line.strip().isdigit()}


def _is_our_repo_python(pid: int) -> bool:
    return any(int(entry["pid"]) == pid for entry in _list_repo_python_pids())


def _kill_pid(pid: int) -> bool:
    if os.name == "nt":
        try:
            subprocess.check_call(
                ["taskkill", "/F", "/PID", str(pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
            return True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            return False
    try:
        os.kill(pid, 15)
        return True
    except OSError:
        return False
