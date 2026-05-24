"""Tests for the subprocess environment sanitizer.

Background: on Windows we saw every child Python job die instantly with
STATUS_DLL_INIT_FAILED (3221225794) because the parent shell's env either
leaked PYTHONHOME or stripped System32 from PATH -- without either, the
child interpreter cannot load its own runtime DLLs. _child_env defends
against both.
"""
from __future__ import annotations

import os

import pytest

from packages.cockpit.proc import _child_env


def test_child_env_strips_pythonhome(monkeypatch: pytest.MonkeyPatch) -> None:
    """PYTHONHOME leaking from the parent shell points the child at the
    wrong Python install -- strip it."""
    monkeypatch.setenv("PYTHONHOME", r"C:\some\other\python")
    env = _child_env()
    assert "PYTHONHOME" not in env


def test_child_env_sets_python_runtime_flags() -> None:
    env = _child_env()
    assert env["PYTHONUNBUFFERED"] == "1"
    assert env["PYTHONIOENCODING"] == "utf-8"
    assert env["PYTHONPATH"]


def test_child_env_preserves_user_pythonpath(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the operator set PYTHONPATH themselves, do not clobber it."""
    monkeypatch.setenv("PYTHONPATH", "./mypath")
    env = _child_env()
    assert env["PYTHONPATH"] == "./mypath"


def test_child_env_preserves_other_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALPACA_PAPER_KEY_ID", "abc123")
    env = _child_env()
    assert env["ALPACA_PAPER_KEY_ID"] == "abc123"


def test_child_env_appends_system_paths_on_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate Windows with a stripped PATH and verify System32 gets added."""
    monkeypatch.setattr(os, "name", "nt")
    monkeypatch.setattr(os, "pathsep", ";")
    monkeypatch.setenv("PATH", r"C:\Users\me\venv\Scripts")
    env = _child_env()
    parts = env["PATH"].split(";")
    # The user's venv path must still be first.
    assert parts[0] == r"C:\Users\me\venv\Scripts"
    # System32 must be present somewhere.
    lowered = [p.lower() for p in parts]
    assert r"c:\windows\system32" in lowered


def test_child_env_no_duplicate_system_paths_on_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """If PATH already contains System32, do not add a duplicate entry."""
    monkeypatch.setattr(os, "name", "nt")
    monkeypatch.setattr(os, "pathsep", ";")
    monkeypatch.setenv("PATH", r"C:\Windows\System32;C:\other")
    env = _child_env()
    parts = [p.lower() for p in env["PATH"].split(";")]
    assert parts.count(r"c:\windows\system32") == 1


def test_child_env_leaves_path_alone_on_posix(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-Windows hosts must not pick up Windows directories."""
    monkeypatch.setattr(os, "name", "posix")
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    env = _child_env()
    assert env["PATH"] == "/usr/bin:/bin"
    assert "System32" not in env["PATH"]


def test_start_writes_diagnostic_header_before_child_runs(
    tmp_path: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Even if the child dies instantly, the log must show what was tried.

    This is the test we wished we had when paper_loop / pretrain were
    failing on Windows with zero-byte log files: the header now records
    argv, cwd, and key env vars so the operator can see exactly what the
    cockpit attempted.
    """
    import sys

    from packages.cockpit import proc

    monkeypatch.setattr(proc, "LOG_DIR", tmp_path)
    monkeypatch.setattr(proc, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(proc, "_jobs", {})
    monkeypatch.setattr(proc, "_procs", {})
    monkeypatch.setattr(proc, "_persist", lambda: None)

    info = proc.start("diag_smoke", [sys.executable, "-c", "print('child ran')"])
    assert info.pid is not None

    # Wait briefly for the child + tee thread to finish.
    import time

    for _ in range(50):
        if info.exit_code is not None:
            break
        time.sleep(0.05)

    log_text = (tmp_path / "diag_smoke.log").read_text(encoding="utf-8")
    # Header is present
    assert "=== launching diag_smoke at" in log_text
    assert "argv = " in log_text
    assert "PYTHONPATH = " in log_text
    assert "=== child output below ===" in log_text
    # And the child's actual output made it through the tee
    assert "child ran" in log_text
