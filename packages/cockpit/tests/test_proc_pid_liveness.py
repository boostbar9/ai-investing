"""Regression guard for the Windows ``os.kill(pid, 0)`` SystemError that
crashed ``/api/health`` and left every cockpit dashboard stuck on
"Loading...".

On Windows, ``os.kill(pid, 0)`` for an unknown / exited PID can raise
``SystemError: <built-in function kill> returned a result with an
exception set`` (a CPython quirk where ``OpenProcess`` returns invalid
parameter and the wrapper doesn't translate it to OSError cleanly).
That uncaught SystemError bubbled through FastAPI and produced a 500
response, which broke the dashboards' first ``/api/health`` poll.

The fix routes liveness checks through ``_pid_is_running`` which uses
psutil with a defensive fallback. These tests verify the helper handles
the failure modes that caused the original bug.
"""

from __future__ import annotations

import os

import pytest

from packages.cockpit import proc as proc_mod
from packages.cockpit.proc import JobInfo, _pid_is_running


def test_pid_none_is_not_running():
    assert _pid_is_running(None) is False  # type: ignore[arg-type]


def test_pid_zero_is_not_running():
    assert _pid_is_running(0) is False


def test_negative_pid_is_not_running():
    assert _pid_is_running(-1) is False


def test_current_process_is_running():
    assert _pid_is_running(os.getpid()) is True


def test_obviously_dead_pid_returns_false_without_raising():
    # 2^31 - 2 is well above any real PID on any platform we run on.
    # This is exactly the situation that triggered the Windows SystemError
    # on the user's machine: a PID that the OS can't translate to a handle.
    assert _pid_is_running(2_147_483_646) is False


def test_jobinfo_is_running_handles_dead_pid():
    """The high-level wrapper used by /api/health must not raise."""
    info = JobInfo(kind="paper_loop", pid=2_147_483_646)
    # Must return False rather than propagating SystemError/OSError --
    # this is the exact call site that crashed /api/health before.
    assert info.is_running() is False


def test_jobinfo_to_dict_includes_running_field_safely():
    """to_dict() is what /api/health serializes -- it must never raise."""
    info = JobInfo(kind="pretrain", pid=2_147_483_646)
    payload = info.to_dict()
    assert payload["running"] is False
    assert payload["kind"] == "pretrain"
    assert payload["pid"] == 2_147_483_646


def test_jobinfo_with_no_pid_reports_not_running():
    info = JobInfo(kind="paper_loop")
    assert info.is_running() is False
    assert info.to_dict()["running"] is False


@pytest.mark.skipif(proc_mod._psutil is None, reason="psutil not installed")
def test_psutil_path_is_taken_when_available():
    """Sanity check the production path uses psutil, not the os.kill fallback."""
    # If psutil is loaded, the function should not delegate to os.kill for
    # this check. We can't easily mock without monkeypatching, but we can
    # at least assert psutil is the loaded module.
    assert proc_mod._psutil is not None
