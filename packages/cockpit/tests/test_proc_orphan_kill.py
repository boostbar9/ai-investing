"""Tests for the orphaned-PID kill path in proc.stop() (Phase 36d hotfix).

When the cockpit restarts while a child job (e.g. paper_loop) is still
running, the Popen handle is lost but the OS process keeps living. Before
this fix, stop() would early-return without killing the orphan, leaving
the job slot wedged forever.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest import mock

import pytest

from packages.cockpit import proc as proc_mod


@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset the in-memory registry between tests."""
    with proc_mod._lock:
        proc_mod._jobs.clear()
        proc_mod._procs.clear()
    yield
    with proc_mod._lock:
        proc_mod._jobs.clear()
        proc_mod._procs.clear()


def _seed_orphan(kind: str, pid: int) -> proc_mod.JobInfo:
    """Seed an orphan job entry: PID present, no Popen handle."""
    info = proc_mod.JobInfo(
        kind=kind,
        pid=pid,
        started_at=datetime.now(UTC).isoformat(timespec="seconds"),
        command=["pretend", "args"],
        log_file=f"/tmp/{kind}.log",
    )
    with proc_mod._lock:
        proc_mod._jobs[kind] = info
        # Critical: NO entry in _procs -- this is what makes it an orphan.
    return info


class TestOrphanStop:
    def test_orphan_with_live_pid_calls_kill_os(self) -> None:
        """Live orphan PID: stop() must call _kill_pid_os to clean up."""
        _seed_orphan("paper_loop", pid=12345)

        with mock.patch.object(proc_mod, "_pid_is_running", return_value=True) as alive, \
             mock.patch.object(proc_mod, "_kill_pid_os", return_value=True) as killer:
            result = proc_mod.stop("paper_loop")

        # _pid_is_running gets called by both is_running() (via JobInfo) and
        # by stop()'s Case-2 guard; what matters is _kill_pid_os ran.
        assert alive.called
        killer.assert_called_once_with(12345)
        assert result.exit_code == -9
        assert result.finished_at is not None

    def test_orphan_with_dead_pid_just_cleans_state(self) -> None:
        """Dead orphan PID: stop() should clear state without calling kill."""
        _seed_orphan("paper_loop", pid=99999)

        with mock.patch.object(proc_mod, "_pid_is_running", return_value=False), \
             mock.patch.object(proc_mod, "_kill_pid_os") as killer:
            result = proc_mod.stop("paper_loop")

        killer.assert_not_called()
        assert result.finished_at is not None

    def test_orphan_kill_failure_records_no_exit(self) -> None:
        """If _kill_pid_os returns False, exit_code stays None (failure)."""
        _seed_orphan("paper_loop", pid=12345)

        with mock.patch.object(proc_mod, "_pid_is_running", return_value=True), \
             mock.patch.object(proc_mod, "_kill_pid_os", return_value=False):
            result = proc_mod.stop("paper_loop")

        assert result.exit_code is None
        assert result.finished_at is not None  # still records the attempt

    def test_no_job_at_all_returns_empty_info(self) -> None:
        """stop() on an unknown kind should not crash."""
        result = proc_mod.stop("never_existed")
        assert result.kind == "never_existed"
        assert result.pid is None

    def test_stop_unblocks_start(self) -> None:
        """After orphan kill, start() can spawn a fresh process in the slot."""
        _seed_orphan("paper_loop", pid=12345)

        with mock.patch.object(proc_mod, "_pid_is_running", return_value=True), \
             mock.patch.object(proc_mod, "_kill_pid_os", return_value=True):
            proc_mod.stop("paper_loop")

        # After stop, is_running() should report False -- start() relies on this.
        with proc_mod._lock:
            info = proc_mod._jobs["paper_loop"]
        assert info.finished_at is not None
        # Now simulate a fresh start: _pid_is_running for the OLD pid is False.
        with mock.patch.object(proc_mod, "_pid_is_running", return_value=False):
            assert not info.is_running()


class TestKillPidOsHelper:
    def test_invalid_pid_returns_false(self) -> None:
        assert proc_mod._kill_pid_os(0) is False
        assert proc_mod._kill_pid_os(-1) is False
        assert proc_mod._kill_pid_os(None) is False  # type: ignore[arg-type]

    def test_uses_psutil_terminate_then_kill_on_timeout(self) -> None:
        if proc_mod._psutil is None:
            pytest.skip("psutil not available")
        fake_proc = mock.MagicMock()
        fake_proc.wait.side_effect = [proc_mod._psutil.TimeoutExpired(5), None]
        fake_proc.is_running.return_value = False
        fake_proc.status.return_value = "running"
        with mock.patch.object(proc_mod._psutil, "Process", return_value=fake_proc):
            ok = proc_mod._kill_pid_os(42)
        assert ok is True
        fake_proc.terminate.assert_called_once()
        fake_proc.kill.assert_called_once()
