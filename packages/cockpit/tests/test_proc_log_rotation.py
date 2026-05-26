"""Tests for the job-log rotation helper in cockpit.proc.

A long-running pretrain or chatty ollama_setup will append to the same
``<kind>.log`` file forever between cockpit restarts. Without rotation, that
file can grow without bound and eventually starve the disk. ``_rotate_if_needed``
moves the active log to ``<path>.1`` (overwriting any previous archive) once
it exceeds the configured byte cap and starts a fresh file with a marker line
so the operator knows where output continues from.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from packages.cockpit.proc import MAX_LOG_BYTES, _rotate_if_needed


def test_rotate_noop_when_file_missing(tmp_path: Path) -> None:
    """A path that doesn't exist isn't an error — just return False."""
    assert _rotate_if_needed(tmp_path / "missing.log") is False


def test_rotate_noop_when_under_cap(tmp_path: Path) -> None:
    """Files smaller than the cap are left untouched."""
    log = tmp_path / "tiny.log"
    log.write_text("hello\n", encoding="utf-8")
    assert _rotate_if_needed(log, max_bytes=1024) is False
    assert log.read_text(encoding="utf-8") == "hello\n"
    assert not (tmp_path / "tiny.log.1").exists()


def test_rotate_moves_oversize_log_to_archive(tmp_path: Path) -> None:
    """When the file is over the cap, it becomes the .1 archive."""
    log = tmp_path / "big.log"
    payload = "x" * 2048
    log.write_text(payload, encoding="utf-8")
    assert _rotate_if_needed(log, max_bytes=1024) is True
    archive = tmp_path / "big.log.1"
    assert archive.exists()
    assert archive.read_text(encoding="utf-8") == payload
    # The fresh active log carries a single marker line so operators can see
    # where rotation happened and what the previous size was.
    active = log.read_text(encoding="utf-8")
    assert "log rotated at" in active
    assert "2,048 bytes" in active
    assert "big.log.1" in active


def test_rotate_overwrites_previous_archive(tmp_path: Path) -> None:
    """Only one generation is kept — repeated rotations replace the archive."""
    log = tmp_path / "loop.log"
    archive = tmp_path / "loop.log.1"
    log.write_text("y" * 2048, encoding="utf-8")
    _rotate_if_needed(log, max_bytes=1024)
    first_archive = archive.read_text(encoding="utf-8")
    assert first_archive.startswith("y")

    # Second rotation: write fresh oversize content and roll again.
    log.write_text("z" * 2048, encoding="utf-8")
    _rotate_if_needed(log, max_bytes=1024)
    second_archive = archive.read_text(encoding="utf-8")
    assert second_archive.startswith("z")
    assert second_archive != first_archive


def test_max_log_bytes_constant_is_reasonable() -> None:
    """Guard against accidentally shipping a 1-byte or 10-GB cap.

    2 MiB is enough to capture a full pretrain run's tail without letting the
    log file dominate the data directory.
    """
    assert 1 * 1024 * 1024 <= MAX_LOG_BYTES <= 16 * 1024 * 1024


def test_rotate_handles_locked_archive_gracefully(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If we can't write the archive (e.g. Windows file lock), bail cleanly.

    The tee thread should keep running rather than crash the cockpit. We
    simulate a locked archive by making ``Path.rename`` raise OSError once.
    """
    log = tmp_path / "locked.log"
    log.write_text("a" * 2048, encoding="utf-8")

    original_rename = Path.rename

    def boom(self: Path, target: Path) -> Path:
        raise OSError("simulated archive lock")

    monkeypatch.setattr(Path, "rename", boom)
    assert _rotate_if_needed(log, max_bytes=1024) is False
    # Restore so the test runner's teardown isn't surprised.
    monkeypatch.setattr(Path, "rename", original_rename)
    # Active log is left intact when rotation fails.
    assert log.exists()
    assert log.stat().st_size == 2048
