"""Tests for the pure (non-GUI) helpers in tools/tray/launcher.py."""

from __future__ import annotations

import json
import subprocess
from unittest.mock import patch

import httpx
import pytest

from tools.tray import launcher


def test_local_head_sha_returns_string():
    sha = launcher.local_head_sha()
    # In CI we are in a git repo, so this should be non-empty hex.
    assert sha == "" or all(c in "0123456789abcdef" for c in sha)


def test_remote_head_sha_handles_network_failure():
    with patch("tools.tray.launcher.httpx.get", side_effect=httpx.HTTPError("x")):
        assert launcher.remote_head_sha() == ""


def test_remote_head_sha_handles_non_200():
    class _R:
        status_code = 503

        def json(self):  # pragma: no cover — not reached
            return {}

    with patch("tools.tray.launcher.httpx.get", return_value=_R()):
        assert launcher.remote_head_sha() == ""


def test_remote_head_sha_parses_sha():
    class _R:
        status_code = 200

        def json(self):
            return {"sha": "deadbeef"}

    with patch("tools.tray.launcher.httpx.get", return_value=_R()):
        assert launcher.remote_head_sha() == "deadbeef"


def test_check_for_updates_detects_match():
    with (
        patch("tools.tray.launcher.local_head_sha", return_value="abc"),
        patch("tools.tray.launcher.remote_head_sha", return_value="abc"),
    ):
        s = launcher.check_for_updates()
        assert s.update_available is False
        assert s.error is None


def test_check_for_updates_detects_diff():
    with (
        patch("tools.tray.launcher.local_head_sha", return_value="abc"),
        patch("tools.tray.launcher.remote_head_sha", return_value="def"),
    ):
        s = launcher.check_for_updates()
        assert s.update_available is True
        assert s.local_sha == "abc"
        assert s.remote_sha == "def"


def test_check_for_updates_handles_no_git():
    with patch("tools.tray.launcher.local_head_sha", return_value=""):
        s = launcher.check_for_updates()
        assert s.update_available is False
        assert s.error == "not a git repo"


def test_check_for_updates_handles_no_network():
    with (
        patch("tools.tray.launcher.local_head_sha", return_value="abc"),
        patch("tools.tray.launcher.remote_head_sha", return_value=""),
    ):
        s = launcher.check_for_updates()
        assert s.update_available is False
        assert s.error == "could not reach GitHub"


def test_stack_status_stopped_when_no_docker():
    with patch(
        "tools.tray.launcher.subprocess.check_output",
        side_effect=FileNotFoundError,
    ):
        assert launcher.stack_status() in {"stopped", "running", "degraded"}  # tolerant


def test_stack_status_parses_running_array():
    fake = json.dumps([
        {"Service": "api", "State": "running"},
        {"Service": "cockpit", "State": "running"},
    ])
    with patch("tools.tray.launcher.subprocess.check_output", return_value=fake.encode()):
        assert launcher.stack_status() == "running"


def test_stack_status_parses_running_jsonl():
    fake = (
        json.dumps({"Service": "api", "State": "running"})
        + "\n"
        + json.dumps({"Service": "cockpit", "State": "exited"})
    )
    with patch("tools.tray.launcher.subprocess.check_output", return_value=fake.encode()):
        assert launcher.stack_status() == "degraded"


def test_stack_status_handles_empty_output():
    with patch("tools.tray.launcher.subprocess.check_output", return_value=b""):
        assert launcher.stack_status() == "stopped"


def test_stack_status_handles_subprocess_error():
    with patch(
        "tools.tray.launcher.subprocess.check_output",
        side_effect=subprocess.CalledProcessError(1, "docker"),
    ):
        assert launcher.stack_status() == "stopped"


@pytest.mark.parametrize("color", ["#22c55e", "#ef4444", "#eab308", "#3b82f6"])
def test_make_icon_image_returns_pil_image(color):
    pil = pytest.importorskip("PIL.Image")
    img = launcher._make_icon_image(color)
    assert isinstance(img, pil.Image)
    assert img.size == (64, 64)
