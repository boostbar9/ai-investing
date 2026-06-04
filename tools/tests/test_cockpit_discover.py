"""Tests for tools/cockpit_discover.py (Phase 36d)."""

from __future__ import annotations

import base64
import json
import subprocess
from unittest import mock

import pytest

from tools import cockpit_discover


def _gh_ok(payload: dict) -> mock.MagicMock:
    """Build a successful CompletedProcess for `gh api`."""
    encoded = base64.b64encode(json.dumps(payload).encode("utf-8")).decode("ascii")
    return mock.MagicMock(
        returncode=0,
        stdout=json.dumps({"content": encoded, "encoding": "base64"}),
        stderr="",
    )


def _gh_fail(stderr: str = "404 Not Found", rc: int = 1) -> mock.MagicMock:
    return mock.MagicMock(returncode=rc, stdout="", stderr=stderr)


class TestFetchHandle:
    def test_returns_parsed_handle(self) -> None:
        handle = {
            "url": "https://abc.trycloudflare.com",
            "token": "0123456789abcdef" * 2,
            "started_at": "2026-06-04T00:00:00Z",
        }
        with mock.patch.object(subprocess, "run", return_value=_gh_ok(handle)) as run:
            out = cockpit_discover.fetch_handle("boostbar9", "ai-investing")
        assert out == handle
        args, kwargs = run.call_args
        cmd = args[0]
        assert cmd[:2] == ["gh", "api"]
        assert "repos/boostbar9/ai-investing/contents/data/cockpit/remote_handle.json" in cmd[2]
        assert "ref=cockpit-handle" in cmd[2]

    def test_custom_branch(self) -> None:
        handle = {"url": "x", "token": "y"}
        with mock.patch.object(subprocess, "run", return_value=_gh_ok(handle)) as run:
            cockpit_discover.fetch_handle("o", "r", branch="other-branch")
        cmd = run.call_args[0][0]
        assert "ref=other-branch" in cmd[2]

    def test_raises_on_gh_failure(self) -> None:
        with mock.patch.object(subprocess, "run", return_value=_gh_fail("nope", rc=1)):
            with pytest.raises(RuntimeError, match="gh api failed"):
                cockpit_discover.fetch_handle("o", "r")

    def test_raises_on_missing_content_field(self) -> None:
        bad = mock.MagicMock(
            returncode=0,
            stdout=json.dumps({"name": "remote_handle.json"}),  # no 'content'
            stderr="",
        )
        with mock.patch.object(subprocess, "run", return_value=bad):
            with pytest.raises(RuntimeError, match="unexpected payload"):
                cockpit_discover.fetch_handle("o", "r")


class TestMainCLI:
    def test_full_json_output(self, capsys: pytest.CaptureFixture[str]) -> None:
        handle = {"url": "https://x.trycloudflare.com", "token": "t" * 32}
        with mock.patch.object(subprocess, "run", return_value=_gh_ok(handle)):
            rc = cockpit_discover.main([])
        assert rc == 0
        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert parsed == handle

    def test_single_field(self, capsys: pytest.CaptureFixture[str]) -> None:
        handle = {"url": "https://x.trycloudflare.com", "token": "t" * 32}
        with mock.patch.object(subprocess, "run", return_value=_gh_ok(handle)):
            rc = cockpit_discover.main(["--field", "url"])
        assert rc == 0
        assert capsys.readouterr().out.strip() == "https://x.trycloudflare.com"

    def test_unknown_field_returns_2(self, capsys: pytest.CaptureFixture[str]) -> None:
        with mock.patch.object(subprocess, "run", return_value=_gh_ok({"url": "x"})):
            rc = cockpit_discover.main(["--field", "nope"])
        assert rc == 2
        assert "no field 'nope'" in capsys.readouterr().err

    def test_fetch_failure_returns_1(self, capsys: pytest.CaptureFixture[str]) -> None:
        with mock.patch.object(subprocess, "run", return_value=_gh_fail()):
            rc = cockpit_discover.main([])
        assert rc == 1
        assert "discover failed" in capsys.readouterr().err

    def test_owner_repo_overrides(self) -> None:
        handle = {"url": "u", "token": "t"}
        with mock.patch.object(subprocess, "run", return_value=_gh_ok(handle)) as run:
            cockpit_discover.main(["--owner", "alt", "--repo", "rep"])
        cmd = run.call_args[0][0]
        assert "repos/alt/rep/contents" in cmd[2]
