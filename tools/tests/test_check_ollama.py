"""Tests for the Ollama auto-setup tool.

These tests pin three things that are easy to silently break:
  * Tag matching across quantization variants (deepseek-r1:32b matches
    deepseek-r1:32b-q4_K_M, but does NOT match a totally different family).
  * HTTP-pull stream parsing: a success status produces True; an error in
    any jsonl line produces False; malformed lines are ignored.
  * ensure_daemon never raises and returns False cleanly when the ollama
    CLI is not installed (so --auto can still produce a helpful message).
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from unittest.mock import patch

# Make `tools/` importable.
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

import check_ollama  # noqa: E402

# ---------------------------------------------------------------------------
# _matches — tag matching
# ---------------------------------------------------------------------------


def test_matches_exact_tag() -> None:
    assert check_ollama._matches("deepseek-r1:32b", ["deepseek-r1:32b"]) is True


def test_matches_quantization_variant() -> None:
    """A q4 quantization of the same base+size should be accepted."""
    assert check_ollama._matches(
        "deepseek-r1:32b", ["deepseek-r1:32b-q4_K_M"]
    ) is True


def test_matches_rejects_wrong_size() -> None:
    """A 7B variant must NOT satisfy a 32B requirement."""
    # 7b and 32b are different tags, so this is a miss.
    assert check_ollama._matches(
        "deepseek-r1:32b", ["deepseek-r1:7b"]
    ) is False


def test_matches_rejects_wrong_family() -> None:
    """Qwen does not satisfy a DeepSeek requirement."""
    assert check_ollama._matches(
        "deepseek-r1:32b", ["qwen2.5:32b"]
    ) is False


def test_matches_no_tag_required_accepts_any_tag() -> None:
    """Required 'deepseek-r1' (no tag) should accept any installed tag of
    that family."""
    assert check_ollama._matches("deepseek-r1", ["deepseek-r1:latest"]) is True
    assert check_ollama._matches("deepseek-r1", ["deepseek-r1:70b"]) is True


def test_matches_empty_installed() -> None:
    assert check_ollama._matches("deepseek-r1:32b", []) is False


# ---------------------------------------------------------------------------
# _pull_via_http — stream parsing
# ---------------------------------------------------------------------------


class _FakeResponse:
    """Minimal stand-in for the urlopen() return value: iterates lines."""

    def __init__(self, lines: list[str]) -> None:
        self._buf = io.BytesIO(b"\n".join(line.encode("utf-8") for line in lines))

    def __iter__(self):
        yield from self._buf

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _fake_urlopen(lines: list[str]):
    def _stub(req, timeout=None):
        return _FakeResponse(lines)
    return _stub


def test_http_pull_success() -> None:
    """A stream that ends with status=success returns True."""
    lines = [
        json.dumps({"status": "pulling manifest"}),
        json.dumps({"status": "downloading", "total": 1000, "completed": 500}),
        json.dumps({"status": "downloading", "total": 1000, "completed": 1000}),
        json.dumps({"status": "success"}),
    ]
    with patch.object(check_ollama.urllib.request, "urlopen", _fake_urlopen(lines)):
        ok = check_ollama._pull_via_http("http://x", "deepseek-r1:32b", verbose=False)
    assert ok is True


def test_http_pull_error_in_stream_fails() -> None:
    """An error field anywhere in the jsonl must cause failure."""
    lines = [
        json.dumps({"status": "pulling manifest"}),
        json.dumps({"error": "manifest not found"}),
    ]
    with patch.object(check_ollama.urllib.request, "urlopen", _fake_urlopen(lines)):
        ok = check_ollama._pull_via_http("http://x", "fake:model", verbose=False)
    assert ok is False


def test_http_pull_ignores_malformed_lines() -> None:
    """Garbage lines must not crash the parser — they're silently dropped."""
    lines = [
        json.dumps({"status": "pulling manifest"}),
        "not-json-at-all",
        "{incomplete",
        json.dumps({"status": "success"}),
    ]
    with patch.object(check_ollama.urllib.request, "urlopen", _fake_urlopen(lines)):
        ok = check_ollama._pull_via_http("http://x", "deepseek-r1:32b", verbose=False)
    assert ok is True


def test_http_pull_handles_url_error() -> None:
    """A connection error on the urlopen call must return False, not raise."""
    def boom(req, timeout=None):
        raise check_ollama.urllib.error.URLError("connection refused")

    with patch.object(check_ollama.urllib.request, "urlopen", boom):
        ok = check_ollama._pull_via_http("http://x", "deepseek-r1:32b", verbose=False)
    assert ok is False


# ---------------------------------------------------------------------------
# ensure_daemon — autostart guardrails
# ---------------------------------------------------------------------------


def test_ensure_daemon_returns_true_when_alive() -> None:
    """If the daemon already responds, we must NOT try to spawn another one."""
    with patch.object(check_ollama, "_daemon_alive", return_value=True), \
         patch.object(check_ollama, "_start_daemon_background") as spawn:
        assert check_ollama.ensure_daemon("http://x", verbose=False) is True
        spawn.assert_not_called()


def test_ensure_daemon_returns_false_if_cli_missing() -> None:
    """No CLI on PATH and daemon down -> friendly False (not crash)."""
    with patch.object(check_ollama, "_daemon_alive", return_value=False), \
         patch.object(check_ollama, "_start_daemon_background", return_value=None):
        assert check_ollama.ensure_daemon("http://x", verbose=False) is False


def test_ensure_daemon_starts_and_waits_for_ready() -> None:
    """The happy autostart path: spawn the daemon then wait for it to be alive."""
    class _FakeProc:
        pid = 1234
    with patch.object(check_ollama, "_daemon_alive", return_value=False), \
         patch.object(check_ollama, "_start_daemon_background", return_value=_FakeProc()), \
         patch.object(check_ollama, "_wait_for_daemon", return_value=True):
        assert check_ollama.ensure_daemon("http://x", verbose=False) is True


def test_ensure_daemon_reports_failure_on_timeout() -> None:
    """If the daemon never comes up within the timeout, surface the failure."""
    class _FakeProc:
        pid = 1234
    with patch.object(check_ollama, "_daemon_alive", return_value=False), \
         patch.object(check_ollama, "_start_daemon_background", return_value=_FakeProc()), \
         patch.object(check_ollama, "_wait_for_daemon", return_value=False):
        assert check_ollama.ensure_daemon("http://x", verbose=False) is False


# ---------------------------------------------------------------------------
# pull_model — HTTP-first with CLI fallback
# ---------------------------------------------------------------------------


def test_pull_model_uses_http_first() -> None:
    """If HTTP pull succeeds, CLI fallback must NOT be invoked."""
    with patch.object(check_ollama, "_pull_via_http", return_value=True), \
         patch.object(check_ollama, "_pull_via_cli") as cli:
        assert check_ollama.pull_model("http://x", "deepseek-r1:32b", verbose=False) is True
        cli.assert_not_called()


def test_pull_model_falls_back_to_cli() -> None:
    """If HTTP fails, the CLI fallback runs and its result is returned."""
    with patch.object(check_ollama, "_pull_via_http", return_value=False), \
         patch.object(check_ollama, "_pull_via_cli", return_value=True) as cli:
        assert check_ollama.pull_model("http://x", "deepseek-r1:32b", verbose=False) is True
        cli.assert_called_once()


def test_pull_model_both_paths_fail_returns_false() -> None:
    with patch.object(check_ollama, "_pull_via_http", return_value=False), \
         patch.object(check_ollama, "_pull_via_cli", return_value=False):
        assert check_ollama.pull_model("http://x", "deepseek-r1:32b", verbose=False) is False
