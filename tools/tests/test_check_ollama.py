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
# _start_daemon_background -- Windows-safe Popen kwargs
# ---------------------------------------------------------------------------


def test_start_daemon_uses_creationflags_on_windows() -> None:
    """On Windows we must NOT pass start_new_session=True. Instead we pass
    DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP via creationflags. This
    sidesteps the Python 3.12.0..3.12.5 Windows subprocess.Popen abort."""
    captured: dict = {}

    class _FakeProc:
        pid = 999

    def fake_popen(args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return _FakeProc()

    with patch.object(check_ollama, "resolve_ollama_binary",
                      return_value=("C:\\Ollama\\ollama.exe", "standard")), \
         patch.object(check_ollama.sys, "platform", "win32"), \
         patch.object(check_ollama.subprocess, "Popen", side_effect=fake_popen):
        proc = check_ollama._start_daemon_background()

    assert proc is not None
    assert captured["args"] == ["C:\\Ollama\\ollama.exe", "serve"]
    kw = captured["kwargs"]
    # The exact bug we are dodging:
    assert "start_new_session" not in kw, \
        "start_new_session must NOT be passed on Windows (Py 3.12.0 abort)"
    # The fix:
    assert "creationflags" in kw
    # DETACHED_PROCESS (0x08) and CREATE_NEW_PROCESS_GROUP (0x200) must be set.
    assert kw["creationflags"] & 0x00000008  # DETACHED_PROCESS
    assert kw["creationflags"] & 0x00000200  # CREATE_NEW_PROCESS_GROUP
    assert kw.get("close_fds") is True


def test_start_daemon_uses_start_new_session_on_posix() -> None:
    """On POSIX we still want start_new_session=True so the daemon
    survives the parent shell's SIGHUP. Verify the platform branch."""
    captured: dict = {}

    class _FakeProc:
        pid = 999

    def fake_popen(args, **kwargs):
        captured["kwargs"] = kwargs
        return _FakeProc()

    with patch.object(check_ollama, "resolve_ollama_binary",
                      return_value=("/usr/local/bin/ollama", "path")), \
         patch.object(check_ollama.sys, "platform", "linux"), \
         patch.object(check_ollama.subprocess, "Popen", side_effect=fake_popen):
        proc = check_ollama._start_daemon_background()

    assert proc is not None
    kw = captured["kwargs"]
    assert kw.get("start_new_session") is True
    assert "creationflags" not in kw  # POSIX has no such concept


def test_start_daemon_swallows_baseexception_from_popen() -> None:
    """If subprocess.Popen raises ANYTHING (including non-Exception
    subclasses that Python 3.12.0 Windows has been seen to emit), we
    must return None and let the caller treat it as degraded -- never
    let the failure escape and kill the boot orchestrator."""
    class WeirdAbort(BaseException):
        """Simulates a non-Exception bubble from native code."""

    def boom(*args, **kwargs):
        raise WeirdAbort("native abort")

    with patch.object(check_ollama, "resolve_ollama_binary",
                      return_value=("/usr/bin/ollama", "path")), \
         patch.object(check_ollama.subprocess, "Popen", side_effect=boom):
        # Must NOT raise even though Popen raised a BaseException.
        assert check_ollama._start_daemon_background() is None


def test_start_daemon_returns_none_when_no_binary() -> None:
    """If resolve_ollama_binary returns None we must short-circuit
    without touching subprocess at all."""
    with patch.object(check_ollama, "resolve_ollama_binary",
                      return_value=(None, "missing")), \
         patch.object(check_ollama.subprocess, "Popen") as popen:
        assert check_ollama._start_daemon_background() is None
        popen.assert_not_called()


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


# ---------------------------------------------------------------------------
# status_snapshot — read-only inventory used by the cockpit GUI
# ---------------------------------------------------------------------------


def test_status_snapshot_daemon_down() -> None:
    """When the daemon is unreachable, every required model is reported as missing."""
    with patch.object(check_ollama, "_daemon_alive", return_value=False):
        snap = check_ollama.status_snapshot(host="http://nope")
    assert snap["daemon_alive"] is False
    assert snap["ready"] is False
    assert snap["installed"] == []
    # Worst-case reporting: required == missing when we can't verify.
    assert snap["missing"] == snap["required"]
    assert isinstance(snap["profile"]["name"], str)


def test_status_snapshot_daemon_up_all_present() -> None:
    """When every required model is installed, ready=True and missing is empty."""
    fake_required = ["deepseek-r1:32b", "qwen2.5:14b"]
    with patch.object(check_ollama, "_daemon_alive", return_value=True), \
         patch.object(check_ollama, "_list_installed", return_value=list(fake_required)), \
         patch.object(check_ollama, "all_models", return_value=fake_required):
        snap = check_ollama.status_snapshot(host="http://x")
    assert snap["daemon_alive"] is True
    assert snap["ready"] is True
    assert snap["missing"] == []
    assert set(snap["installed"]) == set(fake_required)


def test_status_snapshot_daemon_up_some_missing() -> None:
    """Partial install: only truly-missing tags appear in `missing`."""
    with patch.object(check_ollama, "_daemon_alive", return_value=True), \
         patch.object(check_ollama, "_list_installed", return_value=["deepseek-r1:32b"]), \
         patch.object(check_ollama, "all_models", return_value=["deepseek-r1:32b", "qwen2.5:14b"]):
        snap = check_ollama.status_snapshot(host="http://x")
    assert snap["daemon_alive"] is True
    assert snap["ready"] is False
    assert snap["missing"] == ["qwen2.5:14b"]


def test_status_snapshot_handles_list_errors_gracefully() -> None:
    """If /api/tags blows up after the alive-probe says yes, fall back to empty
    installed list — never propagate the exception to the cockpit."""
    import urllib.error

    with patch.object(check_ollama, "_daemon_alive", return_value=True), \
         patch.object(check_ollama, "_list_installed",
                       side_effect=urllib.error.URLError("boom")), \
         patch.object(check_ollama, "all_models", return_value=["deepseek-r1:32b"]):
        snap = check_ollama.status_snapshot(host="http://x")
    assert snap["daemon_alive"] is True
    assert snap["installed"] == []
    assert snap["missing"] == ["deepseek-r1:32b"]
    assert snap["ready"] is False


# ---------------------------------------------------------------------------
# resolve_ollama_binary — picks the right ollama.exe on Windows
# ---------------------------------------------------------------------------
#
# On RX 7700+ machines with the January 2026+ Adrenalin driver, the user
# typically ends up with TWO ollama.exe binaries on PATH:
#
#   1. C:\Users\<user>\AppData\Local\Programs\Ollama\ollama.exe   (standard)
#   2. C:\Users\<user>\AppData\Local\AMD\AI_Bundle\Ollama\ollama.exe (Adrenalin)
#
# We MUST pick the Adrenalin one because its ROCm libs are pre-patched for
# gfx1100/gfx1103/gfx1201 by AMD. PATH-order resolution silently picks the
# wrong one and the cockpit falls back to CPU.


def _fake_localappdata(monkeypatch, root: str) -> None:
    monkeypatch.setenv("LOCALAPPDATA", root)


def test_resolve_prefers_adrenalin_when_both_present(monkeypatch, tmp_path) -> None:
    """The killer scenario: both binaries on disk, Adrenalin must win."""
    _fake_localappdata(monkeypatch, str(tmp_path))
    adren = tmp_path / "AMD" / "AI_Bundle" / "Ollama"
    adren.mkdir(parents=True)
    (adren / "ollama.exe").write_text("")
    std = tmp_path / "Programs" / "Ollama"
    std.mkdir(parents=True)
    (std / "ollama.exe").write_text("")
    monkeypatch.delenv("COCKPIT_OLLAMA_BIN", raising=False)

    path, flavor = check_ollama.resolve_ollama_binary()
    assert flavor == "adrenalin"
    assert path is not None and "AI_Bundle" in path


def test_resolve_falls_back_to_standard_when_only_one(monkeypatch, tmp_path) -> None:
    """Adrenalin not installed - use the standard ollama.com install."""
    _fake_localappdata(monkeypatch, str(tmp_path))
    std = tmp_path / "Programs" / "Ollama"
    std.mkdir(parents=True)
    (std / "ollama.exe").write_text("")
    monkeypatch.delenv("COCKPIT_OLLAMA_BIN", raising=False)

    path, flavor = check_ollama.resolve_ollama_binary()
    assert flavor == "standard"
    assert path is not None and "Programs" in path


def test_resolve_env_override_wins_over_disk(monkeypatch, tmp_path) -> None:
    """COCKPIT_OLLAMA_BIN must beat everything else - it's the user's escape hatch."""
    _fake_localappdata(monkeypatch, str(tmp_path))
    adren = tmp_path / "AMD" / "AI_Bundle" / "Ollama"
    adren.mkdir(parents=True)
    (adren / "ollama.exe").write_text("")

    pinned = tmp_path / "my-custom" / "ollama.exe"
    pinned.parent.mkdir(parents=True)
    pinned.write_text("")
    monkeypatch.setenv("COCKPIT_OLLAMA_BIN", str(pinned))

    path, flavor = check_ollama.resolve_ollama_binary()
    assert flavor == "override"
    assert path == str(pinned)


def test_resolve_env_override_ignored_if_file_missing(monkeypatch, tmp_path) -> None:
    """A stale COCKPIT_OLLAMA_BIN pointing at a deleted file must NOT short-circuit;
    we should fall through to the next resolver tier instead of returning None."""
    _fake_localappdata(monkeypatch, str(tmp_path))
    adren = tmp_path / "AMD" / "AI_Bundle" / "Ollama"
    adren.mkdir(parents=True)
    (adren / "ollama.exe").write_text("")
    monkeypatch.setenv("COCKPIT_OLLAMA_BIN", str(tmp_path / "does-not-exist.exe"))

    path, flavor = check_ollama.resolve_ollama_binary()
    assert flavor == "adrenalin"
    assert path is not None


def test_resolve_falls_back_to_path_on_linux(monkeypatch, tmp_path) -> None:
    """No LOCALAPPDATA and no override - trust whatever's on PATH (Linux/Mac)."""
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    monkeypatch.delenv("COCKPIT_OLLAMA_BIN", raising=False)
    with patch.object(check_ollama.shutil, "which", return_value="/usr/local/bin/ollama"):
        path, flavor = check_ollama.resolve_ollama_binary()
    assert flavor == "path"
    assert path == "/usr/local/bin/ollama"


def test_resolve_returns_missing_when_nothing_found(monkeypatch) -> None:
    """No binary anywhere - report missing, don't crash."""
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    monkeypatch.delenv("COCKPIT_OLLAMA_BIN", raising=False)
    with patch.object(check_ollama.shutil, "which", return_value=None):
        path, flavor = check_ollama.resolve_ollama_binary()
    assert flavor == "missing"
    assert path is None


def test_status_snapshot_includes_binary_metadata(monkeypatch, tmp_path) -> None:
    """The cockpit UI needs to know which binary is in play to surface the
    'AMD Adrenalin' badge - the snapshot must include ollama_binary + flavor."""
    _fake_localappdata(monkeypatch, str(tmp_path))
    adren = tmp_path / "AMD" / "AI_Bundle" / "Ollama"
    adren.mkdir(parents=True)
    (adren / "ollama.exe").write_text("")
    monkeypatch.delenv("COCKPIT_OLLAMA_BIN", raising=False)

    with patch.object(check_ollama, "_daemon_alive", return_value=False), \
         patch.object(check_ollama, "all_models", return_value=["x:1"]):
        snap = check_ollama.status_snapshot(host="http://x")
    assert snap["ollama_flavor"] == "adrenalin"
    assert "AI_Bundle" in (snap["ollama_binary"] or "")
