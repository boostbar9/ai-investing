"""Tests for the Windows NT status code translator in cockpit.proc.

When a background helper like ``ollama_setup`` crashes on Windows, the bare
exit code is gibberish to most operators (``3221225794`` instead of
``STATUS_DLL_INIT_FAILED``). These tests pin the friendly translation so
the cockpit error log stays human-readable.
"""
from __future__ import annotations

from packages.cockpit.proc import describe_exit, exit_hint


def test_describe_exit_dll_init_failed_decodes_user_seen_code() -> None:
    """The exact code Devin saw on Windows: 3221225794 = 0xC0000142."""
    msg = describe_exit("ollama_setup", 3221225794)
    assert "3221225794" in msg
    assert "STATUS_DLL_INIT_FAILED" in msg
    # Operator-actionable hint must appear in the message.
    assert "DLL" in msg


def test_describe_exit_access_violation() -> None:
    msg = describe_exit("ollama_setup", 3221225477)
    assert "STATUS_ACCESS_VIOLATION" in msg


def test_describe_exit_ctrl_c() -> None:
    msg = describe_exit("nightly", 3221225786)
    assert "STATUS_CONTROL_C_EXIT" in msg


def test_describe_exit_unknown_code_falls_through() -> None:
    """Unknown codes must keep the bare integer so we never hide a failure."""
    msg = describe_exit("ollama_setup", 1)
    assert msg == "ollama_setup exited with code 1"


def test_describe_exit_none_is_handled() -> None:
    assert describe_exit("x", None) == "x exited with code None"


def test_exit_hint_returns_dict_for_known_code() -> None:
    hint = exit_hint(3221225794)
    assert hint["exit_status_name"] == "STATUS_DLL_INIT_FAILED"
    assert "DLL" in hint["exit_status_hint"]


def test_exit_hint_empty_for_unknown_code() -> None:
    """Empty dict so the caller can spread it without conditionals."""
    assert exit_hint(1) == {}
    assert exit_hint(None) == {}


def test_exit_hint_mergeable_into_context() -> None:
    """Pattern the proc._watch uses: ``{**other, **exit_hint(rc)}``."""
    base = {"exit_code": 3221225794, "log_file": "x.log"}
    merged = {**base, **exit_hint(3221225794)}
    assert merged["exit_code"] == 3221225794
    assert merged["log_file"] == "x.log"
    assert merged["exit_status_name"] == "STATUS_DLL_INIT_FAILED"
