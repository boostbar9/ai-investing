"""Tests for the flip-event notifier backends and dispatch fan-out."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import patch
from urllib.error import HTTPError

import pytest

from packages.shadow import notifiers as nmod
from packages.shadow.notifiers import (
    NullNotifier,
    WebhookNotifier,
    WindowsToastNotifier,
    _format_toast_lines,
    build_default_notifiers,
    dispatch_flip_event,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


@dataclass
class _Recorder:
    """Notifier stub that records every call without side-effects."""

    name: str = "recorder"
    raise_on_call: Exception | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)

    def notify(self, event: dict[str, Any]) -> dict[str, Any]:
        if self.raise_on_call is not None:
            raise self.raise_on_call
        self.calls.append(event)
        return {"ok": True, "backend": self.name, "delivered": True}


_SAMPLE = {
    "ts": "2026-05-28T19:00:00+00:00",
    "from": "shadow",
    "to": "ready",
    "streak_days": 14,
    "reasons": ["14 of 14 days green"],
}


# ---------------------------------------------------------------------------
# format_toast_lines
# ---------------------------------------------------------------------------


def test_format_toast_lines_includes_streak_and_reasons() -> None:
    title, body = _format_toast_lines(_SAMPLE)
    assert "Greenlight" in title
    assert "14" in body
    assert "14 of 14 days green" in body


def test_format_toast_lines_tolerates_missing_fields() -> None:
    title, body = _format_toast_lines({})
    assert title
    assert "0 day(s)" in body
    # No reasons line when reasons is missing or empty.
    assert "Reasons:" not in body


def test_format_toast_lines_truncates_reasons_to_three() -> None:
    event = {**_SAMPLE, "reasons": ["alpha", "bravo", "charlie", "delta", "echo"]}
    _, body = _format_toast_lines(event)
    assert "alpha; bravo; charlie" in body
    assert "delta" not in body  # truncated
    assert "echo" not in body


# ---------------------------------------------------------------------------
# NullNotifier
# ---------------------------------------------------------------------------


def test_null_notifier_returns_undelivered_ok() -> None:
    result = NullNotifier().notify(_SAMPLE)
    assert result == {"ok": True, "backend": "null", "delivered": False}


# ---------------------------------------------------------------------------
# WindowsToastNotifier
# ---------------------------------------------------------------------------


def test_windows_toast_falls_back_when_no_backend_available() -> None:
    """On a Linux test runner, winrt + win10toast both fail to import."""
    result = WindowsToastNotifier().notify(_SAMPLE)
    assert result["ok"] is True
    assert result["delivered"] is False
    assert "errors" in result
    # Both attempts should have been recorded.
    assert any("winrt" in e for e in result["errors"])
    assert any("win10toast" in e for e in result["errors"])


def test_windows_toast_succeeds_via_winrt_when_available() -> None:
    def fake_winrt(title: str, body: str) -> str:
        assert "Greenlight" in title
        return "winrt"

    with patch.object(WindowsToastNotifier, "_notify_winrt", staticmethod(fake_winrt)):
        result = WindowsToastNotifier().notify(_SAMPLE)
    assert result == {
        "ok": True,
        "backend": "windows_toast",
        "delivered": True,
        "via": "winrt",
    }


def test_windows_toast_falls_back_to_win10toast() -> None:
    def fail(title: str, body: str) -> str:
        raise ImportError("winrt missing")

    def ok(title: str, body: str) -> str:
        return "win10toast"

    with (
        patch.object(WindowsToastNotifier, "_notify_winrt", staticmethod(fail)),
        patch.object(WindowsToastNotifier, "_notify_win10toast", staticmethod(ok)),
    ):
        result = WindowsToastNotifier().notify(_SAMPLE)
    assert result["delivered"] is True
    assert result["via"] == "win10toast"


# ---------------------------------------------------------------------------
# WebhookNotifier
# ---------------------------------------------------------------------------


class _FakeResp:
    def __init__(self, status: int = 200) -> None:
        self.status = status

    def __enter__(self) -> _FakeResp:
        return self

    def __exit__(self, *a: Any) -> None:
        return None


def test_webhook_returns_error_on_empty_url() -> None:
    result = WebhookNotifier(url="").notify(_SAMPLE)
    assert result["ok"] is False
    assert result["error"] == "empty_url"
    assert result["delivered"] is False


def test_webhook_posts_event_payload() -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(req: Any, timeout: float = 0) -> _FakeResp:
        captured["url"] = req.full_url
        captured["data"] = req.data
        captured["headers"] = dict(req.headers)
        captured["timeout"] = timeout
        return _FakeResp(status=204)

    with patch("packages.shadow.notifiers.urllib.request.urlopen", side_effect=fake_urlopen):
        result = WebhookNotifier(url="https://example.com/hook", timeout_seconds=2.5).notify(_SAMPLE)

    assert result == {"ok": True, "backend": "webhook", "delivered": True, "status": 204}
    assert captured["url"] == "https://example.com/hook"
    body = json.loads(captured["data"])
    assert body["kind"] == "shadow_flip"
    assert body["event"] == _SAMPLE
    assert "Greenlight" in body["title"]
    # urllib normalizes header keys.
    ct_key = next(k for k in captured["headers"] if k.lower() == "content-type")
    assert captured["headers"][ct_key] == "application/json"
    assert captured["timeout"] == 2.5


def test_webhook_extra_headers_are_forwarded() -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(req: Any, timeout: float = 0) -> _FakeResp:
        captured["headers"] = dict(req.headers)
        return _FakeResp(status=200)

    with patch("packages.shadow.notifiers.urllib.request.urlopen", side_effect=fake_urlopen):
        notifier = WebhookNotifier(
            url="https://example.com/hook",
            extra_headers={"Authorization": "Bearer xyz"},
        )
        result = notifier.notify(_SAMPLE)
    assert result["delivered"] is True
    auth_key = next(k for k in captured["headers"] if k.lower() == "authorization")
    assert captured["headers"][auth_key] == "Bearer xyz"


def test_webhook_handles_http_error() -> None:
    err = HTTPError("https://x", 503, "boom", hdrs=None, fp=None)  # type: ignore[arg-type]
    with patch("packages.shadow.notifiers.urllib.request.urlopen", side_effect=err):
        result = WebhookNotifier(url="https://example.com/hook").notify(_SAMPLE)
    assert result["ok"] is False
    assert result["delivered"] is False
    assert result["error"] == "HTTP 503"


def test_webhook_handles_network_error() -> None:
    with patch(
        "packages.shadow.notifiers.urllib.request.urlopen",
        side_effect=OSError("conn refused"),
    ):
        result = WebhookNotifier(url="https://example.com/hook").notify(_SAMPLE)
    assert result["ok"] is False
    assert "OSError" in result["error"]


def test_webhook_treats_non_2xx_as_undelivered() -> None:
    with patch(
        "packages.shadow.notifiers.urllib.request.urlopen",
        return_value=_FakeResp(status=302),
    ):
        result = WebhookNotifier(url="https://example.com/hook").notify(_SAMPLE)
    assert result["ok"] is False
    assert result["delivered"] is False
    assert result["status"] == 302


# ---------------------------------------------------------------------------
# dispatch_flip_event
# ---------------------------------------------------------------------------


def test_dispatch_calls_every_notifier() -> None:
    a, b = _Recorder(name="a"), _Recorder(name="b")
    results = dispatch_flip_event(_SAMPLE, [a, b])
    assert len(results) == 2
    assert {r["backend"] for r in results} == {"a", "b"}
    assert a.calls == [_SAMPLE]
    assert b.calls == [_SAMPLE]


def test_dispatch_isolates_failures() -> None:
    boom = _Recorder(name="boom", raise_on_call=RuntimeError("nope"))
    ok = _Recorder(name="ok")
    results = dispatch_flip_event(_SAMPLE, [boom, ok])
    by_backend = {r["backend"]: r for r in results}
    assert by_backend["boom"]["ok"] is False
    assert "RuntimeError" in by_backend["boom"]["error"]
    assert by_backend["ok"]["ok"] is True
    assert ok.calls == [_SAMPLE]


def test_dispatch_falls_back_to_null_when_empty() -> None:
    results = dispatch_flip_event(_SAMPLE, [])
    assert len(results) == 1
    assert results[0]["backend"] == "null"


def test_dispatch_handles_none_as_empty() -> None:
    results = dispatch_flip_event(_SAMPLE, None)
    assert len(results) == 1
    assert results[0]["backend"] == "null"


# ---------------------------------------------------------------------------
# build_default_notifiers
# ---------------------------------------------------------------------------


def test_build_default_includes_toast_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SHADOW_FLIP_WEBHOOK_URL", raising=False)
    monkeypatch.delenv("COCKPIT_DESKTOP_TOAST", raising=False)
    sinks = build_default_notifiers()
    assert any(isinstance(s, WindowsToastNotifier) for s in sinks)
    assert not any(isinstance(s, WebhookNotifier) for s in sinks)


def test_build_default_adds_webhook_when_env_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHADOW_FLIP_WEBHOOK_URL", "https://example.com/hook")
    monkeypatch.setenv("COCKPIT_DESKTOP_TOAST", "1")
    sinks = build_default_notifiers()
    webhooks = [s for s in sinks if isinstance(s, WebhookNotifier)]
    assert len(webhooks) == 1
    assert webhooks[0].url == "https://example.com/hook"


def test_build_default_disables_toast_via_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COCKPIT_DESKTOP_TOAST", "0")
    monkeypatch.delenv("SHADOW_FLIP_WEBHOOK_URL", raising=False)
    sinks = build_default_notifiers()
    assert all(not isinstance(s, WindowsToastNotifier) for s in sinks)
    # Falls back to NullNotifier.
    assert any(isinstance(s, NullNotifier) for s in sinks)


def test_build_default_strips_whitespace_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHADOW_FLIP_WEBHOOK_URL", "   ")
    monkeypatch.setenv("COCKPIT_DESKTOP_TOAST", "0")
    sinks = build_default_notifiers()
    assert all(not isinstance(s, WebhookNotifier) for s in sinks)
    assert any(isinstance(s, NullNotifier) for s in sinks)


# ---------------------------------------------------------------------------
# XML escape helper (defense-in-depth for toast XML body)
# ---------------------------------------------------------------------------


def test_escape_xml_handles_specials() -> None:
    out = nmod._escape_xml("<>&\"'foo")
    assert out == "&lt;&gt;&amp;&quot;&apos;foo"


# Sanity: smoke around env keys to confirm tests aren't leaking state.
def test_env_does_not_leak_between_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SHADOW_FLIP_WEBHOOK_URL", raising=False)
    assert os.environ.get("SHADOW_FLIP_WEBHOOK_URL") is None
