"""Flip-event notification backends (Phase 8).

When :func:`packages.shadow.notify.detect_flip` emits an upward shadow ->
ready transition, the user wants to know **immediately** -- not on the
next time they happen to open the cockpit. This module owns the
side-effect: turn an in-process :class:`FlipEvent` into a desktop toast
+ a webhook POST (Telegram / Discord / ntfy.sh / etc.).

The design is deliberately lightweight:

* :class:`Notifier` is a Protocol with one method, ``notify(event)``.
* Three concrete backends ship:
    - :class:`NullNotifier` -- logs only, never raises (default).
    - :class:`WindowsToastNotifier` -- best-effort Windows toast; falls
      back to ``NullNotifier`` behavior if neither ``winrt`` nor
      ``win10toast`` is installed. This is the *only* OS-specific bit
      and it is wrapped so a missing dep can never crash boot.
    - :class:`WebhookNotifier` -- generic JSON POST to any URL.
* :func:`dispatch_flip_event` fans an event out to every notifier; one
  failing backend never blocks the others.

The notifier layer is intentionally *pull-driven*: the cockpit's
``flip_notify_loop`` (see :mod:`packages.shadow.notify_loop`) tails the
flip-events JSONL and feeds new rows through here. That keeps the
snapshot writer pure I/O-free and makes redelivery on restart trivial
(the loop owns a cursor file).
"""
from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol + default
# ---------------------------------------------------------------------------


class Notifier(Protocol):
    """Single-method contract for flip-event sinks.

    Concrete backends additionally expose a ``name`` attribute used for
    diagnostic results, but it's not part of the Protocol so frozen
    dataclasses (read-only attrs) remain valid implementations.
    """

    def notify(self, event: dict[str, Any]) -> dict[str, Any]:  # pragma: no cover - protocol
        ...


@dataclass(frozen=True)
class NullNotifier:
    """Default no-op notifier -- logs the event and returns ok=True."""

    name: str = "null"

    def notify(self, event: dict[str, Any]) -> dict[str, Any]:
        log.info("flip event (null sink): %s", event)
        return {"ok": True, "backend": self.name, "delivered": False}


# ---------------------------------------------------------------------------
# Windows desktop toast
# ---------------------------------------------------------------------------


def _format_toast_lines(event: dict[str, Any]) -> tuple[str, str]:
    """Return (title, body) text for a flip-event toast."""
    streak = event.get("streak_days") or 0
    body_lines = [
        f"Shadow soak complete after {streak} day(s).",
        "Open the cockpit to arm live trading.",
    ]
    reasons = event.get("reasons") or []
    if isinstance(reasons, list) and reasons:
        body_lines.append("Reasons: " + "; ".join(str(r) for r in reasons[:3]))
    return ("AI Investing — Greenlight ready", "\n".join(body_lines))


@dataclass(frozen=True)
class WindowsToastNotifier:
    """Best-effort Windows desktop notification.

    Tries ``winrt`` first (the modern WinRT bridge); falls back to
    ``win10toast``; if neither import succeeds we log and report
    ``delivered=False`` -- we *do not* raise, because the soak can be
    running on a Linux box just as easily.
    """

    name: str = "windows_toast"

    def notify(self, event: dict[str, Any]) -> dict[str, Any]:
        title, body = _format_toast_lines(event)
        # Try winrt first.
        backend_used: str | None = None
        err_msgs: list[str] = []
        try:
            backend_used = self._notify_winrt(title, body)
        except Exception as exc:
            err_msgs.append(f"winrt: {type(exc).__name__}: {exc}")
        if backend_used is None:
            try:
                backend_used = self._notify_win10toast(title, body)
            except Exception as exc:
                err_msgs.append(f"win10toast: {type(exc).__name__}: {exc}")
        if backend_used is None:
            log.info("WindowsToastNotifier: no backend available (%s)", "; ".join(err_msgs))
            return {
                "ok": True,
                "backend": self.name,
                "delivered": False,
                "errors": err_msgs,
            }
        return {"ok": True, "backend": self.name, "delivered": True, "via": backend_used}

    @staticmethod
    def _notify_winrt(title: str, body: str) -> str:
        # winrt is an optional Windows-only dep.
        from winrt.windows.data.xml.dom import XmlDocument
        from winrt.windows.ui.notifications import (
            ToastNotification,
            ToastNotificationManager,
        )

        xml = (
            "<toast><visual><binding template='ToastGeneric'>"
            f"<text>{_escape_xml(title)}</text><text>{_escape_xml(body)}</text>"
            "</binding></visual></toast>"
        )
        doc = XmlDocument()
        doc.load_xml(xml)
        notifier = ToastNotificationManager.create_toast_notifier("AI Investing")
        notifier.show(ToastNotification(doc))
        return "winrt"

    @staticmethod
    def _notify_win10toast(title: str, body: str) -> str:
        from win10toast import ToastNotifier

        ToastNotifier().show_toast(title, body, duration=10, threaded=True)
        return "win10toast"


def _escape_xml(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


# ---------------------------------------------------------------------------
# Generic webhook
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WebhookNotifier:
    """POST the event JSON to an arbitrary URL.

    Works for Telegram (``https://api.telegram.org/bot<token>/sendMessage``
    with a ``chat_id`` template), Discord webhooks, ntfy.sh, or any
    custom collector. The body schema is::

        {
          "kind": "shadow_flip",
          "event": {<flip event row>},
          "title": "...",
          "body": "..."
        }

    Callers wanting platform-specific formatting can wrap or subclass.
    """

    url: str
    timeout_seconds: float = 5.0
    name: str = "webhook"
    extra_headers: dict[str, str] = field(default_factory=dict)

    def notify(self, event: dict[str, Any]) -> dict[str, Any]:
        if not self.url:
            return {"ok": False, "backend": self.name, "delivered": False, "error": "empty_url"}
        title, body = _format_toast_lines(event)
        payload = json.dumps(
            {"kind": "shadow_flip", "event": event, "title": title, "body": body},
            sort_keys=True,
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        headers.update(self.extra_headers)
        req = urllib.request.Request(self.url, data=payload, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                status = int(getattr(resp, "status", 200) or 200)
        except urllib.error.HTTPError as exc:
            log.warning("webhook %s HTTP %s: %s", self.url, exc.code, exc.reason)
            return {
                "ok": False,
                "backend": self.name,
                "delivered": False,
                "error": f"HTTP {exc.code}",
            }
        except Exception as exc:
            log.warning("webhook %s failed: %s", self.url, exc)
            return {
                "ok": False,
                "backend": self.name,
                "delivered": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        ok = 200 <= status < 300
        return {
            "ok": ok,
            "backend": self.name,
            "delivered": ok,
            "status": status,
        }


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def dispatch_flip_event(
    event: dict[str, Any],
    notifiers: list[Notifier] | None = None,
) -> list[dict[str, Any]]:
    """Fan ``event`` out to every notifier; never raises.

    Returns the list of per-notifier result dicts so the caller (the
    notify loop) can persist a summary into its cursor file.
    """
    sinks = list(notifiers or [])
    if not sinks:
        sinks = [NullNotifier()]
    results: list[dict[str, Any]] = []
    for n in sinks:
        try:
            res = n.notify(event)
        except Exception as exc:
            log.warning("notifier %s raised: %s", getattr(n, "name", "?"), exc)
            res = {
                "ok": False,
                "backend": getattr(n, "name", "?"),
                "delivered": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        results.append(res)
    return results


# ---------------------------------------------------------------------------
# Env-driven factory
# ---------------------------------------------------------------------------


def build_default_notifiers() -> list[Notifier]:
    """Construct notifiers from environment configuration.

    * ``SHADOW_FLIP_WEBHOOK_URL`` -- if set, add a :class:`WebhookNotifier`.
    * ``COCKPIT_DESKTOP_TOAST`` -- "1" (default) enables
      :class:`WindowsToastNotifier`. The toast notifier itself no-ops
      gracefully on non-Windows / missing-dep systems, so it's safe to
      include by default.

    If neither produces an active backend, falls back to NullNotifier so
    the dispatch path still logs.
    """
    out: list[Notifier] = []
    if os.environ.get("COCKPIT_DESKTOP_TOAST", "1") in ("1", "true", "True"):
        out.append(WindowsToastNotifier())
    webhook_url = (os.environ.get("SHADOW_FLIP_WEBHOOK_URL") or "").strip()
    if webhook_url:
        out.append(WebhookNotifier(url=webhook_url))
    if not out:
        out.append(NullNotifier())
    return out
