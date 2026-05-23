"""Push-notification client (§12 mobile, issue #6).

Two-channel design \u2014 the call sites do not care which backend is used:

  - OneSignal (default for v1.0): one REST POST per push.
  - Firebase Cloud Messaging (FCM): future fallback when OneSignal quota
    is exceeded. Stubbed today so the call site is stable.

All sends are idempotent on ``dedupe_key`` and emit OTel spans. When neither
provider is configured (``ONESIGNAL_APP_ID`` / ``FCM_PROJECT_ID`` unset) the
client logs and returns immediately so dev / CI runs do not flake.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import httpx

from packages.shared.otel import span

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PushPayload:
    title: str
    body: str
    url: str | None = None        # deep link into the cockpit
    dedupe_key: str | None = None  # suppresses duplicates server-side
    data: dict[str, Any] | None = None  # arbitrary metadata


class PushError(RuntimeError):
    """Raised when the provider rejects a send (network errors are surfaced)."""


# ---------------------------------------------------------------------------
# OneSignal
# ---------------------------------------------------------------------------


class OneSignalClient:
    """Thin REST wrapper for OneSignal's ``/notifications`` endpoint.

    https://documentation.onesignal.com/reference/create-notification
    """

    def __init__(
        self,
        app_id: str | None = None,
        api_key: str | None = None,
        client: httpx.AsyncClient | None = None,
        base_url: str = "https://api.onesignal.com",
    ) -> None:
        self.app_id = app_id or os.getenv("ONESIGNAL_APP_ID", "")
        self.api_key = api_key or os.getenv("ONESIGNAL_API_KEY", "")
        self.base_url = base_url
        self._client = client or httpx.AsyncClient(timeout=10)

    @property
    def configured(self) -> bool:
        return bool(self.app_id and self.api_key)

    async def send(self, payload: PushPayload, *, segments: list[str] | None = None) -> dict[str, Any]:
        if not self.configured:
            log.info("OneSignal not configured; dropping push %r", payload.title)
            return {"id": None, "skipped": True}

        body: dict[str, Any] = {
            "app_id": self.app_id,
            "headings": {"en": payload.title},
            "contents": {"en": payload.body},
            "included_segments": segments or ["Subscribed Users"],
        }
        if payload.url:
            body["url"] = payload.url
        if payload.dedupe_key:
            # Provider de-dupes identical external_ids within ~24h.
            body["external_id"] = payload.dedupe_key
        if payload.data:
            body["data"] = payload.data

        with span("push.onesignal.send", {"title": payload.title}):
            r = await self._client.post(
                f"{self.base_url}/notifications",
                headers={
                    "Authorization": f"Basic {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=body,
            )
            if r.status_code >= 300:
                raise PushError(f"onesignal {r.status_code}: {r.text[:200]}")
            return r.json()

    async def aclose(self) -> None:
        await self._client.aclose()


# ---------------------------------------------------------------------------
# Composite client \u2014 used everywhere
# ---------------------------------------------------------------------------


class PushClient:
    """Multi-provider push facade. Today: OneSignal.

    Designed so call sites never need to change when we add FCM:

        client = PushClient()
        await client.send(PushPayload(title=\"DD halt\", body=\"...\"))
    """

    def __init__(self, onesignal: OneSignalClient | None = None) -> None:
        self.onesignal = onesignal or OneSignalClient()

    @property
    def configured(self) -> bool:
        return self.onesignal.configured

    async def send(self, payload: PushPayload) -> dict[str, Any]:
        return await self.onesignal.send(payload)

    async def aclose(self) -> None:
        await self.onesignal.aclose()
