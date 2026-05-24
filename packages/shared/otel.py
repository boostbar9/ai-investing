"""OpenTelemetry helpers — every action emits a structured span (v3.1)."""
from __future__ import annotations

import logging
import os
import socket
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from urllib.parse import urlparse

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

log = logging.getLogger(__name__)

_INITIALIZED = False
# Tracks the active exporter status so /api/health and tests can see it.
_EXPORTER_STATUS: dict[str, Any] = {"enabled": False, "endpoint": None, "reason": ""}

# Tunable probe timeout. Kept tiny so a missing collector never adds more
# than a fraction of a second to cockpit startup.
_PROBE_TIMEOUT_S = 0.4


def _collector_reachable(endpoint: str, *, timeout: float = _PROBE_TIMEOUT_S) -> bool:
    """Best-effort TCP probe of the OTLP HTTP endpoint host:port.

    Returns True iff a TCP connect succeeds within ``timeout`` seconds. We
    do NOT validate that the listener is actually an OTLP collector — any
    process willing to accept the connection counts. The point is to avoid
    BatchSpanProcessor retry spam when nobody is listening at all (e.g.
    operator started the cockpit without docker-compose).
    """
    try:
        parsed = urlparse(endpoint if "://" in endpoint else f"http://{endpoint}")
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except Exception:
        return False
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def init_tracing(service_name: str | None = None) -> trace.Tracer:
    """Idempotent OTel init. Safe to call from every entrypoint.

    Decision tree:

    * ``OTEL_SDK_DISABLED=true`` (or ``OTEL_EXPORTER_OTLP_ENDPOINT`` unset):
      tracer is wired up in-process only. No exporter attached. CI / local
      runs stay quiet.
    * Endpoint set but TCP-unreachable: log a single friendly warning and
      skip the exporter so we don't flood logs with retries when the
      operator forgot ``-WithDocker`` on the start script.
    * Endpoint set and reachable: attach :class:`BatchSpanProcessor` and
      ship spans as before.
    """
    global _INITIALIZED
    name = service_name or os.getenv("OTEL_SERVICE_NAME", "ai-investing")
    if _INITIALIZED:
        return trace.get_tracer(name)

    provider = TracerProvider(resource=Resource.create({"service.name": name}))
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    disabled = os.getenv("OTEL_SDK_DISABLED", "").lower() in ("1", "true", "yes")

    if disabled:
        _EXPORTER_STATUS.update({"enabled": False, "endpoint": endpoint, "reason": "OTEL_SDK_DISABLED"})
    elif not endpoint:
        _EXPORTER_STATUS.update({"enabled": False, "endpoint": None, "reason": "endpoint not set"})
    elif not _collector_reachable(endpoint):
        log.warning(
            "OTel collector at %s is unreachable; tracing exporter disabled "
            "(set OTEL_SDK_DISABLED=true to silence this, or start docker-compose to enable).",
            endpoint,
        )
        _EXPORTER_STATUS.update({"enabled": False, "endpoint": endpoint, "reason": "collector unreachable"})
    else:
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint.rstrip("/") + "/v1/traces"))
        )
        _EXPORTER_STATUS.update({"enabled": True, "endpoint": endpoint, "reason": "ok"})

    trace.set_tracer_provider(provider)
    _INITIALIZED = True
    return trace.get_tracer(name)


def exporter_status() -> dict[str, Any]:
    """Snapshot of the last :func:`init_tracing` decision, for diagnostics."""
    return dict(_EXPORTER_STATUS)


@contextmanager
def span(name: str, attrs: dict[str, Any] | None = None) -> Iterator[trace.Span]:
    tracer = init_tracing()
    with tracer.start_as_current_span(name) as s:
        for k, v in (attrs or {}).items():
            s.set_attribute(k, v)
        yield s
