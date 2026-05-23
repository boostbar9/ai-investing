"""OpenTelemetry helpers — every action emits a structured span (v3.1)."""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

_INITIALIZED = False


def init_tracing(service_name: str | None = None) -> trace.Tracer:
    """Idempotent OTel init. Safe to call from every entrypoint.

    If ``OTEL_EXPORTER_OTLP_ENDPOINT`` is unset the tracer is still wired up
    (spans are recorded in-process) but no exporter is attached — CI and
    local unit tests stay quiet, production gets full traces.
    """
    global _INITIALIZED
    name = service_name or os.getenv("OTEL_SERVICE_NAME", "ai-investing")
    if not _INITIALIZED:
        provider = TracerProvider(resource=Resource.create({"service.name": name}))
        endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
        if endpoint:
            provider.add_span_processor(
                BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint.rstrip("/") + "/v1/traces"))
            )
        trace.set_tracer_provider(provider)
        _INITIALIZED = True
    return trace.get_tracer(name)


@contextmanager
def span(name: str, attrs: dict[str, Any] | None = None) -> Iterator[trace.Span]:
    tracer = init_tracing()
    with tracer.start_as_current_span(name) as s:
        for k, v in (attrs or {}).items():
            s.set_attribute(k, v)
        yield s
