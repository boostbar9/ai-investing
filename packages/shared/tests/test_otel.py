"""Tests for the OTel exporter decision tree (v3.1).

The goal here is operational: cockpit must never spam its own logs with
exporter retries when the operator forgot to start docker-compose. We
verify each branch of :func:`packages.shared.otel.init_tracing` makes the
correct decision and records a status that diagnostics endpoints can
surface.
"""
from __future__ import annotations

import importlib
import socket
from contextlib import closing

import pytest


def _reload_otel():
    """Re-import the module so each test starts with a clean ``_INITIALIZED``."""
    import packages.shared.otel as otel_mod

    return importlib.reload(otel_mod)


def _free_port() -> int:
    """Grab a TCP port that is *currently* free, then release it.

    There is a tiny race window before the test asks the probe to connect,
    but on a developer box the kernel won't hand the same port out twice
    inside a few milliseconds, so this is reliable enough for unit tests.
    """
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_probe_returns_false_on_closed_port() -> None:
    otel = _reload_otel()
    port = _free_port()
    assert otel._collector_reachable(f"http://127.0.0.1:{port}") is False


def test_probe_returns_true_on_open_port() -> None:
    otel = _reload_otel()
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        assert otel._collector_reachable(f"http://127.0.0.1:{port}") is True


def test_probe_handles_bare_hostport() -> None:
    """Endpoints in env files are sometimes written without a scheme."""
    otel = _reload_otel()
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        assert otel._collector_reachable(f"127.0.0.1:{port}") is True


def test_init_skips_exporter_when_endpoint_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)
    otel = _reload_otel()
    otel.init_tracing("test-svc")
    status = otel.exporter_status()
    assert status["enabled"] is False
    assert status["endpoint"] is None
    assert "not set" in status["reason"]


def test_init_skips_exporter_when_disabled_flag_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:4318")
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    otel = _reload_otel()
    otel.init_tracing("test-svc")
    status = otel.exporter_status()
    assert status["enabled"] is False
    assert status["reason"] == "OTEL_SDK_DISABLED"


def test_init_skips_exporter_when_collector_unreachable(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The killer scenario: a default endpoint with nothing listening.

    We expect exactly one warning (not a retry storm) and ``enabled`` False.
    """
    port = _free_port()
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", f"http://127.0.0.1:{port}")
    monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)
    otel = _reload_otel()
    with caplog.at_level("WARNING", logger="packages.shared.otel"):
        otel.init_tracing("test-svc")
    status = otel.exporter_status()
    assert status["enabled"] is False
    assert status["reason"] == "collector unreachable"
    # Exactly one warning *from our logger*, mentioning the endpoint so the
    # operator can act. (The OTel SDK itself may log a TracerProvider
    # override warning when tests reload the module — we ignore those.)
    warnings = [
        r for r in caplog.records
        if r.levelname == "WARNING" and r.name == "packages.shared.otel"
    ]
    assert len(warnings) == 1
    assert str(port) in warnings[0].getMessage()


def test_init_attaches_exporter_when_collector_reachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", f"http://127.0.0.1:{port}")
        otel = _reload_otel()
        otel.init_tracing("test-svc")
        status = otel.exporter_status()
    assert status["enabled"] is True
    assert status["reason"] == "ok"
    assert f":{port}" in status["endpoint"]


def test_init_tracing_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Second call must not change status or re-warn."""
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    otel = _reload_otel()
    otel.init_tracing("test-svc")
    first = otel.exporter_status()
    otel.init_tracing("test-svc")
    second = otel.exporter_status()
    assert first == second
