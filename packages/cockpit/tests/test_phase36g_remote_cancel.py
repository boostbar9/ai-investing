"""Phase 36g — /api/remote/cancel_orders endpoint tests.

Verifies the auth gate, the broker call, error mapping, and state
audit entry. The actual Alpaca HTTP call is stubbed via a fake
``cancel_all_orders`` coroutine on a fake AlpacaPaperBroker class.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from packages.cockpit import state as st
from packages.cockpit.web import remote as remote_mod
from packages.cockpit.web import server as srv
from packages.execution import broker as broker_mod

GOOD_TOKEN = "x" * 32


@pytest.fixture
def fake_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "state.json"
    monkeypatch.setattr(st, "STATE_PATH", path)
    monkeypatch.setattr(st.load_state, "__defaults__", (path,))
    monkeypatch.setattr(st.save_state, "__defaults__", (path,))
    return path


@pytest.fixture
def token_env(monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv(remote_mod.ENV_TOKEN, GOOD_TOKEN)
    return GOOD_TOKEN


@pytest.fixture
def client() -> TestClient:
    return TestClient(srv.app)


def _auth() -> dict[str, str]:
    return {"Authorization": f"Bearer {GOOD_TOKEN}"}


def _install_fake_broker(
    monkeypatch: pytest.MonkeyPatch,
    *,
    result: dict[str, Any] | None = None,
    raise_with: Exception | None = None,
) -> dict[str, int]:
    """Replace AlpacaPaperBroker in the remote module with a fake."""
    calls = {"cancel": 0, "aclose": 0}

    class _FakeBroker:
        def __init__(self, *a, **kw) -> None:
            pass

        async def cancel_all_orders(self) -> dict[str, Any]:
            calls["cancel"] += 1
            if raise_with is not None:
                raise raise_with
            return result or {"cancelled_orders": 0, "orders_response": []}

        async def aclose(self) -> None:
            calls["aclose"] += 1

    # The endpoint imports lazily; patch on the source module so the
    # local import picks up the fake.
    monkeypatch.setattr(broker_mod, "AlpacaPaperBroker", _FakeBroker)
    return calls


def test_cancel_orders_requires_auth(client: TestClient, token_env: str) -> None:
    r = client.post("/api/remote/cancel_orders")  # no auth header
    assert r.status_code == 401


def test_cancel_orders_happy_path(
    client: TestClient,
    token_env: str,
    fake_state: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_fake_broker(
        monkeypatch,
        result={"cancelled_orders": 7, "orders_response": [{"id": f"o{i}"} for i in range(7)]},
    )
    r = client.post("/api/remote/cancel_orders", headers=_auth())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["result"]["cancelled_orders"] == 7
    assert calls["cancel"] == 1
    assert calls["aclose"] == 1, "broker must be closed even on success"
    # State audit entry includes the count for the snapshot timeline.
    assert "7 cancelled" in body["state"]["last_action"]


def test_cancel_orders_zero_when_nothing_open(
    client: TestClient,
    token_env: str,
    fake_state: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_broker(monkeypatch, result={"cancelled_orders": 0, "orders_response": []})
    r = client.post("/api/remote/cancel_orders", headers=_auth())
    assert r.status_code == 200
    assert r.json()["result"]["cancelled_orders"] == 0


def test_cancel_orders_broker_error_maps_to_502(
    client: TestClient,
    token_env: str,
    fake_state: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_fake_broker(
        monkeypatch, raise_with=broker_mod.BrokerError("alpaca 401: bad keys")
    )
    r = client.post("/api/remote/cancel_orders", headers=_auth())
    assert r.status_code == 502
    assert "alpaca 401" in r.json()["detail"]
    # Broker should still be closed even on the error path.
    assert calls["aclose"] == 1


def test_cancel_orders_disabled_when_no_token(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(remote_mod.ENV_TOKEN, raising=False)
    r = client.post("/api/remote/cancel_orders", headers=_auth())
    assert r.status_code == 503
