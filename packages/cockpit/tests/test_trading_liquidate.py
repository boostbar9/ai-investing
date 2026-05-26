"""End-to-end tests for the /api/trading/liquidate recovery endpoint.

This endpoint exists to break the deadlock the user hit on 2026-05-25:
$100k equity but $71.86 buying power because old positions held all the
cash. Hitting Alpaca's DELETE /v2/positions atomically frees the cash so
the §16 soak streak can resume. We pause the loop as part of the same
request so the agents can't immediately re-fill.

The tests below patch :class:`AlpacaPaperBroker.liquidate_all` so we never
touch the real network -- we only verify the HTTP wiring (confirm guard,
creds guard, pause side-effect, error mapping).
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from packages.cockpit.web import server as server_module
from packages.cockpit.web.server import app
from packages.execution.broker import BrokerError


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("ALPACA_PAPER_KEY_ID", "key")
    monkeypatch.setenv("ALPACA_PAPER_SECRET", "secret")
    with TestClient(app) as c:
        yield c


class _FakeBroker:
    def __init__(self, *, result: dict[str, Any] | None = None, raises: BaseException | None = None) -> None:
        self._result = result or {
            "cancelled_orders": 2,
            "closed_positions": 3,
            "orders_response": [],
            "positions_response": [],
        }
        self._raises = raises
        self.calls: list[bool] = []

    async def liquidate_all(self, cancel_orders: bool = True) -> dict[str, Any]:
        self.calls.append(cancel_orders)
        if self._raises is not None:
            raise self._raises
        return self._result

    async def aclose(self) -> None:
        pass


def _patch_broker(monkeypatch, fake: _FakeBroker) -> None:
    monkeypatch.setattr(server_module, "AlpacaPaperBroker", lambda: fake)


def test_liquidate_requires_confirm(client):
    r = client.post("/api/trading/liquidate", json={"confirm": False})
    assert r.status_code == 400
    assert "confirm" in r.json()["detail"].lower()


def test_liquidate_requires_paper_creds(client, monkeypatch):
    monkeypatch.delenv("ALPACA_PAPER_KEY_ID", raising=False)
    monkeypatch.delenv("ALPACA_PAPER_SECRET", raising=False)
    r = client.post("/api/trading/liquidate", json={"confirm": True})
    assert r.status_code == 400
    assert "ALPACA_PAPER_KEY_ID" in r.json()["detail"]


def test_liquidate_happy_path_returns_counts_and_pauses(client, monkeypatch):
    fake = _FakeBroker(result={
        "cancelled_orders": 4,
        "closed_positions": 7,
        "orders_response": [],
        "positions_response": [],
    })
    _patch_broker(monkeypatch, fake)
    r = client.post("/api/trading/liquidate", json={"confirm": True})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["closed_positions"] == 7
    assert body["cancelled_orders"] == 4
    assert body["paused"] is True
    # The endpoint should have called liquidate_all exactly once with the
    # default cancel_orders=True.
    assert fake.calls == [True]

    # And the bot should now be paused on the dashboard.
    state = client.get("/api/state").json()
    assert state["control"]["paused"] is True


def test_liquidate_propagates_cancel_orders_flag(client, monkeypatch):
    fake = _FakeBroker()
    _patch_broker(monkeypatch, fake)
    r = client.post(
        "/api/trading/liquidate", json={"confirm": True, "cancel_orders": False}
    )
    assert r.status_code == 200
    assert fake.calls == [False]


def test_liquidate_maps_broker_error_to_502(client, monkeypatch):
    fake = _FakeBroker(raises=BrokerError("alpaca 503: service unavailable"))
    _patch_broker(monkeypatch, fake)
    r = client.post("/api/trading/liquidate", json={"confirm": True})
    assert r.status_code == 502
    assert "alpaca" in r.json()["detail"].lower()
