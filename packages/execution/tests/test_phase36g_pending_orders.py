"""Phase 36g — pending-order guard + cancel-orders helper.

Covers the two new broker methods (``open_orders`` and
``cancel_all_orders``) and confirms each properly maps to the Alpaca
endpoints. The integration with the planner is exercised separately in
``tools/tests/test_paper_trade_pending_guard.py``.
"""
from __future__ import annotations

import httpx
import pytest

from packages.execution.broker import AlpacaPaperBroker, BrokerError


@pytest.mark.asyncio
async def test_open_orders_hits_status_open_endpoint() -> None:
    seen: dict[str, object] = {}

    class _T(httpx.AsyncBaseTransport):
        async def handle_async_request(
            self, request: httpx.Request
        ) -> httpx.Response:
            seen["method"] = request.method
            seen["path"] = request.url.path
            seen["query"] = dict(request.url.params)
            return httpx.Response(
                200,
                json=[
                    {"symbol": "SPY", "side": "buy", "qty": "5", "status": "new"},
                    {"symbol": "NVDA", "side": "sell", "qty": "1", "status": "new"},
                ],
            )

    client = httpx.AsyncClient(transport=_T(), base_url="https://x.invalid")
    b = AlpacaPaperBroker(key_id="k", secret="s", base_url="https://x.invalid", client=client)
    rows = await b.open_orders()
    assert seen["method"] == "GET"
    assert seen["path"] == "/v2/orders"
    # Must filter to open orders only — otherwise we'd see filled history
    # and skip planning forever.
    assert seen["query"]["status"] == "open"
    assert len(rows) == 2
    assert {r["symbol"] for r in rows} == {"SPY", "NVDA"}


@pytest.mark.asyncio
async def test_open_orders_raises_on_http_error() -> None:
    class _T(httpx.AsyncBaseTransport):
        async def handle_async_request(
            self, request: httpx.Request
        ) -> httpx.Response:
            return httpx.Response(500, text="boom")

    client = httpx.AsyncClient(transport=_T(), base_url="https://x.invalid")
    b = AlpacaPaperBroker(key_id="k", secret="s", base_url="https://x.invalid", client=client)
    with pytest.raises(BrokerError):
        await b.open_orders()


@pytest.mark.asyncio
async def test_open_orders_returns_empty_list_when_json_not_list() -> None:
    """Defensive: Alpaca normally returns [], but if a future API quirk
    sends an object we shouldn't blow up — just treat as no orders."""

    class _T(httpx.AsyncBaseTransport):
        async def handle_async_request(
            self, request: httpx.Request
        ) -> httpx.Response:
            return httpx.Response(200, json={"unexpected": "shape"})

    client = httpx.AsyncClient(transport=_T(), base_url="https://x.invalid")
    b = AlpacaPaperBroker(key_id="k", secret="s", base_url="https://x.invalid", client=client)
    rows = await b.open_orders()
    assert rows == []


@pytest.mark.asyncio
async def test_cancel_all_orders_hits_delete_orders_endpoint() -> None:
    seen: dict[str, object] = {}

    class _T(httpx.AsyncBaseTransport):
        async def handle_async_request(
            self, request: httpx.Request
        ) -> httpx.Response:
            seen["method"] = request.method
            seen["path"] = request.url.path
            return httpx.Response(
                207,
                json=[
                    {"id": "o1", "status": 200},
                    {"id": "o2", "status": 200},
                    {"id": "o3", "status": 200},
                ],
            )

    client = httpx.AsyncClient(transport=_T(), base_url="https://x.invalid")
    b = AlpacaPaperBroker(key_id="k", secret="s", base_url="https://x.invalid", client=client)
    result = await b.cancel_all_orders()
    assert seen["method"] == "DELETE"
    assert seen["path"] == "/v2/orders"
    assert result["cancelled_orders"] == 3
    assert isinstance(result["orders_response"], list)


@pytest.mark.asyncio
async def test_cancel_all_orders_does_not_touch_positions() -> None:
    """Critical safety property: cancel must NEVER hit /v2/positions.

    If a future refactor accidentally calls liquidate_all this test
    fails fast — protects user from a one-character typo wiping out
    holdings.
    """
    paths_seen: list[str] = []

    class _T(httpx.AsyncBaseTransport):
        async def handle_async_request(
            self, request: httpx.Request
        ) -> httpx.Response:
            paths_seen.append(request.url.path)
            return httpx.Response(200, json=[])

    client = httpx.AsyncClient(transport=_T(), base_url="https://x.invalid")
    b = AlpacaPaperBroker(key_id="k", secret="s", base_url="https://x.invalid", client=client)
    await b.cancel_all_orders()
    assert paths_seen == ["/v2/orders"]
    assert "/v2/positions" not in paths_seen


@pytest.mark.asyncio
async def test_cancel_all_orders_raises_on_4xx() -> None:
    class _T(httpx.AsyncBaseTransport):
        async def handle_async_request(
            self, request: httpx.Request
        ) -> httpx.Response:
            return httpx.Response(401, text="unauthorized")

    client = httpx.AsyncClient(transport=_T(), base_url="https://x.invalid")
    b = AlpacaPaperBroker(key_id="k", secret="s", base_url="https://x.invalid", client=client)
    with pytest.raises(BrokerError):
        await b.cancel_all_orders()


@pytest.mark.asyncio
async def test_cancel_all_orders_handles_empty_body() -> None:
    """Alpaca returns an empty body when no orders to cancel — must
    not crash with a JSON decode error."""

    class _T(httpx.AsyncBaseTransport):
        async def handle_async_request(
            self, request: httpx.Request
        ) -> httpx.Response:
            return httpx.Response(200, text="")

    client = httpx.AsyncClient(transport=_T(), base_url="https://x.invalid")
    b = AlpacaPaperBroker(key_id="k", secret="s", base_url="https://x.invalid", client=client)
    result = await b.cancel_all_orders()
    assert result["cancelled_orders"] == 0
