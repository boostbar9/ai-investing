import json

import httpx
import pytest

from packages.execution.broker import (
    AlpacaPaperBroker,
    Broker,
    BrokerError,
    BrokerPosition,
    BrokerRouter,
    OrderAck,
    OrderRequest,
)


class _Healthy(Broker):
    name = "healthy"

    async def health(self) -> bool:
        return True

    async def submit(self, req: OrderRequest) -> OrderAck:
        return OrderAck(broker=self.name, broker_order_id="ok-1", status="accepted", submitted_at="now")

    async def positions(self) -> list[BrokerPosition]:
        return [BrokerPosition(symbol="SPY", qty=10, avg_price=500.0, last_price=510.0, pnl_pct=0.02)]


class _Down(Broker):
    name = "down"

    async def health(self) -> bool:
        return False

    async def submit(self, req: OrderRequest) -> OrderAck:
        raise BrokerError("nope")

    async def positions(self) -> list[BrokerPosition]:
        raise BrokerError("nope")


@pytest.mark.asyncio
async def test_failover_prefers_first_healthy():
    router = BrokerRouter([_Down(), _Healthy()])
    ack = await router.submit(OrderRequest(symbol="SPY", side="buy", qty=1))
    assert ack.broker == "healthy"


@pytest.mark.asyncio
async def test_all_down_raises():
    router = BrokerRouter([_Down(), _Down()])
    with pytest.raises(BrokerError):
        await router.submit(OrderRequest(symbol="SPY", side="buy", qty=1))


def test_router_requires_brokers():
    with pytest.raises(ValueError):
        BrokerRouter([])


@pytest.mark.asyncio
async def test_router_positions_uses_first_healthy():
    router = BrokerRouter([_Down(), _Healthy()])
    ps = await router.positions()
    assert ps[0].symbol == "SPY"
    assert ps[0].pnl_pct == 0.02


@pytest.mark.asyncio
async def test_alpaca_positions_parses_response():
    class _T(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/v2/positions"):
                return httpx.Response(
                    200,
                    json=[
                        {
                            "symbol": "SPY",
                            "qty": "10",
                            "avg_entry_price": "500.0",
                            "current_price": "510.0",
                        },
                        {
                            "symbol": "BROKEN",
                            "qty": "oops",  # parse failure — skipped
                            "avg_entry_price": "1",
                        },
                    ],
                )
            return httpx.Response(404)

    client = httpx.AsyncClient(transport=_T(), base_url="http://x")
    broker = AlpacaPaperBroker(key_id="k", secret="s", base_url="http://x", client=client)
    try:
        ps = await broker.positions()
        assert len(ps) == 1
        assert ps[0].symbol == "SPY"
        assert ps[0].pnl_pct is not None and abs(ps[0].pnl_pct - 0.02) < 1e-9
    finally:
        await broker.aclose()


@pytest.mark.asyncio
async def test_alpaca_positions_raises_on_http_error():
    class _T(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            return httpx.Response(403, content=json.dumps({"error": "nope"}).encode())

    client = httpx.AsyncClient(transport=_T(), base_url="http://x")
    broker = AlpacaPaperBroker(key_id="k", secret="s", base_url="http://x", client=client)
    try:
        with pytest.raises(BrokerError):
            await broker.positions()
    finally:
        await broker.aclose()
