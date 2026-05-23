import pytest

from packages.execution.broker import Broker, BrokerError, BrokerRouter, OrderAck, OrderRequest


class _Healthy(Broker):
    name = "healthy"

    async def health(self) -> bool:
        return True

    async def submit(self, req: OrderRequest) -> OrderAck:
        return OrderAck(broker=self.name, broker_order_id="ok-1", status="accepted", submitted_at="now")


class _Down(Broker):
    name = "down"

    async def health(self) -> bool:
        return False

    async def submit(self, req: OrderRequest) -> OrderAck:
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
