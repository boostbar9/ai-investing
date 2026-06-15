"""Tests for deterministic idempotency keys (P0-1) and fill
reconciliation (P0-3).

Idempotency: the same logical order retried must produce the SAME
client_order_id so the broker dedupes instead of double-filling. A
different order (or one with no identity) must NOT collide.

Reconciliation: after a submit we poll order status a bounded number of
times and surface any shortfall (filled < intended) as a structured,
non-silent mismatch. The poll loop must be read-only and bounded.
"""

from __future__ import annotations

import json

import httpx
import pytest

from packages.execution.broker import (
    AlpacaPaperBroker,
    OrderRequest,
    deterministic_client_order_id,
    reconcile_fill_via_poll,
)

# ---------------------------------------------------------------------------
# Deterministic client_order_id
# ---------------------------------------------------------------------------


def test_same_identity_produces_same_id():
    a = deterministic_client_order_id(
        symbol="SPY", side="buy", qty=3, decision_id="dec-1", bar_ts="2026-06-15T10:00"
    )
    b = deterministic_client_order_id(
        symbol="SPY", side="buy", qty=3, decision_id="dec-1", bar_ts="2026-06-15T10:00"
    )
    assert a == b
    assert a.startswith("seer-")


def test_different_qty_changes_id():
    a = deterministic_client_order_id(
        symbol="SPY", side="buy", qty=3, decision_id="dec-1"
    )
    b = deterministic_client_order_id(
        symbol="SPY", side="buy", qty=4, decision_id="dec-1"
    )
    assert a != b


def test_different_side_changes_id():
    a = deterministic_client_order_id(symbol="SPY", side="buy", qty=3, bar_ts="t")
    b = deterministic_client_order_id(symbol="SPY", side="sell", qty=3, bar_ts="t")
    assert a != b


def test_bar_ts_only_is_still_deterministic():
    """No decision_id but a bar_ts is enough to dedupe identical retries
    within the same bar."""
    a = deterministic_client_order_id(symbol="SPY", side="buy", qty=1, bar_ts="t1")
    b = deterministic_client_order_id(symbol="SPY", side="buy", qty=1, bar_ts="t1")
    assert a == b


def test_no_identity_falls_back_to_uuid():
    """With NO decision_id and NO bar_ts we cannot dedupe, so we get a
    fresh (non-equal) id each call -- documented fallback."""
    a = deterministic_client_order_id(symbol="SPY", side="buy", qty=1)
    b = deterministic_client_order_id(symbol="SPY", side="buy", qty=1)
    assert a != b


def test_prefix_is_respected():
    out = deterministic_client_order_id(
        symbol="SPY", side="buy", qty=1, bar_ts="t", prefix="rh"
    )
    assert out.startswith("rh-")


# ---------------------------------------------------------------------------
# Alpaca submit uses the deterministic key (retry dedupe)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_alpaca_submit_sends_deterministic_id_on_retry():
    seen: list[str] = []

    class _T(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/v2/orders") and request.method == "POST":
                body = json.loads(request.content.decode())
                seen.append(body["client_order_id"])
                return httpx.Response(
                    200, content=json.dumps({"id": "o1", "status": "accepted"}).encode()
                )
            return httpx.Response(404)

    client = httpx.AsyncClient(transport=_T(), base_url="http://x")
    broker = AlpacaPaperBroker(key_id="k", secret="s", base_url="http://x", client=client)
    req = OrderRequest(
        symbol="SPY", side="buy", qty=2, decision_id="d-9", bar_ts="2026-06-15"
    )
    try:
        await broker.submit(req)
        await broker.submit(req)  # retry of the SAME logical order
    finally:
        await broker.aclose()
    assert len(seen) == 2
    assert seen[0] == seen[1]  # deterministic -> broker can dedupe


# ---------------------------------------------------------------------------
# Reconciliation core
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reconcile_match():
    async def poll():
        return {"filled_qty": "5", "status": "filled"}

    recon = await reconcile_fill_via_poll(
        poll=poll, broker_order_id="o1", intended_qty=5, max_polls=3, sleep=_no_sleep
    )
    assert recon.matched is True
    assert recon.filled_qty == 5.0
    assert recon.polls == 1  # terminal on first poll


@pytest.mark.asyncio
async def test_reconcile_mismatch_partial_fill():
    async def poll():
        return {"filled_qty": "2", "status": "partially_filled"}

    recon = await reconcile_fill_via_poll(
        poll=poll, broker_order_id="o2", intended_qty=5, max_polls=3, sleep=_no_sleep
    )
    assert recon.matched is False
    assert recon.filled_qty == 2.0
    assert recon.polls == 3  # never terminal -> exhausts the bound


@pytest.mark.asyncio
async def test_reconcile_is_bounded_on_poll_error():
    async def poll():
        raise RuntimeError("transient")

    recon = await reconcile_fill_via_poll(
        poll=poll, broker_order_id="o3", intended_qty=5, max_polls=5, sleep=_no_sleep
    )
    # First poll raises -> we break immediately, never trade, report mismatch.
    assert recon.matched is False
    assert recon.polls == 1


@pytest.mark.asyncio
async def test_alpaca_reconcile_fill_polls_order_status():
    class _T(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            if "/v2/orders/o-7" in request.url.path:
                return httpx.Response(
                    200,
                    content=json.dumps(
                        {"filled_qty": "10", "status": "filled"}
                    ).encode(),
                )
            return httpx.Response(404)

    client = httpx.AsyncClient(transport=_T(), base_url="http://x")
    broker = AlpacaPaperBroker(key_id="k", secret="s", base_url="http://x", client=client)
    try:
        recon = await broker.reconcile_fill("o-7", intended_qty=10, max_polls=2)
        assert recon.matched is True
        assert recon.filled_qty == 10.0
    finally:
        await broker.aclose()


async def _no_sleep(_s: float) -> None:
    return None
