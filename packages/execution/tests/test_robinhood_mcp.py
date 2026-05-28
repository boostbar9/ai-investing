"""Tests for the JSON-RPC 2.0 MCP-over-HTTP client.

We mock httpx via a transport so we can assert the *exact* wire format
without ever opening a socket. The Robinhood server contract is what
keeps these tests honest -- if Robinhood changes the protocol version
or required headers, the failure surfaces here first.
"""

from __future__ import annotations

import json

import httpx
import pytest

from packages.execution.robinhood_mcp import (
    MCP_PROTOCOL_VERSION,
    McpError,
    RobinhoodMcpClient,
)


def _make_client(handler) -> RobinhoodMcpClient:
    """Build an MCP client whose underlying httpx routes every request
    through ``handler`` (a callable taking httpx.Request -> httpx.Response).
    """

    class _T(httpx.AsyncBaseTransport):
        async def handle_async_request(
            self, request: httpx.Request
        ) -> httpx.Response:
            return handler(request)

    httpx_client = httpx.AsyncClient(transport=_T(), base_url="http://x")
    return RobinhoodMcpClient(
        bearer_token="test-token",
        url="http://x/mcp",
        client=httpx_client,
    )


# ---------------------------------------------------------------------------
# initialize handshake
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_initialize_sends_required_envelope_and_headers():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content.decode())
        return httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": captured["body"]["id"],
                "result": {"capabilities": {"tools": {}}},
            },
        )

    client = _make_client(handler)
    try:
        result = await client.initialize()
    finally:
        await client.aclose()

    # JSON-RPC envelope
    assert captured["body"]["jsonrpc"] == "2.0"
    assert captured["body"]["method"] == "initialize"
    assert captured["body"]["params"]["protocolVersion"] == MCP_PROTOCOL_VERSION
    assert "clientInfo" in captured["body"]["params"]
    # Headers
    assert captured["headers"]["authorization"] == "Bearer test-token"
    assert captured["headers"]["mcp-protocol-version"] == MCP_PROTOCOL_VERSION
    assert captured["headers"]["content-type"] == "application/json"
    # Result
    assert result == {"capabilities": {"tools": {}}}


@pytest.mark.asyncio
async def test_initialize_is_cached():
    """A second initialize() must not hit the network -- the contract
    is one handshake per client."""
    call_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        body = json.loads(request.content.decode())
        return httpx.Response(
            200, json={"jsonrpc": "2.0", "id": body["id"], "result": {}}
        )

    client = _make_client(handler)
    try:
        await client.initialize()
        again = await client.initialize()
    finally:
        await client.aclose()

    assert call_count["n"] == 1
    assert again == {"cached": True}


# ---------------------------------------------------------------------------
# list_tools / call_tool
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_tools_auto_initializes():
    """Calling list_tools() before initialize() must transparently do
    the handshake first -- callers shouldn't have to remember the
    ordering."""
    methods: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        methods.append(body["method"])
        if body["method"] == "initialize":
            return httpx.Response(
                200, json={"jsonrpc": "2.0", "id": body["id"], "result": {}}
            )
        return httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": body["id"],
                "result": {
                    "tools": [
                        {"name": "submit_order"},
                        {"name": "list_positions"},
                    ]
                },
            },
        )

    client = _make_client(handler)
    try:
        tools = await client.list_tools()
    finally:
        await client.aclose()

    assert methods == ["initialize", "tools/list"]
    assert [t["name"] for t in tools] == ["submit_order", "list_positions"]


@pytest.mark.asyncio
async def test_call_tool_forwards_arguments_and_parses_result():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        if body["method"] == "initialize":
            return httpx.Response(
                200, json={"jsonrpc": "2.0", "id": body["id"], "result": {}}
            )
        captured["params"] = body["params"]
        return httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": body["id"],
                "result": {
                    "content": {"order_id": "rh-42", "status": "accepted"},
                    "isError": False,
                },
            },
        )

    client = _make_client(handler)
    try:
        out = await client.call_tool(
            "submit_order", {"symbol": "SPY", "qty": 1}
        )
    finally:
        await client.aclose()

    assert captured["params"]["name"] == "submit_order"
    assert captured["params"]["arguments"] == {"symbol": "SPY", "qty": 1}
    assert out.tool == "submit_order"
    assert out.is_error is False
    assert out.content["order_id"] == "rh-42"


@pytest.mark.asyncio
async def test_call_tool_propagates_is_error_flag():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        if body["method"] == "initialize":
            return httpx.Response(
                200, json={"jsonrpc": "2.0", "id": body["id"], "result": {}}
            )
        return httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": body["id"],
                "result": {
                    "content": {"reason": "insufficient buying power"},
                    "isError": True,
                },
            },
        )

    client = _make_client(handler)
    try:
        out = await client.call_tool("submit_order", {})
    finally:
        await client.aclose()
    assert out.is_error is True


# ---------------------------------------------------------------------------
# error paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_http_error_raises_mcp_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text='{"error":"unauthorized"}')

    client = _make_client(handler)
    try:
        with pytest.raises(McpError) as excinfo:
            await client.initialize()
    finally:
        await client.aclose()
    assert "401" in str(excinfo.value)


@pytest.mark.asyncio
async def test_json_rpc_error_envelope_raises_mcp_error():
    """A 200 OK with an ``error`` field in the body is still a failure
    per JSON-RPC 2.0 -- we must surface it as McpError."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        return httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": body["id"],
                "error": {"code": -32601, "message": "method not found"},
            },
        )

    client = _make_client(handler)
    try:
        with pytest.raises(McpError) as excinfo:
            await client.initialize()
    finally:
        await client.aclose()
    assert "method not found" in str(excinfo.value)


@pytest.mark.asyncio
async def test_non_json_body_raises_mcp_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="this is not json")

    client = _make_client(handler)
    try:
        with pytest.raises(McpError):
            await client.initialize()
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_transport_error_raises_mcp_error():
    """Network errors must come back as McpError, not raw httpx ones,
    so the broker layer can wrap them in BrokerError cleanly."""

    class _Boom(httpx.AsyncBaseTransport):
        async def handle_async_request(
            self, request: httpx.Request
        ) -> httpx.Response:
            raise httpx.ConnectError("nope")

    httpx_client = httpx.AsyncClient(transport=_Boom(), base_url="http://x")
    client = RobinhoodMcpClient(
        bearer_token="t", url="http://x/mcp", client=httpx_client
    )
    try:
        with pytest.raises(McpError) as excinfo:
            await client.initialize()
    finally:
        await client.aclose()
    assert "transport" in str(excinfo.value).lower()


@pytest.mark.asyncio
async def test_aclose_only_closes_owned_client():
    """If the caller injected an httpx client, aclose() must not close
    it (the caller owns the lifetime)."""

    class _T(httpx.AsyncBaseTransport):
        async def handle_async_request(
            self, request: httpx.Request
        ) -> httpx.Response:
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": 1, "result": {}})

    httpx_client = httpx.AsyncClient(transport=_T(), base_url="http://x")
    client = RobinhoodMcpClient(
        bearer_token="t", url="http://x/mcp", client=httpx_client
    )
    await client.aclose()
    # Caller's client should still be usable.
    assert httpx_client.is_closed is False
    await httpx_client.aclose()
