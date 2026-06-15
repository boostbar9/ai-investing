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
        body = json.loads(request.content.decode())
        if "id" not in body:  # notifications/initialized
            return httpx.Response(202)
        captured["headers"] = dict(request.headers)
        captured["body"] = body
        return httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": body["id"],
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
        body = json.loads(request.content.decode())
        if "id" not in body:  # notifications/initialized
            return httpx.Response(202)
        call_count["n"] += 1
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
        if body["method"] == "notifications/initialized":
            return httpx.Response(202)
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

    assert methods == ["initialize", "notifications/initialized", "tools/list"]
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
        if body["method"] == "notifications/initialized":
            return httpx.Response(202)
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
        if body["method"] == "notifications/initialized":
            return httpx.Response(202)
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


# ---------------------------------------------------------------------------
# Streamable HTTP transport: Accept header, session id, notifications, SSE
# ---------------------------------------------------------------------------


def _init_response(request: httpx.Request, **kwargs) -> httpx.Response:
    """Build a minimal initialize result response, passing extra kwargs
    (e.g. headers) straight through to httpx.Response. Notifications (which
    carry no ``id``) get a bare 202 ack."""
    body = json.loads(request.content.decode())
    if "id" not in body:  # notifications/initialized
        return httpx.Response(202)
    return httpx.Response(
        200, json={"jsonrpc": "2.0", "id": body["id"], "result": {}}, **kwargs
    )


@pytest.mark.asyncio
async def test_rpc_sends_accept_header():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = dict(request.headers)
        return _init_response(request)

    client = _make_client(handler)
    try:
        await client.initialize()
    finally:
        await client.aclose()

    assert (
        captured["headers"]["accept"]
        == "application/json, text/event-stream"
    )


@pytest.mark.asyncio
async def test_initialize_captures_and_echoes_session_id():
    """The session id assigned on the initialize response must be echoed
    on every subsequent request (and on notifications/initialized)."""
    seen_session_headers: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        seen_session_headers.append(request.headers.get("mcp-session-id"))
        if body.get("method") == "initialize":
            return _init_response(request, headers={"Mcp-Session-Id": "sess-123"})
        if body.get("method") == "notifications/initialized":
            return httpx.Response(202)
        # tools/list
        return httpx.Response(
            200,
            json={"jsonrpc": "2.0", "id": body["id"], "result": {"tools": []}},
        )

    client = _make_client(handler)
    try:
        await client.initialize()
        assert client._session_id == "sess-123"
        await client.list_tools()
    finally:
        await client.aclose()

    # First request (initialize) carries no session id yet; the
    # notification and tools/list both echo the captured one.
    assert seen_session_headers[0] is None
    assert seen_session_headers[1] == "sess-123"  # notifications/initialized
    assert seen_session_headers[2] == "sess-123"  # tools/list


@pytest.mark.asyncio
async def test_no_session_id_header_when_unset():
    """Before the server assigns a session id, the header must be absent
    entirely -- not sent as empty."""
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = dict(request.headers)
        return _init_response(request)  # no Mcp-Session-Id in response

    client = _make_client(handler)
    try:
        await client.initialize()
    finally:
        await client.aclose()

    assert "mcp-session-id" not in captured["headers"]
    assert client._session_id is None


@pytest.mark.asyncio
async def test_notifications_initialized_sent_after_initialize():
    """A JSON-RPC notification (no id, method notifications/initialized)
    must be POSTed after initialize, and a 202/empty body must not raise."""
    notifications: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        if body.get("method") == "initialize":
            return _init_response(request, headers={"Mcp-Session-Id": "s-9"})
        if body.get("method") == "notifications/initialized":
            notifications.append(
                {"body": body, "session": request.headers.get("mcp-session-id")}
            )
            # Empty 202 -- the canonical notification ack.
            return httpx.Response(202)
        return _init_response(request)

    client = _make_client(handler)
    try:
        await client.initialize()
    finally:
        await client.aclose()

    assert len(notifications) == 1
    note = notifications[0]["body"]
    assert note["method"] == "notifications/initialized"
    assert "id" not in note
    assert notifications[0]["session"] == "s-9"


@pytest.mark.asyncio
async def test_sse_response_parsed_for_tools_list():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        if body.get("method") == "initialize":
            return _init_response(request)
        if body.get("method") == "notifications/initialized":
            return httpx.Response(202)
        rpc = {
            "jsonrpc": "2.0",
            "id": body["id"],
            "result": {"tools": [{"name": "place_equity_order"}]},
        }
        sse = f"event: message\ndata: {json.dumps(rpc)}\n\n"
        return httpx.Response(
            200, text=sse, headers={"Content-Type": "text/event-stream"}
        )

    client = _make_client(handler)
    try:
        tools = await client.list_tools()
    finally:
        await client.aclose()

    assert [t["name"] for t in tools] == ["place_equity_order"]


@pytest.mark.asyncio
async def test_sse_response_parsed_for_call_tool():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        if body.get("method") == "initialize":
            return _init_response(request)
        if body.get("method") == "notifications/initialized":
            return httpx.Response(202)
        rpc = {
            "jsonrpc": "2.0",
            "id": body["id"],
            "result": {
                "content": {"order_id": "rh-7"},
                "isError": False,
            },
        }
        sse = f"data: {json.dumps(rpc)}\n\n"
        return httpx.Response(
            200, text=sse, headers={"Content-Type": "text/event-stream"}
        )

    client = _make_client(handler)
    try:
        out = await client.call_tool("place_equity_order", {"symbol": "SPY"})
    finally:
        await client.aclose()

    assert out.tool == "place_equity_order"
    assert out.is_error is False
    assert out.content["order_id"] == "rh-7"


@pytest.mark.asyncio
async def test_sse_multiple_data_lines_concatenated():
    """Per the SSE spec, multiple data: lines in one event are joined with
    newlines before being interpreted as the message payload."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        if body.get("method") == "initialize":
            return _init_response(request)
        if body.get("method") == "notifications/initialized":
            return httpx.Response(202)
        rpc = {
            "jsonrpc": "2.0",
            "id": body["id"],
            "result": {"tools": [{"name": "get_accounts"}]},
        }
        # Split the JSON across two data: lines; they must be rejoined
        # with a newline to reconstruct valid JSON.
        full = json.dumps(rpc)
        mid = len(full) // 2
        sse = f"data: {full[:mid]}\ndata: {full[mid:]}\n\n"
        return httpx.Response(
            200, text=sse, headers={"Content-Type": "text/event-stream"}
        )

    client = _make_client(handler)
    try:
        tools = await client.list_tools()
    finally:
        await client.aclose()

    assert [t["name"] for t in tools] == ["get_accounts"]


@pytest.mark.asyncio
async def test_sse_no_data_lines_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        if body.get("method") == "initialize":
            # SSE framing with only a comment line -- no data payload.
            return httpx.Response(
                200,
                text=": keep-alive\n\n",
                headers={"Content-Type": "text/event-stream"},
            )
        return httpx.Response(202)

    client = _make_client(handler)
    try:
        with pytest.raises(McpError):
            await client.initialize()
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_plain_json_path_still_works():
    """Regression: the application/json response path is unchanged."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        if body.get("method") == "initialize":
            return _init_response(request)
        if body.get("method") == "notifications/initialized":
            return httpx.Response(202)
        return httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": body["id"],
                "result": {"tools": [{"name": "get_equity_positions"}]},
            },
        )

    client = _make_client(handler)
    try:
        tools = await client.list_tools()
    finally:
        await client.aclose()

    assert [t["name"] for t in tools] == ["get_equity_positions"]
