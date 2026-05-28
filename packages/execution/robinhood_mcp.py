"""Thin MCP-over-HTTP client for Robinhood's agentic-trading server.

MCP is JSON-RPC 2.0 over a transport (stdio for desktop hosts, streamable
HTTP for remote servers). Robinhood publishes a remote HTTP server at
https://agent.robinhood.com/mcp/trading.

Why we hand-roll this instead of pulling in a full MCP client library:
  * The Python MCP SDK churns rapidly and pulls in heavy deps we don't
    otherwise need.
  * We only call three methods (``initialize``, ``tools/list``,
    ``tools/call``) -- nothing complex enough to justify the surface.
  * Keeping the implementation small means we can mock httpx in tests
    without wrestling with an SDK.

If Robinhood's contract drifts (new ``tools/call`` argument shape, etc.)
the change lives in *one* place: ``RobinhoodMcpClient.call_tool``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Official endpoint per Robinhood's May 27, 2026 announcement.
DEFAULT_MCP_URL = os.getenv(
    "ROBINHOOD_MCP_URL", "https://agent.robinhood.com/mcp/trading"
)

# Generous default for live trading calls (Robinhood may go through
# market data + risk checks before acking) but still bounded so we
# never wedge a worker forever.
DEFAULT_TIMEOUT_S = 15.0

# Required MCP protocol version per the 2025-06-18 spec, which is what
# Robinhood targets at launch. Bumping this requires testing against
# whatever their server expects -- don't change blindly.
MCP_PROTOCOL_VERSION = "2025-06-18"


class McpError(RuntimeError):
    """Raised for MCP-level failures: HTTP non-2xx, JSON-RPC error
    response, malformed payload. The broker wraps these in BrokerError
    before raising to user code."""


@dataclass
class McpCallResult:
    """One ``tools/call`` response. ``content`` is whatever the server
    returned in the JSON-RPC ``result`` field; we don't try to type it
    here because each Robinhood tool has its own shape."""

    tool: str
    content: Any
    is_error: bool = False


class RobinhoodMcpClient:
    """Async JSON-RPC 2.0 client for an MCP-over-HTTP server.

    Lifecycle:
        client = RobinhoodMcpClient(bearer_token=...)
        await client.initialize()         # one-time handshake
        tools = await client.list_tools() # discover available tools
        ack   = await client.call_tool("submit_order", {...})

    The client does NOT cache anything across calls. The broker layer
    owns retries, token refresh, and shadow-mode interception -- this
    class only knows about JSON-RPC.
    """

    def __init__(
        self,
        bearer_token: str,
        *,
        url: str = DEFAULT_MCP_URL,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._url = url
        self._timeout = timeout_s
        self._bearer = bearer_token
        self._client = client  # injectable for tests
        self._owns_client = client is None
        self._request_id = 0
        # Track whether we've completed the initialize handshake. Some
        # MCP servers (Robinhood included) refuse other calls until
        # initialize has succeeded.
        self._initialized = False

    # ---- internal -------------------------------------------------------

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    async def _rpc(self, method: str, params: dict[str, Any]) -> Any:
        """Send a single JSON-RPC request. Returns the ``result`` field
        on success, raises ``McpError`` on any non-success path."""
        client = await self._ensure_client()
        body = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": method,
            "params": params,
        }
        headers = {
            "Authorization": f"Bearer {self._bearer}",
            "Content-Type": "application/json",
            # MCP-over-HTTP requires this so the server can negotiate.
            "MCP-Protocol-Version": MCP_PROTOCOL_VERSION,
        }
        try:
            r = await client.post(self._url, json=body, headers=headers)
        except httpx.HTTPError as exc:
            raise McpError(f"transport error: {exc.__class__.__name__}") from exc

        if r.status_code >= 400:
            raise McpError(
                f"mcp {method} returned HTTP {r.status_code}: "
                f"{r.text[:200] if r.text else ''}"
            )

        try:
            payload = r.json()
        except ValueError as exc:
            raise McpError(f"mcp {method} returned non-JSON body") from exc

        if "error" in payload:
            err = payload["error"]
            code = err.get("code", "?") if isinstance(err, dict) else "?"
            msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
            raise McpError(f"mcp {method} error {code}: {msg}")

        if "result" not in payload:
            raise McpError(f"mcp {method} returned no result/error field")

        return payload["result"]

    # ---- public --------------------------------------------------------

    async def initialize(self) -> dict[str, Any]:
        """Run the MCP handshake. Must be called once before any other
        method. Returns the server's capabilities dict.

        Subsequent calls are no-ops (we cache success).
        """
        if self._initialized:
            return {"cached": True}
        result = await self._rpc(
            "initialize",
            {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {
                    "name": "ai-investing",
                    "version": "0.1.0",
                },
            },
        )
        self._initialized = True
        return result if isinstance(result, dict) else {}

    async def list_tools(self) -> list[dict[str, Any]]:
        """Return the server's tool catalog. Robinhood exposes things
        like ``submit_order``, ``list_positions``, ``get_account``."""
        if not self._initialized:
            await self.initialize()
        result = await self._rpc("tools/list", {})
        if not isinstance(result, dict):
            return []
        tools = result.get("tools", [])
        return tools if isinstance(tools, list) else []

    async def call_tool(
        self, name: str, arguments: dict[str, Any]
    ) -> McpCallResult:
        """Invoke a tool by name. The caller knows the expected argument
        shape for each Robinhood tool; this method just relays."""
        if not self._initialized:
            await self.initialize()
        result = await self._rpc(
            "tools/call",
            {"name": name, "arguments": arguments},
        )
        if not isinstance(result, dict):
            return McpCallResult(tool=name, content=result)
        return McpCallResult(
            tool=name,
            content=result.get("content"),
            is_error=bool(result.get("isError", False)),
        )

    async def aclose(self) -> None:
        """Close the underlying client if we own it. Safe to call
        multiple times."""
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None
