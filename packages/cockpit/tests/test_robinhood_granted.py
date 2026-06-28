"""Tests for the now-implemented Robinhood 'granted' detector.

``_check_granted_via_token`` was a stub returning None; it now exercises
the stored token against the MCP server. These tests mock the broker +
MCP client so no network/keychain is touched.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from packages.cockpit import robinhood_access as ra


def test_returns_none_when_not_connected():
    """No usable token -> fall through to the public discovery probe."""
    with patch(
        "packages.execution.robinhood.is_connected", return_value=False
    ):
        assert ra._check_granted_via_token() is None


def test_granted_when_mcp_handshake_succeeds():
    """A token that authenticates against the MCP server -> granted."""
    fake_client = MagicMock()
    fake_client.initialize = AsyncMock(return_value={})
    fake_client.list_tools = AsyncMock(
        return_value=[{"name": "get_account"}, {"name": "submit_order"}]
    )
    fake_client.aclose = AsyncMock()

    fake_tokens = MagicMock()
    fake_tokens.access_token = "acc"
    fake_broker = MagicMock()
    fake_broker._require_token.return_value = fake_tokens

    with patch(
        "packages.execution.robinhood.is_connected", return_value=True
    ), patch(
        "packages.execution.robinhood.RobinhoodAgenticBroker",
        return_value=fake_broker,
    ), patch(
        "packages.execution.robinhood_mcp.RobinhoodMcpClient",
        return_value=fake_client,
    ):
        result = ra._check_granted_via_token()

    assert result is not None
    assert result.outcome == "granted"
    assert "2 tools" in result.detail


def test_waitlist_when_mcp_rejects_token():
    """Token present but the agentic sub-account isn't provisioned yet
    (MCP 403) -> waitlist, not granted."""
    from packages.execution.robinhood_mcp import McpError

    fake_client = MagicMock()
    fake_client.initialize = AsyncMock(
        side_effect=McpError("mcp initialize returned HTTP 403: forbidden")
    )
    fake_client.aclose = AsyncMock()

    fake_tokens = MagicMock()
    fake_tokens.access_token = "acc"
    fake_broker = MagicMock()
    fake_broker._require_token.return_value = fake_tokens

    with patch(
        "packages.execution.robinhood.is_connected", return_value=True
    ), patch(
        "packages.execution.robinhood.RobinhoodAgenticBroker",
        return_value=fake_broker,
    ), patch(
        "packages.execution.robinhood_mcp.RobinhoodMcpClient",
        return_value=fake_client,
    ):
        result = ra._check_granted_via_token()

    assert result is not None
    assert result.outcome == "waitlist"


def test_falls_through_on_ambiguous_mcp_error():
    """A non-auth MCP error is ambiguous -> return None so the caller
    uses the reachability probe instead of mislabeling."""
    from packages.execution.robinhood_mcp import McpError

    fake_client = MagicMock()
    fake_client.initialize = AsyncMock(
        side_effect=McpError("mcp initialize returned HTTP 500: oops")
    )
    fake_client.aclose = AsyncMock()

    fake_tokens = MagicMock()
    fake_tokens.access_token = "acc"
    fake_broker = MagicMock()
    fake_broker._require_token.return_value = fake_tokens

    with patch(
        "packages.execution.robinhood.is_connected", return_value=True
    ), patch(
        "packages.execution.robinhood.RobinhoodAgenticBroker",
        return_value=fake_broker,
    ), patch(
        "packages.execution.robinhood_mcp.RobinhoodMcpClient",
        return_value=fake_client,
    ):
        assert ra._check_granted_via_token() is None


def test_waitlist_when_token_refresh_fails():
    """is_connected() true but the broker can't produce a usable token
    (refresh failed) -> waitlist with an explanatory detail."""
    from packages.execution.broker import BrokerError

    fake_broker = MagicMock()
    fake_broker._require_token.side_effect = BrokerError("reconnect needed")

    with patch(
        "packages.execution.robinhood.is_connected", return_value=True
    ), patch(
        "packages.execution.robinhood.RobinhoodAgenticBroker",
        return_value=fake_broker,
    ):
        result = ra._check_granted_via_token()

    assert result is not None
    assert result.outcome == "waitlist"
    assert "not usable" in result.detail
