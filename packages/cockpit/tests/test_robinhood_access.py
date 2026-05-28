"""Tests for the Robinhood agentic-trading access detector.

The detector must never block boot or crash the cockpit, so the suite
deliberately covers timeout, connect-error, and unexpected-status paths.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from packages.cockpit.robinhood_access import (
    ProbeResult,
    _probe_discovery,
    detect_access,
)

# ---------------------------------------------------------------------------
# detect_access: declined short-circuit
# ---------------------------------------------------------------------------


def test_declined_short_circuits_without_network() -> None:
    """If the user already opted out, we MUST NOT hit the network and we
    MUST surface a ``declined`` outcome so the caller doesn't overwrite
    their choice."""
    with patch(
        "packages.cockpit.robinhood_access._probe_discovery"
    ) as probe:
        result = detect_access(declined_already=True)
    probe.assert_not_called()
    assert result.outcome == "declined"


# ---------------------------------------------------------------------------
# detect_access: token branch (Phase 2 stub)
# ---------------------------------------------------------------------------


def test_granted_via_token_branch_returns_granted() -> None:
    """When the token introspection stub is filled in, its result should
    take precedence over the public discovery probe."""
    granted = ProbeResult(outcome="granted", detail="token introspect ok")
    with patch(
        "packages.cockpit.robinhood_access._check_granted_via_token",
        return_value=granted,
    ), patch(
        "packages.cockpit.robinhood_access._probe_discovery"
    ) as probe:
        result = detect_access()
    probe.assert_not_called()
    assert result.outcome == "granted"


# ---------------------------------------------------------------------------
# _probe_discovery: HTTP outcomes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status_code", [200, 204, 301, 404, 503])
def test_probe_returns_waitlist_for_any_response(
    status_code: int,
) -> None:
    """Reachability is what we care about: ANY HTTP response means we're
    not on a black-hole network and Robinhood is up. Bucket = waitlist."""
    mock_response = MagicMock()
    mock_response.status_code = status_code

    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = False
    mock_client.head.return_value = mock_response

    with patch(
        "packages.cockpit.robinhood_access.httpx.Client",
        return_value=mock_client,
    ):
        result = _probe_discovery()

    assert result.outcome == "waitlist"
    assert result.http_status == status_code


def test_probe_timeout_returns_unknown() -> None:
    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = False
    mock_client.head.side_effect = httpx.TimeoutException("slow")

    with patch(
        "packages.cockpit.robinhood_access.httpx.Client",
        return_value=mock_client,
    ):
        result = _probe_discovery(timeout_s=0.1)

    assert result.outcome == "unknown"
    assert "timed out" in result.detail
    assert result.http_status is None


def test_probe_connect_error_returns_offline() -> None:
    """Connect error == 'no internet' == don't scare the user."""
    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = False
    mock_client.head.side_effect = httpx.ConnectError("dns")

    with patch(
        "packages.cockpit.robinhood_access.httpx.Client",
        return_value=mock_client,
    ):
        result = _probe_discovery()

    assert result.outcome == "offline"
    assert result.http_status is None


def test_probe_generic_http_error_returns_unknown() -> None:
    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = False
    mock_client.head.side_effect = httpx.HTTPError("boom")

    with patch(
        "packages.cockpit.robinhood_access.httpx.Client",
        return_value=mock_client,
    ):
        result = _probe_discovery()

    assert result.outcome == "unknown"


# ---------------------------------------------------------------------------
# detect_access: end-to-end with mocked probe
# ---------------------------------------------------------------------------


def test_detect_access_delegates_to_probe_when_no_token() -> None:
    with patch(
        "packages.cockpit.robinhood_access._check_granted_via_token",
        return_value=None,
    ), patch(
        "packages.cockpit.robinhood_access._probe_discovery",
        return_value=ProbeResult(outcome="waitlist", http_status=200),
    ):
        result = detect_access()
    assert result.outcome == "waitlist"
    assert result.http_status == 200
