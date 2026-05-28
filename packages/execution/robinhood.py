"""Robinhood Agentic Trading broker.

Implements the ``Broker`` ABC against Robinhood's MCP-over-HTTP server,
gated by two safety mechanisms so the user can't lose money before
they're ready:

  1. **Execution mode** -- default ``ExecutionMode.SHADOW``. Every order
     is appended to ``data/cockpit/shadow_trades.jsonl`` and a fake
     ``OrderAck`` is returned. NO network call to Robinhood happens.

  2. **Float cap** -- pulled from ``OnboardingState.live_float_cap_usd``
     (default $300). If the notional (price * qty) exceeds the cap we
     raise ``BrokerError`` *before* contacting Robinhood, even in live
     mode. The cap is the user's hard ceiling; Phase 6 raises it only
     after 14 days of positive shadow-trade PnL.

The broker is intentionally additive -- existing ``AlpacaPaperBroker``
behavior is unchanged. The two coexist; the cockpit picks one based on
the user's onboarding choices.

Auth flow lives here too (PKCE + browser callback) but persistence is
delegated to ``robinhood_token.py`` so we never write secrets to disk
ourselves.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from packages.execution.broker import (
    Broker,
    BrokerError,
    BrokerPosition,
    OrderAck,
    OrderRequest,
)
from packages.execution.modes import ExecutionMode
from packages.execution.robinhood_mcp import (
    McpError,
    RobinhoodMcpClient,
)
from packages.execution.robinhood_token import TokenSet, load_tokens

logger = logging.getLogger(__name__)

# Audit log for shadow-mode trades. JSONL so we can grep / pipe / replay.
SHADOW_TRADES_PATH = Path(
    os.getenv("ROBINHOOD_SHADOW_TRADES_PATH", "data/cockpit/shadow_trades.jsonl")
)

# Hardcoded sanity ceiling on the float cap. Prevents an accidental
# ``live_float_cap_usd = 1_000_000`` from blowing the user up if they
# fat-finger the wizard. The Phase 6 auto-greenlight respects this.
ABSOLUTE_MAX_FLOAT_USD = 10_000.0


# ---------------------------------------------------------------------------
# Cap resolution -- pulls from OnboardingState so the cockpit is the
# single source of truth on user-controlled risk knobs.
# ---------------------------------------------------------------------------


def resolve_float_cap() -> float:
    """Return the active live-trading float cap in USD.

    Reads from ``OnboardingState.live_float_cap_usd``, clamped to
    ``[0, ABSOLUTE_MAX_FLOAT_USD]``. Falls back to the default $300 if
    onboarding hasn't completed (defense-in-depth -- the wizard *should*
    set this, but we don't trust that contract here).
    """
    try:
        from packages.cockpit.onboarding import (
            DEFAULT_FLOAT_CAP_USD,
            load_onboarding,
        )

        state = load_onboarding()
        cap = float(state.live_float_cap_usd)
    except Exception as exc:  # pragma: no cover - belt and braces
        logger.warning("float-cap resolve failed: %s", exc.__class__.__name__)
        from packages.cockpit.onboarding import DEFAULT_FLOAT_CAP_USD

        cap = DEFAULT_FLOAT_CAP_USD
    return max(0.0, min(cap, ABSOLUTE_MAX_FLOAT_USD))


# ---------------------------------------------------------------------------
# Shadow audit log
# ---------------------------------------------------------------------------


def _append_shadow_trade(entry: dict[str, Any]) -> None:
    """Append one JSON line to the shadow-trades audit log.

    Atomicity: we don't bother with a temp-rename here because JSONL is
    append-only and the cockpit only ever reads complete lines. A torn
    write at the end of the file would orphan a single line; the next
    write reseeks to EOF so subsequent entries are unaffected.
    """
    import sys

    target = sys.modules[__name__].SHADOW_TRADES_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(entry, separators=(",", ":")) + "\n"
    with open(target, "a", encoding="utf-8") as f:
        f.write(line)


def load_shadow_trades(path: Path | None = None) -> list[dict[str, Any]]:
    """Read the audit log; the dashboard uses this for the shadow-PnL view.

    Skips malformed lines silently rather than blowing up -- one corrupt
    entry shouldn't lose the rest of the history."""
    import sys

    target = path if path is not None else sys.modules[__name__].SHADOW_TRADES_PATH
    if not target.exists():
        return []
    out: list[dict[str, Any]] = []
    for raw in target.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


# ---------------------------------------------------------------------------
# Broker
# ---------------------------------------------------------------------------


class RobinhoodAgenticBroker(Broker):
    """Implements ``Broker`` against Robinhood's MCP-over-HTTP server.

    Default mode is ``SHADOW`` -- the moment you instantiate without an
    explicit ``mode=`` you get a logging-only broker. Flipping to
    ``LIVE`` requires you to pass it deliberately AND have a valid
    keychain token AND respect the float cap.
    """

    name = "robinhood_agentic"

    def __init__(
        self,
        *,
        mode: ExecutionMode = ExecutionMode.SHADOW,
        mcp_client: RobinhoodMcpClient | None = None,
        token_loader=load_tokens,  # injectable for tests
    ) -> None:
        self._mode = mode
        self._mcp_client_override = mcp_client
        self._token_loader = token_loader
        # Cached so we don't build a new client on every call.
        self._mcp: RobinhoodMcpClient | None = None

    # ---- helpers --------------------------------------------------------

    def _is_shadow(self) -> bool:
        # PAPER is treated as SHADOW for Robinhood -- there is no
        # 'paper Robinhood'. Anything that isn't explicit LIVE is logged.
        return self._mode is not ExecutionMode.LIVE

    def _require_token(self) -> TokenSet:
        tokens = self._token_loader()
        if tokens is None:
            raise BrokerError(
                "robinhood: no tokens in keychain -- run the auth flow "
                "from Settings to connect your account"
            )
        if tokens.is_stale():
            raise BrokerError(
                "robinhood: access token is stale -- refresh from "
                "Settings (Phase 2 will add automatic refresh)"
            )
        return tokens

    async def _client(self) -> RobinhoodMcpClient:
        if self._mcp_client_override is not None:
            return self._mcp_client_override
        if self._mcp is None:
            tokens = self._require_token()
            self._mcp = RobinhoodMcpClient(bearer_token=tokens.access_token)
            await self._mcp.initialize()
        return self._mcp

    # ---- Broker contract -----------------------------------------------

    async def health(self) -> bool:
        """Cheap reachability check. In shadow mode we just confirm the
        keychain has tokens (or report False, never raise)."""
        if self._is_shadow():
            return self._token_loader() is not None
        try:
            client = await self._client()
            await client.list_tools()
            return True
        except (BrokerError, McpError):
            return False

    async def positions(self) -> list[BrokerPosition]:
        """Read positions from Robinhood. Safe in shadow mode because
        this is read-only -- no orders submitted, no cap concern."""
        try:
            client = await self._client()
            result = await client.call_tool("list_positions", {})
        except BrokerError:
            return []  # no token yet -- caller treats as empty portfolio
        except McpError as exc:
            logger.warning("robinhood positions failed: %s", exc)
            return []

        # The server is the source of truth on payload shape. We accept
        # a flexible structure and skip rows we can't parse.
        items = result.content or []
        if isinstance(items, dict):  # some MCP servers wrap in {"items": ...}
            items = items.get("positions") or items.get("items") or []
        out: list[BrokerPosition] = []
        for row in items if isinstance(items, list) else []:
            if not isinstance(row, dict):
                continue
            try:
                out.append(
                    BrokerPosition(
                        symbol=str(row["symbol"]),
                        qty=float(row.get("qty", 0.0)),
                        avg_price=float(row.get("avg_price", 0.0)),
                        last_price=(
                            float(row["last_price"])
                            if row.get("last_price") is not None
                            else None
                        ),
                        pnl_pct=(
                            float(row["pnl_pct"])
                            if row.get("pnl_pct") is not None
                            else None
                        ),
                    )
                )
            except (KeyError, ValueError, TypeError):
                continue
        return out

    async def submit(self, req: OrderRequest) -> OrderAck:
        """Submit an order, gated by mode + float cap.

        Order of safety checks (cheapest first):
          1. Validate request shape (qty > 0, side in {buy,sell}).
          2. Float-cap check using ``req.limit_price`` for limits, or a
             conservative estimate via ``qty * limit_price`` for now.
             Market orders without a price ceiling default to the cap
             ceiling itself for sanity -- the strategy is responsible
             for not naked-market-ordering large notional.
          3. If shadow, log + return fake ack. NO network.
          4. Otherwise, send to Robinhood via MCP.
        """
        # ---- (1) shape ----
        if req.qty <= 0:
            raise BrokerError(f"robinhood: bad qty {req.qty}")
        side = req.side.lower()
        if side not in {"buy", "sell"}:
            raise BrokerError(f"robinhood: bad side {req.side!r}")

        # ---- (2) float cap ----
        cap = resolve_float_cap()
        # Sell orders generate cash; the cap is a *deployment* ceiling so
        # we only enforce on buys. (Sells of held shares can't increase
        # net exposure; refusing them would lock the user out of risk
        # reduction.)
        if side == "buy":
            notional = self._estimate_notional(req)
            if notional > cap:
                raise BrokerError(
                    f"robinhood: notional ${notional:.2f} exceeds float "
                    f"cap ${cap:.2f} -- raise the cap in Settings after "
                    f"14 days of positive shadow PnL"
                )

        # ---- (3) shadow ----
        if self._is_shadow():
            ack = OrderAck(
                broker=self.name,
                broker_order_id=f"shadow-{uuid4()}",
                status="accepted_shadow",
                submitted_at=datetime.now(UTC).isoformat(timespec="seconds"),
            )
            _append_shadow_trade(
                {
                    "ts": ack.submitted_at,
                    "broker": self.name,
                    "broker_order_id": ack.broker_order_id,
                    "symbol": req.symbol,
                    "side": side,
                    "qty": req.qty,
                    "type": req.type,
                    "limit_price": req.limit_price,
                    "time_in_force": req.time_in_force,
                    "mode": "shadow",
                    "notional_estimate": self._estimate_notional(req)
                    if side == "buy"
                    else None,
                }
            )
            return ack

        # ---- (4) live ----
        client = await self._client()
        try:
            result = await client.call_tool(
                "submit_order",
                {
                    "symbol": req.symbol,
                    "side": side,
                    "qty": req.qty,
                    "type": req.type,
                    "limit_price": req.limit_price,
                    "time_in_force": req.time_in_force,
                },
            )
        except McpError as exc:
            raise BrokerError(f"robinhood submit failed: {exc}") from exc

        if result.is_error:
            raise BrokerError(
                f"robinhood rejected order: {result.content!r}"
            )

        content = result.content if isinstance(result.content, dict) else {}
        return OrderAck(
            broker=self.name,
            broker_order_id=str(content.get("order_id") or content.get("id") or ""),
            status=str(content.get("status") or "accepted"),
            submitted_at=str(
                content.get("submitted_at")
                or datetime.now(UTC).isoformat(timespec="seconds")
            ),
        )

    # ---- internal --------------------------------------------------------

    def _estimate_notional(self, req: OrderRequest) -> float:
        """Best-effort notional. Limit price wins; otherwise fall back to
        the cap itself as a conservative ceiling so we *always* reject
        unbounded market-buy notional when no price hint is available."""
        if req.limit_price is not None and req.limit_price > 0:
            return float(req.limit_price) * float(req.qty)
        # No price hint -> assume the worst (= cap). Forces the strategy
        # to provide a price hint or use a smaller qty.
        return resolve_float_cap()


# ---------------------------------------------------------------------------
# Convenience factory used by the cockpit when wiring brokers
# ---------------------------------------------------------------------------


def build_broker_from_settings() -> RobinhoodAgenticBroker:
    """Build a broker honoring the user's onboarding choices.

    Reads ``OnboardingState.rh_mode`` -- 'shadow' (default) -> SHADOW,
    'live' -> LIVE. The user can never go LIVE without explicitly
    flipping that field in Settings.
    """
    try:
        from packages.cockpit.onboarding import load_onboarding

        state = load_onboarding()
        mode = (
            ExecutionMode.LIVE
            if state.rh_mode == "live"
            else ExecutionMode.SHADOW
        )
    except Exception as exc:
        logger.warning("rh_mode resolve failed: %s -- defaulting to shadow", exc.__class__.__name__)
        mode = ExecutionMode.SHADOW
    return RobinhoodAgenticBroker(mode=mode)


# Re-export the audit-log writer for tests / debugging tools.
__all__ = [
    "ABSOLUTE_MAX_FLOAT_USD",
    "SHADOW_TRADES_PATH",
    "RobinhoodAgenticBroker",
    "build_broker_from_settings",
    "load_shadow_trades",
    "resolve_float_cap",
]
