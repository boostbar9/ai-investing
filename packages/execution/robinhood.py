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

import contextlib
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid4, uuid5

from packages.execution import daily_notional
from packages.execution.broker import (
    Broker,
    BrokerError,
    BrokerPosition,
    OrderAck,
    OrderRequest,
    deterministic_client_order_id,
    reconcile_fill_via_poll,
)
from packages.execution.modes import ExecutionMode, resolve_mode
from packages.execution.robinhood_mcp import (
    McpError,
    RobinhoodMcpClient,
)
from packages.execution.robinhood_token import (
    OAuthEndpoints,
    TokenSet,
    build_authorize_url,
    clear_client_id,
    clear_pending_auth,
    clear_tokens,
    discover_endpoints,
    exchange_code,
    load_client_id,
    load_pending_auth,
    load_tokens,
    new_pkce_pair,
    new_state,
    refresh_access_token,
    register_client,
    save_client_id,
    save_pending_auth,
    save_tokens,
)

logger = logging.getLogger(__name__)

# Audit log for shadow-mode trades. JSONL so we can grep / pipe / replay.
SHADOW_TRADES_PATH = Path(
    os.getenv("ROBINHOOD_SHADOW_TRADES_PATH", "data/cockpit/shadow_trades.jsonl")
)

# Hardcoded sanity ceiling on the float cap. Prevents an accidental
# ``live_float_cap_usd = 1_000_000`` from blowing the user up if they
# fat-finger the wizard. The Phase 6 auto-greenlight respects this.
ABSOLUTE_MAX_FLOAT_USD = 10_000.0

# Fixed UUIDv5 namespace for converting our deterministic idempotency
# identity into Robinhood's required ``ref_id`` UUID format. Frozen
# forever -- changing it would change every derived ref_id and break
# retry-dedupe against the gateway.
_REF_ID_NAMESPACE = uuid5(NAMESPACE_URL, "https://the-seer.local/robinhood/ref_id")

# Robinhood live trading is gated by the SAME resolve_mode promotion gate
# that guards the Alpaca/ENABLE_LIVE_TRADING path (P0-5). We resolve the
# mode under this strategy key so the cockpit's mode store stays coherent.
ROBINHOOD_STRATEGY_KEY = os.getenv("ROBINHOOD_STRATEGY_KEY", "robinhood_agentic")


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
# Live-promotion gate bridge (P0-5)
# ---------------------------------------------------------------------------


def _live_promotion_passed() -> bool:
    """Best-effort read of the live-readiness verdict.

    Reuses ``packages.backtests.live_promotion.live_readiness_gate`` over
    the cockpit's paper equity curve -- the SAME gate the Alpaca live path
    consults. Fails safe: ANY error (missing curve, import failure, too
    few paper days) returns ``False`` so we never accidentally greenlight
    live trading on an exception.

    Env override ``ROBINHOOD_FORCE_LIVE_GATE=true`` exists ONLY for tests
    and explicit operator override; it cannot enable live without
    ``ENABLE_LIVE_TRADING`` also being set (resolve_mode enforces that).
    """
    forced = os.getenv("ROBINHOOD_FORCE_LIVE_GATE", "").strip().lower()
    if forced in {"true", "1", "yes", "on"}:
        return True
    try:
        import pandas as pd

        from packages.backtests import live_promotion as lp
        from packages.cockpit.web.server import equity_curve_points

        points = equity_curve_points(window=200)
        series = pd.Series([float(p.get("equity", 0.0)) for p in points])
        verdict = lp.live_readiness_gate(series)
        return bool(verdict.ready)
    except Exception as exc:
        logger.warning(
            "live-promotion read failed (%s) -- treating gate as not passed",
            exc.__class__.__name__,
        )
        return False


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
# MCP payload normalization
#
# The MCP spec wraps every ``tools/call`` result in a list of content
# blocks, e.g. ``[{"type": "text", "text": "<json string>"}]`` or a
# structured ``{"type": "json", "json": {...}}``. Robinhood's tools
# return their domain payload inside that envelope, and the exact shape
# varies by tool + server version. These helpers unwrap the envelope so
# the broker's parsing code can treat results as plain dicts / lists
# regardless of how the server framed them.
# ---------------------------------------------------------------------------


def _unwrap_content(content: Any) -> Any:
    """Best-effort unwrap of an MCP content payload to a Python object.

    Handles three layouts:
      1. Already a dict / list -> returned as-is (unless it's a list of
         content blocks, which we drill into).
      2. A list of content blocks ``[{"type": "text", "text": ...}, ...]``
         -> the first block carrying a ``text`` (JSON-decoded when it
         parses) or ``json`` field wins.
      3. A JSON string -> decoded.
    Returns the original value when nothing better can be extracted.
    """
    if content is None:
        return None
    if isinstance(content, str):
        try:
            return json.loads(content)
        except (json.JSONDecodeError, ValueError):
            return content
    if isinstance(content, dict):
        # A single content block masquerading as the whole payload.
        if "json" in content and isinstance(content.get("json"), (dict, list)):
            return content["json"]
        if "text" in content and isinstance(content.get("text"), str):
            return _unwrap_content(content["text"])
        return content
    if isinstance(content, list):
        # MCP content-block list: prefer json blocks, then text blocks.
        for block in content:
            if isinstance(block, dict) and isinstance(
                block.get("json"), (dict, list)
            ):
                return block["json"]
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                unwrapped = _unwrap_content(block["text"])
                if isinstance(unwrapped, (dict, list)):
                    return unwrapped
        # Otherwise it may already be a list of domain rows.
        return content
    return content


def _unwrap_mcp_payload(content: Any) -> Any:
    """Unwrap a Robinhood MCP ``tools/call`` result to its domain payload.

    Single reusable entry point used by every read tool (``get_accounts``,
    ``get_portfolio``, ``get_equity_positions``, order polling). It peels
    TWO layers the agentic-trading server wraps around the real data:

      1. **MCP content blocks** -- ``result.content`` is a list like
         ``[{"type": "text", "text": "<json string>"}]``; the JSON string
         is decoded (handled by :func:`_unwrap_content`).
      2. **Presentation envelope** -- the decoded object nests the domain
         payload under ``"data"`` alongside a human-readable ``"guide"``
         string, e.g. ``{"data": {"accounts": [...]}, "guide": "..."}``.
         We descend into ``"data"`` so callers read ``accounts`` /
         portfolio fields directly instead of finding them missing at the
         top level (the bug that left the account unresolved).

    Returns a dict or list (or the raw unwrapped value when neither layer
    applies, e.g. legacy/test payloads that are already flat)."""
    obj = _unwrap_content(content)
    if isinstance(obj, dict) and isinstance(obj.get("data"), (dict, list)):
        return obj["data"]
    return obj


def _normalize_rows(content: Any, *, keys: tuple[str, ...]) -> list[Any]:
    """Normalize an MCP payload to a list of rows.

    Unwraps the full MCP envelope (content blocks + ``data`` presentation
    wrapper), then if the result is a dict, pulls the first matching
    ``keys`` entry that holds a list. Returns ``[]`` when no list is found.
    """
    obj = _unwrap_mcp_payload(content)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for key in keys:
            val = obj.get(key)
            if isinstance(val, list):
                return val
    return []


def _normalize_obj(content: Any) -> dict[str, Any]:
    """Normalize an MCP payload to a single dict (``{}`` when not a dict)."""
    obj = _unwrap_mcp_payload(content)
    if isinstance(obj, dict):
        return obj
    # Some tools return a single-element list wrapping the object.
    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        return obj[0]
    return {}


def _first_float(obj: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    """Return the first key in ``keys`` that parses to a float, else None."""
    for key in keys:
        if key in obj and obj[key] is not None:
            try:
                return float(obj[key])
            except (TypeError, ValueError):
                continue
    return None


def _shape_hint(content: Any) -> str:
    """Describe the STRUCTURE of an MCP payload without leaking any values.

    Returns something like ``dict(keys=[accounts,results])`` or
    ``list[dict(keys=[account_number,type])]`` so an unexpected empty
    response is debuggable from the cockpit's ``errors`` array without ever
    exposing account numbers, balances, or other PII. Reports the shape
    AFTER envelope-unwrapping so the hint reflects the descended ``data``
    payload, not the presentation wrapper."""
    obj = _unwrap_mcp_payload(content)
    if isinstance(obj, dict):
        return f"dict(keys=[{','.join(sorted(map(str, obj.keys())))}])"
    if isinstance(obj, list):
        if not obj:
            return "list[empty]"
        first = obj[0]
        if isinstance(first, dict):
            return f"list[dict(keys=[{','.join(sorted(map(str, first.keys())))}])]"
        return f"list[{type(first).__name__}]"
    return type(obj).__name__


def _mask_account(number: str) -> str:
    """Mask an account number to its last 4 chars for display (``••••3863``).
    Short/empty numbers degrade safely."""
    s = str(number or "").strip()
    if len(s) <= 4:
        return s
    return f"••••{s[-4:]}"


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
        account_number: str | None = None,
    ) -> None:
        self._mode = mode
        self._mcp_client_override = mcp_client
        self._token_loader = token_loader
        # The agentic-allowed Robinhood account to target. Robinhood's MCP
        # tools require an ``account_number`` and reject trades on the
        # non-agentic accounts, so reads + orders must carry this. ``None``
        # leaves the arg off (read tools degrade gracefully); the live order
        # path refuses to submit without it (fail safe).
        #
        # When not passed explicitly, fall back to the onboarding-stored
        # agentic account so a directly-constructed broker still targets the
        # right account (the factory passes it explicitly). This read never
        # hits the network and fails safe to ``None``.
        if account_number:
            self._account_number: str | None = str(account_number).strip()
        else:
            self._account_number = resolve_agentic_account_number()
        # Cached so we don't build a new client on every call.
        self._mcp: RobinhoodMcpClient | None = None

    def _acct_args(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        """Build a tool-args dict including ``account_number`` when set."""
        args: dict[str, Any] = dict(extra or {})
        if self._account_number:
            args["account_number"] = self._account_number
        return args

    # ---- helpers --------------------------------------------------------

    def _is_shadow(self) -> bool:
        # PAPER is treated as SHADOW for Robinhood -- there is no
        # 'paper Robinhood'. Anything that isn't explicit LIVE is logged.
        # An explicit-LIVE request is ALSO treated as shadow unless the
        # resolve_mode promotion gate clears it (P0-5): the float cap is a
        # blast-radius limiter, not a readiness gate.
        if self._mode is not ExecutionMode.LIVE:
            return True
        return not self._live_gate_clears()

    def _live_gate_clears(self) -> bool:
        """Route the Robinhood live path through the SAME promotion gate
        that guards the Alpaca/ENABLE_LIVE_TRADING path.

        Robinhood live requires BOTH ``ENABLE_LIVE_TRADING=true`` AND a
        passed live-promotion gate. We fail safe: any error resolving the
        gate downgrades to shadow. We never auto-upgrade shadow->live --
        only an explicit ``ExecutionMode.LIVE`` request reaches here.
        """
        try:
            # Pin the strategy mode to LIVE for this resolution so
            # resolve_mode evaluates the gates (it reads get_mode, which
            # defaults to PAPER otherwise). We mirror the operator's
            # explicit LIVE intent here.
            from packages.execution import modes as modes_mod

            modes_mod.set_mode(ROBINHOOD_STRATEGY_KEY, ExecutionMode.LIVE)
            decision = resolve_mode(
                ROBINHOOD_STRATEGY_KEY,
                live_gate_passed=_live_promotion_passed(),
            )
            if decision.effective is not ExecutionMode.LIVE:
                logger.warning(
                    "robinhood live gate downgraded to %s: %s",
                    decision.effective.value,
                    decision.reason,
                )
            return decision.effective is ExecutionMode.LIVE
        except Exception as exc:  # pragma: no cover - fail safe
            logger.warning(
                "robinhood live gate resolution failed (%s) -- staying shadow",
                exc.__class__.__name__,
            )
            return False

    def _require_token(self) -> TokenSet:
        tokens = self._token_loader()
        if tokens is None:
            raise BrokerError(
                "robinhood: no tokens in keychain -- run the auth flow "
                "from Settings to connect your account"
            )
        if tokens.is_stale():
            tokens = self._refresh_or_die(tokens)
        return tokens

    def _refresh_or_die(self, tokens: TokenSet) -> TokenSet:
        """Refresh a stale access token using the stored refresh token.

        On success the rotated token set is persisted and returned. On
        any failure we raise ``BrokerError`` with a 'reconnect' hint --
        the user must re-run the browser flow (refresh tokens can be
        revoked or expire too).
        """
        if not tokens.refresh_token:
            raise BrokerError(
                "robinhood: access token expired and no refresh token "
                "available -- reconnect your account from Settings"
            )
        client_id = load_client_id()
        if not client_id:
            raise BrokerError(
                "robinhood: missing client_id for refresh -- reconnect "
                "your account from Settings"
            )
        try:
            fresh = refresh_access_token(
                tokens.refresh_token, client_id=client_id
            )
        except Exception as exc:
            raise BrokerError(
                f"robinhood: token refresh failed ({exc.__class__.__name__}) "
                "-- reconnect your account from Settings"
            ) from exc
        save_tokens(fresh)
        # Reset the cached MCP client so the next call uses the new token.
        self._mcp = None
        return fresh

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
        this is read-only -- no orders submitted, no cap concern.

        Uses the real Robinhood MCP tool name ``get_equity_positions``
        (the agentic-trading server exposes ``get_equity_positions``,
        not the legacy ``list_positions`` guess).
        """
        try:
            client = await self._client()
            result = await client.call_tool(
                "get_equity_positions", self._acct_args()
            )
        except BrokerError:
            return []  # no token yet -- caller treats as empty portfolio
        except McpError as exc:
            logger.warning("robinhood positions failed: %s", exc)
            return []

        # The server is the source of truth on payload shape. We accept
        # a flexible structure and skip rows we can't parse. MCP wraps
        # tool results in content blocks, so normalize first.
        items = _normalize_rows(
            result.content, keys=("positions", "equity_positions", "items", "results")
        )
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

    async def account_snapshot(self) -> dict[str, Any]:
        """Read a live, read-only snapshot of the connected Robinhood
        account so the AI has real account context (buying power,
        cash, total equity, and current positions) when reasoning
        about the market.

        This is ALWAYS read-only and therefore safe in shadow mode --
        it never submits, reviews, or cancels an order. It calls the
        real Robinhood agentic-trading read tools:

          * ``get_accounts``   -> account list (cash / buying power)
          * ``get_portfolio``  -> portfolio equity + day change
          * ``get_equity_positions`` (via :meth:`positions`)

        Returns a plain dict (never raises) shaped for the cockpit +
        agent context. On any failure the relevant section is left
        empty / ``None`` and ``connected`` reflects whether we have a
        usable token, so the UI can degrade gracefully:

            {
              "connected": bool,
              "mode": "shadow" | "live",
              "as_of": ISO-8601 str,
              "accounts": [ {...}, ... ],
              "portfolio": {...} | None,
              "positions": [ {position dict}, ... ],
              "buying_power": float | None,
              "cash": float | None,
              "total_equity": float | None,
              "errors": [ str, ... ],
            }
        """
        snap: dict[str, Any] = {
            "connected": is_connected(),
            "mode": "shadow" if self._is_shadow() else "live",
            "as_of": datetime.now(UTC).isoformat(timespec="seconds"),
            "accounts": [],
            "account_masked": None,
            "portfolio": None,
            "positions": [],
            "buying_power": None,
            "cash": None,
            "total_equity": None,
            "errors": [],
        }
        if not snap["connected"]:
            return snap

        try:
            client = await self._client()
        except BrokerError as exc:
            # No usable token / refresh failed -- report, don't raise.
            snap["connected"] = False
            snap["errors"].append(f"client: {exc}")
            return snap
        except McpError as exc:  # pragma: no cover -- defensive
            snap["errors"].append(f"client: {exc}")
            return snap

        # ---- accounts (cash + buying power) -----------------------------
        # MUST run first: the agentic-trading server's account-scoped tools
        # (``get_portfolio``, ``get_equity_positions``) require an
        # ``account_number`` argument, so we resolve it from the account
        # list before any downstream call.
        try:
            res = await client.call_tool("get_accounts", {})
            accounts = _normalize_rows(
                res.content, keys=("accounts", "results", "items")
            )
            # Late-bind the agentic account number if we don't have one yet
            # (read-only; never enables trading on its own).
            if not self._account_number and isinstance(accounts, list):
                picked = select_agentic_account(accounts)
                if picked:
                    self._account_number = picked
            if isinstance(accounts, list):
                snap["accounts"] = [a for a in accounts if isinstance(a, dict)]
            if not snap["accounts"]:
                # Valid tool, empty list -- record a keys-only structural
                # diagnostic (never values/secrets) so a shape change is
                # debuggable from the cockpit without re-deploying.
                snap["errors"].append(
                    f"get_accounts: empty; payload shape="
                    f"{_shape_hint(res.content)}"
                )
        except McpError as exc:
            snap["errors"].append(f"get_accounts: {exc}")

        # Expose a masked form of the resolved account so the UI can show
        # *which* account is wired up without leaking the full number.
        if self._account_number:
            snap["account_masked"] = _mask_account(self._account_number)

        # Derive top-line cash / buying power from the first account that
        # exposes them (field names vary across server versions).
        for acct in snap["accounts"]:
            if snap["buying_power"] is None:
                snap["buying_power"] = _first_float(
                    acct, ("buying_power", "buyingPower", "buying_power_usd")
                )
            if snap["cash"] is None:
                snap["cash"] = _first_float(
                    acct, ("cash", "cash_balance", "cashBalance", "uncleared_deposits")
                )

        # ---- portfolio (total equity + day change) ----------------------
        # Real agentic-trading tool name is ``get_portfolio`` (NOT the bare
        # ``portfolio`` we guessed before, which 404'd as an unknown tool).
        # Carries ``account_number`` via _acct_args now that it's resolved.
        try:
            res = await client.call_tool("get_portfolio", self._acct_args())
            portfolio = _normalize_obj(res.content)
            if portfolio:
                snap["portfolio"] = portfolio
                snap["total_equity"] = _first_float(
                    portfolio,
                    (
                        "equity",
                        "total_equity",
                        "market_value",
                        "extended_hours_equity",
                        "portfolio_value",
                    ),
                )
                if snap["buying_power"] is None:
                    snap["buying_power"] = _first_float(
                        portfolio,
                        ("buying_power", "buyingPower", "buying_power_usd"),
                    )
                if snap["cash"] is None:
                    snap["cash"] = _first_float(
                        portfolio,
                        ("cash", "cash_balance", "cashBalance", "cash_available"),
                    )
        except McpError as exc:
            snap["errors"].append(f"get_portfolio: {exc}")

        # ---- positions (reuse the parsed BrokerPosition path) -----------
        try:
            snap["positions"] = [p.to_dict() for p in await self.positions()]
        except Exception as exc:  # pragma: no cover -- positions() is defensive
            snap["errors"].append(f"positions: {exc}")

        return snap

    # ---- read-only market-data tools (safe in shadow) -------------------
    #
    # These back the Robinhood-realistic paper simulator
    # (``packages/execution/robinhood_paper.py``). They are STRICTLY
    # read-only: each calls a Robinhood read tool and NEVER touches
    # ``place_equity_order`` / ``cancel_*``. Every one fails safe -- any
    # error or missing field returns ``None`` / a "not known" result so a
    # caller can skip (it must never fabricate a price or treat a missing
    # quote as a fill).

    async def equity_quote(self, symbol: str) -> dict[str, Any] | None:
        """Live bid/ask/last for ``symbol`` via the ``get_equity_quotes``
        read tool. Returns ``{"symbol","bid","ask","last","mid"}`` (floats,
        any of which may be ``None`` if the server omitted it) or ``None``
        on any failure / unparseable payload. Never raises, never trades.

        FAIL SAFE: a missing quote yields ``None`` so the simulator skips
        the order rather than inventing a price.
        """
        sym = str(symbol or "").strip().upper()
        if not sym:
            return None
        try:
            client = await self._client()
            res = await client.call_tool(
                "get_equity_quotes", self._acct_args({"symbols": [sym]})
            )
        except (BrokerError, McpError) as exc:
            logger.warning(
                "robinhood quote failed for %s: %s", sym, exc.__class__.__name__
            )
            return None

        rows = _normalize_rows(
            res.content, keys=("quotes", "results", "items", "data")
        )
        row: dict[str, Any] | None = None
        for r in rows:
            if not isinstance(r, dict):
                continue
            rsym = str(r.get("symbol") or r.get("instrument") or "").upper()
            if not rsym or rsym == sym:
                row = r
                break
        if row is None:
            obj = _normalize_obj(res.content)
            row = obj or None
        if not isinstance(row, dict):
            return None

        bid = _first_float(row, ("bid_price", "bid", "bidPrice"))
        ask = _first_float(row, ("ask_price", "ask", "askPrice"))
        last = _first_float(
            row,
            (
                "last_trade_price",
                "last_price",
                "last",
                "lastTradePrice",
                "price",
                "mark_price",
            ),
        )
        bid_size = _first_float(row, ("bid_size", "bidSize", "bid_quantity"))
        ask_size = _first_float(row, ("ask_size", "askSize", "ask_quantity"))
        mid: float | None = None
        if bid is not None and ask is not None and bid > 0 and ask > 0:
            mid = (bid + ask) / 2.0
        if bid is None and ask is None and last is None and mid is None:
            return None
        return {
            "symbol": sym,
            "bid": bid,
            "ask": ask,
            "last": last,
            "mid": mid,
            "bid_size": bid_size,
            "ask_size": ask_size,
        }

    async def equity_tradability(self, symbol: str) -> dict[str, Any]:
        """Tradability/fractional rules for ``symbol`` via the
        ``get_equity_tradability`` read tool. Returns
        ``{"tradable","fractional","known"}``.

        ``known`` is ``False`` when the call fails or omits the flag -- in
        that case the caller MUST NOT block trading on a read failure (we
        only ever skip on an *explicit* untradable flag), but it may decline
        fractional sizing when fractional support is unknown.
        """
        out = {"tradable": True, "fractional": False, "known": False}
        sym = str(symbol or "").strip().upper()
        if not sym:
            return out
        try:
            client = await self._client()
            res = await client.call_tool(
                "get_equity_tradability", self._acct_args({"symbols": [sym]})
            )
        except (BrokerError, McpError) as exc:
            logger.warning(
                "robinhood tradability failed for %s: %s",
                sym,
                exc.__class__.__name__,
            )
            return out

        rows = _normalize_rows(
            res.content, keys=("tradability", "results", "items", "data")
        )
        row: dict[str, Any] | None = None
        for r in rows:
            if isinstance(r, dict):
                rsym = str(r.get("symbol") or r.get("instrument") or "").upper()
                if not rsym or rsym == sym:
                    row = r
                    break
        if row is None:
            row = _normalize_obj(res.content) or None
        if not isinstance(row, dict):
            return out

        tradable_raw = None
        for key in ("tradable", "tradeable", "is_tradable", "tradability"):
            if key in row and row[key] is not None:
                tradable_raw = row[key]
                break
        if isinstance(tradable_raw, str):
            tradable = tradable_raw.strip().lower() not in {
                "false", "untradable", "halted", "no", "0", "inactive"
            }
        elif tradable_raw is not None:
            tradable = bool(tradable_raw)
        else:
            tradable = True  # field absent -> don't block on a read gap

        frac_raw = None
        for key in (
            "fractional",
            "fractional_tradable",
            "fractionalTradable",
            "fractional_eligible",
        ):
            if key in row and row[key] is not None:
                frac_raw = row[key]
                break
        fractional = bool(frac_raw) if frac_raw is not None else False

        return {
            "tradable": tradable,
            "fractional": fractional,
            "known": tradable_raw is not None,
        }

    async def review_order(
        self,
        *,
        symbol: str,
        side: str,
        qty: float,
        order_type: str = "market",
        limit_price: float | None = None,
    ) -> dict[str, Any] | None:
        """READ-ONLY pre-trade review via ``review_equity_order``.

        This anchors the simulator's pricing/acceptance to Robinhood's own
        response WITHOUT placing anything. It calls ``review_equity_order``
        (a validation/preview tool) -- NEVER ``place_equity_order``. Returns
        the server's review dict, or ``None`` on any failure. Callers
        rate-limit this (it is opt-in) so it doesn't slow the loop.
        """
        sym = str(symbol or "").strip().upper()
        if not sym or qty <= 0:
            return None
        args = self._acct_args(
            {
                "symbol": sym,
                "side": str(side).lower(),
                "quantity": float(qty),
                "type": str(order_type).lower(),
            }
        )
        if limit_price is not None:
            args["limit_price"] = float(limit_price)
        try:
            client = await self._client()
            res = await client.call_tool("review_equity_order", args)
        except (BrokerError, McpError) as exc:
            logger.warning(
                "robinhood order-review failed for %s: %s",
                sym,
                exc.__class__.__name__,
            )
            return None
        obj = _normalize_obj(res.content)
        return obj or None

    async def aclose(self) -> None:
        """Close the cached MCP client if we own one. Safe to call
        multiple times and when no client was ever built. The injected
        override client is left alone -- its owner is responsible for it.
        """
        if self._mcp is not None:
            with contextlib.suppress(Exception):
                await self._mcp.aclose()
            self._mcp = None

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

        is_shadow = self._is_shadow()

        # ---- (2) float cap ----
        cap = resolve_float_cap()
        # Sell orders generate cash; the cap is a *deployment* ceiling so
        # we only enforce on buys. (Sells of held shares can't increase
        # net exposure; refusing them would lock the user out of risk
        # reduction.)
        notional = self._estimate_notional(req) if side == "buy" else 0.0
        if side == "buy":
            if notional > cap:
                raise BrokerError(
                    f"robinhood: notional ${notional:.2f} exceeds float "
                    f"cap ${cap:.2f} -- raise the cap in Settings after "
                    f"14 days of positive shadow PnL"
                )
            # Cumulative daily-notional gate (P0-4). Shadow buys are
            # recorded but NEVER blocked; only a *live* buy is rejected
            # when today's aggregate would breach the cap.
            if not is_shadow:
                exceeds, projected = daily_notional.would_exceed_cap(notional, cap)
                if exceeds:
                    raise BrokerError(
                        f"robinhood: today's deployed buy notional "
                        f"${projected:.2f} (incl. this ${notional:.2f} order) "
                        f"exceeds daily float cap ${cap:.2f} -- the cap is a "
                        f"per-day aggregate, not per-order"
                    )

        # ---- (3) shadow ----
        if is_shadow:
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
                    "notional_estimate": notional if side == "buy" else None,
                }
            )
            if side == "buy":
                daily_notional.record_buy(
                    symbol=req.symbol, notional=notional, mode="shadow"
                )
            return ack

        # ---- (4) live ----
        # Fail safe: Robinhood rejects orders that don't target the
        # agentic-allowed account, so refuse to submit a live order without
        # a resolved account_number rather than letting it bounce at the
        # gateway (or worse, hit a wrong account). Reads above degrade
        # gracefully; only the order path is hard-gated.
        if not self._account_number:
            raise BrokerError(
                "robinhood: no agentic account_number resolved -- connect "
                "and select your Agentic account before live trading"
            )
        client = await self._client()
        # Deterministic idempotency key (P0-1): a retried logical order
        # produces the SAME key so Robinhood dedupes instead of
        # double-filling.
        idempotency_key = deterministic_client_order_id(
            symbol=req.symbol,
            side=side,
            qty=req.qty,
            decision_id=req.decision_id,
            bar_ts=req.bar_ts,
            prefix="rh",
        )
        # ref_id is Robinhood's CONFIRMED idempotency field (optional UUID,
        # verified against a live authenticated session): re-send the SAME
        # ref_id on transient retries of one logical order, a new one only
        # for a new order. Our deterministic key is a sha256-derived
        # ``rh-<hex>`` string, NOT a UUID, so we fold it through uuid5 over
        # a fixed namespace -- that stays deterministic (same identity ->
        # same UUID, distinct identity -> distinct UUID) while satisfying
        # the UUID-format requirement.
        ref_id = str(uuid5(_REF_ID_NAMESPACE, idempotency_key))
        try:
            result = await client.call_tool(
                "place_equity_order",
                self._acct_args(
                    {
                        "symbol": req.symbol,
                        "side": side,
                        "qty": req.qty,
                        "type": req.type,
                        "limit_price": req.limit_price,
                        "time_in_force": req.time_in_force,
                        "ref_id": ref_id,
                    }
                ),
            )
        except McpError as exc:
            raise BrokerError(f"robinhood submit failed: {exc}") from exc

        if result.is_error:
            raise BrokerError(
                f"robinhood rejected order: {result.content!r}"
            )

        content = _normalize_obj(result.content)
        ack = OrderAck(
            broker=self.name,
            broker_order_id=str(content.get("order_id") or content.get("id") or ""),
            status=str(content.get("status") or "accepted"),
            submitted_at=str(
                content.get("submitted_at")
                or datetime.now(UTC).isoformat(timespec="seconds")
            ),
        )
        if side == "buy":
            daily_notional.record_buy(
                symbol=req.symbol, notional=notional, mode="live"
            )
        return ack

    # ---- fill reconciliation (P0-3) -------------------------------------

    async def reconcile_fill(
        self,
        broker_order_id: str,
        intended_qty: float,
        *,
        max_polls: int = 5,
        delay_s: float = 1.0,
    ) -> dict[str, Any]:
        """Poll Robinhood for an order's fill status and surface mismatches.

        Reads ``get_equity_order`` a BOUNDED number of times until the
        order reaches a terminal state or we exhaust ``max_polls``. Returns
        a structured result describing filled vs intended qty. NEVER places
        an order -- read-only by construction. Logs a structured warning on
        any shortfall so a partial fill can't silently pass.

        Safe in shadow mode: returns a synthetic 'matched' result without
        touching the network (shadow fills are assumed complete).
        """
        if self._is_shadow():
            return {
                "broker_order_id": broker_order_id,
                "intended_qty": float(intended_qty),
                "filled_qty": float(intended_qty),
                "status": "shadow",
                "matched": True,
                "polls": 0,
            }
        recon = await reconcile_fill_via_poll(
            poll=lambda: self._poll_equity_order(broker_order_id),
            broker_order_id=broker_order_id,
            intended_qty=intended_qty,
            max_polls=max_polls,
            delay_s=delay_s,
        )
        return recon.to_dict()

    async def _poll_equity_order(self, broker_order_id: str) -> dict[str, Any]:
        """Fetch one order snapshot via the ``get_equity_order`` MCP tool.
        Returns a dict with at least ``filled_qty`` and ``status`` keys
        (best-effort -- shapes vary; missing fields default safely)."""
        client = await self._client()
        try:
            result = await client.call_tool(
                "get_equity_order",
                self._acct_args({"order_id": broker_order_id}),
            )
        except McpError as exc:
            raise BrokerError(f"robinhood get_equity_order failed: {exc}") from exc
        return _normalize_obj(result.content)

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
# OAuth browser flow (loopback, PKCE) -- the "Connect your agent" path
# ---------------------------------------------------------------------------
#
# Robinhood's MCP server uses the OAuth 2.1 authorization-code-with-PKCE
# native-app flow (RFC 8252). The cockpit:
#   1. begin_auth()    -> discover endpoints, register client, build the
#                         authorize URL, and stash the in-flight verifier +
#                         state BOTH in memory AND in a short-lived encrypted
#                         file (see below for why on-disk).
#   2. user opens the URL, approves, Robinhood redirects to our loopback
#      redirect_uri (http://localhost:PORT/callback?code=...&state=...).
#   3. complete_auth() -> verify state (memory first, then the encrypted
#                         file), exchange code+verifier for tokens, persist
#                         tokens + client_id, then DELETE the pending-auth
#                         file (single-use consumption).
#
# Why persist the pending-auth (a deliberate, documented tradeoff):
#   The cockpit launcher (tools/start_cockpit.ps1) runs the web server under
#   an AUTO-RESTART loop. The process can restart between begin_auth and the
#   /callback redirect, wiping the in-memory ``_PENDING_AUTH`` global -> every
#   callback then fails state validation ("OAuth state mismatch"). Keeping the
#   verifier in memory only (the OAuth 2.1 / RFC 7636 ideal) is therefore not
#   workable here. We persist the blob with the SAME encrypted-file store as
#   the tokens (Fernet, key in keyring) and bound the exposure with: a short
#   TTL (PENDING_AUTH_TTL_S), encryption at rest, 0600 perms, and single-use
#   deletion on consumption. The cockpit owns the actual HTTP listener (it
#   already runs a web server). There is at most one auth flow in flight.


# Max age of a persisted pending-auth before it's treated as expired (and
# its replay window closed). 10 minutes comfortably covers a human approving
# in the browser while keeping the CSRF/replay exposure small.
PENDING_AUTH_TTL_S = 600


@dataclass
class PendingAuth:
    """State for one in-flight authorization.

    Held in the module-level ``_PENDING_AUTH`` global AND mirrored to a
    short-lived encrypted file (``data/cockpit/.rh_pending_auth.enc``) so the
    flow survives a cockpit server auto-restart between ``begin_auth`` and the
    ``/callback`` redirect. ``created_at`` (unix ts) stamps when the flow
    began so ``complete_auth`` can expire a stale blob (TTL)."""

    state: str
    code_verifier: str
    client_id: str
    endpoints: OAuthEndpoints
    redirect_uri: str
    authorize_url: str
    created_at: float = 0.0


# Module-level holder for the single in-flight flow. Set by begin_auth,
# consumed + cleared by complete_auth. Mirrored to an encrypted file so a
# server restart between begin and callback doesn't drop the flow.
_PENDING_AUTH: PendingAuth | None = None


def _serialize_pending(pending: PendingAuth) -> str:
    """Serialize a ``PendingAuth`` to the JSON string we persist on disk.

    ``OAuthEndpoints`` is flattened to a dict so it rehydrates cleanly; the
    code_verifier is included because ``complete_auth`` needs it for the PKCE
    exchange (the file is encrypted + 0600 + single-use + TTL'd)."""
    return json.dumps(
        {
            "state": pending.state,
            "code_verifier": pending.code_verifier,
            "client_id": pending.client_id,
            "redirect_uri": pending.redirect_uri,
            "endpoints": pending.endpoints.to_dict(),
            "authorize_url": pending.authorize_url,
            "created_at": pending.created_at,
        },
        separators=(",", ":"),
    )


def _deserialize_pending(raw: str) -> PendingAuth | None:
    """Rehydrate a ``PendingAuth`` from a persisted JSON string, or ``None`` if
    the blob is malformed (never raises -- a corrupt file degrades to 'no
    pending flow')."""
    try:
        data = json.loads(raw)
        eps = data["endpoints"]
        endpoints = OAuthEndpoints(
            issuer=str(eps.get("issuer", "")),
            authorization_endpoint=str(eps["authorization_endpoint"]),
            token_endpoint=str(eps["token_endpoint"]),
            registration_endpoint=str(eps.get("registration_endpoint", "")),
        )
        return PendingAuth(
            state=str(data["state"]),
            code_verifier=str(data["code_verifier"]),
            client_id=str(data["client_id"]),
            endpoints=endpoints,
            redirect_uri=str(data["redirect_uri"]),
            authorize_url=str(data.get("authorize_url", "")),
            created_at=float(data.get("created_at", 0.0)),
        )
    except (KeyError, ValueError, TypeError) as exc:
        logger.warning("pending-auth blob malformed: %s", exc.__class__.__name__)
        return None


def _load_pending_from_disk() -> PendingAuth | None:
    """Best-effort load of the persisted pending-auth. Never raises."""
    try:
        raw = load_pending_auth()
    except Exception as exc:  # pragma: no cover - storage fail-safe
        logger.warning(
            "pending-auth disk read failed (%s)", exc.__class__.__name__
        )
        return None
    if not raw:
        return None
    return _deserialize_pending(raw)


def _clear_pending() -> None:
    """Clear the in-flight flow from BOTH memory and disk. Never raises."""
    global _PENDING_AUTH
    _PENDING_AUTH = None
    try:
        clear_pending_auth()
    except Exception as exc:  # pragma: no cover - storage fail-safe
        logger.warning(
            "pending-auth clear failed (%s)", exc.__class__.__name__
        )


def begin_auth(*, redirect_uri: str | None = None) -> PendingAuth:
    """Start the OAuth flow: discover, register, build authorize URL.

    Returns a ``PendingAuth`` whose ``authorize_url`` the caller surfaces
    to the user (open in browser). Stores the pending state in memory so
    ``complete_auth`` can finish the exchange. Raises ``BrokerError`` on
    any discovery/registration failure so the cockpit can show a clear
    message instead of a dead 'Connect' button.
    """
    global _PENDING_AUTH
    from packages.execution.robinhood_token import RH_OAUTH_REDIRECT_URI

    redirect_uri = redirect_uri or RH_OAUTH_REDIRECT_URI
    try:
        endpoints = discover_endpoints()
        client_id = load_client_id() or register_client(
            endpoints, redirect_uri=redirect_uri
        )
        verifier, challenge = new_pkce_pair()
        state = new_state()
        url = build_authorize_url(
            endpoints,
            client_id=client_id,
            code_challenge=challenge,
            state=state,
            redirect_uri=redirect_uri,
        )
    except Exception as exc:
        raise BrokerError(
            f"robinhood: could not start auth flow ({exc.__class__.__name__}: {exc})"
        ) from exc

    _PENDING_AUTH = PendingAuth(
        state=state,
        code_verifier=verifier,
        client_id=client_id,
        endpoints=endpoints,
        redirect_uri=redirect_uri,
        authorize_url=url,
        created_at=time.time(),
    )
    # Mirror to the encrypted file so the flow survives a server restart
    # between here and the /callback redirect. Storage failure must never
    # crash connect -- the in-memory global still works if the process
    # doesn't restart, so we degrade quietly.
    try:
        save_pending_auth(_serialize_pending(_PENDING_AUTH))
    except Exception as exc:  # pragma: no cover - storage fail-safe
        logger.warning(
            "pending-auth persist failed (%s) -- continuing in-memory only",
            exc.__class__.__name__,
        )
    return _PENDING_AUTH


def complete_auth(*, code: str, state: str) -> TokenSet:
    """Finish the flow: verify state, exchange code, persist tokens.

    Must be called with the ``code`` and ``state`` query params Robinhood
    sent to the loopback redirect. Raises ``BrokerError`` if there is no
    pending flow, the flow has expired (TTL), or the state doesn't match
    (CSRF guard).

    The in-memory global is consulted first. If it's empty or its state
    doesn't match -- the common case after a cockpit server auto-restart
    wiped the global between begin_auth and this callback -- we fall back to
    the encrypted pending-auth file and validate against that. On success we
    delete the file and clear the global (single-use consumption).
    """
    pending = _PENDING_AUTH
    # Fall back to the persisted blob when memory is gone (restart) or its
    # state doesn't match the callback. We re-validate state below against
    # whichever source we end up using, so this fallback can't weaken the
    # CSRF guard.
    if pending is None or not state or state != pending.state:
        from_disk = _load_pending_from_disk()
        if from_disk is not None:
            pending = from_disk
    if pending is None:
        raise BrokerError(
            "robinhood: no auth flow in progress -- click Connect first"
        )
    # TTL / replay window: an old blob is expired regardless of state. Clear
    # it (memory + file) so a stale flow can't be replayed.
    if pending.created_at and (time.time() - pending.created_at) > PENDING_AUTH_TTL_S:
        _clear_pending()
        raise BrokerError(
            "robinhood: auth flow expired -- click Connect again"
        )
    if not state or state != pending.state:
        # Mismatch against BOTH memory and the file. Do NOT clear the pending
        # flow -- it may be a stray/replayed callback while the real one is
        # still coming.
        raise BrokerError(
            "robinhood: OAuth state mismatch -- ignoring callback (possible "
            "CSRF or stale tab)"
        )
    try:
        tokens = exchange_code(
            pending.endpoints,
            code=code,
            code_verifier=pending.code_verifier,
            client_id=pending.client_id,
            redirect_uri=pending.redirect_uri,
        )
    except Exception as exc:
        raise BrokerError(
            f"robinhood: code exchange failed ({exc.__class__.__name__}: {exc})"
        ) from exc

    save_tokens(tokens)
    save_client_id(pending.client_id)
    _clear_pending()  # consume the one-shot flow (memory + encrypted file)
    return tokens


def pending_auth() -> PendingAuth | None:
    """Expose the current in-flight auth (cockpit uses this to re-show the
    authorize URL if the user closed the tab)."""
    return _PENDING_AUTH


def disconnect() -> None:
    """Full Robinhood disconnect: wipe tokens + client_id + any pending
    flow (in memory AND the encrypted file). Backs the 'Disconnect
    Robinhood' button."""
    global _PENDING_AUTH
    _PENDING_AUTH = None
    clear_tokens()
    clear_client_id()
    with contextlib.suppress(Exception):
        clear_pending_auth()


def is_connected() -> bool:
    """True if a (possibly stale-but-refreshable) token set is stored.
    Stale-with-refresh still counts as connected -- the broker will
    refresh on first use."""
    tokens = load_tokens()
    if tokens is None:
        return False
    # Stale-with-refresh still counts as connected; stale-without does not.
    return not (tokens.is_stale() and not tokens.refresh_token)


# ---------------------------------------------------------------------------
# Agentic-account discovery + resolution
#
# Robinhood enforces ``agentic_allowed`` at the API level: orders placed on
# a non-agentic account are rejected. So the broker MUST target the single
# account flagged ``agentic_allowed=true`` (and active / not deactivated).
# We discover it via ``get_accounts`` and persist the chosen account_number
# in onboarding state so we don't re-discover on every call. We NEVER
# hardcode an account number in source.
# ---------------------------------------------------------------------------


def _is_agentic_account(acct: dict[str, Any]) -> bool:
    """True if an account dict is flagged agentic-allowed AND active.

    Field names vary across server versions, so we accept a few aliases.
    ``agentic_allowed`` must be truthy; if any of the deactivated/closed
    flags are truthy the account is skipped (fail safe -- never target a
    deactivated account)."""
    allowed = acct.get("agentic_allowed")
    if allowed is None:
        allowed = acct.get("agenticAllowed")
    if not bool(allowed):
        return False
    for dead_key in ("deactivated", "is_deactivated", "closed", "is_closed"):
        if bool(acct.get(dead_key)):
            return False
    # If an explicit status/state is present, require it to look active.
    status = str(acct.get("status") or acct.get("state") or "active").lower()
    return status not in {"deactivated", "closed", "inactive", "disabled"}


def _account_number_of(acct: dict[str, Any]) -> str:
    """Pull the account number from an account dict (field name varies)."""
    for key in ("account_number", "accountNumber", "account_id", "number"):
        val = acct.get(key)
        if val:
            return str(val)
    return ""


def _account_is_active(acct: dict[str, Any]) -> bool:
    """True if an account dict isn't flagged deactivated/closed/inactive.
    (Looser than :func:`_is_agentic_account` -- does NOT require the agentic
    flag; used only for the single-account auto-select fallback.)"""
    for dead_key in ("deactivated", "is_deactivated", "closed", "is_closed"):
        if bool(acct.get(dead_key)):
            return False
    status = str(acct.get("status") or acct.get("state") or "active").lower()
    return status not in {"deactivated", "closed", "inactive", "disabled"}


def select_agentic_account(accounts: list[Any]) -> str | None:
    """Pick the account number to target from a ``get_accounts`` list.

    Resolution order:
      1. The FIRST active ``agentic_allowed=true`` account (the canonical
         case; warns if several exist and takes the first active one).
      2. **Single-account auto-select** -- if NO account carries the agentic
         flag but exactly ONE active account exists, target it. The user is
         often remote (phone) and can't click a chooser, and Robinhood only
         exposes agentic-eligible accounts through this server anyway. Reads
         are harmless; the live order path is still gated independently and
         Robinhood itself rejects orders on a non-agentic account.
      3. Otherwise ``None`` (fail safe -- ambiguous multi-account with no
         agentic flag must be resolved by an explicit user choice).
    """
    agentic = [
        a
        for a in accounts
        if isinstance(a, dict) and _is_agentic_account(a) and _account_number_of(a)
    ]
    if agentic:
        if len(agentic) > 1:
            logger.warning(
                "robinhood: %d agentic accounts found -- using the first active one",
                len(agentic),
            )
        chosen = _account_number_of(agentic[0])
        logger.info("robinhood: selected agentic account ...%s", chosen[-4:])
        return chosen

    # Fallback: exactly one active account with a number, no agentic flag set.
    active = [
        a
        for a in accounts
        if isinstance(a, dict) and _account_is_active(a) and _account_number_of(a)
    ]
    if len(active) == 1:
        chosen = _account_number_of(active[0])
        logger.info(
            "robinhood: single active account ...%s auto-selected (no agentic "
            "flag present)",
            chosen[-4:],
        )
        return chosen
    return None


async def discover_agentic_account_number(
    broker: RobinhoodAgenticBroker | None = None,
) -> str | None:
    """Call ``get_accounts`` and return the agentic-allowed account number.

    Read-only (safe in shadow). Returns ``None`` on any failure or when no
    agentic account exists -- the caller treats that as "do not enable
    Robinhood trading" (fail safe). Builds a default broker honoring the
    user's onboarding mode when one isn't supplied."""
    owns = broker is None
    if broker is None:
        broker = build_broker_from_settings()
    try:
        client = await broker._client()
        res = await client.call_tool("get_accounts", {})
        accounts = _normalize_rows(
            res.content, keys=("accounts", "results", "items")
        )
        return select_agentic_account(accounts if isinstance(accounts, list) else [])
    except (BrokerError, McpError) as exc:
        logger.warning(
            "robinhood: agentic-account discovery failed (%s)",
            exc.__class__.__name__,
        )
        return None
    finally:
        if owns:
            with contextlib.suppress(Exception):
                await broker.aclose()


def resolve_agentic_account_number() -> str | None:
    """Return the stored agentic account number from onboarding, if any.

    Pure read of ``OnboardingState.rh_account_number`` -- never hits the
    network. ``None`` when unset so the broker/factory can fail safe."""
    try:
        from packages.cockpit.onboarding import load_onboarding

        num = load_onboarding().rh_account_number.strip()
        return num or None
    except Exception as exc:  # pragma: no cover - belt and braces
        logger.warning(
            "robinhood: account-number resolve failed (%s)",
            exc.__class__.__name__,
        )
        return None


async def ensure_agentic_account_number() -> str | None:
    """Resolve the agentic account number, discovering + persisting if unset.

    1. If onboarding already stores one, return it (no network).
    2. Otherwise discover it via ``get_accounts``; on success persist it to
       onboarding and return it.
    3. On any failure / no agentic account, return ``None`` (fail safe).
    """
    stored = resolve_agentic_account_number()
    if stored:
        return stored
    discovered = await discover_agentic_account_number()
    if discovered:
        try:
            from packages.cockpit.onboarding import (
                load_onboarding,
                save_onboarding,
            )

            state = load_onboarding()
            state.rh_account_number = discovered
            save_onboarding(state)
        except Exception as exc:  # pragma: no cover - persistence best-effort
            logger.warning(
                "robinhood: failed to persist agentic account (%s)",
                exc.__class__.__name__,
            )
    return discovered


# ---------------------------------------------------------------------------
# Convenience factory used by the cockpit when wiring brokers
# ---------------------------------------------------------------------------


def build_broker_from_settings() -> RobinhoodAgenticBroker:
    """Build a broker honoring the user's onboarding choices.

    Reads ``OnboardingState.rh_mode`` -- 'shadow' (default) -> SHADOW,
    'live' -> LIVE. The user can never go LIVE without explicitly
    flipping that field in Settings.

    Also threads the stored agentic ``account_number`` into the broker so
    its reads + orders target the agentic-allowed account (Robinhood
    rejects trades on non-agentic accounts). ``None`` when unset -- the
    broker stays read-safe and the factory's live path refuses to enable.
    """
    try:
        from packages.cockpit.onboarding import load_onboarding

        state = load_onboarding()
        mode = (
            ExecutionMode.LIVE
            if state.rh_mode == "live"
            else ExecutionMode.SHADOW
        )
        account_number = state.rh_account_number.strip() or None
    except Exception as exc:
        logger.warning("rh_mode resolve failed: %s -- defaulting to shadow", exc.__class__.__name__)
        mode = ExecutionMode.SHADOW
        account_number = None
    return RobinhoodAgenticBroker(mode=mode, account_number=account_number)


async def robinhood_account_snapshot() -> dict[str, Any]:
    """Convenience wrapper used by the cockpit + agent context.

    Builds a broker honoring the user's onboarding mode and returns a
    read-only :meth:`RobinhoodAgenticBroker.account_snapshot`. Always
    closes the MCP client afterwards and never raises -- on any failure
    it returns a snapshot with ``connected=False`` and an ``errors``
    entry so callers (UI / agent) can degrade gracefully.
    """
    if not is_connected():
        return {
            "connected": False,
            "mode": "shadow",
            "as_of": datetime.now(UTC).isoformat(timespec="seconds"),
            "accounts": [],
            "portfolio": None,
            "positions": [],
            "buying_power": None,
            "cash": None,
            "total_equity": None,
            "errors": [],
        }
    broker = build_broker_from_settings()
    try:
        return await broker.account_snapshot()
    except Exception as exc:  # pragma: no cover -- account_snapshot is defensive
        return {
            "connected": is_connected(),
            "mode": "shadow" if broker._is_shadow() else "live",
            "as_of": datetime.now(UTC).isoformat(timespec="seconds"),
            "accounts": [],
            "portfolio": None,
            "positions": [],
            "buying_power": None,
            "cash": None,
            "total_equity": None,
            "errors": [f"snapshot: {exc}"],
        }
    finally:
        with contextlib.suppress(Exception):
            await broker.aclose()


# Re-export the audit-log writer for tests / debugging tools.
__all__ = [
    "ABSOLUTE_MAX_FLOAT_USD",
    "SHADOW_TRADES_PATH",
    "PendingAuth",
    "RobinhoodAgenticBroker",
    "begin_auth",
    "build_broker_from_settings",
    "complete_auth",
    "disconnect",
    "discover_agentic_account_number",
    "ensure_agentic_account_number",
    "is_connected",
    "load_shadow_trades",
    "pending_auth",
    "resolve_agentic_account_number",
    "resolve_float_cap",
    "robinhood_account_snapshot",
    "select_agentic_account",
]
