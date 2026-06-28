"""Phase 36c — Remote control bridge for the cockpit.

A small, opinionated surface that lives at ``/api/remote/*`` and lets a
trusted external operator (the Perplexity agent over a Cloudflare tunnel,
say) observe and steer the local cockpit. The surface deliberately
mirrors the existing in-process API rather than reinventing it, so the
remote and the local UI are always feature-parity.

Security model
--------------

Every route under ``/api/remote/*`` requires a shared-secret bearer
token, supplied either as:

* ``Authorization: Bearer <token>`` header, or
* ``X-Cockpit-Token: <token>`` header.

The expected token is read from the ``COCKPIT_REMOTE_TOKEN`` environment
variable at request time. If that env var is unset or empty the entire
remote surface is **disabled** — every route returns 503. This is the
fail-closed default: a fresh checkout cannot accidentally expose remote
control without an explicit opt-in.

The comparison uses :func:`hmac.compare_digest` to avoid timing leaks.

Surface
-------

============================  ====  ===========================================
Path                          Verb  Purpose
============================  ====  ===========================================
``/api/remote/health``        GET   Liveness probe + token sanity check.
``/api/remote/snapshot``      GET   Full operator-facing state in one payload.
``/api/remote/log``           GET   Tail (or full download) of the paper loop log.
``/api/remote/pause``         POST  Pause the bot.
``/api/remote/resume``        POST  Resume the bot (unpause).
``/api/remote/loop/start``    POST  Start the paper-trade loop.
``/api/remote/loop/stop``     POST  Stop the paper-trade loop.
``/api/remote/liquidate``     POST  Liquidate all positions. Requires
                                    ``{"confirm": "LIQUIDATE"}`` in the body.
============================  ====  ===========================================

Each mutating route returns the post-action :class:`CockpitState` dict so
the caller (me) can verify the effect immediately without a follow-up
poll.
"""

from __future__ import annotations

import contextlib
import hmac
import logging
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from packages.cockpit import proc as cockpit_proc
from packages.cockpit import updater as cockpit_updater
from packages.cockpit.state import load_state, record_action, save_state

log = logging.getLogger(__name__)


# Path to the web-server's own log file. The Robinhood/OAuth onboarding
# routes log to the uvicorn process (this app), whose output otherwise only
# reaches the console window. server.py attaches a RotatingFileHandler that
# mirrors that output here so the remote bridge can surface connect errors.
# Defined in this module (rather than imported from server.py) to avoid a
# circular import — server.py imports this module at startup.
# packages/cockpit/web/remote.py -> parents[3] == repo root.
WEB_LOG_PATH = Path(__file__).resolve().parents[3] / "data" / "cockpit" / "logs" / "cockpit_web.log"

# Default and cap for the /weblog tail size, in bytes. The default mirrors
# proc.tail_log (64KB); the cap bounds a hostile/buggy caller.
_WEBLOG_DEFAULT_BYTES = 64 * 1024
_WEBLOG_MAX_BYTES = 1024 * 1024


def _tail_web_log(max_bytes: int) -> str:
    """Return up to ``max_bytes`` of the tail of the web-server log file.

    Empty string if the file doesn't exist yet (the server may not have
    written anything, or file logging may be disabled). Mirrors the
    byte-tail approach of :func:`packages.cockpit.proc.tail_log`.
    """
    from packages.data.redact import redact

    path = WEB_LOG_PATH
    if not path.exists():
        return ""
    try:
        size = path.stat().st_size
        with path.open("rb") as f:
            if size > max_bytes:
                f.seek(size - max_bytes)
            data = f.read()
        # Defense in depth: redact on read too. The logging filter masks new
        # lines, but historical lines (or lines written before the filter was
        # installed) could still carry a token in a URL.
        return redact(data.decode("utf-8", errors="replace"))
    except OSError:
        return ""


# Env var that gates the entire surface. When unset/empty, every route
# returns 503 — a fail-closed posture so a fresh deploy can't leak
# control without an explicit opt-in.
ENV_TOKEN = "COCKPIT_REMOTE_TOKEN"

# Header names we accept for the bearer token. We prefer the standard
# Authorization header but also accept X-Cockpit-Token for callers that
# can't easily set Authorization (some webhooks, browser extensions).
AUTH_HEADER = "authorization"
ALT_HEADER = "x-cockpit-token"


# Minimum token length to accept. Short enough to be usable but long
# enough to make brute force impractical given normal rate limits.
MIN_TOKEN_LEN = 16


def _expected_token() -> str | None:
    """Return the configured remote token, or None if the surface is off."""
    tok = os.environ.get(ENV_TOKEN, "").strip()
    return tok or None


def _extract_provided_token(
    authorization: str | None, x_cockpit_token: str | None
) -> str | None:
    """Pull the bearer token out of either accepted header."""
    if authorization:
        s = authorization.strip()
        # Accept "Bearer xxx" or a bare token (some clients drop the scheme).
        if s.lower().startswith("bearer "):
            return s[7:].strip() or None
        return s or None
    if x_cockpit_token:
        return x_cockpit_token.strip() or None
    return None


def require_remote_auth(
    authorization: str | None = Header(default=None),
    x_cockpit_token: str | None = Header(default=None),
) -> None:
    """FastAPI dependency that enforces the shared-secret token.

    Returns ``None`` on success; raises ``HTTPException`` otherwise.

    Failure modes (each maps to a distinct status code so callers can
    diagnose without a debug round-trip):

    * **503** — surface disabled (``COCKPIT_REMOTE_TOKEN`` unset/empty
      or too short to be considered a real secret).
    * **401** — no token supplied.
    * **403** — token supplied but does not match.
    """
    expected = _expected_token()
    if not expected or len(expected) < MIN_TOKEN_LEN:
        raise HTTPException(
            status_code=503,
            detail=(
                "remote control disabled: set COCKPIT_REMOTE_TOKEN to a "
                f"secret of at least {MIN_TOKEN_LEN} characters to enable"
            ),
        )
    provided = _extract_provided_token(authorization, x_cockpit_token)
    if not provided:
        raise HTTPException(status_code=401, detail="missing remote token")
    # Constant-time comparison to avoid timing oracles.
    if not hmac.compare_digest(provided.encode("utf-8"), expected.encode("utf-8")):
        raise HTTPException(status_code=403, detail="invalid remote token")


# ---------------------------------------------------------------------------
# Pydantic request bodies
# ---------------------------------------------------------------------------


class RemoteStartLoop(BaseModel):
    strategy: str = "ensemble"
    dry_run: bool = False


class RemoteLiquidate(BaseModel):
    confirm: str = ""


class RemoteRestart(BaseModel):
    """Body for /restart.

    ``pull``: whether the detached helper should pull from origin before
    relaunching uvicorn. Default True (the whole point of restart).
    ``delay_sec``: how long the helper waits before killing uvicorn so
    this HTTP response can flush. Default 2s. Bounded [1, 10].
    """

    pull: bool = True
    delay_sec: int = 2


class RemoteUpdate(BaseModel):
    """Body for /update/apply.

    ``restart_loop``: after a successful pull+pip, stop and respawn the
    paper-trade loop so the new code is actually running. Default True
    because that's the whole point of a remote update; the operator can
    pass False to apply code changes that don't affect the loop (UI,
    docs, tests).
    ``dry_run``: when the loop is restarted, pass through the dry-run
    flag. Defaults to False to match the operator's current intent.
    ``strategy``: strategy name for the respawn. Defaults to ensemble.
    """

    restart_loop: bool = True
    dry_run: bool = False
    strategy: str = "ensemble"


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


def build_router() -> APIRouter:
    """Build the ``/api/remote`` router.

    Factoring this into a builder (instead of a module-level router) lets
    tests instantiate fresh routers and lets the main server module wire
    it up without import-order surprises.
    """
    router = APIRouter(prefix="/api/remote", tags=["remote"])

    @router.get("/health")
    def remote_health(_: None = None) -> dict[str, Any]:  # pragma: no cover
        # We want /health to be reachable WITHOUT auth so the operator
        # can verify the surface is up. But it should NOT leak whether
        # the token is configured \u2014 just return enabled/disabled status.
        return {
            "ok": True,
            "enabled": bool(_expected_token() and len(_expected_token() or "") >= MIN_TOKEN_LEN),
        }

    @router.get("/whoami")
    def remote_whoami(
        request: Request,
        authorization: str | None = Header(default=None),
        x_cockpit_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        """Authenticated probe \u2014 returns 200 only with a valid token."""
        require_remote_auth(authorization, x_cockpit_token)
        return {
            "ok": True,
            "client": request.client.host if request.client else None,
            "surface": "remote",
        }

    @router.get("/snapshot")
    def remote_snapshot(
        authorization: str | None = Header(default=None),
        x_cockpit_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        """All operator-facing context in a single payload.

        Built best-effort: each sub-section is wrapped so a single
        failing subsystem doesn't blank the whole response.
        """
        require_remote_auth(authorization, x_cockpit_token)
        out: dict[str, Any] = {"errors": {}}
        try:
            out["state"] = load_state().to_dict()
        except Exception as e:  # pragma: no cover
            out["errors"]["state"] = str(e)
        try:
            info = cockpit_proc.status("paper_loop")
            out["paper_loop"] = info.to_dict()
        except Exception as e:  # pragma: no cover
            out["errors"]["paper_loop"] = str(e)
        try:
            out["log_tail"] = cockpit_proc.tail_log("paper_loop")
        except Exception as e:  # pragma: no cover
            out["errors"]["log_tail"] = str(e)
        return out

    @router.get("/log")
    def remote_log(
        download: int = 0,
        authorization: str | None = Header(default=None),
        x_cockpit_token: str | None = Header(default=None),
    ):
        """Tail or download the paper loop log file."""
        require_remote_auth(authorization, x_cockpit_token)
        if download:
            path = cockpit_proc.log_path("paper_loop")
            try:
                body = path.read_text(encoding="utf-8", errors="replace")
            except FileNotFoundError:
                body = "(no log on disk yet)"
            return PlainTextResponse(body)
        return {"kind": "paper_loop", "tail": cockpit_proc.tail_log("paper_loop")}

    @router.get("/weblog")
    def remote_weblog(
        lines: int = 0,
        bytes: int = 0,
        authorization: str | None = Header(default=None),
        x_cockpit_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        """Tail the web-server's own log (where Robinhood/OAuth errors land).

        The paper-loop log (/log) only carries the child trading process's
        output. Robinhood onboarding logs to *this* uvicorn process, so we
        expose its log file separately for remote debugging.

        Tail size is controlled by an optional ``?bytes=`` (preferred) or
        ``?lines=`` query param, both clamped to a sane cap. Returns an
        empty tail (still 200) if the log file doesn't exist yet, rather
        than erroring.
        """
        require_remote_auth(authorization, x_cockpit_token)
        # Resolve the requested tail size to a byte budget. ``bytes`` wins
        # if given; otherwise approximate ``lines`` at ~256 bytes/line.
        if bytes > 0:
            max_bytes = bytes
        elif lines > 0:
            max_bytes = lines * 256
        else:
            max_bytes = _WEBLOG_DEFAULT_BYTES
        max_bytes = max(1, min(max_bytes, _WEBLOG_MAX_BYTES))
        return {"ok": True, "kind": "cockpit_web", "tail": _tail_web_log(max_bytes)}

    @router.post("/pause")
    def remote_pause(
        authorization: str | None = Header(default=None),
        x_cockpit_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        require_remote_auth(authorization, x_cockpit_token)
        state = load_state()
        state.paused = True
        state = record_action(state, "Bot paused via remote bridge")
        save_state(state)
        return state.to_dict()

    @router.post("/resume")
    def remote_resume(
        authorization: str | None = Header(default=None),
        x_cockpit_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        require_remote_auth(authorization, x_cockpit_token)
        state = load_state()
        state.paused = False
        state = record_action(state, "Bot resumed via remote bridge")
        save_state(state)
        return state.to_dict()

    @router.post("/loop/start")
    def remote_loop_start(
        body: RemoteStartLoop,
        authorization: str | None = Header(default=None),
        x_cockpit_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        """Start the paper-trade loop and persist intent."""
        require_remote_auth(authorization, x_cockpit_token)
        state = load_state()
        state.paper_loop_intended = True
        state.paper_loop_strategy = body.strategy
        state.paper_loop_dry_run = bool(body.dry_run)
        state.paper_loop_user_touched = True
        state = record_action(
            state,
            f"Loop started via remote: strategy={body.strategy} dry_run={body.dry_run}",
        )
        save_state(state)
        # Best-effort spawn \u2014 the auto-resume/start hook also handles
        # this on the next boot, so a transient spawn failure here
        # doesn't lose the intent.
        spawn_info: dict[str, Any] = {"running": False}
        try:
            import sys as _sys

            cmd = [
                _sys.executable,
                "tools/paper_trade.py",
                "--strategy",
                body.strategy,
                "--loop",
            ]
            if body.dry_run:
                cmd.append("--dry-run")
            info = cockpit_proc.start("paper_loop", cmd)
            spawn_info = info.to_dict()
        except Exception as e:
            spawn_info = {"running": False, "error": str(e)}
        return {"state": state.to_dict(), "job": spawn_info}

    @router.post("/loop/stop")
    def remote_loop_stop(
        authorization: str | None = Header(default=None),
        x_cockpit_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        require_remote_auth(authorization, x_cockpit_token)
        state = load_state()
        state.paper_loop_intended = False
        state.paper_loop_user_touched = True
        state = record_action(state, "Loop stopped via remote bridge")
        save_state(state)
        stop_info: dict[str, Any] = {"running": False}
        try:
            info = cockpit_proc.stop("paper_loop")
            stop_info = info.to_dict()
        except Exception as e:
            stop_info = {"running": False, "error": str(e)}
        return {"state": state.to_dict(), "job": stop_info}

    @router.post("/cancel_orders")
    def remote_cancel_orders(
        authorization: str | None = Header(default=None),
        x_cockpit_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        """Phase 36g — cancel every open Alpaca order; positions untouched.

        Surgical recovery for the buying-power 403 cascade: pre-market
        orders held buying power, the planner kept re-queueing the same
        symbols, every cycle 403'd. Cancelling open orders releases the
        reserved cash without disturbing any holdings.

        Unlike /liquidate, this is reversible (the bot will simply
        re-plan the same orders on the next cycle if state still calls
        for them) so we don't require a confirm token.
        """
        require_remote_auth(authorization, x_cockpit_token)
        import asyncio

        from packages.execution.broker import AlpacaPaperBroker, BrokerError

        async def _run() -> dict[str, Any]:
            broker = AlpacaPaperBroker()
            try:
                return await broker.cancel_all_orders()
            finally:
                await broker.aclose()

        try:
            result = asyncio.run(_run())
        except BrokerError as e:
            raise HTTPException(status_code=502, detail=str(e)) from e
        except Exception as e:  # pragma: no cover — defensive
            raise HTTPException(status_code=500, detail=f"cancel failed: {e}") from e

        state = load_state()
        state = record_action(
            state,
            f"Cancel-all-orders via remote bridge ({result.get('cancelled_orders', 0)} cancelled)",
        )
        save_state(state)
        return {"ok": True, "result": result, "state": state.to_dict()}

    @router.post("/liquidate")
    def remote_liquidate(
        body: RemoteLiquidate,
        authorization: str | None = Header(default=None),
        x_cockpit_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        """Liquidate all positions. Requires explicit confirm token.

        Irreversible, so the body must contain ``{"confirm": "LIQUIDATE"}``
        as a small protection against accidental triggers via a leaked
        token or a curl with the wrong path.
        """
        require_remote_auth(authorization, x_cockpit_token)
        if body.confirm != "LIQUIDATE":
            raise HTTPException(
                status_code=400,
                detail='liquidate requires {"confirm": "LIQUIDATE"} in the body',
            )
        state = load_state()
        # Set paused + clear intent so the auto-resume hook doesn't
        # immediately respawn the loop after the liquidation.
        state.paused = True
        state.paper_loop_intended = False
        state.paper_loop_user_touched = True
        state = record_action(state, "Liquidate-all via remote bridge")
        save_state(state)
        # Best-effort stop of any running loop; the actual liquidation
        # of broker positions is the operator's responsibility via the
        # existing /api/trading/liquidate path or the Alpaca UI \u2014 we
        # don't duplicate that broker call here because it would split
        # the audit trail.
        with contextlib.suppress(Exception):
            cockpit_proc.stop("paper_loop")
        return {
            "state": state.to_dict(),
            "note": (
                "Loop stopped + state paused. Use the local cockpit "
                "/api/trading/liquidate (or the Alpaca UI) to actually "
                "close broker positions."
            ),
        }

    @router.get("/rh/tools")
    def remote_rh_tools(
        authorization: str | None = Header(default=None),
        x_cockpit_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        """Discovery helper: dump the LIVE Robinhood MCP ``tools/list``.

        Read-only. Builds a broker from the user's onboarding settings
        (which honors SHADOW default), runs the MCP ``initialize`` +
        ``tools/list`` handshake with the OAuth token the cockpit already
        holds, and returns the raw tool catalog (names + input schemas) so
        we can see the server's real contract. Never places an order and
        never logs the bearer token.
        """
        require_remote_auth(authorization, x_cockpit_token)
        import asyncio

        from packages.execution.robinhood import (
            build_broker_from_settings,
            is_connected,
        )
        from packages.execution.robinhood_mcp import McpError

        if not is_connected():
            return {"ok": False, "reason": "not_connected", "tools": []}

        async def _run() -> list[dict[str, Any]]:
            broker = build_broker_from_settings()
            try:
                client = await broker._client()
                return await client.list_tools()
            finally:
                with contextlib.suppress(Exception):
                    await broker.aclose()

        try:
            tools = asyncio.run(_run())
        except (McpError, Exception) as e:  # noqa: BLE001 - report, never crash
            return {"ok": False, "error": f"{e.__class__.__name__}: {e}", "tools": []}
        # Surface name + description + required params so the contract is
        # legible without dumping huge schemas.
        summary = []
        for t in tools if isinstance(tools, list) else []:
            if not isinstance(t, dict):
                continue
            schema = t.get("inputSchema") or t.get("input_schema") or {}
            required = schema.get("required") if isinstance(schema, dict) else None
            props = schema.get("properties") if isinstance(schema, dict) else None
            summary.append(
                {
                    "name": t.get("name"),
                    "description": t.get("description"),
                    "required": required,
                    "properties": list(props.keys()) if isinstance(props, dict) else None,
                }
            )
        return {"ok": True, "count": len(summary), "tools": summary, "raw": tools}

    @router.get("/rh/probe")
    def remote_rh_probe(
        tool: str,
        account_number: str = "",
        authorization: str | None = Header(default=None),
        x_cockpit_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        """Discovery helper: call ONE read-only Robinhood MCP tool and dump
        its raw result. Refuses any tool whose name hints at a mutating
        action (order/trade/buy/sell/cancel) so this can never place a
        trade. Optionally threads an ``account_number`` arg for tools that
        require it.
        """
        require_remote_auth(authorization, x_cockpit_token)
        import asyncio

        from packages.execution.robinhood import (
            build_broker_from_settings,
            is_connected,
        )
        from packages.execution.robinhood_mcp import McpError

        lname = (tool or "").strip().lower()
        if not lname:
            raise HTTPException(status_code=400, detail="tool name required")
        BLOCKED = ("order", "trade", "buy", "sell", "cancel", "place", "submit")
        if any(b in lname for b in BLOCKED):
            raise HTTPException(
                status_code=400,
                detail=f"refusing potentially-mutating tool {tool!r} (read-only probe)",
            )
        if not is_connected():
            return {"ok": False, "reason": "not_connected"}

        args: dict[str, Any] = {}
        if account_number.strip():
            args["account_number"] = account_number.strip()

        async def _run() -> Any:
            broker = build_broker_from_settings()
            try:
                client = await broker._client()
                res = await client.call_tool(tool, args)
                return {"is_error": res.is_error, "content": res.content}
            finally:
                with contextlib.suppress(Exception):
                    await broker.aclose()

        try:
            out = asyncio.run(_run())
        except McpError as e:
            return {"ok": False, "error": f"McpError: {e}"}
        except Exception as e:  # noqa: BLE001 - report, never crash
            return {"ok": False, "error": f"{e.__class__.__name__}: {e}"}
        return {"ok": True, "tool": tool, "args": args, "result": out}

    @router.get("/version")
    def remote_version(
        authorization: str | None = Header(default=None),
        x_cockpit_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        """Local HEAD commit \u2014 lets the agent verify what's running.

        Wraps :func:`updater.current_commit`. Always returns 200 with the
        best-effort fields the underlying git calls produce; if git is
        unavailable the fields are empty strings rather than an error.
        """
        require_remote_auth(authorization, x_cockpit_token)
        try:
            return {"ok": True, "current": cockpit_updater.current_commit()}
        except Exception as e:  # pragma: no cover - defensive
            return {"ok": False, "error": str(e)}

    @router.get("/update/check")
    def remote_update_check(
        authorization: str | None = Header(default=None),
        x_cockpit_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        """Fetch origin and report how far behind HEAD is.

        Read-only: runs ``git fetch`` + ``git rev-list --count`` but never
        moves HEAD. Safe to poll from a watchdog.
        """
        require_remote_auth(authorization, x_cockpit_token)
        try:
            return cockpit_updater.check_updates()
        except Exception as e:  # pragma: no cover - defensive
            return {"ok": False, "error": str(e)}

    @router.post("/update/apply")
    def remote_update_apply(
        body: RemoteUpdate,
        authorization: str | None = Header(default=None),
        x_cockpit_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        """Pull origin (fast-forward only), reinstall, optionally restart loop.

        The flow is:

        1. Call :func:`updater.apply_update` which does ``git pull --ff-only``
           then ``pip install -e .``. Both are bounded by timeouts inside
           the helper. If either step fails the response is ``ok=False``
           with the captured log; the running loop is left untouched.
        2. If ``restart_loop`` is True (default), stop the existing
           paper_loop process and spawn a new one using the body's
           strategy + dry_run. State intent is set so the auto-resume
           hook respects the new config across cockpit restarts.

        Returns a dict with three top-level keys:

        * ``update``  \u2014 the updater result (ok/log/current).
        * ``state``   \u2014 the post-action cockpit state.
        * ``job``     \u2014 the post-action paper_loop process info
          (or ``{"skipped": True}`` if restart_loop was False).
        """
        require_remote_auth(authorization, x_cockpit_token)
        # Step 1: pull + pip. This is the only step that can fail in a
        # way that should abort the rest \u2014 we don't want to restart
        # the loop into a half-applied state.
        try:
            update_result = cockpit_updater.apply_update()
        except Exception as e:
            update_result = {"ok": False, "step": "exception", "log": str(e)}
        if not update_result.get("ok"):
            # Leave the running loop alone, return the failure log.
            return {
                "update": update_result,
                "state": load_state().to_dict(),
                "job": {"skipped": True, "reason": "update failed"},
            }

        # Step 2: optional restart. Mirrors loop_start logic so the
        # surface stays consistent.
        if not body.restart_loop:
            return {
                "update": update_result,
                "state": load_state().to_dict(),
                "job": {"skipped": True, "reason": "restart_loop=False"},
            }

        # Best-effort stop of any existing loop. We don't bail on stop
        # failure \u2014 start() will detect a still-running process and
        # surface it in the returned info dict.
        with contextlib.suppress(Exception):
            cockpit_proc.stop("paper_loop")

        state = load_state()
        state.paper_loop_intended = True
        state.paper_loop_strategy = body.strategy
        state.paper_loop_dry_run = bool(body.dry_run)
        state.paper_loop_user_touched = True
        state = record_action(
            state,
            (
                "Loop restarted via remote update: "
                f"strategy={body.strategy} dry_run={body.dry_run}"
            ),
        )
        save_state(state)

        spawn_info: dict[str, Any] = {"running": False}
        try:
            import sys as _sys

            cmd = [
                _sys.executable,
                "tools/paper_trade.py",
                "--strategy",
                body.strategy,
                "--loop",
            ]
            if body.dry_run:
                cmd.append("--dry-run")
            info = cockpit_proc.start("paper_loop", cmd)
            spawn_info = info.to_dict()
        except Exception as e:
            spawn_info = {"running": False, "error": str(e)}

        return {
            "update": update_result,
            "state": state.to_dict(),
            "job": spawn_info,
        }

    @router.post("/restart")
    def remote_restart(
        body: RemoteRestart,
        authorization: str | None = Header(default=None),
        x_cockpit_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        """Spawn a detached helper that kills this cockpit and relaunches.

        Returns immediately (before the kill happens) so the HTTP
        response can flush. The helper waits ``delay_sec`` seconds,
        kills uvicorn (this process), optionally git pulls, pip
        installs, then relaunches uvicorn in a new window with the same
        token.

        Windows-only by design \u2014 uses powershell.exe + Start-Process
        for true detachment. On non-Windows hosts we return 501.
        """
        require_remote_auth(authorization, x_cockpit_token)

        import os as _os
        import subprocess as _sp
        import sys as _sys
        from pathlib import Path as _Path

        if _sys.platform != "win32":
            raise HTTPException(
                status_code=501,
                detail="remote restart is Windows-only (uses powershell.exe Start-Process)",
            )

        # Resolve repo root from this file's location: packages/cockpit/web/remote.py
        # -> parents[3] == repo root.
        repo_root = _Path(__file__).resolve().parents[3]
        helper = repo_root / "tools" / "cockpit_restart_helper.ps1"
        if not helper.exists():
            raise HTTPException(
                status_code=500,
                detail=f"helper script missing at {helper}",
            )

        venv_python = repo_root / ".venv" / "Scripts" / "python.exe"
        token = _os.environ.get(ENV_TOKEN, "")
        delay = max(1, min(10, int(body.delay_sec)))
        # PID 0 means "this process" \u2014 the helper resolves it.
        my_pid = _os.getpid()

        ps_args = [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(helper),
            "-UvicornPid",
            str(my_pid),
            "-RepoRoot",
            str(repo_root),
            "-VenvPython",
            str(venv_python),
            "-Token",
            token,
            "-DelaySec",
            str(delay),
        ]
        if not body.pull:
            ps_args.append("-NoPull")

        # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW
        # so the helper survives our death. close_fds + no stdio so we
        # don't hold pipes back into a dying parent.
        DETACHED_PROCESS = 0x00000008  # noqa: N806 - mirrors Windows API constant name
        CREATE_NEW_PROCESS_GROUP = 0x00000200  # noqa: N806 - mirrors Windows API constant name
        try:
            _sp.Popen(
                ps_args,
                cwd=str(repo_root),
                stdin=_sp.DEVNULL,
                stdout=_sp.DEVNULL,
                stderr=_sp.DEVNULL,
                close_fds=True,
                creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP,
            )
        except Exception as e:  # pragma: no cover - hard to simulate in CI
            raise HTTPException(
                status_code=500,
                detail=f"failed to spawn restart helper: {e}",
            ) from e

        state = load_state()
        state = record_action(
            state,
            f"Restart helper spawned via remote (pull={body.pull}, delay={delay}s)",
        )
        save_state(state)

        return {
            "ok": True,
            "helper": "spawned",
            "delay_sec": delay,
            "pull": body.pull,
            "pid_to_kill": my_pid,
            "note": (
                f"Cockpit will go down in ~{delay}s, then come back up in "
                "~10-30s. Poll /api/remote/version to confirm new SHA."
            ),
            "state": state.to_dict(),
        }

    return router
