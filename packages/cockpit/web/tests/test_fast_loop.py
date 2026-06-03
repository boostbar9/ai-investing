"""Tests for Phase 25.1 — autonomy fast loop + parallel Phase 25 hooks.

The slow loop runs research every 15min during market hours. The fast
loop runs ONLY exit_rules + dip_watch every 60s so price-sensitive
decisions don't have to wait a full sweep cycle. Both Phase 25 hooks
are invoked in parallel via asyncio.gather inside ``_run_phase25_hooks``.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from packages.cockpit.web import autonomy


@pytest.fixture(autouse=True)
def _reset_state():
    autonomy.reset_for_tests()
    yield
    autonomy.reset_for_tests()


def _force_market_open(monkeypatch: pytest.MonkeyPatch, *, open_: bool = True) -> None:
    monkeypatch.setattr(autonomy, "is_market_open", lambda *a, **kw: open_)


@pytest.mark.asyncio
async def test_run_phase25_hooks_runs_in_parallel(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both hooks should overlap; total wall time ≈ max(exit, dip), not sum."""
    delay = 0.15  # seconds — long enough to detect sequential vs parallel.

    async def slow_exit() -> dict[str, Any]:
        await asyncio.sleep(delay)
        return {"evaluated": 3, "sells_triggered": 0}

    async def slow_dip() -> dict[str, Any]:
        await asyncio.sleep(delay)
        return {"checked": 2, "fired": 0}

    autonomy.configure(exit_rules_tick=slow_exit, dip_watch_tick=slow_dip)
    cfg = autonomy.STATE._config

    started = time.perf_counter()
    exit_r, dip_r, eod_r = await autonomy._run_phase25_hooks(cfg)
    elapsed = time.perf_counter() - started

    assert exit_r == {"evaluated": 3, "sells_triggered": 0}
    assert dip_r == {"checked": 2, "fired": 0}
    # eod_flatten_tick unset — should be None.
    assert eod_r is None
    # Sequential would be ≥ 2*delay = 0.30s. Parallel should be ≈ delay
    # plus a tiny overhead. We allow a generous ceiling for CI jitter.
    assert elapsed < delay * 1.8, f"Hooks did not run in parallel: {elapsed:.3f}s"


@pytest.mark.asyncio
async def test_run_phase25_hooks_isolates_exceptions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If exit_rules raises, dip_watch still runs and returns its result."""

    async def boom_exit() -> dict[str, Any]:
        raise RuntimeError("alpaca 500")

    async def good_dip() -> dict[str, Any]:
        return {"checked": 1, "fired": 1}

    autonomy.configure(exit_rules_tick=boom_exit, dip_watch_tick=good_dip)
    cfg = autonomy.STATE._config

    exit_r, dip_r, eod_r = await autonomy._run_phase25_hooks(cfg)
    assert exit_r is not None and "error" in exit_r
    assert dip_r == {"checked": 1, "fired": 1}
    assert eod_r is None


@pytest.mark.asyncio
async def test_run_phase25_hooks_skips_unset_hooks() -> None:
    """When a hook is None, we don't error — we return None for that slot."""
    cfg = autonomy.STATE._config
    cfg.exit_rules_tick = None
    cfg.dip_watch_tick = None
    cfg.eod_flatten_tick = None

    exit_r, dip_r, eod_r = await autonomy._run_phase25_hooks(cfg)
    assert exit_r is None
    assert dip_r is None
    assert eod_r is None


@pytest.mark.asyncio
async def test_run_phase25_hooks_respects_enabled_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Disabling exit_rules_enabled skips that hook even when set."""
    calls = {"exit": 0, "dip": 0}

    async def exit_hook() -> dict[str, Any]:
        calls["exit"] += 1
        return {"evaluated": 0}

    async def dip_hook() -> dict[str, Any]:
        calls["dip"] += 1
        return {"checked": 0}

    autonomy.configure(
        exit_rules_tick=exit_hook,
        dip_watch_tick=dip_hook,
        exit_rules_enabled=False,
    )
    cfg = autonomy.STATE._config

    exit_r, dip_r, eod_r = await autonomy._run_phase25_hooks(cfg)
    assert exit_r is None  # disabled
    assert dip_r == {"checked": 0}
    assert eod_r is None  # unset
    assert calls == {"exit": 0, "dip": 1}


@pytest.mark.asyncio
async def test_run_fast_tick_skips_outside_market_hours(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_market_open(monkeypatch, open_=False)
    called = {"n": 0}

    async def exit_hook() -> dict[str, Any]:
        called["n"] += 1
        return {}

    autonomy.configure(exit_rules_tick=exit_hook, dip_watch_tick=None)

    out = await autonomy.run_fast_tick()
    assert out["skipped"] is True
    assert out["reason"] == "market_closed"
    assert called["n"] == 0
    assert autonomy.STATE.last_fast_tick_status == "skipped_closed"


@pytest.mark.asyncio
async def test_run_fast_tick_skips_when_paused(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_market_open(monkeypatch, open_=True)
    monkeypatch.setattr(autonomy, "_default_pause_check", lambda: True)

    async def exit_hook() -> dict[str, Any]:
        raise AssertionError("should not be called when paused")

    autonomy.configure(exit_rules_tick=exit_hook, dip_watch_tick=None)
    out = await autonomy.run_fast_tick()
    assert out["skipped"] is True
    assert out["reason"] == "paused"
    assert autonomy.STATE.last_fast_tick_status == "skipped_paused"


@pytest.mark.asyncio
async def test_run_fast_tick_invokes_both_hooks(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_market_open(monkeypatch, open_=True)
    monkeypatch.setattr(autonomy, "_default_pause_check", lambda: False)
    calls = {"exit": 0, "dip": 0}

    async def exit_hook() -> dict[str, Any]:
        calls["exit"] += 1
        return {"evaluated": 5, "sells_triggered": 1}

    async def dip_hook() -> dict[str, Any]:
        calls["dip"] += 1
        return {"checked": 2, "fired": 0}

    autonomy.configure(exit_rules_tick=exit_hook, dip_watch_tick=dip_hook)

    out = await autonomy.run_fast_tick()
    assert out["ok"] is True
    assert out["exit_rules"]["evaluated"] == 5
    assert out["dip_watch"]["checked"] == 2
    assert calls == {"exit": 1, "dip": 1}
    assert autonomy.STATE.last_fast_tick_status == "ok"
    assert autonomy.STATE.last_fast_tick_exit == {"evaluated": 5, "sells_triggered": 1}
    assert autonomy.STATE.last_fast_tick_dip == {"checked": 2, "fired": 0}


@pytest.mark.asyncio
async def test_snapshot_exposes_fast_loop_telemetry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_market_open(monkeypatch, open_=True)
    monkeypatch.setattr(autonomy, "_default_pause_check", lambda: False)

    async def exit_hook() -> dict[str, Any]:
        return {"evaluated": 1}

    autonomy.configure(exit_rules_tick=exit_hook, dip_watch_tick=None)
    await autonomy.run_fast_tick()

    snap = autonomy.snapshot()
    fast = snap["fast_loop"]
    assert fast["interval_s"] == 60
    assert fast["last_tick_status"] == "ok"
    assert fast["last_exit"] == {"evaluated": 1}
    # config block also surfaces the interval for /api/autonomy consumers
    assert snap["config"]["fast_loop_seconds"] == 60


@pytest.mark.asyncio
async def test_slow_loop_still_runs_hooks_via_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run_one_tick should call _run_phase25_hooks once during market hours."""
    _force_market_open(monkeypatch, open_=True)
    monkeypatch.setattr(autonomy, "_default_pause_check", lambda: False)
    # Disable self-improvement to keep the test focused & fast.
    calls = {"hooks": 0}

    async def fake_sweep() -> dict[str, Any]:
        return {"status": "ok", "candidates": []}

    async def hook_proxy() -> dict[str, Any]:
        calls["hooks"] += 1
        return {"ok": True}

    autonomy.configure(
        exit_rules_tick=hook_proxy,
        dip_watch_tick=hook_proxy,
        self_improve_enabled=False,
    )
    out = await autonomy.run_one_tick(sweep_runner=fake_sweep)
    assert out["ok"] is True
    # Both hooks fired in parallel — both increments observed.
    assert calls["hooks"] == 2
    assert out["exit_rules"] == {"ok": True}
    assert out["dip_watch"] == {"ok": True}
    # eod_flatten_tick unset → None
    assert out["eod_flatten"] is None


@pytest.mark.asyncio
async def test_run_fast_tick_invokes_eod_flatten_hook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Phase 28-R step 2 — the EOD flattener runs as a 3rd parallel branch."""
    _force_market_open(monkeypatch, open_=True)
    monkeypatch.setattr(autonomy, "_default_pause_check", lambda: False)
    calls = {"exit": 0, "dip": 0, "eod": 0}

    async def exit_hook() -> dict[str, Any]:
        calls["exit"] += 1
        return {"evaluated": 1}

    async def dip_hook() -> dict[str, Any]:
        calls["dip"] += 1
        return {"checked": 1}

    async def eod_hook() -> dict[str, Any]:
        calls["eod"] += 1
        return {"action": "flatten", "session": "2026-06-03"}

    autonomy.configure(
        exit_rules_tick=exit_hook,
        dip_watch_tick=dip_hook,
        eod_flatten_tick=eod_hook,
    )

    out = await autonomy.run_fast_tick()
    assert out["ok"] is True
    assert out["eod_flatten"] == {"action": "flatten", "session": "2026-06-03"}
    assert calls == {"exit": 1, "dip": 1, "eod": 1}
    assert autonomy.STATE.last_fast_tick_eod_flatten == {
        "action": "flatten",
        "session": "2026-06-03",
    }


@pytest.mark.asyncio
async def test_eod_flatten_disabled_flag_skips_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_market_open(monkeypatch, open_=True)
    monkeypatch.setattr(autonomy, "_default_pause_check", lambda: False)
    fired = {"n": 0}

    async def eod_hook() -> dict[str, Any]:
        fired["n"] += 1
        return {"action": "flatten"}

    autonomy.configure(
        exit_rules_tick=None,
        dip_watch_tick=None,
        eod_flatten_tick=eod_hook,
        eod_flatten_enabled=False,
    )
    out = await autonomy.run_fast_tick()
    assert out["eod_flatten"] is None
    assert fired["n"] == 0


@pytest.mark.asyncio
async def test_eod_flatten_exception_isolated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """EOD-flatten errors must not take down the other fast-loop branches."""
    _force_market_open(monkeypatch, open_=True)
    monkeypatch.setattr(autonomy, "_default_pause_check", lambda: False)

    async def boom() -> dict[str, Any]:
        raise RuntimeError("broker timeout")

    async def good_exit() -> dict[str, Any]:
        return {"evaluated": 2}

    autonomy.configure(
        exit_rules_tick=good_exit,
        dip_watch_tick=None,
        eod_flatten_tick=boom,
    )
    out = await autonomy.run_fast_tick()
    assert out["exit_rules"] == {"evaluated": 2}
    assert out["eod_flatten"] is not None
    assert "error" in out["eod_flatten"]
    assert "broker timeout" in out["eod_flatten"]["error"]


@pytest.mark.asyncio
async def test_slow_loop_skips_phase25_when_market_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_market_open(monkeypatch, open_=False)
    monkeypatch.setattr(autonomy, "_default_pause_check", lambda: False)

    async def fake_sweep() -> dict[str, Any]:
        return {"status": "ok", "candidates": []}

    fired = {"n": 0}

    async def exit_hook() -> dict[str, Any]:
        fired["n"] += 1
        return {}

    autonomy.configure(
        exit_rules_tick=exit_hook,
        dip_watch_tick=exit_hook,
        self_improve_enabled=False,
    )
    out = await autonomy.run_one_tick(sweep_runner=fake_sweep)
    assert out["ok"] is True
    assert out["exit_rules"] is None
    assert out["dip_watch"] is None
    assert fired["n"] == 0


# ---------------------------------------------------------------------------
# Phase 35 — adaptive cadence + watchdog heartbeat
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_fast_tick_writes_heartbeat_on_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fast tick must stamp last_fast_tick_heartbeat_at on success
    so the curiosity watchdog can detect a stalled loop."""
    _force_market_open(monkeypatch, open_=True)
    monkeypatch.setattr(autonomy, "_default_pause_check", lambda: False)

    async def exit_hook() -> dict[str, Any]:
        return {"ok": True}

    autonomy.configure(exit_rules_tick=exit_hook, dip_watch_tick=None)
    assert autonomy.STATE.last_fast_tick_heartbeat_at is None
    out = await autonomy.run_fast_tick()
    assert out["ok"] is True
    assert autonomy.STATE.last_fast_tick_heartbeat_at is not None
    # Heartbeat matches the tick timestamp.
    assert autonomy.STATE.last_fast_tick_heartbeat_at == autonomy.STATE.last_fast_tick_at


@pytest.mark.asyncio
async def test_run_fast_tick_skips_do_not_advance_heartbeat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A market-closed skip should NOT advance the heartbeat — the watchdog
    must still see staleness if the loop hasn't actually run."""
    _force_market_open(monkeypatch, open_=False)
    monkeypatch.setattr(autonomy, "_default_pause_check", lambda: False)

    async def exit_hook() -> dict[str, Any]:
        return {}

    autonomy.configure(exit_rules_tick=exit_hook, dip_watch_tick=None)
    autonomy.STATE.last_fast_tick_heartbeat_at = None
    out = await autonomy.run_fast_tick()
    assert out["skipped"] is True
    # Heartbeat must remain None (no successful tick happened).
    assert autonomy.STATE.last_fast_tick_heartbeat_at is None


def test_autonomy_config_has_hot_cadence_default() -> None:
    """fast_loop_hot_seconds defaults to 10s — fast enough to catch a
    sub-minute price spike without slamming the broker API."""
    cfg = autonomy.AutonomyConfig()
    assert cfg.fast_loop_hot_seconds == 10
    # And the regular interval is 60s by default.
    assert cfg.fast_loop_seconds == 60
    # FAST_LOOP_STALE_S is exposed for the curiosity watchdog.
    assert autonomy.FAST_LOOP_STALE_S == 300


def test_autonomy_config_hot_cadence_env_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Operator can tune the hot cadence via AUTONOMY_FAST_LOOP_HOT_S."""
    monkeypatch.setenv("AUTONOMY_FAST_LOOP_HOT_S", "3")
    # AutonomyConfig reads env at default-eval time, so we have to
    # rebuild the dataclass to see the override.
    import importlib

    import packages.cockpit.web.autonomy as autonomy_mod

    importlib.reload(autonomy_mod)
    try:
        cfg = autonomy_mod.AutonomyConfig()
        assert cfg.fast_loop_hot_seconds == 3
    finally:
        # Reload again WITHOUT the env override so downstream tests see
        # the default and the patched module references stay sane.
        monkeypatch.delenv("AUTONOMY_FAST_LOOP_HOT_S", raising=False)
        importlib.reload(autonomy_mod)
