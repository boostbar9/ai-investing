"""Phase 25.5 — Tests for `latest_regime()` autonomy-brain fallback.

The hero card on the cockpit was showing "—" because the paper runs
ledger does not carry a ``regime`` key on every run. The brain knows
the regime in memory; the fallback is to read it from the brain
snapshot when the ledger lacks it.

Contract this test locks in:
  * When the latest run has a `regime` value, it wins (no fallback).
  * When the latest run has no `regime` (typical), we fall back to
    ``autonomy_brain.snapshot()["regime"]["label"]``.
  * The brain's confidence is surfaced when the run lacks one.
  * A failing/missing snapshot does not crash the endpoint — it just
    returns ``auto: None`` and the UI continues to display "—".
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from packages.cockpit.web import server as srv


@pytest.fixture
def fake_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from packages.cockpit import state as st

    path = tmp_path / "state.json"
    monkeypatch.setattr(st, "STATE_PATH", path)
    monkeypatch.setattr(st.load_state, "__defaults__", (path,))
    monkeypatch.setattr(st.save_state, "__defaults__", (path,))
    return path


def _seed_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, runs: list[dict]) -> None:
    log = tmp_path / "runs.jsonl"
    log.write_text("\n".join(json.dumps(r) for r in runs) + "\n")
    monkeypatch.setattr(srv, "PAPER_LOG", log)


def test_latest_regime_uses_ledger_when_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_state: Path
) -> None:
    """When the most recent run carries a `regime` field, prefer it."""
    _seed_runs(
        tmp_path,
        monkeypatch,
        [
            {
                "ts": "2026-06-01T20:00:00+00:00",
                "regime": "bull",
                "regime_confidence": 0.42,
                "account_equity": 100000.0,
            }
        ],
    )
    # Brain snapshot must NOT be consulted in this branch.
    def _boom() -> dict:
        raise AssertionError("brain.snapshot should not be called")

    monkeypatch.setattr(srv.autonomy_brain, "snapshot", _boom)

    out = srv.latest_regime()
    assert out["auto"] == "bull"
    assert out["confidence"] == 0.42
    assert out["effective"] == "bull"


def test_latest_regime_falls_back_to_brain_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_state: Path
) -> None:
    """When the run lacks `regime`, surface the brain's live label."""
    _seed_runs(
        tmp_path,
        monkeypatch,
        [
            {
                "ts": "2026-06-01T20:00:00+00:00",
                "account_equity": 100000.0,
                # No regime field — exactly the bug.
            }
        ],
    )
    # The live brain publishes regime under `last_regime` (see
    # autonomy.snapshot()). Verify the fallback consumes that key.
    monkeypatch.setattr(
        srv.autonomy_brain,
        "snapshot",
        lambda: {"last_regime": {"label": "risk_on", "confidence": 1.0}},
    )

    out = srv.latest_regime()
    assert out["auto"] == "risk_on"
    assert out["confidence"] == 1.0
    assert out["effective"] == "risk_on"


def test_latest_regime_accepts_regime_key_for_forward_compat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_state: Path
) -> None:
    """If the brain is ever renamed to publish under `regime`, support it."""
    _seed_runs(
        tmp_path,
        monkeypatch,
        [{"ts": "2026-06-01T20:00:00+00:00", "account_equity": 100000.0}],
    )
    monkeypatch.setattr(
        srv.autonomy_brain,
        "snapshot",
        lambda: {"regime": {"label": "risk_off", "confidence": 0.8}},
    )

    out = srv.latest_regime()
    assert out["auto"] == "risk_off"
    assert out["confidence"] == 0.8


def test_latest_regime_brain_failure_is_swallowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_state: Path
) -> None:
    """If the brain throws, the endpoint must NOT crash."""
    _seed_runs(
        tmp_path,
        monkeypatch,
        [{"ts": "2026-06-01T20:00:00+00:00", "account_equity": 100000.0}],
    )

    def _explode() -> dict:
        raise RuntimeError("brain unavailable")

    monkeypatch.setattr(srv.autonomy_brain, "snapshot", _explode)

    out = srv.latest_regime()
    # Degraded but not broken — UI will render "—" in this case.
    assert out["auto"] is None
    assert out["confidence"] is None
    assert out["effective"] is None  # state.regime_override defaults to "auto"
