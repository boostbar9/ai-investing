"""Phase 13: smoke tests for the 'policy' strategy wiring in paper_trade.

We don't exercise the live data path -- those tests live in the
package-specific suites for HMM, sweep loader, etc. Here we just verify
the three contract pieces that the rest of Phase 13 relies on:

  1. ``"policy"`` is a registered strategy with a non-empty universe.
  2. ``compute_target_weights("policy")`` populates the module-level
     ``_LAST_POLICY_DECISIONS`` side-effect holder, so the decision log
     can capture calibration data.
  3. The returned weights dict drops explicit zeros (zeros pollute the
     sentiment overlay multiplication downstream).
"""
from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from tools import paper_trade as pt


def test_policy_is_a_registered_strategy() -> None:
    """Phase 13 wires 'policy' as a third strategy alongside ensemble."""
    assert "policy" in pt.STRATEGY_CHOICES
    assert "policy" in pt.STRATEGY_UNIVERSE
    assert len(pt.STRATEGY_UNIVERSE["policy"]) > 0


def test_compute_target_weights_routes_policy_to_policy_function(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """compute_target_weights('policy') must dispatch to compute_policy_weights,
    not fall through to the ensemble/single-strategy branches."""
    sentinel = {"SENTINEL": 0.5}
    called: dict[str, bool] = {"policy": False}

    def fake_policy() -> dict[str, float]:
        called["policy"] = True
        return sentinel

    # Seed stale state -- it must be cleared by the routing function's
    # entry guard regardless of which strategy we route to.
    pt._LAST_POLICY_DECISIONS = [
        {"symbol": "STALE", "action": "buy", "confidence": 0.9}
    ]

    monkeypatch.setattr(pt, "compute_policy_weights", fake_policy)
    out = pt.compute_target_weights("policy")

    assert called["policy"] is True
    assert out == sentinel
    # The routing function clears _LAST_POLICY_DECISIONS at the top. Our
    # fake policy doesn't repopulate it, so the cleared state is what
    # leaks through here.
    assert pt._LAST_POLICY_DECISIONS == []


def test_compute_policy_weights_populates_decisions_and_drops_zeros(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """compute_policy_weights must call the policy, log full decisions
    (including SELLs and HOLDs), and return weights with zeros stripped."""
    # Build a small fake panel covering the policy universe.
    universe = pt.STRATEGY_UNIVERSE["policy"]
    idx = pd.date_range("2026-01-01", periods=30, freq="D")
    fake_panel = pd.DataFrame(
        {s: [100.0 + (i % 3) for i in range(30)] for s in universe}, index=idx
    )
    monkeypatch.setattr(pt, "load_panel", lambda symbols: fake_panel)

    # Force a deterministic regime + posterior so threshold math is
    # predictable. Bull regime + high posterior -> strong BUYs trigger.
    class _Reading:
        regime = "bull"
        confidence = 0.9

    import packages.regime.hmm as hmm_mod

    monkeypatch.setattr(hmm_mod, "detect_regime", lambda *a, **k: _Reading())

    # Seed sweep candidates: one strong (will BUY), one moderate (will HOLD).
    strong_sym = universe[0]
    mid_sym = universe[1] if len(universe) > 1 else universe[0]
    fake_cands: list[dict[str, Any]] = [
        {
            "symbol": strong_sym,
            "confidence": 1.0,
            "reddit_trust": 0.9,
            "corroborated": True,
        },
        {"symbol": mid_sym, "confidence": 0.4, "reddit_trust": 0.3},
    ]
    monkeypatch.setattr(pt, "_load_latest_sweep_candidates", lambda: fake_cands)

    # Ensemble: agree on the strong name, silent on the rest. The
    # alignment vote nudges composite over the BUY threshold.
    monkeypatch.setattr(
        pt, "compute_ensemble_weights", lambda: {strong_sym: 0.10}
    )

    # Important: clear holder first so we can assert it was *repopulated*.
    pt._LAST_POLICY_DECISIONS = []

    weights = pt.compute_policy_weights()

    # Side effect: per-symbol decision list captured for calibration log.
    assert len(pt._LAST_POLICY_DECISIONS) >= 1
    actions = {d["symbol"]: d["action"] for d in pt._LAST_POLICY_DECISIONS}
    assert actions.get(strong_sym) == "buy"

    # The strong candidate must produce a positive weight.
    assert strong_sym in weights
    assert weights[strong_sym] > 0.0

    # No zero weights in the returned dict -- zeros confuse the
    # sentiment overlay which multiplies weight by a sentiment factor.
    assert all(v > 0.0 for v in weights.values())
