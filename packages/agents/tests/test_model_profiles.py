"""Tests for the hardware-aware model profile selector."""
from __future__ import annotations

import pytest

from packages.agents.model_profiles import (
    BALANCED,
    CPU_ONLY,
    HIGH_END,
    PROFILES,
    RX_7900_XT,
    WORKSTATION,
    active_profile,
    all_models,
    chain_for,
)


def test_known_profiles_are_registered() -> None:
    assert set(PROFILES) == {
        "cpu_only", "balanced", "rx_7900_xt", "high_end", "workstation"
    }


def test_env_override_picks_named_profile() -> None:
    assert active_profile(env_value="rx_7900_xt") is RX_7900_XT
    assert active_profile(env_value="cpu_only") is CPU_ONLY
    assert active_profile(env_value="workstation") is WORKSTATION


def test_env_override_is_case_insensitive() -> None:
    assert active_profile(env_value="RX_7900_XT") is RX_7900_XT


def test_unknown_profile_raises() -> None:
    with pytest.raises(ValueError, match="HARDWARE_PROFILE"):
        active_profile(env_value="rtx_5090_super")


def test_vram_heuristic_steps_through_tiers() -> None:
    assert active_profile(vram_gb=0) is CPU_ONLY
    assert active_profile(vram_gb=8) is BALANCED
    assert active_profile(vram_gb=20) is RX_7900_XT
    assert active_profile(vram_gb=24) is HIGH_END
    assert active_profile(vram_gb=48) is WORKSTATION
    assert active_profile(vram_gb=80) is WORKSTATION  # nothing bigger


def test_chain_for_rx_7900_xt_research_uses_deepseek_32b() -> None:
    """May-2026 upgrade: heavy-reasoning agents on 20GB cards get DeepSeek R1 32B.

    The 32B distill is the largest model that fits comfortably in 20GB VRAM
    alongside Ollama's own overhead and a small KV cache. Anything bigger
    (70B) gets evicted at load time.
    """
    chain = chain_for("research", RX_7900_XT)
    assert chain.primary == "deepseek-r1:32b"
    # Backup must NOT be the 70B which won't fit.
    assert "70b" not in chain.backup.lower()


def test_chain_for_rx_7900_xt_risk_uses_deepseek_32b() -> None:
    """Risk gate is the most safety-critical agent — top-tier model wins."""
    assert chain_for("risk", RX_7900_XT).primary == "deepseek-r1:32b"


def test_chain_for_rx_7900_xt_discovery_uses_deepseek_32b() -> None:
    """Discovery needs to spot novel patterns — heavy reasoner."""
    assert chain_for("discovery", RX_7900_XT).primary == "deepseek-r1:32b"


def test_chain_for_rx_7900_xt_strategy_and_execution_stay_mid_tier() -> None:
    """Mid-tier agents on rx_7900_xt run a smaller, faster model so the
    32B reasoners aren't blocked waiting for VRAM."""
    assert "14b" in chain_for("strategy", RX_7900_XT).primary.lower()
    assert "14b" in chain_for("execution", RX_7900_XT).primary.lower()


def test_every_profile_has_a_discovery_chain() -> None:
    """Adding Discovery as a 5th agent means every profile must register a
    chain for it — otherwise active_profile() would KeyError at runtime."""
    for prof in (CPU_ONLY, BALANCED, RX_7900_XT, HIGH_END, WORKSTATION):
        chain = chain_for("discovery", prof)
        assert chain.primary, f"profile {prof.name} missing discovery primary"
        assert chain.backup, f"profile {prof.name} missing discovery backup"
        assert chain.quantized, f"profile {prof.name} missing discovery quantized"


def test_chain_for_workstation_matches_spec_defaults() -> None:
    chain = chain_for("research", WORKSTATION)
    assert chain.primary == "deepseek-r1:70b"
    assert chain.backup == "qwen2.5:72b"


def test_all_models_returns_dedup_list_per_profile() -> None:
    models = all_models(RX_7900_XT)
    assert len(models) == len(set(models)), "all_models must dedupe"
    # Sanity: the heavy reasoner, its 14B backup, and the mid-tier fast
    # model all show up so ``ollama pull`` knows to grab them.
    assert any("deepseek-r1:32b" in m for m in models)
    assert any("deepseek-r1:14b" in m for m in models)
    assert any("qwen3:14b" in m for m in models)


def test_chain_for_unknown_agent_raises() -> None:
    with pytest.raises(KeyError, match="unknown_agent"):
        chain_for("unknown_agent", RX_7900_XT)


def test_default_profile_falls_back_to_balanced(monkeypatch) -> None:
    monkeypatch.delenv("HARDWARE_PROFILE", raising=False)
    monkeypatch.delenv("DETECTED_VRAM_GB", raising=False)
    assert active_profile() is BALANCED


def test_env_var_path_via_os_environ(monkeypatch) -> None:
    monkeypatch.setenv("HARDWARE_PROFILE", "rx_7900_xt")
    assert active_profile() is RX_7900_XT


def test_detected_vram_env_path(monkeypatch) -> None:
    monkeypatch.delenv("HARDWARE_PROFILE", raising=False)
    monkeypatch.setenv("DETECTED_VRAM_GB", "20")
    assert active_profile() is RX_7900_XT
