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


def test_chain_for_rx_7900_xt_uses_14b_class() -> None:
    chain = chain_for("research", RX_7900_XT)
    # Spec calls for DeepSeek R1 reasoning; on this tier we use the 14B
    # distill. Critical assertion: NOT the 70B which won't fit.
    assert "70b" not in chain.primary.lower()
    assert "14b" in chain.primary.lower()


def test_chain_for_workstation_matches_spec_defaults() -> None:
    chain = chain_for("research", WORKSTATION)
    assert chain.primary == "deepseek-r1:70b"
    assert chain.backup == "qwen2.5:72b"


def test_all_models_returns_dedup_list_per_profile() -> None:
    models = all_models(RX_7900_XT)
    assert len(models) == len(set(models)), "all_models must dedupe"
    # Sanity: a handful of expected tags appear
    assert any("deepseek-r1:14b" in m for m in models)
    assert any("qwen2.5:14b" in m for m in models)
    assert any("llama3.2:3b" in m for m in models)


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
