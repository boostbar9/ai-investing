"""Hardware-aware LLM model profiles.

The §5 spec table assumes a 48GB+ VRAM rig (DeepSeek R1 671B / Qwen 2.5 72B
both loaded). Most operator hardware is smaller than that, so the router
now resolves each agent's chain through a **profile** chosen for the
operator's box.

Active profile is picked by, in order of precedence:

1. ``HARDWARE_PROFILE`` env var (explicit override — e.g. ``rx_7900_xt``)
2. A small heuristic on detected total VRAM if available
3. Default to ``balanced`` (Qwen 14B class) \u2014 works on anything with 16GB
   system RAM, slow on CPU but functional

Profiles live in this file as data so they review in PRs like config, not
like code. The chain shape (primary \u2192 backup \u2192 quantized) and the agent
names match the existing ``LLMChain`` contract in ``llm_router``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class LLMChain:
    """Mirror of llm_router.LLMChain so profiles import cleanly.

    Kept duplicated here (instead of importing from llm_router) so this
    module has no runtime dependency on the router \u2014 profiles can be loaded
    by tests, CLIs, or docs generators without spinning up the router.
    """

    primary: str
    backup: str
    quantized: str


@dataclass(frozen=True)
class HardwareProfile:
    name: str
    description: str
    min_vram_gb: int
    min_ram_gb: int
    chains: dict[str, LLMChain]


# ---------------------------------------------------------------------------
# Profile catalogue
# ---------------------------------------------------------------------------
#
# Sizing notes (q4_K_M quantization unless noted):
#   *  3B  weights \u2248  2 GB on disk, \u2248  3 GB VRAM
#   *  7B  weights \u2248  4 GB on disk, \u2248  5 GB VRAM
#   * 14B  weights \u2248  9 GB on disk, \u2248 11 GB VRAM
#   * 32B  weights \u2248 20 GB on disk, \u2248 24 GB VRAM
#   * 70B  weights \u2248 42 GB on disk, \u2248 48 GB VRAM
#   * 671B weights \u2248 380 GB \u2014 H100-class only


# Operator default: Devin's box (Ryzen 7 5700X3D, RX 7900 XT 20GB, 32GB RAM).
# Sized so two 14B models can coexist in VRAM (~22GB peak; one swaps when
# the other is hot). Quantized fallback drops to 7B for cold-start latency.
RX_7900_XT = HardwareProfile(
    name="rx_7900_xt",
    description="20GB AMD GPU class (RX 7900 XT/XTX), 32GB system RAM",
    min_vram_gb=20,
    min_ram_gb=32,
    chains={
        "research":  LLMChain("deepseek-r1:14b",            "qwen2.5:14b-instruct-q4_K_M", "qwen2.5:7b-instruct-q4_K_M"),
        "strategy":  LLMChain("qwen2.5:14b-instruct-q4_K_M", "llama3.1:8b-instruct-q4_K_M", "qwen2.5:7b-instruct-q4_K_M"),
        "risk":      LLMChain("deepseek-r1:14b",            "qwen2.5:14b-instruct-q4_K_M", "qwen2.5:7b-instruct-q4_K_M"),
        "execution": LLMChain("llama3.1:8b-instruct-q4_K_M", "qwen2.5:7b-instruct-q4_K_M", "llama3.2:3b-instruct-q4_K_M"),
    },
)


# CPU-only / no GPU / small NUC. Everything 7B or smaller. Slow but works.
CPU_ONLY = HardwareProfile(
    name="cpu_only",
    description="No usable GPU \u2014 7B-class on CPU, 16GB+ RAM",
    min_vram_gb=0,
    min_ram_gb=16,
    chains={
        "research":  LLMChain("qwen2.5:7b-instruct-q4_K_M", "llama3.1:8b-instruct-q4_K_M", "llama3.2:3b-instruct-q4_K_M"),
        "strategy":  LLMChain("qwen2.5:7b-instruct-q4_K_M", "llama3.1:8b-instruct-q4_K_M", "llama3.2:3b-instruct-q4_K_M"),
        "risk":      LLMChain("qwen2.5:7b-instruct-q4_K_M", "llama3.1:8b-instruct-q4_K_M", "llama3.2:3b-instruct-q4_K_M"),
        "execution": LLMChain("llama3.2:3b-instruct-q4_K_M", "qwen2.5:7b-instruct-q4_K_M", "llama3.2:3b-instruct-q4_K_M"),
    },
)


# Middle of the road: any 8-12GB NVIDIA card (RTX 4060 Ti, 3060 12GB, 4070).
BALANCED = HardwareProfile(
    name="balanced",
    description="8-12GB GPU class, 16GB+ system RAM",
    min_vram_gb=8,
    min_ram_gb=16,
    chains={
        "research":  LLMChain("qwen2.5:7b-instruct-q4_K_M", "llama3.1:8b-instruct-q4_K_M", "llama3.2:3b-instruct-q4_K_M"),
        "strategy":  LLMChain("qwen2.5:7b-instruct-q4_K_M", "llama3.1:8b-instruct-q4_K_M", "llama3.2:3b-instruct-q4_K_M"),
        "risk":      LLMChain("qwen2.5:7b-instruct-q4_K_M", "llama3.1:8b-instruct-q4_K_M", "llama3.2:3b-instruct-q4_K_M"),
        "execution": LLMChain("llama3.1:8b-instruct-q4_K_M", "qwen2.5:7b-instruct-q4_K_M", "llama3.2:3b-instruct-q4_K_M"),
    },
)


# Enthusiast: RTX 4090 24GB, RTX 6000 Ada, etc.
HIGH_END = HardwareProfile(
    name="high_end",
    description="24GB GPU class (4090, 7900 XTX), 64GB+ system RAM",
    min_vram_gb=24,
    min_ram_gb=64,
    chains={
        "research":  LLMChain("deepseek-r1:32b",            "qwen2.5:32b-instruct-q4_K_M", "qwen2.5:14b-instruct-q4_K_M"),
        "strategy":  LLMChain("qwen2.5:32b-instruct-q4_K_M", "llama3.3:70b-q4_K_M",        "qwen2.5:14b-instruct-q4_K_M"),
        "risk":      LLMChain("deepseek-r1:32b",            "qwen2.5:32b-instruct-q4_K_M", "qwen2.5:14b-instruct-q4_K_M"),
        "execution": LLMChain("llama3.3:70b-q4_K_M",        "qwen2.5:14b-instruct-q4_K_M", "llama3.2:3b-instruct-q4_K_M"),
    },
)


# Workstation: dual 4090 / H100 80GB / A6000. Matches the §5 spec defaults.
WORKSTATION = HardwareProfile(
    name="workstation",
    description="48GB+ VRAM, 128GB+ RAM \u2014 \u00a75 spec defaults",
    min_vram_gb=48,
    min_ram_gb=128,
    chains={
        "research":  LLMChain("deepseek-r1:70b", "qwen2.5:72b",     "qwen2.5:7b-instruct-q4_K_M"),
        "strategy":  LLMChain("qwen2.5:72b",     "llama3.3:70b",    "qwen2.5:7b-instruct-q4_K_M"),
        "risk":      LLMChain("deepseek-r1:70b", "mistral-large",   "deepseek-r1:7b-q4_K_M"),
        "execution": LLMChain("llama3.3:70b",    "mistral-large",   "llama3.2:3b-q4_K_M"),
    },
)


PROFILES: dict[str, HardwareProfile] = {
    p.name: p
    for p in (CPU_ONLY, BALANCED, RX_7900_XT, HIGH_END, WORKSTATION)
}


def _vram_to_profile(vram_gb: int) -> HardwareProfile:
    """Pick the largest profile whose ``min_vram_gb`` we satisfy."""
    candidates = sorted(PROFILES.values(), key=lambda p: p.min_vram_gb, reverse=True)
    for p in candidates:
        if vram_gb >= p.min_vram_gb:
            return p
    return CPU_ONLY


def active_profile(
    *,
    env_value: str | None = None,
    vram_gb: int | None = None,
) -> HardwareProfile:
    """Resolve the active hardware profile.

    Override order:
      1. Explicit ``env_value`` (or ``$HARDWARE_PROFILE``)
      2. ``vram_gb`` argument (or ``$DETECTED_VRAM_GB``)
      3. ``BALANCED`` (safe middle default)
    """
    env_value = env_value if env_value is not None else os.getenv("HARDWARE_PROFILE")
    if env_value:
        key = env_value.strip().lower()
        if key in PROFILES:
            return PROFILES[key]
        raise ValueError(
            f"HARDWARE_PROFILE={env_value!r} is not one of {sorted(PROFILES)}"
        )

    if vram_gb is None:
        raw = os.getenv("DETECTED_VRAM_GB")
        if raw:
            try:
                vram_gb = int(raw)
            except ValueError:
                vram_gb = None

    if vram_gb is not None:
        return _vram_to_profile(vram_gb)

    return BALANCED


def chain_for(agent: str, profile: HardwareProfile | None = None) -> LLMChain:
    """Return the (primary, backup, quantized) chain for one agent."""
    p = profile or active_profile()
    if agent not in p.chains:
        raise KeyError(f"profile {p.name!r} has no chain for agent {agent!r}")
    return p.chains[agent]


def all_models(profile: HardwareProfile | None = None) -> list[str]:
    """Flat, deduplicated list of every model the active profile needs.

    Used by the ``make pull-models`` target so the operator knows exactly
    which Ollama tags to download up front.
    """
    p = profile or active_profile()
    seen: dict[str, None] = {}
    for chain in p.chains.values():
        for m in (chain.primary, chain.backup, chain.quantized):
            seen.setdefault(m, None)
    return list(seen.keys())
