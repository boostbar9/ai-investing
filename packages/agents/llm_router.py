"""LLM router with auto-fallback chain (§5 + §18 mitigations).

Primary model → backup model → quantized GGUF fallback. Each agent declares
its chain and the router walks it on timeout / OOM / non-JSON output.

When the declared chain has not been pulled yet (fresh install, big model
still downloading), the router falls back to an emergency tier of tiny
quantized models that ship with every profile so the agents at least
run in a degraded-but-working mode rather than 404-spamming for an hour.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time

import httpx

from packages.agents.model_profiles import LLMChain, chain_for
from packages.shared.otel import span

log = logging.getLogger(__name__)

# Re-exported so existing call sites that imported LLMChain from here keep
# working.
__all__ = [
    "EMERGENCY_FALLBACK_MODELS",
    "LLMChain",
    "LLMError",
    "LLMRouter",
    "installed_matches",
]

# Connection retry knobs. Ollama can hiccup for a fraction of a second when
# loading a fresh model into VRAM (it briefly stops accepting new connections
# while it mmap()s the weights). One retry with a tiny backoff is enough to
# ride that out without papering over a daemon that's truly dead.
_CONNECT_RETRIES = 1
_CONNECT_RETRY_DELAY_S = 0.5

# Loading a cold 32B model from disk can take 20-30s on a spinning disk before
# Ollama even starts generating tokens. The first call to a freshly-installed
# model gets this longer ceiling so we don't spuriously fall back when the
# operator's box is just paging the weights in. Subsequent calls fall back to
# the caller-supplied timeout because by then the model is warm.
_COLD_START_TIMEOUT_S = 90

# Universal fallback tier. Picked because:
#   - qwen2.5:7b-instruct-q4_K_M appears in every profile's quantized slot,
#     so any operator who ran ``--auto`` for any profile already has it.
#   - llama3.2:3b-instruct-q4_K_M is ~2GB and runs on a potato; absolute
#     last-resort so the soak still produces JSON even on a half-broken
#     Ollama install.
# Order: best emergency first. The router walks them after the declared chain.
EMERGENCY_FALLBACK_MODELS: tuple[str, ...] = (
    "qwen2.5:7b-instruct-q4_K_M",
    "llama3.2:3b-instruct-q4_K_M",
)


def installed_matches(required: str, installed: frozenset[str]) -> bool:
    """Liberal name match between a declared model and Ollama's tag list.

    Mirrors :func:`tools.check_ollama._matches` so the router accepts the same
    "close enough" inventory as the cockpit health row. Cases we want to
    accept:

    * ``deepseek-r1:32b`` matches ``deepseek-r1:32b`` (exact)
    * ``deepseek-r1:32b`` matches ``deepseek-r1:32b-q4_K_M`` (quantized variant)
    * ``qwen2.5:7b-instruct-q4_K_M`` matches ``qwen2.5:7b-instruct-q4_K_M``
    * ``qwen2.5:7b-instruct-q4_K_M`` matches ``qwen2.5:7b-instruct-q4_K_S``
      (different K-quant; weights are interchangeable enough for fallback)
    * Bare base ``qwen3`` matches ``qwen3:14b`` (profile sometimes omits tag)
    """
    if required in installed:
        return True
    base = required.split(":", 1)[0]
    target_tag = required.split(":", 1)[1] if ":" in required else ""
    for tag in installed:
        if tag == base:
            return True
        if not tag.startswith(base + ":"):
            continue
        installed_tag = tag.split(":", 1)[1] if ":" in tag else ""
        if not target_tag:
            return True
        # Either side is a prefix of the other — e.g. "32b" vs "32b-q4_K_M",
        # or "7b-instruct-q4_K_M" vs "7b-instruct-q4".
        if installed_tag.startswith(target_tag) or target_tag.startswith(installed_tag):
            return True
    return False


class LLMError(RuntimeError):
    """Raised when every model in the chain fails."""


# Backwards-compat shim: callers that imported ``CHAINS`` get the active
# profile's chains lazily on every access (so HARDWARE_PROFILE env changes
# take effect without a re-import).
class _ChainsView:
    def __getitem__(self, agent: str) -> LLMChain:
        return chain_for(agent)

    def __contains__(self, agent: str) -> bool:
        try:
            chain_for(agent)
        except KeyError:
            return False
        return True


CHAINS = _ChainsView()


class LLMRouter:
    """Async router talking to Ollama's HTTP API."""

    # Cache the installed-models inventory for a short while so we don't
    # hammer /api/tags on every agent run. 30s is short enough that a
    # newly-pulled model becomes usable inside the same soak window.
    _INSTALLED_TTL_SECONDS = 30.0

    def __init__(
        self,
        host: str | None = None,
        client: httpx.AsyncClient | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.2,
    ) -> None:
        self.host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self._client = client or httpx.AsyncClient(timeout=120)
        self.max_tokens = max_tokens
        self.temperature = temperature
        # (cached_at, frozenset_of_installed_names). None until first fetch.
        self._installed_cache: tuple[float, frozenset[str]] | None = None
        # Models that have served at least one successful response in this
        # process — used to grant a longer cold-start timeout on first use
        # so we don't fall back just because the weights are paging in.
        self._warmed: set[str] = set()

    async def generate_json(
        self,
        agent: str,
        prompt: str,
        *,
        decision_id: str,
        timeout_seconds: int = 30,
    ) -> dict:
        """Walk the chain until one model returns parseable JSON.

        The walk order is:
          1. The agent's declared chain (primary → backup → quantized),
             filtered to models actually installed on the Ollama daemon
             so we don't 404 our way through three calls per agent.
          2. The universal emergency fallback tier, again filtered to
             installed models, so a fresh install still produces JSON
             while the bigger declared models are still downloading.
          3. The full declared chain unfiltered, as a final safety net
             in case ``/api/tags`` itself is the broken thing.
        """
        chain = chain_for(agent)
        declared = (chain.primary, chain.backup, chain.quantized)
        installed = await self._installed_models()

        # Build the walk order without duplicates while preserving priority.
        walk: list[str] = []
        seen: set[str] = set()

        def _push(model: str) -> None:
            if model and model not in seen:
                walk.append(model)
                seen.add(model)

        # Tier 1: declared chain, installed-first.
        if installed is not None:
            for m in declared:
                if installed_matches(m, installed):
                    _push(m)
            # Tier 2: emergency fallback, installed-first.
            for m in EMERGENCY_FALLBACK_MODELS:
                if installed_matches(m, installed):
                    _push(m)
        # Tier 3: everything declared, even if /api/tags said it's not
        # there. Catches the case where /api/tags itself is broken.
        for m in declared:
            _push(m)
        for m in EMERGENCY_FALLBACK_MODELS:
            _push(m)

        if not walk:
            raise LLMError(f"no usable models for {agent}")

        last_err: Exception | None = None
        for model in walk:
            is_fallback = model not in declared
            # First call to a cold model gets a longer ceiling so a 32B
            # mmap doesn't trigger a spurious fall-back. Subsequent calls
            # use the caller-supplied timeout.
            effective_timeout = (
                max(timeout_seconds, _COLD_START_TIMEOUT_S)
                if model not in self._warmed
                else timeout_seconds
            )
            with span(
                "llm.generate",
                {
                    "agent": agent,
                    "model": model,
                    "decision_id": decision_id,
                    "fallback": is_fallback,
                    "cold": model not in self._warmed,
                },
            ) as s:
                try:
                    text = await asyncio.wait_for(
                        self._call(model, prompt), timeout=effective_timeout
                    )
                    s.set_attribute("llm.chars_out", len(text))
                    self._warmed.add(model)
                    if is_fallback:
                        log.warning(
                            "agent %s served by emergency fallback model %s "
                            "-- declared chain not installed",
                            agent, model,
                        )
                    return json.loads(text)
                except Exception as e:
                    last_err = e
                    s.set_attribute("llm.error", str(e)[:200])
                    # If a model 404s mid-session it was uninstalled — evict
                    # the inventory cache so the next call re-reads /api/tags
                    # and stops trying this model.
                    if isinstance(e, httpx.HTTPStatusError) and e.response.status_code == 404:
                        self._installed_cache = None
                    continue

        raise LLMError(f"all models failed for {agent}: {last_err}")

    async def warmup(self, prompt: str = "ping") -> str | None:
        """Pre-load the smallest installed model so the first real agent call
        doesn't pay the cold-start latency.

        Picks the first installed model from the emergency fallback tier
        (tiniest first) since the goal is to be cheap, not capable. Returns
        the model name that was warmed, or ``None`` if no installed model
        could be reached. Never raises — warmup is best-effort.
        """
        installed = await self._installed_models()
        if installed is None:
            return None
        # Iterate emergency tier in reverse so smallest (llama3.2:3b) is
        # warmed first — fastest to load and runs on every box.
        for candidate in reversed(EMERGENCY_FALLBACK_MODELS):
            if not installed_matches(candidate, installed):
                continue
            try:
                await asyncio.wait_for(
                    self._call(candidate, prompt), timeout=_COLD_START_TIMEOUT_S
                )
                self._warmed.add(candidate)
                log.info("router warmed %s", candidate)
                return candidate
            except Exception as e:
                log.debug("warmup of %s failed: %s", candidate, e)
                continue
        return None

    async def _installed_models(self) -> frozenset[str] | None:
        """Return the set of installed model names, or ``None`` if Ollama is
        unreachable / ``/api/tags`` is broken.

        Cached for ``_INSTALLED_TTL_SECONDS`` to avoid hammering the daemon.
        Names are returned with their full ``name:tag`` form (and without
        tag for entries that didn't ship a tag) so simple ``in`` works.
        """
        now = time.monotonic()
        if (
            self._installed_cache is not None
            and now - self._installed_cache[0] < self._INSTALLED_TTL_SECONDS
        ):
            return self._installed_cache[1]
        try:
            r = await self._client.get(f"{self.host}/api/tags", timeout=2.5)
            r.raise_for_status()
            payload = r.json() or {}
        except Exception as e:
            log.debug("router could not fetch /api/tags: %s", e)
            # Don't poison the cache on a transient miss — we want the next
            # call to retry rather than re-fall-back blind.
            return None
        names: set[str] = set()
        for entry in payload.get("models", []):
            n = entry.get("name")
            if isinstance(n, str) and n:
                names.add(n)
                # Also store the bare base name so ``deepseek-r1`` matches
                # ``deepseek-r1:32b`` should a profile ever omit the tag.
                base = n.split(":", 1)[0]
                names.add(base)
        frozen = frozenset(names)
        self._installed_cache = (now, frozen)
        return frozen

    async def _call(self, model: str, prompt: str) -> str:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "num_predict": self.max_tokens,
                "temperature": self.temperature,
            },
        }
        # Retry transient connection errors once with a tiny backoff. We do
        # NOT retry HTTP errors (404 / 500) — those are loud failures the
        # caller should see immediately so the chain walks to the next model.
        last_conn_err: Exception | None = None
        for attempt in range(_CONNECT_RETRIES + 1):
            try:
                r = await self._client.post(f"{self.host}/api/generate", json=payload)
                r.raise_for_status()
                return r.json().get("response", "")
            except (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError) as e:
                last_conn_err = e
                if attempt < _CONNECT_RETRIES:
                    await asyncio.sleep(_CONNECT_RETRY_DELAY_S)
                    continue
                raise
        # Unreachable, but keeps type-checkers happy.
        raise last_conn_err  # type: ignore[misc]

    async def aclose(self) -> None:
        await self._client.aclose()
