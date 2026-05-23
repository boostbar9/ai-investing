"""LLM router with auto-fallback chain (§5 + §18 mitigations).

Primary model → backup model → quantized GGUF fallback. Each agent declares
its chain and the router walks it on timeout / OOM / non-JSON output.
"""
from __future__ import annotations

import asyncio
import json
import os

import httpx

from packages.agents.model_profiles import LLMChain, chain_for
from packages.shared.otel import span

# Re-exported so existing call sites that imported LLMChain from here keep
# working.
__all__ = ["LLMChain", "LLMError", "LLMRouter"]


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

    async def generate_json(
        self,
        agent: str,
        prompt: str,
        *,
        decision_id: str,
        timeout_seconds: int = 30,
    ) -> dict:
        """Walk the chain until one model returns parseable JSON."""
        chain = chain_for(agent)
        models = (chain.primary, chain.backup, chain.quantized)
        last_err: Exception | None = None

        for model in models:
            with span(
                "llm.generate",
                {"agent": agent, "model": model, "decision_id": decision_id},
            ) as s:
                try:
                    text = await asyncio.wait_for(
                        self._call(model, prompt), timeout=timeout_seconds
                    )
                    s.set_attribute("llm.chars_out", len(text))
                    return json.loads(text)
                except Exception as e:
                    last_err = e
                    s.set_attribute("llm.error", str(e)[:200])
                    continue

        raise LLMError(f"all models failed for {agent}: {last_err}")

    async def _call(self, model: str, prompt: str) -> str:
        r = await self._client.post(
            f"{self.host}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {
                    "num_predict": self.max_tokens,
                    "temperature": self.temperature,
                },
            },
        )
        r.raise_for_status()
        return r.json().get("response", "")

    async def aclose(self) -> None:
        await self._client.aclose()
