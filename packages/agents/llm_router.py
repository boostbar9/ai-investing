"""LLM router with auto-fallback chain (§5 + §18 mitigations).

Primary model → backup model → quantized GGUF fallback. Each agent declares
its chain and the router walks it on timeout / OOM / non-JSON output.
"""
from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass

import httpx

from packages.shared.otel import span


class LLMError(RuntimeError):
    """Raised when every model in the chain fails."""


@dataclass(frozen=True)
class LLMChain:
    primary: str
    backup: str
    quantized: str  # last-resort, e.g. ``qwen2.5:7b-q4_K_M``


# Per §5 table — paired with §18 hardware-too-weak mitigation.
CHAINS: dict[str, LLMChain] = {
    "research":  LLMChain("deepseek-r1:70b", "qwen2.5:72b",     "qwen2.5:7b-instruct-q4_K_M"),
    "strategy":  LLMChain("qwen2.5:72b",     "llama3.3:70b",    "qwen2.5:7b-instruct-q4_K_M"),
    "risk":      LLMChain("deepseek-r1:70b", "mistral-large",   "deepseek-r1:7b-q4_K_M"),
    "execution": LLMChain("llama3.3:70b",    "mistral-large",   "llama3.2:3b-q4_K_M"),
}


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
        chain = CHAINS[agent]
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
                except Exception as e:  # noqa: BLE001
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
