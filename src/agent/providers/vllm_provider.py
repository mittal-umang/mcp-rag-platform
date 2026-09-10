"""Phase 2: self-hosted inference via vLLM's OpenAI-compatible server.

vLLM exposes an OpenAI-compatible /v1 API, so this reuses the OpenAI client
pointed at the in-cluster vLLM service. Enable with LLM_PROVIDER=vllm and
vllm.enabled=true in the Helm values (needs a GPU node).

Continuous batching and paged-attention are handled server-side by vLLM; this
client stays thin. Streaming can be layered on later for time-to-first-token.
"""
from __future__ import annotations

from src.agent.providers.base import Completion
from src.common.config import settings


class VLLMProvider:
    name = "vllm"

    def __init__(self) -> None:
        from openai import OpenAI

        # vLLM ignores the key but the client requires one
        self._client = OpenAI(base_url=settings.vllm_base_url, api_key="not-needed")
        self._model = settings.vllm_model

    def generate(self, system: str, prompt: str) -> Completion:
        resp = self._client.chat.completions.create(
            model=self._model,
            max_tokens=800,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        )
        usage = resp.usage
        return Completion(
            text=resp.choices[0].message.content or "",
            completion_tokens=usage.completion_tokens if usage else 0,
            provider=self.name,
        )
