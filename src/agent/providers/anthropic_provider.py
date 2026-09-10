"""Anthropic Messages API backend."""
from __future__ import annotations

from src.agent.providers.base import Completion
from src.common.config import settings


class AnthropicProvider:
    name = "anthropic"

    def __init__(self) -> None:
        import anthropic

        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self._model = settings.llm_model

    def generate(self, system: str, prompt: str) -> Completion:
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=800,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in resp.content if block.type == "text")
        return Completion(
            text=text,
            completion_tokens=resp.usage.output_tokens,
            provider=self.name,
        )
