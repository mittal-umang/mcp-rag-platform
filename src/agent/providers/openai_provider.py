"""OpenAI Chat Completions backend."""
from __future__ import annotations

from src.agent.providers.base import Completion
from src.common.config import settings


class OpenAIProvider:
    name = "openai"

    def __init__(self) -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key=settings.openai_api_key)
        self._model = settings.llm_model

    def generate(self, system: str, prompt: str) -> Completion:
        resp = self._client.chat.completions.create(
            model=self._model,
            max_tokens=800,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        )
        return Completion(
            text=resp.choices[0].message.content or "",
            completion_tokens=resp.usage.completion_tokens,
            provider=self.name,
        )
