"""The one interface every generation backend implements.

Keeping this deliberately small is what lets a self-hosted vLLM backend (Phase 2)
drop in next to the API providers with zero changes to the RAG pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class Completion:
    text: str
    completion_tokens: int
    provider: str


class LLMProvider(Protocol):
    name: str

    def generate(self, system: str, prompt: str) -> Completion:
        """Return a grounded answer for the assembled prompt."""
        ...
