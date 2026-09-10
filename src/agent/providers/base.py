"""The interface every generation backend implements.

Kept deliberately small: a backend is a class exposing ``name`` and ``generate``.
This is what lets a self-hosted vLLM backend (Phase 2) drop in next to the API
providers with no changes to the RAG pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from src.common.enums import LLMProviderName

__all__ = ["Completion", "LLMProvider", "LLMProviderName"]


@dataclass(frozen=True, slots=True)
class Completion:
    """A single generation result, plus the accounting the pipeline reports on."""

    text: str
    completion_tokens: int
    provider: LLMProviderName


@runtime_checkable
class LLMProvider(Protocol):
    name: LLMProviderName

    def generate(self, system: str, prompt: str) -> Completion:
        """Return a grounded answer for the assembled prompt."""
        ...
