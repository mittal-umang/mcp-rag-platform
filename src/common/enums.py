"""Enumerations shared across configuration and the agent runtime."""
from __future__ import annotations

from enum import StrEnum


class LLMProviderName(StrEnum):
    """Supported generation backends.

    The single source of truth for provider selection: ``settings.llm_provider`` is
    validated against this enum, and the provider registry is checked for completeness
    against it at import time. To add a backend: add a member here, register a factory
    in ``providers.registry``, and ship the provider class.
    """

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    VLLM = "vllm"
