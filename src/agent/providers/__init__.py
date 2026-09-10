"""Pluggable LLM generation backends.

Public surface: the ``LLMProvider`` interface, the ``LLMProviderName`` enum of
supported backends, and ``get_provider`` to resolve the configured one.
"""
from __future__ import annotations

from src.agent.providers.base import Completion, LLMProvider, LLMProviderName
from src.agent.providers.registry import get_provider

__all__ = ["Completion", "LLMProvider", "LLMProviderName", "get_provider"]
