"""Provider registry: pick the generation backend from config."""
from __future__ import annotations

from src.agent.providers.base import LLMProvider
from src.common.config import settings


def get_provider() -> LLMProvider:
    name = settings.llm_provider.lower()
    if name == "anthropic":
        from src.agent.providers.anthropic_provider import AnthropicProvider
        return AnthropicProvider()
    if name == "openai":
        from src.agent.providers.openai_provider import OpenAIProvider
        return OpenAIProvider()
    if name == "vllm":
        from src.agent.providers.vllm_provider import VLLMProvider
        return VLLMProvider()
    raise ValueError(f"unknown LLM_PROVIDER: {settings.llm_provider}")
