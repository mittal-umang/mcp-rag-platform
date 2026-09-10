"""Provider registry: resolve the configured backend from the ``LLMProviderName`` enum.

Selection is a dict lookup keyed by the enum, not a branch ladder. Each factory
imports its client lazily, so the provider SDKs stay optional dependencies - only
the selected provider's package needs to be installed.
"""
from __future__ import annotations

from collections.abc import Callable

from src.agent.providers.base import LLMProvider, LLMProviderName
from src.common.config import settings

ProviderFactory = Callable[[], LLMProvider]


def _anthropic() -> LLMProvider:
    from src.agent.providers.anthropic_provider import AnthropicProvider

    return AnthropicProvider()


def _openai() -> LLMProvider:
    from src.agent.providers.openai_provider import OpenAIProvider

    return OpenAIProvider()


def _vllm() -> LLMProvider:
    from src.agent.providers.vllm_provider import VLLMProvider

    return VLLMProvider()


# Every LLMProviderName maps to exactly one factory.
_FACTORIES: dict[LLMProviderName, ProviderFactory] = {
    LLMProviderName.ANTHROPIC: _anthropic,
    LLMProviderName.OPENAI: _openai,
    LLMProviderName.VLLM: _vllm,
}

# Fail fast: a provider declared in the enum but never registered is a bug we want
# to surface at import time, not on the first request that happens to select it.
_unregistered = set(LLMProviderName) - _FACTORIES.keys()
if _unregistered:
    raise RuntimeError(
        f"LLM providers declared in {LLMProviderName.__name__} but not registered: "
        f"{sorted(p.value for p in _unregistered)}"
    )


def get_provider(name: LLMProviderName | None = None) -> LLMProvider:
    """Instantiate the configured generation backend.

    Args:
        name: override the configured provider (used by tests); defaults to
            ``settings.llm_provider``, which pydantic has already validated against
            the enum, so the lookup below cannot miss in normal operation.
    """
    selected = name if name is not None else settings.llm_provider
    return _FACTORIES[selected]()
