"""Central config, loaded from environment (.env). Single source of truth."""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict

from src.common.enums import LLMProviderName


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # generation
    llm_provider: LLMProviderName = LLMProviderName.ANTHROPIC
    llm_model: str = "claude-3-5-sonnet-latest"
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    vllm_base_url: str = "http://vllm:8000/v1"
    vllm_model: str = "Qwen/Qwen2.5-3B-Instruct"

    # vector store
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "docs"

    # embeddings (local)
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # retrieval guardrail
    min_score: float = 0.30
    top_k: int = 5


settings = Settings()
