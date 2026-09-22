"""Central config, loaded from environment (.env). Single source of truth."""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict

from src.common.enums import LLMProviderName, RetrievalMode, SourceName


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
    embed_batch_size: int = 64

    # retrieval
    retrieval_mode: RetrievalMode = RetrievalMode.HYBRID
    top_k: int = 5
    hybrid_prefetch: int = 50          # candidates each arm fetches before fusion
    sparse_model: str = "Qdrant/bm25"  # fastembed sparse model (stateless BM25)
    min_score: float = 0.30            # guardrail: min dense cosine of the best match

    # reranking (optional cross-encoder stage on top of retrieval)
    rerank_enabled: bool = False
    reranker_model: str = "BAAI/bge-reranker-base"
    rerank_top_n: int = 30             # candidates fetched and fed to the reranker

    # ingestion
    ingestion_source: SourceName = SourceName.WIKIPEDIA
    ingestion_concurrency: int = 8
    ingestion_state_path: str = ".cache/ingestion_state.json"
    request_timeout_seconds: float = 30.0
    user_agent: str = (
        "mcp-rag-platform/0.1 (https://github.com/mittal-umang/mcp-rag-platform)"
    )

    # wikipedia source
    wikipedia_language: str = "en"
    wikipedia_category: str = "Member states of the United Nations"
    wikipedia_batch_titles: int = 20  # MediaWiki extracts exlimit maximum


settings = Settings()
