"""Retriever: turns a query into ranked chunks, plus a grounding confidence.

Composes the retrieval strategy (dense or hybrid) with an optional cross-encoder rerank
stage, keeping that policy out of the RAG pipeline and the MCP tools. Grounding confidence
is always the top-1 dense cosine similarity, computed independently of fusion/rerank, so
the guardrail threshold means the same thing in every mode.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.agent.rerank import rerank
from src.common.config import settings
from src.common.enums import RetrievalMode
from src.common.models import ScoredChunk
from src.indexing.embeddings import embed_query
from src.indexing.qdrant_store import VectorStore
from src.indexing.sparse import embed_query_sparse


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    chunks: list[ScoredChunk]
    confidence: float | None  # top-1 dense cosine; None when the store is empty


class Retriever:
    def __init__(
        self,
        store: VectorStore | None = None,
        mode: RetrievalMode | None = None,
        rerank_enabled: bool | None = None,
    ) -> None:
        self.store = store or VectorStore()
        self.mode = mode or settings.retrieval_mode
        self.rerank_enabled = (
            settings.rerank_enabled if rerank_enabled is None else rerank_enabled
        )

    def retrieve(self, query: str, k: int) -> RetrievalResult:
        dense_q = embed_query(query)
        # when reranking, fetch a wider candidate set, then let the cross-encoder cut to k
        fetch_k = settings.rerank_top_n if self.rerank_enabled else k

        if self.mode is RetrievalMode.HYBRID:
            candidates = self.store.hybrid_search(
                dense_q, embed_query_sparse(query), fetch_k, settings.hybrid_prefetch
            )
        else:
            candidates = self.store.dense_search(dense_q, fetch_k)

        if self.rerank_enabled:
            candidates = rerank(query, candidates, k)
        else:
            candidates = candidates[:k]

        confidence = self.store.dense_top_score(dense_q)
        return RetrievalResult(chunks=candidates, confidence=confidence)
