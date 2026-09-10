"""The RAG pipeline: retrieve -> ground -> generate, with a weak-retrieval guardrail."""
from __future__ import annotations

import time

from src.agent.providers import get_provider
from src.common.config import settings
from src.common.logging import get_logger
from src.common.metrics import (
    GENERATION_LATENCY,
    GENERATION_TOKENS,
    QUERIES,
    RETRIEVAL_LATENCY,
)
from src.common.models import Citation, QueryResponse, ScoredChunk
from src.indexing.embeddings import embed_query
from src.indexing.qdrant_store import VectorStore

log = get_logger("rag")

SYSTEM = (
    "You answer strictly from the provided context. Cite sources by [doc_id]. "
    "If the context does not contain the answer, say you do not have enough grounding."
)

REFUSAL = "I do not have enough grounding in the indexed documents to answer that."


class RagPipeline:
    def __init__(self, store: VectorStore | None = None):
        self.store = store or VectorStore()
        self.provider = get_provider()

    def _build_prompt(self, query: str, chunks: list[ScoredChunk]) -> str:
        context = "\n\n".join(
            f"[{c.doc_id}{('#' + c.section) if c.section else ''}]\n{c.text}" for c in chunks
        )
        return f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer with citations:"

    def answer(self, query: str, k: int | None = None) -> QueryResponse:
        k = k or settings.top_k

        t0 = time.perf_counter()
        chunks = self.store.search(embed_query(query), k)
        RETRIEVAL_LATENCY.observe(time.perf_counter() - t0)

        # guardrail: refuse rather than hallucinate when retrieval is weak
        if not chunks or chunks[0].score < settings.min_score:
            QUERIES.labels(grounded="false").inc()
            log.info("guardrail_refused", query=query,
                     top_score=chunks[0].score if chunks else None)
            return QueryResponse(answer=REFUSAL, citations=[], grounded=False)

        prompt = self._build_prompt(query, chunks)
        t1 = time.perf_counter()
        completion = self.provider.generate(SYSTEM, prompt)
        GENERATION_LATENCY.observe(time.perf_counter() - t1)
        GENERATION_TOKENS.labels(provider=completion.provider.value).inc(completion.completion_tokens)
        QUERIES.labels(grounded="true").inc()

        citations = [
            Citation(doc_id=c.doc_id, section=c.section,
                     source_uri=c.source_uri, score=c.score)
            for c in chunks
        ]
        return QueryResponse(answer=completion.text, citations=citations, grounded=True)
