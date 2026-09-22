"""The RAG pipeline: retrieve -> ground -> generate, with a weak-retrieval guardrail.

Retrieval strategy (dense / hybrid / rerank) lives in the Retriever; this module owns
prompt assembly, the guardrail, and generation.
"""
from __future__ import annotations

import time

from src.agent.providers import get_provider
from src.agent.retrieval import Retriever
from src.common.config import settings
from src.common.logging import get_logger
from src.common.metrics import (
    GENERATION_LATENCY,
    GENERATION_TOKENS,
    QUERIES,
    RETRIEVAL_LATENCY,
)
from src.common.models import Citation, QueryResponse, ScoredChunk

log = get_logger("rag")

SYSTEM = (
    "You answer strictly from the provided context. Cite sources by [doc_id]. "
    "If the context does not contain the answer, say you do not have enough grounding."
)

REFUSAL = "I do not have enough grounding in the indexed documents to answer that."


class RagPipeline:
    def __init__(self, retriever: Retriever | None = None):
        self.retriever = retriever or Retriever()
        self.provider = get_provider()

    def _build_prompt(self, query: str, chunks: list[ScoredChunk]) -> str:
        context = "\n\n".join(
            f"[{c.doc_id}{('#' + c.section) if c.section else ''}]\n{c.text}" for c in chunks
        )
        return f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer with citations:"

    def answer(self, query: str, k: int | None = None) -> QueryResponse:
        k = k or settings.top_k

        t0 = time.perf_counter()
        result = self.retriever.retrieve(query, k)
        RETRIEVAL_LATENCY.observe(time.perf_counter() - t0)

        # guardrail: grounding is judged by the strongest dense match, not by the
        # (fusion/rerank) ranking score - refuse rather than hallucinate.
        if not result.chunks or result.confidence is None or result.confidence < settings.min_score:
            QUERIES.labels(grounded="false").inc()
            log.info("guardrail_refused", query=query, confidence=result.confidence)
            return QueryResponse(answer=REFUSAL, citations=[], grounded=False)

        prompt = self._build_prompt(query, result.chunks)
        t1 = time.perf_counter()
        completion = self.provider.generate(SYSTEM, prompt)
        GENERATION_LATENCY.observe(time.perf_counter() - t1)
        GENERATION_TOKENS.labels(provider=completion.provider.value).inc(completion.completion_tokens)
        QUERIES.labels(grounded="true").inc()

        citations = [
            Citation(doc_id=c.doc_id, section=c.section,
                     source_uri=c.source_uri, score=c.score)
            for c in result.chunks
        ]
        return QueryResponse(answer=completion.text, citations=citations, grounded=True)
