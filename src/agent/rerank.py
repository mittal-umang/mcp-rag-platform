"""Optional cross-encoder reranking stage.

A bi-encoder retriever is fast but approximate; a cross-encoder scores each
(query, chunk) pair jointly and is far more precise. So we retrieve a wider candidate set
(``rerank_top_n``) cheaply, then rerank down to the final ``k``. Loaded lazily and gated by
``settings.rerank_enabled`` because it is heavier than embedding lookup.
"""
from __future__ import annotations

from functools import lru_cache

from src.common.config import settings
from src.common.models import ScoredChunk


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(settings.reranker_model)


def rerank(query: str, chunks: list[ScoredChunk], k: int) -> list[ScoredChunk]:
    """Re-score candidates with the cross-encoder and return the top ``k``.

    The returned chunks carry the cross-encoder relevance score (not the retrieval score).
    """
    if not chunks:
        return []
    scores = _model().predict([(query, c.text) for c in chunks])
    ranked = sorted(zip(chunks, scores), key=lambda pair: pair[1], reverse=True)
    return [c.model_copy(update={"score": float(s)}) for c, s in ranked[:k]]
