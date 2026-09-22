"""Sparse (BM25) embeddings via fastembed - the lexical arm of hybrid search.

Uses the stateless ``Qdrant/bm25`` model: term weights are produced per-document with a
fixed IDF, so no corpus-wide fitting step is needed and re-indexing stays incremental.
Qdrant applies the IDF modifier at query time (see the collection's sparse config).

Loaded lazily so importing this module (and the agent) does not require fastembed unless
hybrid retrieval is actually used.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from src.common.config import settings


@dataclass(frozen=True, slots=True)
class SparseEmbedding:
    indices: list[int]
    values: list[float]


@lru_cache(maxsize=1)
def _model():
    from fastembed import SparseTextEmbedding

    return SparseTextEmbedding(settings.sparse_model)


def _to_sparse(embedding) -> SparseEmbedding:
    # fastembed returns numpy arrays for indices/values
    return SparseEmbedding(
        indices=[int(i) for i in embedding.indices],
        values=[float(v) for v in embedding.values],
    )


def embed_texts_sparse(texts: list[str]) -> list[SparseEmbedding]:
    return [_to_sparse(e) for e in _model().embed(texts)]


def embed_query_sparse(text: str) -> SparseEmbedding:
    return _to_sparse(next(iter(_model().query_embed(text))))
