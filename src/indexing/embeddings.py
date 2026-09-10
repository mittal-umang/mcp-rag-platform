"""Local, open embeddings via sentence-transformers. No API cost, no network at query time.

Loaded lazily so importing this module is cheap (matters for tests / the MCP process).
"""
from __future__ import annotations

from functools import lru_cache

from src.common.config import settings


@lru_cache(maxsize=1)
def _model():
    # imported lazily: heavy dependency, only needed when we actually embed
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(settings.embedding_model)


def embed_texts(texts: list[str]) -> list[list[float]]:
    vectors = _model().encode(texts, normalize_embeddings=True)
    return [v.tolist() for v in vectors]


def embed_query(text: str) -> list[float]:
    # bge models want an instruction prefix on the query side
    prefixed = f"Represent this sentence for searching relevant passages: {text}"
    return embed_texts([prefixed])[0]


def embedding_dim() -> int:
    return int(_model().get_sentence_embedding_dimension())
