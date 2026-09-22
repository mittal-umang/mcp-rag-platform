"""Single write path for indexing a document's chunks into the vector store.

Centralizes the dense + (optional) sparse embedding and the idempotent delete-then-upsert,
so every ingest caller (local loader, ingestion pipeline, MCP ``ingest`` tool) shares one
implementation. Sparse vectors are computed only when hybrid retrieval is configured.
"""
from __future__ import annotations

from src.common.config import settings
from src.common.enums import RetrievalMode
from src.indexing.chunking import TextChunk
from src.indexing.embeddings import embed_texts
from src.indexing.qdrant_store import VectorStore
from src.indexing.sparse import embed_texts_sparse


def upsert_document(
    store: VectorStore,
    doc_id: str,
    chunks: list[TextChunk],
    source_uri: str | None,
    metadata: dict | None = None,
) -> int:
    if not chunks:
        return 0
    texts = [c.text for c in chunks]
    dense = embed_texts(texts)
    sparse = (
        embed_texts_sparse(texts)
        if settings.retrieval_mode is RetrievalMode.HYBRID
        else None
    )
    payloads = [
        {"section": c.section, "text": c.text, "source_uri": source_uri, **(metadata or {})}
        for c in chunks
    ]
    store.delete_doc(doc_id)  # idempotent re-index
    return store.upsert(doc_id, dense, payloads, sparse_vectors=sparse)
