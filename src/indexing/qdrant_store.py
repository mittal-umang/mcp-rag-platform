"""Thin wrapper over Qdrant: collection lifecycle, upsert, and search.

doc_id is stable, so re-ingesting a document deletes its old points first
(idempotent per doc_id) before writing the new chunks.
"""
from __future__ import annotations

import uuid

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from src.common.config import settings
from src.common.models import ScoredChunk


class VectorStore:
    def __init__(self, url: str | None = None, collection: str | None = None):
        self.client = QdrantClient(url=url or settings.qdrant_url)
        self.collection = collection or settings.qdrant_collection

    def ensure_collection(self, dim: int) -> None:
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection not in existing:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
            )

    def delete_doc(self, doc_id: str) -> None:
        self.client.delete(
            collection_name=self.collection,
            points_selector=qm.FilterSelector(
                filter=qm.Filter(must=[qm.FieldCondition(
                    key="doc_id", match=qm.MatchValue(value=doc_id))])
            ),
        )

    def upsert(self, doc_id: str, vectors: list[list[float]], payloads: list[dict]) -> int:
        points = [
            qm.PointStruct(id=str(uuid.uuid4()), vector=v, payload={**p, "doc_id": doc_id})
            for v, p in zip(vectors, payloads)
        ]
        self.client.upsert(collection_name=self.collection, points=points)
        return len(points)

    def search(self, vector: list[float], k: int) -> list[ScoredChunk]:
        hits = self.client.query_points(
            collection_name=self.collection, query=vector, limit=k,
        ).points
        return [
            ScoredChunk(
                doc_id=h.payload["doc_id"],
                section=h.payload.get("section"),
                text=h.payload.get("text", ""),
                source_uri=h.payload.get("source_uri"),
                score=float(h.score),
            )
            for h in hits
        ]
