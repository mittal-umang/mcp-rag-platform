"""Qdrant store with named dense + sparse vectors for hybrid retrieval.

Each point carries a ``dense`` vector (cosine) and, when hybrid indexing is on, a
``sparse`` BM25 vector. Dense-only search uses the dense vector; hybrid search runs both
arms as prefetches and lets Qdrant fuse them server-side with Reciprocal Rank Fusion.

doc_id is stable, so re-ingesting a document deletes its old points first (idempotent per
doc_id) before writing the new chunks.
"""
from __future__ import annotations

import uuid

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from src.common.config import settings
from src.common.models import ScoredChunk
from src.indexing.sparse import SparseEmbedding

DENSE = "dense"
SPARSE = "sparse"


class VectorStore:
    def __init__(self, url: str | None = None, collection: str | None = None):
        self.client = QdrantClient(url=url or settings.qdrant_url)
        self.collection = collection or settings.qdrant_collection

    def ensure_collection(self, dim: int) -> None:
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection in existing:
            return
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config={DENSE: qm.VectorParams(size=dim, distance=qm.Distance.COSINE)},
            # IDF modifier => Qdrant applies BM25 inverse-document-frequency at query time
            sparse_vectors_config={SPARSE: qm.SparseVectorParams(modifier=qm.Modifier.IDF)},
        )

    def delete_doc(self, doc_id: str) -> None:
        self.client.delete(
            collection_name=self.collection,
            points_selector=qm.FilterSelector(
                filter=qm.Filter(must=[qm.FieldCondition(
                    key="doc_id", match=qm.MatchValue(value=doc_id))])
            ),
        )

    def upsert(
        self,
        doc_id: str,
        dense_vectors: list[list[float]],
        payloads: list[dict],
        sparse_vectors: list[SparseEmbedding] | None = None,
    ) -> int:
        points: list[qm.PointStruct] = []
        for i, (dense, payload) in enumerate(zip(dense_vectors, payloads)):
            vector: dict = {DENSE: dense}
            if sparse_vectors is not None:
                s = sparse_vectors[i]
                vector[SPARSE] = qm.SparseVector(indices=s.indices, values=s.values)
            points.append(
                qm.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={**payload, "doc_id": doc_id},
                )
            )
        self.client.upsert(collection_name=self.collection, points=points)
        return len(points)

    # ---- retrieval ------------------------------------------------------

    def dense_search(self, dense_vec: list[float], k: int) -> list[ScoredChunk]:
        res = self.client.query_points(
            collection_name=self.collection,
            query=dense_vec, using=DENSE, limit=k, with_payload=True,
        )
        return [self._to_chunk(p) for p in res.points]

    def hybrid_search(
        self, dense_vec: list[float], sparse: SparseEmbedding, k: int, prefetch: int
    ) -> list[ScoredChunk]:
        res = self.client.query_points(
            collection_name=self.collection,
            prefetch=[
                qm.Prefetch(query=dense_vec, using=DENSE, limit=prefetch),
                qm.Prefetch(
                    query=qm.SparseVector(indices=sparse.indices, values=sparse.values),
                    using=SPARSE, limit=prefetch,
                ),
            ],
            query=qm.FusionQuery(fusion=qm.Fusion.RRF),
            limit=k, with_payload=True,
        )
        return [self._to_chunk(p) for p in res.points]

    def dense_top_score(self, dense_vec: list[float]) -> float | None:
        """Top-1 cosine similarity - the grounding signal used by the guardrail,
        computed independently of fusion/rerank so the threshold means the same thing
        in every retrieval mode."""
        res = self.client.query_points(
            collection_name=self.collection,
            query=dense_vec, using=DENSE, limit=1, with_payload=False,
        )
        return float(res.points[0].score) if res.points else None

    @staticmethod
    def _to_chunk(point) -> ScoredChunk:
        payload = point.payload or {}
        return ScoredChunk(
            doc_id=payload["doc_id"],
            section=payload.get("section"),
            text=payload.get("text", ""),
            source_uri=payload.get("source_uri"),
            score=float(point.score),
        )
