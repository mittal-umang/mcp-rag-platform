"""Orchestrates a source into the vector store.

discover -> fetch -> chunk -> embed (batched) -> upsert, skipping documents whose
revision is unchanged since the last run. Retrieval, chunking, and embedding are reused
from ``indexing`` so this module only owns orchestration and the incremental logic.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass

from src.common.config import settings
from src.common.logging import get_logger
from src.indexing.chunking import chunk
from src.indexing.embeddings import embed_texts, embedding_dim
from src.indexing.qdrant_store import VectorStore
from src.ingestion.base import Source, SourceDocument
from src.ingestion.state import IngestionState

log = get_logger("ingestion")


@dataclass(frozen=True, slots=True)
class IngestionReport:
    discovered: int
    fetched: int
    indexed: int
    skipped: int
    chunks: int
    seconds: float


class IngestionPipeline:
    def __init__(
        self,
        source: Source,
        store: VectorStore | None = None,
        state: IngestionState | None = None,
    ) -> None:
        self.source = source
        self.store = store or VectorStore()
        self.state = state or IngestionState(settings.ingestion_state_path)

    async def run(self, *, incremental: bool = True) -> IngestionReport:
        started = time.perf_counter()
        keys = await self.source.discover()
        docs = await self.source.fetch(keys)

        self.store.ensure_collection(embedding_dim())
        indexed = skipped = total_chunks = 0
        for doc in docs:
            if incremental and not self.state.changed(doc.doc_id, doc.revision):
                skipped += 1
                continue
            total_chunks += self._index_document(doc)
            self.state.record(doc.doc_id, doc.revision)
            indexed += 1
        self.state.save()

        report = IngestionReport(
            discovered=len(keys),
            fetched=len(docs),
            indexed=indexed,
            skipped=skipped,
            chunks=total_chunks,
            seconds=round(time.perf_counter() - started, 3),
        )
        log.info("ingestion_complete", **asdict(report))
        return report

    def _index_document(self, doc: SourceDocument) -> int:
        chunks = chunk(doc.text)
        vectors = embed_texts([c.text for c in chunks])
        payloads = [
            {"section": c.section, "text": c.text, "source_uri": doc.source_uri, **doc.metadata}
            for c in chunks
        ]
        self.store.delete_doc(doc.doc_id)  # idempotent re-index
        return self.store.upsert(doc.doc_id, vectors, payloads)
