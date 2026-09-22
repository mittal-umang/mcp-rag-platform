"""Orchestrates a source into the vector store.

discover -> fetch -> chunk -> embed (batched) -> upsert, skipping documents whose
revision is unchanged since the last run. Chunking, embedding, and the write path are
reused from ``indexing`` so this module only owns orchestration and the incremental logic.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass

from src.common.config import settings
from src.common.logging import get_logger
from src.indexing.chunking import chunk
from src.indexing.embeddings import embedding_dim
from src.indexing.qdrant_store import VectorStore
from src.indexing.writer import upsert_document
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
        return upsert_document(
            self.store, doc.doc_id, chunk(doc.text), doc.source_uri, doc.metadata
        )
