"""Ingestion tests - hermetic: no network, no real embedding model, no Qdrant.

Covered:
  - Wikipedia response parsing (against a recorded fixture, incl. a missing page)
  - incremental state (new / unchanged / changed revision)
  - the pipeline's incremental skip logic, with fakes for the source and store
"""
from __future__ import annotations

import asyncio
import json
import pathlib

from src.ingestion.base import SourceDocument
from src.ingestion.state import IngestionState
from src.ingestion.wikipedia import parse_pages

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "wikipedia_query.json"


def test_parse_pages_skips_missing_and_maps_fields():
    data = json.loads(FIXTURE.read_text())
    docs = parse_pages(data, language="en")

    assert [d.title for d in docs] == ["France", "Japan"]  # missing page dropped
    france = docs[0]
    assert france.doc_id == "wikipedia:5843419"
    assert france.revision == "1234567890"
    assert france.source_uri == "https://en.wikipedia.org/?curid=5843419"
    assert "French Republic" in france.text


def test_state_tracks_revisions(tmp_path):
    state = IngestionState(tmp_path / "state.json")
    assert state.changed("wikipedia:1", "rev-a") is True   # unseen
    state.record("wikipedia:1", "rev-a")
    assert state.changed("wikipedia:1", "rev-a") is False  # unchanged
    assert state.changed("wikipedia:1", "rev-b") is True   # revision moved

    state.save()
    reloaded = IngestionState(tmp_path / "state.json")
    assert reloaded.changed("wikipedia:1", "rev-a") is False  # persisted


class _FakeSource:
    def __init__(self, docs):
        self._docs = docs

    async def discover(self):
        return [d.title for d in self._docs]

    async def fetch(self, keys):
        return list(self._docs)


class _FakeStore:
    def __init__(self):
        self.upserts = 0

    def ensure_collection(self, dim):
        pass

    def delete_doc(self, doc_id):
        pass

    def upsert(self, doc_id, vectors, payloads):
        self.upserts += 1
        return len(vectors)


def _doc(doc_id, revision):
    return SourceDocument(
        doc_id=doc_id, title=doc_id, text="body text",
        source_uri="uri", revision=revision, metadata={},
    )


def test_pipeline_incremental_skips_unchanged(tmp_path, monkeypatch):
    # keep the pipeline hermetic: fake out embedding + dimension
    monkeypatch.setattr("src.ingestion.pipeline.embed_texts", lambda texts: [[0.0] for _ in texts])
    monkeypatch.setattr("src.ingestion.pipeline.embedding_dim", lambda: 1)
    from src.ingestion.pipeline import IngestionPipeline

    docs = [_doc("wikipedia:1", "r1"), _doc("wikipedia:2", "r1")]
    store = _FakeStore()
    state = IngestionState(tmp_path / "state.json")

    first = asyncio.run(IngestionPipeline(_FakeSource(docs), store, state).run())
    assert first.indexed == 2 and first.skipped == 0

    # one revision changes; the other is unchanged and must be skipped
    docs[0] = _doc("wikipedia:1", "r2")
    second = asyncio.run(IngestionPipeline(_FakeSource(docs), store, state).run())
    assert second.indexed == 1
    assert second.skipped == 1
