"""Local-file corpus loader (dev path).

Reads a YAML manifest of sources (local paths or URLs), chunks each document, and writes
it into the vector store via the shared writer. Run via `make index`. The primary corpus
path is the source-based ingestion pipeline (``python -m src.ingestion``); this stays as a
zero-network way to index local files.
"""
from __future__ import annotations

import argparse
import hashlib
import pathlib

import httpx
import yaml

from src.common.logging import configure_logging, get_logger
from src.indexing.chunking import chunk
from src.indexing.embeddings import embedding_dim
from src.indexing.qdrant_store import VectorStore
from src.indexing.writer import upsert_document

log = get_logger("indexer")


def _doc_id(source_uri: str) -> str:
    return hashlib.sha1(source_uri.encode()).hexdigest()[:12]


def _read(source_uri: str) -> str:
    if source_uri.startswith(("http://", "https://")):
        return httpx.get(source_uri, timeout=30, follow_redirects=True).text
    return pathlib.Path(source_uri).read_text(encoding="utf-8")


def index_source(store: VectorStore, source_uri: str) -> int:
    doc_id = _doc_id(source_uri)
    n = upsert_document(store, doc_id, chunk(_read(source_uri)), source_uri)
    log.info("indexed", doc_id=doc_id, source=source_uri, chunks=n)
    return n


def main() -> None:
    configure_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/corpus_manifest.yaml")
    args = ap.parse_args()

    manifest = yaml.safe_load(pathlib.Path(args.manifest).read_text())
    store = VectorStore()
    store.ensure_collection(embedding_dim())

    total = sum(index_source(store, s["uri"]) for s in manifest["sources"])
    log.info("index_complete", total_chunks=total, sources=len(manifest["sources"]))


if __name__ == "__main__":
    main()
