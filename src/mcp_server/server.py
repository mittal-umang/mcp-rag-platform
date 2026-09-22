"""MCP server exposing three tools over the RAG stack.

Tools:
  - retrieve(query, k)          -> ranked chunks (dense or hybrid, per config)
  - fetch_source(doc_id, ...)   -> full text behind a chunk
  - ingest(source_uri|text,...) -> add/re-index a document (idempotent per doc_id)

Runs over stdio so an MCP host (e.g. Claude Desktop) can launch it directly. Retrieval,
ingest, and chunking reuse the same code as the agent.
"""
from __future__ import annotations

import hashlib

from mcp.server.fastmcp import FastMCP

from src.agent.retrieval import Retriever
from src.common.logging import configure_logging, get_logger
from src.indexing.chunking import chunk as chunk_text
from src.indexing.embeddings import embed_query, embedding_dim
from src.indexing.qdrant_store import VectorStore
from src.indexing.writer import upsert_document

configure_logging()
log = get_logger("mcp")

mcp = FastMCP("mcp-rag-platform")
store = VectorStore()
retriever = Retriever(store)


@mcp.tool()
def retrieve(query: str, k: int = 5) -> list[dict]:
    """Ranked retrieval (dense or hybrid, per config). Returns chunks with scores + source."""
    return [c.model_dump() for c in retriever.retrieve(query, k).chunks]


@mcp.tool()
def fetch_source(doc_id: str, section: str | None = None) -> dict:
    """Return the full stored text for a doc_id (optionally a single section)."""
    hits = store.dense_search(embed_query(doc_id), k=50)
    parts = [h for h in hits if h.doc_id == doc_id and (section is None or h.section == section)]
    text = "\n\n".join(p.text for p in parts)
    return {"doc_id": doc_id, "section": section, "text": text,
            "source_uri": parts[0].source_uri if parts else None}


@mcp.tool()
def ingest(source_uri: str | None = None, text: str | None = None,
           metadata: dict | None = None) -> dict:
    """Add or re-index a document. Provide either source_uri or raw text."""
    if not text and not source_uri:
        raise ValueError("provide either text or source_uri")
    body = text or ""
    key = source_uri or hashlib.sha1(body.encode()).hexdigest()
    doc_id = hashlib.sha1(key.encode()).hexdigest()[:12]

    store.ensure_collection(embedding_dim())
    n = upsert_document(store, doc_id, chunk_text(body), source_uri, metadata)
    log.info("ingested", doc_id=doc_id, chunks=n)
    return {"doc_id": doc_id, "chunks_indexed": n, "reindexed": True}


if __name__ == "__main__":
    mcp.run()  # stdio transport
