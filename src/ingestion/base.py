"""Ingestion source interface: discover keys, then fetch documents.

A source is anything the pipeline can pull a bounded set of documents from
(Wikipedia today; an S3 prefix or a docs site later). Separating ``discover`` from
``fetch`` lets the pipeline report progress, bound concurrency during fetch, and
re-index incrementally by comparing each document's ``revision``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from src.common.enums import SourceName

__all__ = ["Source", "SourceDocument", "SourceName"]


@dataclass(frozen=True, slots=True)
class SourceDocument:
    """One document pulled from a source.

    ``revision`` is an opaque version marker (a Wikipedia revision id, an S3 ETag,
    a content hash). The pipeline stores it and re-indexes a document only when its
    revision changes, so a re-run touches just what moved.
    """

    doc_id: str
    title: str
    text: str
    source_uri: str
    revision: str
    metadata: dict[str, str] = field(default_factory=dict)


@runtime_checkable
class Source(Protocol):
    name: SourceName

    async def discover(self) -> list[str]:
        """Return the keys (e.g. page titles) of every document to ingest."""
        ...

    async def fetch(self, keys: list[str]) -> list[SourceDocument]:
        """Fetch and normalize the documents for the given keys."""
        ...
