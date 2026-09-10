"""Source-based data ingestion.

Public surface: the ``Source`` interface, the ``SourceName`` enum, ``get_source``
to resolve the configured source, and the ``IngestionPipeline`` that drives a source
into the vector store.
"""
from __future__ import annotations

from src.ingestion.base import Source, SourceDocument, SourceName
from src.ingestion.pipeline import IngestionPipeline, IngestionReport
from src.ingestion.registry import get_source

__all__ = [
    "IngestionPipeline",
    "IngestionReport",
    "Source",
    "SourceDocument",
    "SourceName",
    "get_source",
]
