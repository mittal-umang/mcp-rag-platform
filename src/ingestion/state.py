"""Incremental-ingest state: the last-indexed revision per document.

A JSON file is sufficient at this corpus size. The interface is the point - swapping
in a database table or an object-store manifest for a larger corpus touches only this
module, not the pipeline.
"""
from __future__ import annotations

import json
from pathlib import Path


class IngestionState:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._revisions: dict[str, str] = {}
        if self._path.exists():
            self._revisions = json.loads(self._path.read_text(encoding="utf-8"))

    def changed(self, doc_id: str, revision: str) -> bool:
        """True if this doc_id is new or its revision differs from what we indexed."""
        return self._revisions.get(doc_id) != revision

    def record(self, doc_id: str, revision: str) -> None:
        self._revisions[doc_id] = revision

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(self._revisions, indent=2, sort_keys=True), encoding="utf-8"
        )
