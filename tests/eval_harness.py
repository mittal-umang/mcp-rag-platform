"""Tiny retrieval eval: for a handful of questions, is the expected doc_id in top-k?

Run against a populated Qdrant (after `make index`):  python -m tests.eval_harness
Prints recall@k and exits non-zero if below threshold - usable as a CI gate.
"""
from __future__ import annotations

import sys

from src.common.config import settings
from src.indexing.embeddings import embed_query
from src.indexing.qdrant_store import VectorStore

# (question, expected doc source substring)
CASES = [
    ("how do I roll back an argocd application?", "argocd"),
    ("what does maxUnavailable control?", "kubernetes"),
    ("how does terraform detect drift?", "terraform"),
    ("how do I override chart values at install time?", "helm"),
]

THRESHOLD = 0.75


def main() -> int:
    store = VectorStore()
    hits = 0
    for question, expected in CASES:
        results = store.search(embed_query(question), settings.top_k)
        found = any(expected in (r.source_uri or "") for r in results)
        hits += found
        print(f"[{'PASS' if found else 'FAIL'}] {question}")
    recall = hits / len(CASES)
    print(f"\nrecall@{settings.top_k} = {recall:.2f} (threshold {THRESHOLD})")
    return 0 if recall >= THRESHOLD else 1


if __name__ == "__main__":
    sys.exit(main())
