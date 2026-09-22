"""Retriever tests with fakes - no embeddings model, no Qdrant.

Covers mode dispatch (dense vs hybrid), the rerank stage (wider fetch, reorder, truncate),
and that grounding confidence comes from the dense top score independent of ranking.
"""
import src.agent.retrieval as retrieval_mod
from src.agent.retrieval import Retriever
from src.common.config import settings
from src.common.enums import RetrievalMode
from src.common.models import ScoredChunk


def _chunks(n):
    return [ScoredChunk(doc_id=f"d{i}", text=f"t{i}", score=1.0 - i * 0.1) for i in range(n)]


class FakeStore:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def dense_search(self, vec, k):
        self.calls.append(("dense", k))
        return self.results[:k]

    def hybrid_search(self, vec, sparse, k, prefetch):
        self.calls.append(("hybrid", k, prefetch))
        return self.results[:k]

    def dense_top_score(self, vec):
        return 0.7


def test_dense_mode_uses_dense_search(monkeypatch):
    monkeypatch.setattr(retrieval_mod, "embed_query", lambda q: [0.0])
    store = FakeStore(_chunks(5))
    result = Retriever(store, mode=RetrievalMode.DENSE, rerank_enabled=False).retrieve("q", 3)

    assert store.calls[0] == ("dense", 3)
    assert len(result.chunks) == 3
    assert result.confidence == 0.7


def test_hybrid_mode_uses_hybrid_search(monkeypatch):
    monkeypatch.setattr(retrieval_mod, "embed_query", lambda q: [0.0])
    monkeypatch.setattr(retrieval_mod, "embed_query_sparse", lambda q: object())
    store = FakeStore(_chunks(5))
    result = Retriever(store, mode=RetrievalMode.HYBRID, rerank_enabled=False).retrieve("q", 2)

    assert store.calls[0][0] == "hybrid"
    assert store.calls[0][1] == 2  # fetch k directly when not reranking
    assert len(result.chunks) == 2


def test_rerank_fetches_wide_then_reorders_and_truncates(monkeypatch):
    monkeypatch.setattr(retrieval_mod, "embed_query", lambda q: [0.0])
    # fake cross-encoder: reverse the candidate order, then the retriever truncates to k
    monkeypatch.setattr(retrieval_mod, "rerank", lambda query, chunks, k: list(reversed(chunks))[:k])
    store = FakeStore(_chunks(5))
    result = Retriever(store, mode=RetrievalMode.DENSE, rerank_enabled=True).retrieve("q", 2)

    assert store.calls[0] == ("dense", settings.rerank_top_n)  # widened fetch
    assert [c.doc_id for c in result.chunks] == ["d4", "d3"]   # reranked + cut to 2
