"""Guardrail test with fakes - no network, no real model, no Qdrant.

Shows the weak-retrieval guardrail refuses instead of calling the LLM.
"""
from src.agent.providers.base import Completion, LLMProviderName
from src.agent.rag import REFUSAL, RagPipeline
from src.common.models import ScoredChunk


class FakeStore:
    def __init__(self, chunks):
        self._chunks = chunks

    def search(self, vector, k):
        return self._chunks[:k]


class FakeProvider:
    name = LLMProviderName.ANTHROPIC
    called = False

    def generate(self, system, prompt):
        FakeProvider.called = True
        return Completion(text="grounded answer [d1]", completion_tokens=3, provider=self.name)


def _pipeline(chunks, monkeypatch):
    monkeypatch.setattr("src.agent.rag.embed_query", lambda q: [0.0])
    p = RagPipeline.__new__(RagPipeline)
    p.store = FakeStore(chunks)
    p.provider = FakeProvider()
    return p


def test_guardrail_refuses_on_weak_retrieval(monkeypatch):
    FakeProvider.called = False
    weak = [ScoredChunk(doc_id="d1", text="x", score=0.05)]
    resp = _pipeline(weak, monkeypatch).answer("unrelated question", k=5)
    assert resp.grounded is False
    assert resp.answer == REFUSAL
    assert FakeProvider.called is False  # LLM never invoked


def test_answers_on_strong_retrieval(monkeypatch):
    strong = [ScoredChunk(doc_id="d1", text="relevant", score=0.9)]
    resp = _pipeline(strong, monkeypatch).answer("good question", k=5)
    assert resp.grounded is True
    assert resp.citations[0].doc_id == "d1"
