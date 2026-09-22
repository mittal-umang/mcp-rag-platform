"""RAG guardrail tests with fakes - no network, no model, no Qdrant.

The pipeline judges grounding by the retriever's confidence (top-1 dense cosine); a weak
confidence must refuse without ever calling the LLM.
"""
from src.agent.providers.base import Completion, LLMProviderName
from src.agent.rag import REFUSAL, RagPipeline
from src.agent.retrieval import RetrievalResult
from src.common.models import ScoredChunk


class FakeProvider:
    name = LLMProviderName.ANTHROPIC
    called = False

    def generate(self, system, prompt):
        FakeProvider.called = True
        return Completion(text="grounded answer [d1]", completion_tokens=3, provider=self.name)


class FakeRetriever:
    def __init__(self, chunks, confidence):
        self._result = RetrievalResult(chunks=chunks, confidence=confidence)

    def retrieve(self, query, k):
        return self._result


def _pipeline(chunks, confidence):
    pipeline = RagPipeline.__new__(RagPipeline)
    pipeline.retriever = FakeRetriever(chunks, confidence)
    pipeline.provider = FakeProvider()
    return pipeline


def test_guardrail_refuses_on_weak_confidence():
    FakeProvider.called = False
    chunks = [ScoredChunk(doc_id="d1", text="x", score=0.9)]
    resp = _pipeline(chunks, confidence=0.05).answer("unrelated question", k=5)
    assert resp.grounded is False
    assert resp.answer == REFUSAL
    assert FakeProvider.called is False  # LLM never invoked


def test_answers_on_strong_confidence():
    chunks = [ScoredChunk(doc_id="d1", text="relevant", score=0.9)]
    resp = _pipeline(chunks, confidence=0.8).answer("good question", k=5)
    assert resp.grounded is True
    assert resp.citations[0].doc_id == "d1"
