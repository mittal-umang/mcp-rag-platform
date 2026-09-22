"""Pydantic contracts shared across the agent, MCP tools, and indexing.

Every tool input/output and the HTTP API validate against these models.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A retrievable unit of a source document."""
    doc_id: str
    section: str | None = None
    text: str
    source_uri: str | None = None


class ScoredChunk(Chunk):
    # score is the ranking score for the active retrieval mode: dense cosine, an RRF
    # fusion score, or a cross-encoder relevance score. Not a bounded probability, so
    # it carries no [0, 1] constraint - callers compare within a single result set.
    score: float


class Citation(BaseModel):
    doc_id: str
    section: str | None = None
    source_uri: str | None = None
    score: float


# ---- MCP tool / API request-response models -----------------------------

class RetrieveRequest(BaseModel):
    query: str
    k: int = Field(default=5, ge=1, le=50)


class RetrieveResponse(BaseModel):
    chunks: list[ScoredChunk]


class FetchSourceRequest(BaseModel):
    doc_id: str
    section: str | None = None


class FetchSourceResponse(BaseModel):
    doc_id: str
    section: str | None = None
    text: str
    source_uri: str | None = None


class IngestRequest(BaseModel):
    source_uri: str | None = None
    text: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


class IngestResponse(BaseModel):
    doc_id: str
    chunks_indexed: int
    reindexed: bool


class QueryRequest(BaseModel):
    query: str
    k: int = Field(default=5, ge=1, le=50)


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    grounded: bool  # False => guardrail fired, answer is the refusal string
