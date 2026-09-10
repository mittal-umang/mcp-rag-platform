"""Prometheus metrics. Exposed by the agent at GET /metrics.

Datadog note: point the Datadog Agent's OpenMetrics/Prometheus check at /metrics,
or run the DD Agent as a sidecar and annotate the pod. No app changes needed.
"""
from __future__ import annotations

from prometheus_client import Counter, Histogram

QUERIES = Counter("rag_queries_total", "Total /query requests", ["grounded"])

RETRIEVAL_LATENCY = Histogram(
    "rag_retrieval_seconds", "Time spent in vector retrieval",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)

GENERATION_LATENCY = Histogram(
    "rag_generation_seconds", "Time spent in LLM generation",
    buckets=(0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0),
)

GENERATION_TOKENS = Counter(
    "rag_generation_tokens_total", "Completion tokens produced", ["provider"],
)
