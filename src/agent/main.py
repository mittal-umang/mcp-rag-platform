"""FastAPI app: /query (RAG), health probes, and Prometheus /metrics."""
from __future__ import annotations

from fastapi import FastAPI, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src.agent.rag import RagPipeline
from src.common.logging import configure_logging, get_logger
from src.common.models import QueryRequest, QueryResponse

configure_logging()
log = get_logger("agent")

app = FastAPI(title="mcp-rag-platform agent", version="0.1.0")
_pipeline: RagPipeline | None = None


def pipeline() -> RagPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RagPipeline()
    return _pipeline


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
def readyz() -> dict[str, str]:
    # ready once we can build the pipeline (provider + store wired)
    pipeline()
    return {"status": "ready"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    log.info("query", q=req.query, k=req.k)
    return pipeline().answer(req.query, req.k)


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
