# mcp-rag-platform

A production-style **RAG agent** served behind an **MCP server**, wrapped in the platform
layer that makes it operable: containerized, GitOps-deployed on Kubernetes (Helm + ArgoCD),
and observable. The agent is the payload; the serving/reliability platform around it is the
point.

> Personal project. Not affiliated with any employer.

## Architecture

```
                      ┌─────────────────────────────────────────────┐
   Claude Desktop ───▶│  MCP server  (retrieve / fetch_source /      │
   (MCP host)         │              ingest)                         │
                      └───────────────┬─────────────────────────────┘
                                      │
                      ┌───────────────▼─────────────────────────────┐
   HTTP  ────────────▶│  Agent (FastAPI + Pydantic)                  │
   /query             │   embed → retrieve → ground → generate       │
                      │   weak-retrieval guardrail                   │
                      └───────┬───────────────────────┬─────────────┘
                              │                       │
                     ┌────────▼─────────┐    ┌────────▼──────────────┐
                     │ Qdrant (vectors) │    │ LLM provider          │
                     └──────────────────┘    │  anthropic | openai   │
                                             │  | vllm (Phase 2)     │
                     embeddings: local        └───────────────────────┘
                     (sentence-transformers)
```

## Quickstart (local)

```bash
cp .env.example .env          # add your provider API key
make up                       # docker compose: agent + qdrant
make index                    # embed + upsert the corpus into qdrant
make ask Q="how do I roll back an argocd app?"
```

## Deploy to a local cluster (kind + Helm + ArgoCD)

```bash
make kind-up                  # create kind cluster
make deploy                   # helm install the chart
make argocd                   # apply the ArgoCD Application (GitOps)
make port-forward             # expose the agent locally
```

## MCP tools

| tool | purpose |
|------|---------|
| `retrieve(query, k)` | top-k grounded chunks with source citations |
| `fetch_source(doc_id, section?)` | full document/section behind a chunk |
| `ingest(source_uri \| text, metadata)` | add/re-index a document (idempotent per doc_id) |

## Observability

`GET /metrics` exposes Prometheus counters/histograms (request count, retrieval latency,
generation latency, tokens). Structured JSON logs on stdout. See `docs` note in
`src/common/metrics.py` for wiring Datadog.

## Phase 2 - self-hosted inference (vLLM)

The generation layer is provider-pluggable. Set `LLM_PROVIDER=vllm` and enable the gated
vLLM service (`vllm.enabled=true` in Helm values) to serve a small open model from an
OpenAI-compatible endpoint, with continuous batching and token streaming. Needs a GPU;
the API providers remain the zero-GPU default. See `src/agent/providers/vllm_provider.py`.

## Layout

```
src/agent       FastAPI app, RAG pipeline, pluggable LLM providers
src/mcp_server   MCP tools (retrieve, fetch_source, ingest)
src/indexing     loaders, chunking, embeddings, qdrant store
src/common       pydantic models, config, logging, metrics
deploy/helm      chart: agent, mcp-server, qdrant, (vllm)
deploy/argocd    Application manifest (GitOps)
deploy/kind      local cluster config + bootstrap
data             corpus manifest
tests            unit tests + eval harness
```
