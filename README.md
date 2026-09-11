# mcp-rag-platform

A production-style **RAG agent** served behind an **MCP server**, fed by a real **data
ingestion pipeline**, and wrapped in the platform layer that makes it operable:
containerized, GitOps-deployed on Kubernetes (Helm + ArgoCD), and observable. The agent is
the payload; the ingestion pipeline and the serving/reliability platform around it are the
point.

> Personal project. Not affiliated with any employer.

## Architecture

```
   MediaWiki API ──▶ Ingestion pipeline ──▶ Qdrant (vectors)
                     discover → fetch →              ▲
                     chunk → embed → upsert          │ retrieve
                     (incremental by revision)       │
                                                     │
                      ┌──────────────────────────────┴──────────────┐
   Claude Desktop ───▶│  MCP server (retrieve / fetch_source / ingest)│
   (MCP host)         └───────────────┬──────────────────────────────┘
                                      │
   HTTP /query ──────────────────────▶  Agent (FastAPI + Pydantic)
                                        embed → retrieve → ground → generate
                                        weak-retrieval guardrail
                                        LLM provider: anthropic | openai | vllm
```

## Data ingestion

The corpus is built by a source-based pipeline, not by hand-dropping files. The default
source pulls every UN member-state page from Wikipedia via the **MediaWiki API** (not HTML
scraping): it discovers titles in a category, fetches plain-text extracts plus each page's
**revision id** in bounded-concurrency batches, chunks, embeds locally, and upserts into
Qdrant. Re-runs are **incremental** - a page is re-embedded only when its revision changes.

```bash
python -m src.ingestion            # incremental re-index of the default source
python -m src.ingestion --full     # ignore saved revisions, rebuild everything
```

Sources are pluggable behind a `Source` interface (`discover` + `fetch`), resolved from the
`SourceName` enum through a registry - the same pattern as the LLM providers. Adding an S3
prefix or a docs-site source is a new `Source` implementation plus one enum member.

At ~195 pages this is deliberately a single async batch job. The scale-up path (fan out to
Ray/Spark behind a work queue, swap the JSON revision-state file for a table or object-store
manifest) touches only `pipeline.py` and `state.py` - see the seams noted there. Building the
heavy version for 195 pages would be over-engineering.

## Quickstart (local)

```bash
cp .env.example .env          # add your provider API key
make up                       # docker compose: agent + qdrant
python -m src.ingestion       # populate the vector store from Wikipedia
make ask Q="what is the capital of Japan?"
```

## Deploy to a local cluster (kind + Helm + ArgoCD)

```bash
make kind-up                  # create kind cluster
make deploy                   # helm install the chart (runs the ingestion Job)
make argocd                   # apply the ArgoCD Application (GitOps)
make port-forward             # expose the agent locally
```

Helm runs ingestion as a post-install/upgrade `Job` and schedules incremental re-index as a
`CronJob` (`ingestion.schedule`). Both are gated in `values.yaml`.

## MCP tools

| tool | purpose |
|------|---------|
| `retrieve(query, k)` | top-k grounded chunks with source citations |
| `fetch_source(doc_id, section?)` | full document/section behind a chunk |
| `ingest(source_uri \| text, metadata)` | add/re-index one document ad hoc (idempotent per doc_id) |

The MCP `ingest` tool is the ad-hoc path (drop in a single doc); the pipeline above is the
bulk, repeatable path.

### Register with an MCP host

Point the server's venv interpreter directly at the module - no wrapper script needed.

```bash
# Claude Code (this repo's .mcp.json, shared with anyone who clones it)
claude mcp add mcp-rag-platform --scope project -- \
  /path/to/mcp-rag-platform/venv/bin/python -m src.mcp_server.server
```

For Claude Desktop, add the equivalent entry to `claude_desktop_config.json`'s
`mcpServers` (find it via Settings → Developer):

```json
{
  "mcpServers": {
    "mcp-rag-platform": {
      "command": "/path/to/mcp-rag-platform/venv/bin/python",
      "args": ["-m", "src.mcp_server.server"],
      "cwd": "/path/to/mcp-rag-platform"
    }
  }
}
```

`cwd` matters: config is loaded from a relative `.env`, so the process needs the repo root
as its working directory regardless of where the host launches it from. Qdrant must already
be running (`make up`) before the host connects, since `ingest`/`retrieve` hit it directly.

## Observability

`GET /metrics` exposes Prometheus counters/histograms (request count, retrieval latency,
generation latency, tokens). Structured JSON logs on stdout. See `src/common/metrics.py`
for wiring Datadog.

## Phase 2 - self-hosted inference (vLLM)

The generation layer is provider-pluggable. Set `LLM_PROVIDER=vllm` and enable the gated
vLLM service (`vllm.enabled=true`) to serve a small open model from an OpenAI-compatible
endpoint, with continuous batching and token streaming. Needs a GPU; the API providers
remain the zero-GPU default.

## Layout

```
src/agent         FastAPI app, RAG pipeline, pluggable LLM providers
src/mcp_server    MCP tools (retrieve, fetch_source, ingest)
src/ingestion     source connectors, async fetch, incremental pipeline, CLI
src/indexing      chunking, local embeddings, qdrant store
src/common        pydantic models, enums, config, logging, metrics
deploy/helm       chart: agent, mcp-server, qdrant, ingestion Job/CronJob, (vllm)
deploy/argocd     Application manifest (GitOps)
deploy/kind       local cluster config + bootstrap
tests             unit tests + eval harness (hermetic; no network)
```

A small local corpus under `data/docs/` plus `data/corpus_manifest.yaml` remains as a
zero-network dev path via `python -m src.indexing.loaders`; the Wikipedia pipeline is the
primary corpus.
