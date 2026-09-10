.PHONY: help up down index ask test lint kind-up kind-down deploy argocd port-forward demo

IMAGE ?= mcp-rag-platform:dev
NAMESPACE ?= mcp-rag
Q ?= how do I roll back an argocd application?

help:
	@grep -E '^[a-zA-Z_-]+:.*?# ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?# "}{printf "%-14s %s\n",$$1,$$2}'

up: # start agent + qdrant via docker compose
	docker compose up -d --build

down: # stop the compose stack
	docker compose down -v

index: # embed + upsert the corpus into qdrant
	python -m src.indexing.loaders --manifest data/corpus_manifest.yaml

ask: # one-shot query against the running agent (Q="...")
	curl -s localhost:8000/query -H 'content-type: application/json' \
		-d '{"query": "$(Q)", "k": 5}' | python -m json.tool

test: # run unit tests + eval harness
	pytest
	python -m tests.eval_harness

lint: # ruff + mypy
	ruff check src tests
	mypy src

kind-up: # create the local kind cluster
	kind create cluster --config deploy/kind/kind-config.yaml
	bash deploy/kind/bootstrap.sh

kind-down: # delete the local kind cluster
	kind delete cluster --name mcp-rag

deploy: # helm install/upgrade the chart
	helm upgrade --install mcp-rag deploy/helm -n $(NAMESPACE) --create-namespace

argocd: # register the app with ArgoCD (GitOps)
	kubectl apply -f deploy/argocd/application.yaml

port-forward: # expose the agent on localhost:8000
	kubectl -n $(NAMESPACE) port-forward svc/agent 8000:8000

demo: up index ask # bring it all up and run one grounded query
