#!/usr/bin/env bash
# Install ArgoCD into the local kind cluster and load the local image.
set -euo pipefail

echo ">> loading local image into kind"
kind load docker-image mcp-rag-platform:dev --name mcp-rag

echo ">> installing argocd"
kubectl create namespace argocd --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml

echo ">> creating provider secret (edit with your real keys)"
kubectl create namespace mcp-rag --dry-run=client -o yaml | kubectl apply -f -
kubectl -n mcp-rag create secret generic mcp-rag-secrets \
  --from-literal=ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-}" \
  --from-literal=OPENAI_API_KEY="${OPENAI_API_KEY:-}" \
  --dry-run=client -o yaml | kubectl apply -f -

echo ">> done. next: make deploy && make argocd"
