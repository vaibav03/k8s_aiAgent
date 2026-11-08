#!/usr/bin/env bash
# Deploy kube-state-metrics + Prometheus into the kubeai-lab minikube profile and wait for readiness.
set -euo pipefail
CTX="${KUBE_CONTEXT:-kubeai-lab}"
DIR="$(cd "$(dirname "$0")" && pwd)"
kubectl --context "$CTX" apply -f "$DIR/kube-state-metrics.yaml"
kubectl --context "$CTX" apply -f "$DIR/prometheus.yaml"
"$DIR/metrics-server-15s.sh"
kubectl --context "$CTX" -n kube-system rollout status deploy/kube-state-metrics --timeout=300s
kubectl --context "$CTX" -n monitoring rollout status deploy/prometheus --timeout=300s
echo "monitoring ready. Run $DIR/port-forward.sh to expose Prometheus on localhost:9090"
