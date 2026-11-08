#!/usr/bin/env bash
# minikube's metrics-server addon ships with --metric-resolution=60s; upstream default is 15s.
# At 60s, a new pod has no HPA CPU reading for ~1-2 min, which hides HPA dynamics in short runs.
set -euo pipefail
CTX="${KUBE_CONTEXT:-kubeai-lab}"
IDX=$(kubectl --context "$CTX" -n kube-system get deploy metrics-server -o json |
  python3 -c "import json,sys;a=json.load(sys.stdin)['spec']['template']['spec']['containers'][0]['args'];print([i for i,x in enumerate(a) if x.startswith('--metric-resolution')][0])")
kubectl --context "$CTX" -n kube-system patch deploy metrics-server --type=json \
  -p "[{\"op\":\"replace\",\"path\":\"/spec/template/spec/containers/0/args/$IDX\",\"value\":\"--metric-resolution=15s\"}]"
kubectl --context "$CTX" -n kube-system rollout status deploy/metrics-server --timeout=120s
