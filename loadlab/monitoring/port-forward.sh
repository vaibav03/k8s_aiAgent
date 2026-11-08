#!/usr/bin/env bash
# Keep a port-forward to Prometheus on localhost:9090 alive (restarts if it drops).
# Usage: ./port-forward.sh            (foreground)
#        nohup ./port-forward.sh >/tmp/prom-pf.log 2>&1 &   (background)
CTX="${KUBE_CONTEXT:-kubeai-lab}"
PORT="${PROM_PORT:-9090}"
while true; do
  kubectl --context "$CTX" -n monitoring port-forward svc/prometheus "$PORT:9090" --address 127.0.0.1
  echo "port-forward exited ($?) - restarting in 2s" >&2
  sleep 2
done
