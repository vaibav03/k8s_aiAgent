# Original data collector (Aug 2025)

These are the files that produced `ml_service/data/k8s_metrics.csv` (collected 2025-08-30). Until now they
existed only on a laptop (`kube-AI/dataset_creation/DevTrails`, a local, newer copy of Nirmalkanniaiyan/DevTrails).

- `server-prom.js` polls metrics-server / Prometheus every ~1 s for the `server-image` pod and writes the CSV
  (memory = metrics.k8s.io usage in Ki / 1000, status from `containerStatuses[1]` because of the istio sidecar).
- `k6_test.js`, `sample.sh`, `commands.txt` generate traffic through the Istio ingress gateway.
- `server-image.yaml`, `server-service.yaml`, `istio-gateway.yaml`, `virtual-service.yaml`, `curl-pod.yaml` are the workload.

Kept for provenance only. The new lab in `../loadlab/` replaces it with a reproducible, ground-truth-labelled setup.
