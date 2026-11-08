# KubeAI agent: predicting pod OOM kills

The agent watches the memory of Kubernetes pods. For each pod it asks a trained model how likely an OOM kill is within the next horizon. It alerts only on repeated anomalies (k-of-n). It can also ask Gemini for a remediation, which it checks against an allowlist and runs as a `kubectl --dry-run=server` unless you opt in to real execution.

```
                 +--------------------+   sum by (pod, container)(container_memory_working_set_bytes{...})
                 |    Prometheus      |<-----------------------------------------+
                 +--------------------+                                          |
                                                                                 |
 +----------------+   pods?labelSelector=  +-------------------------------------+------+
 | Kubernetes API |<-----------------------|  agent (Node, index.js + lib/)             |
 | (proxy or SA)  |   limits, restarts,    |  - one sequential loop, POLL_MS            |
 +----------------+   CrashLoopBackOff     |  - per-container ring buffer {ts, bytes}   |
        ^                                  |  - k-of-n hysteresis + cooldown            |
        |  kubectl (allowlisted argv,      |  - Express :3001 /healthz /status /pods/x  |
        |  --dry-run=server by default)    +-----+-------------------------+-----------+
        +----------------------------------------+                         |
                                                   POST /v1/predict/batch  |   strict JSON
                                         +-------------------------+       |   recommendation
                                         | ML service (FastAPI)    |<------+   +-----------+
                                         | ml_service/  :8000      |           |  Gemini   |
                                         | p_oom, time_to_limit,   |           | (optional)|
                                         | forecast, reasons       |           +-----------+
                                         +-------------------------+
```

## Components

| Path | What it is |
|---|---|
| `index.js` | Entry point. Loads config, starts the Express API and the monitoring loop. |
| `lib/config.js` | Reads every setting from the environment (see `.env.example`). |
| `lib/kube.js` | Pod discovery by label selector. Reads each container's own memory limit (real quantity parser; never summed across containers), restart count and seconds since its last termination, CrashLoopBackOff on any container, and the owning Deployment. Sends a bearer token from the service-account file when one is present. |
| `lib/prom.js` | One PromQL query per tick for all pods, per container. Missing data stays missing. |
| `lib/ml.js` | Client for `POST /v1/predict/batch`: chunks of at most `ML_BATCH_MAX` (500, the server's cap); a chunk rejected with 422 is retried item by item via `POST /v1/predict`, so one bad item costs only its own prediction. A limit that is not > 0 is sent as `null`. |
| `lib/alert.js` | k-of-n hysteresis and cooldown on the service's boolean `is_anomaly`. |
| `lib/remediation.js` | Gemini via `@langchain/google-genai` (client created once), the strict JSON schema, the allowlist validator, the kubectl argv builder, and `execFile` (no shell). |
| `lib/quantity.js` | Kubernetes quantity parser (`Ki Mi Gi Ti`, `K M G`, plain bytes, e-notation). |
| `ml_service/` | Training pipeline and FastAPI model server. See `ml_service/README.md`. |
| `k8s/` | Namespace, ML Deployment and Service, agent Deployment, Service, ConfigMap, ServiceAccount, Role and RoleBinding. The LLM key Secret is created out of band (see below), never applied from this directory. |
| `legacy/` | The old stdin LSTM+LightGBM hybrid, and why it was replaced. |
| `test/agent.smoke.mjs` | Runs the real agent against a fake Prometheus, a fake kube API and a fake ML service. |

## The ML service contract

The agent depends only on this contract. The full schema is in `ml_service/`.

- `POST /v1/predict/batch` with body `{"items":[{"pod","namespace","samples":[{"ts","working_set_bytes"}...],"limit_bytes","restart_count","seconds_since_restart"}]}` returns `{"results":[{"pod","namespace","model_version","p_oom","is_anomaly","threshold","horizon_s","usage_fraction","trend_bytes_per_s","time_to_limit_s","forecast","reasons"}]}`.
- `GET /healthz`, `GET /readyz` (503 until the model bundle has loaded), `GET /v1/model` (the manifest: metrics, data hash, label definition), and `GET /metrics` (Prometheus format).

The agent sends **one item per container**, with that container's own samples and its own `limit_bytes`, because the kernel OOM-kills each container against its own limit. (Comparing a pod-level working set with the sum of the container limits hides an app container at its limit behind a sidecar with a large limit, and counts an unlimited sidecar's memory as usage.) The pod's prediction is its most at-risk container (`latest_prediction.container`).

Model caveats, documented in more detail in `ml_service/`: the training data is one pod and one 10.5 h run, and the true memory limit for that run is unknown. Because OOM kills happen at the limit, the observed maximum (887.345 in the collector's `Ki/1000` units) is used as a stand-in for the limit. Cross-dataset results on `nirmal_dataset_1.csv` (900Mi limit) are reported alongside the in-dataset ones.

## The decision loop (one tick)

1. List the pods in `NAMESPACE` that match `POD_SELECTOR`. Forget pods that are gone. When a container's restart count goes up, clear that container's window, because memory resets after a restart and the old samples would fake a trend. (The model is trained on histories cut the same way.)
2. Run `sum by (pod, container) (container_memory_working_set_bytes{namespace="ns",pod=~"a|b",container!="",container!="POD"})` and append to each container's ring buffer (`WINDOW` samples). A container with no result gets no sample.
3. Send every container that has at least 2 samples to the ML service, one item each, in batches of at most `ML_BATCH_MAX`. If the service is down, the agent logs the error and retries on the next tick. A 422 batch is retried item by item.
4. Take each pod's most at-risk container as the pod's prediction and record its `is_anomaly` in the pod's last-N history. An alert fires when at least `ALERT_K` of the last `ALERT_N` predictions are anomalous and the pod is not in cooldown (`COOLDOWN_MS`).
5. When an alert fires:
   - If the pod is already in CrashLoopBackOff, the agent logs `CRITICAL` (notify the administrator) and makes no LLM call.
   - If `GOOGLE_API_KEY` is not set, the recommendation is recorded as `none` with the reason.
   - Otherwise Gemini returns `{"action","target","params","reason"}`. The validator accepts only:
     - `patch_memory_limit` (a container in the pod; the new limit must be above the current one, at least 1.2x the container's latest working set, at least 64Mi, and at most `MAX_MEMORY_LIMIT`; rejected when the container has neither a current limit nor an observed working set)
     - `rollout_restart` (dry run: `kubectl rollout restart` has no `--dry-run` flag, so the agent runs the equivalent `kubectl patch deployment/x -p '{"spec":{"template":{"metadata":{"annotations":{"kubectl.kubernetes.io/restartedAt":"<now>"}}}}}' --dry-run=server`; with `ALLOW_EXEC=true` it runs the real `rollout restart`)
     - `scale` (1 to `MAX_REPLICAS` replicas)
     - `none`

     The target must be the pod's own Deployment. The agent then builds the argv itself, for example `kubectl -n default set resources deployment/x -c app --limits=memory=1024Mi --dry-run=server`. The text the model returns is never executed.

The next tick is scheduled only after the current one finishes, so ticks never overlap.

**Steady pods.** The model's training data has almost no pods that sit flat at high usage, so its `p_oom` for a pod that runs steadily at about 69% or more of its limit is high (0.72-0.93). The service therefore sets `is_anomaly` only when `p_oom >= threshold` **and** memory is on course to reach the limit within 300 s (`time_to_limit_s <= 300`), so steady pods, even at 90% of their limit, do not alert. The flip side: a pod that sits flat just below its limit and is then OOM-killed by a sudden jump gets no early warning. Do not build alerts on raw `p_oom` without a trend condition. See `ml_service/MODEL_CARD.md`.

## Running locally

Prerequisites: Node 18.17+, Python 3.12, and a cluster with Prometheus (kube-prometheus-stack or istio's) and metrics-server.

```bash
# 1. Cluster access
kubectl proxy --port=8001 &
kubectl -n monitoring port-forward svc/prometheus-operated 9090 &   # adjust to your Prometheus svc

# 2. ML service (train + serve) - exact commands in ml_service/README.md
cd ml_service && ...            # serves on :8000; check: curl localhost:8000/readyz

# 3. Agent
cd ..
npm install
cp .env.example .env            # set POD_SELECTOR, optionally GOOGLE_API_KEY
npm start
curl localhost:3001/status | jq
curl localhost:3001/pods/<pod-name> | jq
```

`ALLOW_EXEC` defaults to `false`. In that mode every remediation runs as a server-side dry run: the API server validates it against RBAC and admission, and nothing changes. (`rollout_restart` is dry-run as the equivalent `kubectl patch` of the `restartedAt` annotation, because `kubectl rollout restart` has no `--dry-run` flag.) Set `ALLOW_EXEC=true` only when you want the agent to change Deployments.

## Running in the cluster

```bash
docker build -t kubeai-ml:latest ml_service
docker build -t kubeai-agent:latest .
minikube image load kubeai-ml:latest && minikube image load kubeai-agent:latest   # or push to your registry

kubectl apply -f k8s/            # safe to re-run: k8s/ contains no Secret
# optional LLM key, created once and never committed (the agent's secretRef is optional):
kubectl -n kubeai create secret generic kubeai-agent-secrets --from-literal=GOOGLE_API_KEY=...
kubectl -n kubeai rollout restart deployment/kubeai-agent   # env is read at start
kubectl -n kubeai port-forward svc/kubeai-agent 3001 & curl localhost:3001/status
```

Inside the cluster the agent talks to `https://kubernetes.default.svc`. It uses the projected service-account token (re-read on every call because the token rotates) and the cluster CA (`NODE_EXTRA_CA_CERTS`). The Role is namespace-scoped:
- `pods`: get, list, watch
- `deployments`: get, patch
- `deployments/scale`: get, patch, update

The Role lives in the watched namespace (`default`). If you watch a different namespace, edit `k8s/agent-rbac.yaml` and `NAMESPACE` in `k8s/agent.yaml`. Run the agent as a single replica, because alert and cooldown state is held in memory.

## Agent API

- `GET /healthz` returns `{"status":"ok"}`.
- `GET /status` returns the config (without secrets), the tick count, the last error per dependency, and, for each pod: its containers (limit, restarts, samples, latest working set), its latest prediction (the most at-risk container), alert state (recent history, active, fired_at, cooldown_until), last recommendation, and the kubectl results.
- `GET /pods/:name` returns one pod's view, or 404.

## Tests

```bash
npm run check   # node --check on every source file
npm test        # unit checks + agent child process vs fake Prometheus/kube/ML + fake-LLM dry-run path
```

The smoke test checks the following:
- the label selector is used, and one PromQL query runs per tick
- pods with no data are not sent to the model
- limits are converted to bytes
- the agent recovers after the ML service returns 503
- an alert fires only after 2 of 3 anomalies, and cooldown stops it from re-firing
- a healthy pod stays quiet, and a crash-looping pod logs CRITICAL
- without an API key, kubectl is never executed
- with a fake LLM, the validated recommendation runs as `--dry-run=server`
- every allowlisted action's dry-run argv passes a fake kubectl that, like the real one, rejects `--dry-run` on `rollout restart`
- new memory limits below 64Mi, below 1.2x the working set, or with no limit and no working set are rejected
- items are per container with each container's own limit, and a sidecar with a large limit does not hide an app container at its own limit
- 1,020 items go out in chunks of 500, and one invalid item (422) costs only its own prediction

## Security notice: leaked API key

A real Google API key was committed to this repository's public history (`.env` in commits `c700f68` "Integrated llm call" and `48952ce` "Working", both on `origin/main`). Deleting the file does not remove it from history. **That key must be revoked and rotated in Google Cloud**; treat it as compromised whatever happens to the history. Rewriting history (`git filter-repo --path .env --invert-paths` plus a force-push) is optional and needs every clone to re-fetch. `.env` and `.env.*` are git-ignored; keep the key only in your local `.env` or the cluster Secret.
