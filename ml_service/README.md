# KubeAI ML service

This is an HTTP service that predicts, for many pods at once, the chance that each pod
will be OOMKilled soon. It replaces the old stdin LSTM+LightGBM bridge
(`legacy/hybrid_predict.py`). The service is FastAPI with a small LightGBM model and
needs no TensorFlow.

For what the model does, the real test numbers and its limits, see
[MODEL_CARD.md](MODEL_CARD.md).

```
ml_service/
  kubeai_ml/features.py   one feature definition, shared by training and serving
  kubeai_ml/data.py       CSV loading, unit conversion, crash events, examples
  kubeai_ml/train.py      train + evaluate + baselines + write a versioned bundle
  kubeai_ml/model.py      loads a bundle, turns samples into a prediction
  kubeai_ml/server.py     FastAPI app
  models/<version>/       model.txt, calibrator.json, manifest.json
  models/LATEST           name of the bundle to serve
  data/                   training (k8s_metrics.csv) and cross-check (nirmal_dataset_1.csv) data
```

## Setup

```bash
cd ml_service
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-train.txt     # serving only: requirements.txt
```

## Train

```bash
python -m kubeai_ml.train                 # evaluates H=30/60/120, ships best val F1, writes models/<version>/ + LATEST
python -m kubeai_ml.train --horizon 60    # force a horizon
python -m kubeai_ml.train --dry-run       # evaluate only, write nothing
python -m kubeai_ml.train --no-old-agent  # skip the legacy LSTM baseline (needs tensorflow + joblib)
```

The script prints validation and test metrics for the model and for every baseline:
the usage-fraction rule, the time-to-limit rule, the old agent pair, and the
persistence, trend and old-LSTM forecasts. It writes all of them into
`manifest.json`.

## Serve

```bash
MODEL_DIR=models/LATEST SCALING_MODEL_DIR=models/scaling/LATEST \
  uvicorn kubeai_ml.server:app --port 8000 --workers 1
# or
docker build -t kubeai-ml:latest . && docker run -p 8000:8000 kubeai-ml:latest
# or straight into the minikube node (no registry, no image load):
minikube -p kubeai-lab image build -t kubeai-ml:latest .
```

| Env | Default | |
|---|---|---|
| `MODEL_DIR` | `models/LATEST` | OOM-ahead bundle (`/v1/predict`); a bundle dir or a `LATEST` pointer file |
| `SCALING_MODEL_DIR` | `models/scaling/LATEST` | scaling bundle (`/v1/decide`): classifier, forecaster, capacity, policy |
| `POLICY_MODE` | unset (the bundle's policy) | replica-policy override passed to `DecisionEngine(policy_mode=...)` when the engine supports it (`predictive`, `fault_aware`, `hpa_gated`); an unknown value makes `/readyz` fail rather than silently using a default |

Both bundles are loaded once at startup (the decision engine also gets the OOM bundle as its
optional per-pod memory-risk input). `/readyz` returns 503 until **both** have loaded; if only
the scaling bundle is missing, `/v1/predict` still works and `/v1/decide` returns 503. Run
**one worker**: the decision engine keeps per-deployment stabilisation memory (scale-down
window, restart cooldown) in process, keyed by `namespace/deployment`.

## API

| Method | Path | |
|---|---|---|
| GET | `/healthz` | `{"status":"ok"}` |
| GET | `/readyz` | `{"ready":true,"model_version":...,"scaling_model_version":...,"policy_mode":...}` or 503 |
| GET | `/v1/model` | OOM manifest at the top level (features, horizon, threshold, metrics, baselines, data sha256, label definition) + `bundles: {oom, scaling}` with each bundle's load status, version, endpoint; `scaling.manifest` carries the scaling bundle's headline metrics with baselines |
| POST | `/v1/predict` | one pod |
| POST | `/v1/predict/batch` | `{"items":[...]}` → `{"results":[...]}` (up to 500 items) |
| POST | `/v1/decide` | one Deployment: telemetry window → state, confidence, desired replicas, actions, reasons |
| GET | `/metrics` | Prometheus text: requests, predictions, anomalies, latency histograms (`kubeai_request_latency_seconds`, `kubeai_decide_latency_seconds`), `kubeai_decide_state_total{state}`, `kubeai_decide_actions_total{type}`, `kubeai_decide_errors_total{reason}`, `kubeai_decide_telemetry_insufficient_total`, `kubeai_scaling_model_info` |

Request body:

- `samples`: chronological, between 2 and 1000 samples. The model was trained and
  evaluated on what the agent sends: one sample every 15 s (Prometheus instant query),
  40 samples, starting at the container's last restart. Raw 1 s polling also works (the
  service resamples to 15 s; the model card reports both shapes).
- `limit_bytes`: the **container's** own memory limit (the kernel OOM-kills each
  container against its own limit), or `null`.

`is_anomaly` is `p_oom >= threshold` **and** `time_to_limit_s <= alert_rule.max_time_to_limit_s`
(300 s in the shipped bundle). A pod that sits flat, even at 90% of its limit, gets a high
`p_oom` (the training data has almost no stable high-usage pods) but `is_anomaly: false`;
`reasons` says so. See MODEL_CARD.md, "Why the trend gate".

```bash
curl -s localhost:8000/v1/predict -H 'content-type: application/json' -d '{
  "pod": "web-0", "namespace": "default",
  "limit_bytes": 943718400,
  "restart_count": 2, "seconds_since_restart": 400,
  "samples": [
    {"ts": 1700000000, "working_set_bytes": 471859200},
    {"ts": 1700000060, "working_set_bytes": 566231040},
    {"ts": 1700000120, "working_set_bytes": 660602880},
    {"ts": 1700000180, "working_set_bytes": 754974720},
    {"ts": 1700000240, "working_set_bytes": 849346560}
  ]}'
```

```json
{"pod":"web-0","namespace":"default","model_version":"oom-gbm-h120-20261108121800",
 "p_oom":1.0,"is_anomaly":true,"threshold":0.5,"horizon_s":120,
 "usage_fraction":0.9,"trend_bytes_per_s":1415577.6,"time_to_limit_s":66.7,
 "forecast":[{"in_s":15,"working_set_bytes":865271808.0},{"in_s":30,"working_set_bytes":881197056.0},
             {"in_s":60,"working_set_bytes":913047552.0},{"in_s":120,"working_set_bytes":943718400.0}],
 "reasons":["usage 90% of limit (810 MiB)","rising 1.35 MiB/s, limit in ~67s",
            "P(OOM within 120s) = 1.00 >= threshold 0.50",
            "only 240s of history (model uses up to 300s)","container has restarted 2 times"]}
```

Batch request:

```bash
curl -s localhost:8000/v1/predict/batch -H 'content-type: application/json' \
  -d '{"items":[{"pod":"a","samples":[{"ts":0,"working_set_bytes":1e8},{"ts":15,"working_set_bytes":1e8}],"limit_bytes":1e9}]}'
```

What the service does with some inputs:

- Invalid input, such as fewer than 2 samples, negative bytes, `limit_bytes <= 0` or a
  missing `pod`, returns 422. In a batch, one invalid item makes the whole batch 422,
  and a batch holds at most 500 items: the agent splits into chunks of 500 and retries
  a 422 batch item by item through `/v1/predict`.
- If `limit_bytes` is `null`, then `p_oom` is 0 and `usage_fraction` and
  `time_to_limit_s` are `null`. Trend and forecast are still computed, and `reasons`
  explains why.

## POST /v1/decide

One SRE decision for one Deployment from its recent telemetry (call it every ~15 s). The
agent's deployment loop (`lib/scaler.js`) builds the body with the exact PromQL of
`loadlab/runner/promexport.py`, the exporter that produced the training data; any client
that reproduces those columns can call it. Field definitions, units and PromQL:
`kubeai_ml/scaling/engine.py` (module docstring) and `loadlab/data/DATASET_CARD.md`.

Request:

| Field | Type | |
|---|---|---|
| `deployment` | str, required | Deployment name (engine memory key = `namespace/deployment`) |
| `namespace` | str, `"default"` | |
| `series` | list, 1-2000 rows, required | one row per scrape, 5-15 s apart, oldest first or any order; at least 3 min, 15-30 min recommended (the bad-deploy CPU-per-request baseline looks 2-10 min back) |
| `pods` | list or null, ≤ 100 000 rows | per (ts, pod) rows; without them `noisy_pod` is undetectable and `bad_deploy` has no rollout check |
| `current_replicas` | int ≥ 0 or null | the Deployment's `spec.replicas` (default: newest `replicas_desired`) |
| `min_replicas`, `max_replicas` | int, 1 / 10 | hard bounds, `min ≤ max` |
| `now_ts` | unix s or null | wall clock of the decision; stale telemetry is judged against it; must be ≥ the newest row |
| `revision` | number or null | `deployment.kubernetes.io/revision`; applied to the newest row when the rows carry none |
| `agent_rollout_ts` | list of unix s, optional | times of rollouts the caller triggered itself (its own `rollout undo` / `rollout restart`); a new ReplicaSet appearing within 180 s after one is not counted as rollout evidence, so the caller's own undo cannot re-arm `rollback`. Omit when empty (older servers reject unknown keys). |

`series` rows, grouped by the four signal groups (`null` = no sample: unknown, **not** zero):

| Group | Fields |
|---|---|
| — | `ts` (required) |
| traffic | `rps` (required key, nullable), `goodput_rps`, `inflight` |
| latency / errors | `p95`, `error_rate` (required keys, nullable), `p50`, `p99` |
| CPU | `cpu_cores_used` (required key, nullable), `cpu_util_vs_request`, `throttle_ratio` |
| memory | `mem_ws_max_pod_frac_of_limit`, `mem_growth_bytes_per_s` |
| replicas / restarts | `replicas_ready` (required key, nullable), `replicas_desired`, `hpa_desired`, `pods_not_ready`, `restarts_total`, `oomkills_total`, `crashloop_pods`, `revision` |

`pods` rows: `ts`, `pod` (required), `cpu`, `throttle`, `mem_ws`, `mem_frac_limit`, `rps`,
`ready`, `restarts`. Unknown columns in rows are ignored (so dataset rows can be posted as
they are); unknown top-level fields, negative values, `error_rate`/`throttle_ratio` outside
[0, 1], non-finite numbers, `min_replicas > max_replicas` and `now_ts` older than the newest
row return 422. An optional field that is `null` on every row is left to the engine's
documented fill (e.g. `hpa_desired ← replicas_desired`), and the response's `reasons` lists
it.

Response:

```json
{"deployment":"loadlab","namespace":"lab-sample","model_version":"scaling-20261108194802",
 "policy_mode":"predictive","state":"memory_leak","confidence":0.8979,
 "state_probs":{"healthy":0.013,"memory_leak":0.8979,"noisy_pod":0.0216,"...":"..."},
 "scaling_helps":false,"current_replicas":3,"desired_replicas":3,
 "replicas_needed":{"now":1,"60":1,"120":2,"300":4},
 "forecast":{"60":{"p50":8.449,"p90":13.962},"120":{"p50":8.465,"p90":21.246},"300":{"p50":16.311,"p90":36.481}},
 "safe_rps_per_pod":11.676,"slo_risk":0.0022,"cpu_per_req_s":0.027,
 "actions":[
  {"type":"restart_pod","target":"loadlab-5965489b55-5g76n","params":{"mode":"delete_pod"},
   "reason":"working set grows toward the memory limit (working set +555 KB/s (fastest of the pods)); restart before OOM"},
  {"type":"page","target":"service owner","params":{"severity":"ticket"},"reason":"memory leak: open a bug; restarts only buy time"},
  {"type":"none","target":"loadlab","params":{"scaling":"blocked"},"reason":"do NOT scale: memory_leak is not a capacity problem"}],
 "reasons":["state memory_leak (p=0.90): working-set growth KB/s (1 m median) 554.37; ...","policy: do_not_scale: ...","..."],
 "telemetry_ok":true,
 "inputs":{"deploy_rows":60,"history_s":295.0,"pod_rows":157,"current_replicas":3,"ready_replicas":3},
 "latency_ms":682.43}
```

(That is the real in-cluster answer for the first 300 s of the held-out run
`d-leak-medium-r2`. Latency on kubeai-lab, 1-CPU-limit pod: 682 ms for that first call after
start-up, then 190-460 ms per call as measured by the agent for 8-44-row windows, mean
~240 ms server-side over 12 calls. One call per Deployment per 15 s is fine; this is not a
per-request hot path.)

`actions[].type` is one of `scale` (`params.from`, `params.to`), `restart_pod`
(`params.mode`: `delete_pod` with the pod as `target`, or `rollout_restart` with the
Deployment as `target` when only one pod is ready), `rollback` (only with rollout evidence),
`page` (`params.severity`) and `none` (`params.scaling: "blocked"` = the "do NOT scale" gate).
`telemetry_ok: false` means the newest `rps` is stale or the recent history has a gap: the
engine froze scale-down and ignored the forecast, and `reasons[0]` says why. The service does
not execute anything; the agent decides what to run (`ACT_MODE`).

## Test

```bash
PYTHONPATH=. python -m pytest -q
```

The tests cover:

- the feature definitions;
- the happy path, 422s, batch, and the no-limit fallback;
- that a leak ramp gets a higher `p_oom` than flat low usage, and that flat pods at
  50-99% of the limit are not flagged (trend gate);
- `/metrics`;
- server-side latency under 20 ms for a 600-sample request (measured at about 1.5 ms
  on a laptop);
- `/v1/decide` (`tests/test_decide_server.py`): response schema, steady load holds, the
  minimal required-fields-only body, stale telemetry (`telemetry_ok: false`, no scale-down),
  the top-level `revision`, 12 kinds of invalid input → 422 (also `Infinity`, which used to
  crash FastAPI's default 422 renderer), extra dataset columns ignored, `/readyz` 503 and
  `/v1/decide` 503 without the scaling bundle while `/v1/predict` keeps working, a bad
  `POLICY_MODE` → not ready, `/v1/model` listing both bundles, the decide metrics, and the
  held-out leak window (`d-leak-medium-r2`) → `memory_leak` with no scale-up.
