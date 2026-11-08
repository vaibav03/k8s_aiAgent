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
MODEL_DIR=models/LATEST uvicorn kubeai_ml.server:app --port 8000
# or
docker build -t kubeai-ml:latest . && docker run -p 8000:8000 kubeai-ml:latest
```

`MODEL_DIR` can point to a bundle directory or to a `LATEST` pointer file. By default it
uses `models/LATEST`. The service loads the bundle once at startup. `/readyz` returns 503
until the bundle has loaded.

## API

| Method | Path | |
|---|---|---|
| GET | `/healthz` | `{"status":"ok"}` |
| GET | `/readyz` | `{"ready":true,"model_version":...}` or 503 |
| GET | `/v1/model` | manifest: features, horizon, threshold, metrics, baselines, data sha256, label definition |
| POST | `/v1/predict` | one pod |
| POST | `/v1/predict/batch` | `{"items":[...]}` → `{"results":[...]}` (up to 500 items) |
| GET | `/metrics` | Prometheus text: requests, predictions, anomalies, latency histogram |

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

## Test

```bash
python -m pytest
```

The tests cover:

- the feature definitions;
- the happy path, 422s, batch, and the no-limit fallback;
- that a leak ramp gets a higher `p_oom` than flat low usage, and that flat pods at
  50-99% of the limit are not flagged (trend gate);
- `/metrics`;
- server-side latency under 20 ms for a 600-sample request (measured at about 1.5 ms
  on a laptop).
