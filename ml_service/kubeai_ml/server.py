"""FastAPI service: OOM-risk prediction for many pods.

    MODEL_DIR=models/LATEST uvicorn kubeai_ml.server:app --port 8000
"""
from __future__ import annotations

import math
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field, field_validator

from .model import OomModel

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "LATEST"
MAX_SAMPLES = 1000
MAX_BATCH = 500


# ---------------------------------------------------------------- schemas

class Sample(BaseModel):
    ts: float
    working_set_bytes: float = Field(ge=0)

    @field_validator("ts", "working_set_bytes")
    @classmethod
    def finite(cls, v):
        if not math.isfinite(v):
            raise ValueError("must be finite")
        return v


class PredictRequest(BaseModel):
    pod: str = Field(min_length=1)
    namespace: str = "default"
    samples: List[Sample] = Field(min_length=2, max_length=MAX_SAMPLES)
    limit_bytes: Optional[float] = Field(default=None, gt=0)
    restart_count: Optional[int] = Field(default=None, ge=0)
    seconds_since_restart: Optional[float] = Field(default=None, ge=0)


class ForecastPoint(BaseModel):
    in_s: int
    working_set_bytes: float


class PredictResponse(BaseModel):
    pod: str
    namespace: str
    model_version: str
    p_oom: float
    is_anomaly: bool
    threshold: float
    horizon_s: int
    usage_fraction: Optional[float]
    trend_bytes_per_s: float
    time_to_limit_s: Optional[float]
    forecast: List[ForecastPoint]
    reasons: List[str]


class BatchRequest(BaseModel):
    items: List[PredictRequest] = Field(min_length=1, max_length=MAX_BATCH)


class BatchResponse(BaseModel):
    results: List[PredictResponse]


# ---------------------------------------------------------------- metrics

class Metrics:
    """Tiny Prometheus text-format registry (counters + one latency histogram)."""
    BUCKETS = (0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 1.0)

    def __init__(self):
        self.lock = threading.Lock()
        self.requests = {}          # (endpoint, status) -> count
        self.predictions = 0
        self.anomalies = 0
        self.buckets = [0] * len(self.BUCKETS)
        self.lat_sum = 0.0
        self.lat_count = 0

    def observe(self, endpoint, status, seconds):
        with self.lock:
            key = (endpoint, status)
            self.requests[key] = self.requests.get(key, 0) + 1
            self.lat_sum += seconds
            self.lat_count += 1
            for i, b in enumerate(self.BUCKETS):
                if seconds <= b:
                    self.buckets[i] += 1

    def count_predictions(self, results):
        with self.lock:
            self.predictions += len(results)
            self.anomalies += sum(1 for r in results if r["is_anomaly"])

    def render(self, model_version):
        with self.lock:
            lines = [
                "# HELP kubeai_requests_total HTTP requests by endpoint and status code.",
                "# TYPE kubeai_requests_total counter",
            ]
            for (ep, st), n in sorted(self.requests.items()):
                lines.append(f'kubeai_requests_total{{endpoint="{ep}",status="{st}"}} {n}')
            lines += [
                "# HELP kubeai_predictions_total Pods scored.",
                "# TYPE kubeai_predictions_total counter",
                f"kubeai_predictions_total {self.predictions}",
                "# HELP kubeai_anomalies_total Pods scored with p_oom >= threshold.",
                "# TYPE kubeai_anomalies_total counter",
                f"kubeai_anomalies_total {self.anomalies}",
                "# HELP kubeai_request_latency_seconds Request handling latency.",
                "# TYPE kubeai_request_latency_seconds histogram",
            ]
            for b, n in zip(self.BUCKETS, self.buckets):
                lines.append(f'kubeai_request_latency_seconds_bucket{{le="{b}"}} {n}')
            lines += [
                f'kubeai_request_latency_seconds_bucket{{le="+Inf"}} {self.lat_count}',
                f"kubeai_request_latency_seconds_sum {self.lat_sum:.6f}",
                f"kubeai_request_latency_seconds_count {self.lat_count}",
                "# HELP kubeai_model_info Loaded model bundle.",
                "# TYPE kubeai_model_info gauge",
                f'kubeai_model_info{{model_version="{model_version or ""}"}} {1 if model_version else 0}',
            ]
        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- app

state = {"model": None, "error": None}
metrics = Metrics()


@asynccontextmanager
async def lifespan(app):
    model_dir = os.environ.get("MODEL_DIR", str(DEFAULT_MODEL_DIR))
    try:
        state["model"] = OomModel.load(model_dir)
    except Exception as exc:  # stay up so /readyz can report the problem
        state["error"] = f"could not load model bundle from {model_dir}: {exc}"
    yield


app = FastAPI(title="KubeAI OOM predictor", version="1.0", lifespan=lifespan)

TRACKED = {"/v1/predict", "/v1/predict/batch"}


@app.middleware("http")
async def track(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    if request.url.path in TRACKED:
        metrics.observe(request.url.path, response.status_code, time.perf_counter() - start)
    return response


def require_model():
    model = state["model"]
    if model is None:
        return None, JSONResponse(status_code=503, content={"detail": state["error"] or "model not loaded"})
    return model, None


def run(model, req: PredictRequest):
    return model.predict(
        pod=req.pod, namespace=req.namespace,
        samples=[s.model_dump() for s in req.samples],
        limit_bytes=req.limit_bytes, restart_count=req.restart_count,
        seconds_since_restart=req.seconds_since_restart)


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/readyz")
def readyz():
    model = state["model"]
    if model is None:
        return JSONResponse(status_code=503, content={"ready": False, "error": state["error"]})
    return {"ready": True, "model_version": model.version}


@app.get("/v1/model")
def model_manifest():
    model, err = require_model()
    if err:
        return err
    m = model.manifest
    keys = ["model_version", "features", "horizon_s", "threshold", "metrics", "baselines",
            "cross_dataset", "forecast", "trained_at", "data_sha256", "label_definition",
            "feature_definition", "data", "split"]
    return {k: m.get(k) for k in keys}


@app.post("/v1/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    model, err = require_model()
    if err:
        return err
    result = run(model, req)
    metrics.count_predictions([result])
    return result


@app.post("/v1/predict/batch", response_model=BatchResponse)
def predict_batch(req: BatchRequest):
    model, err = require_model()
    if err:
        return err
    results = [run(model, item) for item in req.items]
    metrics.count_predictions(results)
    return {"results": results}


@app.get("/metrics", response_class=PlainTextResponse)
def prometheus_metrics():
    model = state["model"]
    return PlainTextResponse(metrics.render(model.version if model else None),
                             media_type="text/plain; version=0.0.4")
