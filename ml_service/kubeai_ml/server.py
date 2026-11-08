"""FastAPI service: OOM-risk prediction per pod + deployment-level SRE decisions.

    MODEL_DIR=models/LATEST SCALING_MODEL_DIR=models/scaling/LATEST \
        uvicorn kubeai_ml.server:app --port 8000

Endpoints: POST /v1/predict, POST /v1/predict/batch (OOM-ahead model, per pod/container),
POST /v1/decide (scaling bundle: state + replicas + remediation per Deployment),
GET /v1/model (both bundles), /healthz, /readyz (both bundles loaded), /metrics.

Run with ONE worker process: the decision engine keeps per-deployment stabilisation memory
(scale-down window, restart cooldown) in process.
"""
from __future__ import annotations

import math
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .model import OomModel

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "LATEST"
DEFAULT_SCALING_DIR = Path(__file__).resolve().parent.parent / "models" / "scaling" / "LATEST"
MAX_SAMPLES = 1000
MAX_BATCH = 500
MAX_DEPLOY_ROWS = 2000          # 30 min at 1 s, 2.7 h at 5 s
MAX_POD_ROWS = 100_000          # e.g. 360 scrapes x 250 pods


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


# ---------------------------------------------------------------- /v1/decide schemas
#
# One row per Prometheus evaluation (5-15 s apart), produced by the PromQL of
# loadlab/runner/promexport.py (the agent's lib/promDeploy.js reuses it verbatim). The four
# signal groups: CPU (cpu_cores_used, cpu_util_vs_request, throttle_ratio), memory
# (mem_ws_max_pod_frac_of_limit, mem_growth_bytes_per_s), traffic (rps, goodput_rps, inflight)
# and latency/errors (p50, p95, p99, error_rate), plus replica / restart bookkeeping.
# ``null`` means "no sample" (unknown, NOT zero); the engine freezes scale-down on stale data.

def _finite_or_none(v):
    if v is not None and not math.isfinite(v):
        raise ValueError("must be finite or null")
    return v


class DeployObs(BaseModel):
    model_config = ConfigDict(extra="ignore")   # dataset rows carry extra columns; they are dropped

    ts: float
    # required keys (value may be null = missing sample)
    rps: Optional[float] = Field(ge=0)
    p95: Optional[float] = Field(ge=0)
    error_rate: Optional[float] = Field(ge=0, le=1)
    cpu_cores_used: Optional[float] = Field(ge=0)
    replicas_ready: Optional[float] = Field(ge=0)
    # optional
    goodput_rps: Optional[float] = Field(default=None, ge=0)
    inflight: Optional[float] = Field(default=None, ge=0)
    p50: Optional[float] = Field(default=None, ge=0)
    p99: Optional[float] = Field(default=None, ge=0)
    cpu_util_vs_request: Optional[float] = Field(default=None, ge=0)
    throttle_ratio: Optional[float] = Field(default=None, ge=0, le=1)
    mem_ws_max_pod_frac_of_limit: Optional[float] = Field(default=None, ge=0)
    mem_growth_bytes_per_s: Optional[float] = None
    replicas_desired: Optional[float] = Field(default=None, ge=0)
    restarts_total: Optional[float] = Field(default=None, ge=0)
    oomkills_total: Optional[float] = Field(default=None, ge=0)
    crashloop_pods: Optional[float] = Field(default=None, ge=0)
    hpa_desired: Optional[float] = Field(default=None, ge=0)
    pods_not_ready: Optional[float] = Field(default=None, ge=0)
    revision: Optional[float] = Field(default=None, ge=0)

    @field_validator("*")
    @classmethod
    def finite(cls, v):
        return _finite_or_none(v) if isinstance(v, float) else v


class PodObs(BaseModel):
    model_config = ConfigDict(extra="ignore")

    ts: float
    pod: str = Field(min_length=1, max_length=253)
    cpu: Optional[float] = Field(default=None, ge=0)
    throttle: Optional[float] = Field(default=None, ge=0, le=1)
    mem_ws: Optional[float] = Field(default=None, ge=0)
    mem_frac_limit: Optional[float] = Field(default=None, ge=0)
    rps: Optional[float] = Field(default=None, ge=0)
    ready: Optional[float] = Field(default=None, ge=0, le=1)
    restarts: Optional[float] = Field(default=None, ge=0)

    @field_validator("*")
    @classmethod
    def finite(cls, v):
        return _finite_or_none(v) if isinstance(v, float) else v


class DecideRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    deployment: str = Field(min_length=1, max_length=253)
    namespace: str = Field(default="default", min_length=1, max_length=63)
    series: List[DeployObs] = Field(min_length=1, max_length=MAX_DEPLOY_ROWS)
    pods: Optional[List[PodObs]] = Field(default=None, max_length=MAX_POD_ROWS)
    current_replicas: Optional[int] = Field(default=None, ge=0, le=10_000)
    min_replicas: int = Field(default=1, ge=0, le=10_000)
    max_replicas: int = Field(default=10, ge=1, le=10_000)
    now_ts: Optional[float] = None
    revision: Optional[float] = Field(default=None, ge=0)
    # unix times of rollouts the caller itself triggered (its own rollout undo / rollout restart):
    # a new ReplicaSet that appears right after one of these is not rollout evidence
    agent_rollout_ts: Optional[List[float]] = Field(default=None, max_length=200)

    @field_validator("now_ts")
    @classmethod
    def finite_now(cls, v):
        return _finite_or_none(v)

    @field_validator("agent_rollout_ts")
    @classmethod
    def finite_agent_ts(cls, v):
        if v is None:
            return v
        if not all(math.isfinite(x) for x in v):
            raise ValueError("agent_rollout_ts must be finite")
        return v

    @model_validator(mode="after")
    def bounds(self):
        if self.min_replicas > self.max_replicas:
            raise ValueError(f"min_replicas {self.min_replicas} > max_replicas {self.max_replicas}")
        if self.now_ts is not None and self.now_ts < max(r.ts for r in self.series):
            raise ValueError("now_ts is older than the newest series row")
        return self


class DecideAction(BaseModel):
    type: str                       # scale | restart_pod | rollback | page | none
    target: str
    params: Dict[str, Any]
    reason: str


class DecideResponse(BaseModel):
    deployment: str
    namespace: str
    model_version: str
    policy_mode: Optional[str]
    state: str
    state_probs: Dict[str, float]
    confidence: float
    scaling_helps: bool
    current_replicas: int
    desired_replicas: int
    replicas_needed: Dict[str, Optional[int]]
    forecast: Dict[str, Dict[str, Optional[float]]]
    safe_rps_per_pod: Optional[float]
    slo_risk: Optional[float]
    cpu_per_req_s: Optional[float]
    actions: List[DecideAction]
    reasons: List[str]
    telemetry_ok: bool
    inputs: Dict[str, Any]
    latency_ms: float


# ---------------------------------------------------------------- metrics

class Metrics:
    """Tiny Prometheus text-format registry (counters + latency histograms)."""
    BUCKETS = (0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 1.0)
    DECIDE_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.2, 0.35, 0.5, 1.0, 2.5, 5.0)

    def __init__(self):
        self.lock = threading.Lock()
        self.requests = {}          # (endpoint, status) -> count
        self.predictions = 0
        self.anomalies = 0
        self.buckets = [0] * len(self.BUCKETS)
        self.lat_sum = 0.0
        self.lat_count = 0
        self.decide_buckets = [0] * len(self.DECIDE_BUCKETS)
        self.decide_sum = 0.0
        self.decide_count = 0
        self.decide_states = {}     # state -> count
        self.decide_actions = {}    # action type -> count
        self.decide_errors = {}     # reason -> count
        self.telemetry_bad = 0

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

    def count_decision(self, out, seconds):
        with self.lock:
            self.decide_sum += seconds
            self.decide_count += 1
            for i, b in enumerate(self.DECIDE_BUCKETS):
                if seconds <= b:
                    self.decide_buckets[i] += 1
            st = out["state"]
            self.decide_states[st] = self.decide_states.get(st, 0) + 1
            for a in out["actions"]:
                self.decide_actions[a["type"]] = self.decide_actions.get(a["type"], 0) + 1
            if not out["telemetry_ok"]:
                self.telemetry_bad += 1

    def count_decide_error(self, reason):
        with self.lock:
            self.decide_errors[reason] = self.decide_errors.get(reason, 0) + 1

    def render(self, model_version, scaling_version=None):
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
                "# HELP kubeai_decide_latency_seconds DecisionEngine.decide wall time per /v1/decide call.",
                "# TYPE kubeai_decide_latency_seconds histogram",
            ]
            for b, n in zip(self.DECIDE_BUCKETS, self.decide_buckets):
                lines.append(f'kubeai_decide_latency_seconds_bucket{{le="{b}"}} {n}')
            lines += [
                f'kubeai_decide_latency_seconds_bucket{{le="+Inf"}} {self.decide_count}',
                f"kubeai_decide_latency_seconds_sum {self.decide_sum:.6f}",
                f"kubeai_decide_latency_seconds_count {self.decide_count}",
                "# HELP kubeai_decide_state_total Decisions by classified deployment state.",
                "# TYPE kubeai_decide_state_total counter",
            ]
            for st, n in sorted(self.decide_states.items()):
                lines.append(f'kubeai_decide_state_total{{state="{st}"}} {n}')
            lines += [
                "# HELP kubeai_decide_actions_total Actions emitted by type (scale|restart_pod|rollback|page|none).",
                "# TYPE kubeai_decide_actions_total counter",
            ]
            for t, n in sorted(self.decide_actions.items()):
                lines.append(f'kubeai_decide_actions_total{{type="{t}"}} {n}')
            lines += [
                "# HELP kubeai_decide_errors_total /v1/decide calls that produced no decision.",
                "# TYPE kubeai_decide_errors_total counter",
            ]
            for r, n in sorted(self.decide_errors.items()):
                lines.append(f'kubeai_decide_errors_total{{reason="{r}"}} {n}')
            lines += [
                "# HELP kubeai_decide_telemetry_insufficient_total Decisions made on stale/gappy telemetry.",
                "# TYPE kubeai_decide_telemetry_insufficient_total counter",
                f"kubeai_decide_telemetry_insufficient_total {self.telemetry_bad}",
                "# HELP kubeai_model_info Loaded OOM model bundle.",
                "# TYPE kubeai_model_info gauge",
                f'kubeai_model_info{{model_version="{model_version or ""}"}} {1 if model_version else 0}',
                "# HELP kubeai_scaling_model_info Loaded scaling (decision) bundle.",
                "# TYPE kubeai_scaling_model_info gauge",
                f'kubeai_scaling_model_info{{model_version="{scaling_version or ""}"}} {1 if scaling_version else 0}',
            ]
        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- app

state = {"model": None, "error": None, "engine": None, "engine_error": None}
metrics = Metrics()
engine_lock = threading.Lock()     # DecisionEngine keeps per-deployment memory: one decide at a time


def load_engine(scaling_dir, oom_dir):
    import dataclasses

    from .scaling.engine import DecisionEngine
    kw = {"bundle_dir": scaling_dir, "oom_model_dir": oom_dir}
    mode = os.environ.get("POLICY_MODE", "").strip()
    if mode:   # replica policy override (engines that support it: predictive | fault_aware | hpa_gated)
        if "policy_mode" not in {f.name for f in dataclasses.fields(DecisionEngine)}:
            raise ValueError("POLICY_MODE is set but this DecisionEngine has no policy_mode")
        kw["policy_mode"] = mode
    return DecisionEngine(**kw)


def engine_mode(eng):
    return getattr(eng, "mode", None)


@asynccontextmanager
async def lifespan(app):
    model_dir = os.environ.get("MODEL_DIR", str(DEFAULT_MODEL_DIR))
    scaling_dir = os.environ.get("SCALING_MODEL_DIR", str(DEFAULT_SCALING_DIR))
    state.update(model=None, error=None, engine=None, engine_error=None)
    try:
        state["model"] = OomModel.load(model_dir)
    except Exception as exc:  # stay up so /readyz can report the problem
        state["error"] = f"could not load model bundle from {model_dir}: {exc}"
    try:
        # the engine uses the same OOM bundle (optional input: per-pod memory risk)
        state["engine"] = load_engine(scaling_dir, model_dir if state["model"] is not None else None)
    except Exception as exc:
        state["engine_error"] = f"could not load scaling bundle from {scaling_dir}: {exc}"
    yield


app = FastAPI(title="KubeAI ML service", version="2.0", lifespan=lifespan)

TRACKED = {"/v1/predict", "/v1/predict/batch", "/v1/decide"}


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    """FastAPI's default 422 echoes the input, which crashes JSON encoding on Infinity/NaN."""
    errors = jsonable(jsonable_encoder(exc.errors(), custom_encoder={float: lambda v: v}))
    return JSONResponse(status_code=422, content={"detail": errors})


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


def require_engine():
    eng = state["engine"]
    if eng is None:
        return None, JSONResponse(status_code=503,
                                  content={"detail": state["engine_error"] or "scaling bundle not loaded"})
    return eng, None


def run(model, req: PredictRequest):
    return model.predict(
        pod=req.pod, namespace=req.namespace,
        samples=[s.model_dump() for s in req.samples],
        limit_bytes=req.limit_bytes, restart_count=req.restart_count,
        seconds_since_restart=req.seconds_since_restart)


def jsonable(o):
    """numpy scalars -> python, NaN/inf -> None, dict keys -> str (forecast horizons are ints)."""
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if hasattr(o, "item") and not isinstance(o, (str, bytes)):
        try:
            o = o.item()
        except (AttributeError, ValueError, TypeError):
            pass
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/readyz")
def readyz():
    model, eng = state["model"], state["engine"]
    if model is None or eng is None:
        return JSONResponse(status_code=503, content={
            "ready": False,
            "error": "; ".join(e for e in (state["error"], state["engine_error"]) if e) or "loading",
            "model_version": model.version if model else None,
            "scaling_model_version": eng.version if eng else None})
    return {"ready": True, "model_version": model.version, "scaling_model_version": eng.version,
            "policy_mode": engine_mode(eng)}


OOM_KEYS = ["model_version", "features", "horizon_s", "threshold", "metrics", "baselines",
            "cross_dataset", "forecast", "trained_at", "data_sha256", "label_definition",
            "feature_definition", "data", "split"]
SCALING_KEYS = ["version", "created_at", "git_commit", "status", "components", "policy",
                "headline_metrics_with_baselines", "data", "split"]


@app.get("/v1/model")
def model_manifest():
    """Both bundles. The OOM manifest stays at the top level (backwards compatible);
    ``bundles`` lists every bundle with its load status."""
    model, eng = state["model"], state["engine"]
    out = {k: model.manifest.get(k) for k in OOM_KEYS} if model is not None else {}
    oom = ({"loaded": True, "kind": "oom-ahead", "endpoint": "/v1/predict", "model_version": model.version}
           if model is not None else {"loaded": False, "kind": "oom-ahead", "error": state["error"]})
    if eng is not None:
        m = eng.bundle.manifest
        scaling = {"loaded": True, "kind": "scaling-decision", "endpoint": "/v1/decide",
                   "model_version": eng.version, "path": eng.bundle.path.name,
                   "states": list(_states()), "oom_input": eng._oom is not None,
                   "policy_mode": engine_mode(eng),
                   "manifest": jsonable({k: m.get(k) for k in SCALING_KEYS if k in m})}
    else:
        scaling = {"loaded": False, "kind": "scaling-decision", "error": state["engine_error"]}
    out["bundles"] = {"oom": oom, "scaling": scaling}
    if model is None and eng is None:
        return JSONResponse(status_code=503, content=out)
    return out


def _states():
    from .scaling import states as S
    return S.STATES


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


@app.post("/v1/decide", response_model=DecideResponse)
def decide(req: DecideRequest):
    """One SRE decision for one Deployment from its recent telemetry window."""
    eng, err = require_engine()
    if err:
        metrics.count_decide_error("not_loaded")
        return err
    rows = [r.model_dump(exclude_none=False) for r in req.series]
    if req.revision is not None:
        newest = max(range(len(rows)), key=lambda i: rows[i]["ts"])
        if rows[newest].get("revision") is None:
            rows[newest]["revision"] = req.revision
    # optional columns that are null on every row are dropped, so the engine fills them
    # with its documented derivations (e.g. hpa_desired <- replicas_desired)
    optional = [k for k in DeployObs.model_fields if k not in ("ts", "rps", "p95", "error_rate",
                                                                "cpu_cores_used", "replicas_ready")]
    empty = {k for k in optional if all(r.get(k) is None for r in rows)}
    rows = [{k: (float("nan") if v is None else v) for k, v in r.items() if k not in empty} for r in rows]
    pods = None
    if req.pods:
        pods = [{k: (float("nan") if v is None and k != "pod" else v) for k, v in p.model_dump().items()
                 if not (k == "restarts" and v is None)} for p in req.pods]
    key = f"{req.namespace}/{req.deployment}"
    t0 = time.perf_counter()
    try:
        with engine_lock:
            out = eng.decide(rows, pods, req.current_replicas, req.min_replicas, req.max_replicas,
                             key=key, deployment=req.deployment, now_ts=req.now_ts,
                             agent_rollout_ts=req.agent_rollout_ts)
    except ValueError as exc:
        metrics.count_decide_error("invalid_input")
        return JSONResponse(status_code=422, content={"detail": str(exc)})
    seconds = time.perf_counter() - t0
    out = jsonable(out)
    metrics.count_decision(out, seconds)
    return {
        "deployment": req.deployment,
        "namespace": req.namespace,
        "model_version": out["model_version"],
        "policy_mode": engine_mode(eng),
        "state": out["state"],
        "state_probs": out["state_probs"],
        "confidence": out["confidence"],
        "scaling_helps": bool(out["scaling_helps"]),
        "current_replicas": int(out["inputs"]["current_replicas"]),
        "desired_replicas": out["desired_replicas"],
        "replicas_needed": out["replicas_needed"],
        "forecast": out["forecast"],
        "safe_rps_per_pod": out["safe_rps_per_pod"],
        "slo_risk": out["slo_risk"],
        "cpu_per_req_s": out["cpu_per_req_s"],
        "actions": out["actions"],
        "reasons": out["reasons"],
        "telemetry_ok": bool(out["telemetry_ok"]),
        "inputs": out["inputs"],
        "latency_ms": round(seconds * 1000, 2),
    }


@app.get("/metrics", response_class=PlainTextResponse)
def prometheus_metrics():
    model, eng = state["model"], state["engine"]
    return PlainTextResponse(metrics.render(model.version if model else None, eng.version if eng else None),
                             media_type="text/plain; version=0.0.4")
