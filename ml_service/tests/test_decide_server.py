"""POST /v1/decide, GET /v1/model (both bundles), /readyz and /metrics of the FastAPI service.

Uses the shipped bundles (models/LATEST + models/scaling/LATEST). The recorded-window test also
needs the loadlab dataset and is skipped without it.
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ML_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("MODEL_DIR", str(ML_ROOT / "models" / "LATEST"))

from kubeai_ml import server  # noqa: E402
from kubeai_ml.server import app  # noqa: E402
from kubeai_ml.scaling import states as S  # noqa: E402
from kubeai_ml.scaling.bundle import LATEST  # noqa: E402
from kubeai_ml.scaling.data import LAB_DATA  # noqa: E402

pytestmark = pytest.mark.skipif(not LATEST.exists(), reason="no shipped scaling bundle")

T0 = 1_700_000_000.0
ACTION_TYPES = {"scale", "restart_pod", "rollback", "page", "none"}
RESPONSE_KEYS = {"deployment", "namespace", "model_version", "policy_mode", "state", "state_probs", "confidence",
                 "scaling_helps", "current_replicas", "desired_replicas", "replicas_needed", "forecast",
                 "safe_rps_per_pod", "slo_risk", "cpu_per_req_s", "actions", "reasons", "telemetry_ok",
                 "inputs", "latency_ms"}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def series(n=180, rps=20.0, ready=4, step=5.0, **over):
    rows = []
    for i in range(n):
        r = {"ts": T0 + i * step, "rps": rps, "goodput_rps": rps, "inflight": 1.0, "p50": 0.03, "p95": 0.06,
             "p99": 0.09, "error_rate": 0.0, "cpu_cores_used": rps * 0.0265 + 0.01 * ready,
             "cpu_util_vs_request": (rps * 0.0265 + 0.01 * ready) / (ready * 0.25), "throttle_ratio": 0.0,
             "mem_ws_max_pod_frac_of_limit": 0.3, "mem_growth_bytes_per_s": 0.0, "replicas_desired": ready,
             "replicas_ready": ready, "restarts_total": 0, "oomkills_total": 0, "crashloop_pods": 0,
             "hpa_desired": ready, "pods_not_ready": 0}
        for k, v in over.items():
            r[k] = v(i) if callable(v) else v
        rows.append(r)
    return rows


def pods(n=180, names=("web-abc-0", "web-abc-1", "web-abc-2", "web-abc-3"), step=5.0):
    lim = 256 * 2 ** 20
    return [{"ts": T0 + i * step, "pod": p, "cpu": 0.13, "throttle": 0.0, "mem_ws": 0.3 * lim,
             "mem_frac_limit": 0.3, "rps": 5.0, "ready": 1} for i in range(n) for p in names]


def body(**over):
    b = {"deployment": "web", "namespace": "default", "series": series(), "pods": pods(),
         "current_replicas": 4, "min_replicas": 1, "max_replicas": 8}
    b.update(over)
    return b


def check_schema(out):
    assert set(out) == RESPONSE_KEYS
    assert out["state"] in S.STATES and set(out["state_probs"]) == set(S.STATES)
    assert abs(sum(out["state_probs"].values()) - 1) < 1e-3
    assert 0 <= out["confidence"] <= 1
    assert set(out["forecast"]) == {"60", "120", "300"}
    assert isinstance(out["desired_replicas"], int)
    assert out["actions"] and all(a["type"] in ACTION_TYPES for a in out["actions"])
    assert all(isinstance(r, str) for r in out["reasons"])
    assert isinstance(out["telemetry_ok"], bool)


# ------------------------------------------------------------------ bundles / readiness

def test_ready_reports_both_bundles(client):
    r = client.get("/readyz")
    assert r.status_code == 200
    j = r.json()
    assert j["ready"] is True and j["model_version"] and j["scaling_model_version"].startswith("scaling-")


def test_model_lists_both_bundles_and_keeps_oom_top_level(client):
    m = client.get("/v1/model").json()
    assert m["model_version"] and m["threshold"] is not None          # backwards compatible (OOM)
    b = m["bundles"]
    assert b["oom"]["loaded"] and b["oom"]["endpoint"] == "/v1/predict"
    assert b["scaling"]["loaded"] and b["scaling"]["endpoint"] == "/v1/decide"
    assert b["scaling"]["model_version"] == client.get("/readyz").json()["scaling_model_version"]
    assert set(b["scaling"]["states"]) == set(S.STATES)
    assert b["scaling"]["oom_input"] is True
    assert "headline_metrics_with_baselines" in b["scaling"]["manifest"]


def test_readiness_fails_without_scaling_bundle(tmp_path, monkeypatch):
    monkeypatch.setenv("SCALING_MODEL_DIR", str(tmp_path / "nope"))
    try:
        with TestClient(app) as c:
            r = c.get("/readyz")
            assert r.status_code == 503
            assert r.json()["ready"] is False and "scaling bundle" in r.json()["error"]
            assert r.json()["model_version"]                       # the OOM model itself loaded
            d = c.post("/v1/decide", json=body())
            assert d.status_code == 503
            m = c.get("/v1/model").json()
            assert m["bundles"]["scaling"]["loaded"] is False and m["bundles"]["oom"]["loaded"] is True
            assert c.post("/v1/predict", json={"pod": "p", "samples": [{"ts": 1, "working_set_bytes": 1},
                                                                       {"ts": 2, "working_set_bytes": 2}]}
                          ).status_code == 200                      # OOM endpoint unaffected
    finally:
        monkeypatch.delenv("SCALING_MODEL_DIR")
        with TestClient(app):                                       # reload the real bundles
            pass
    assert server.state["engine"] is not None


def test_policy_mode_env(monkeypatch):
    import dataclasses
    from kubeai_ml.scaling.engine import DecisionEngine
    monkeypatch.setenv("POLICY_MODE", "no-such-mode")
    try:
        with TestClient(app) as c:
            assert c.get("/readyz").status_code == 503              # bad mode -> not ready, never a silent default
    finally:
        monkeypatch.delenv("POLICY_MODE")
        with TestClient(app):
            pass
    if "policy_mode" in {f.name for f in dataclasses.fields(DecisionEngine)}:
        from kubeai_ml.scaling.policy import MODES
        monkeypatch.setenv("POLICY_MODE", MODES[-1])
        try:
            with TestClient(app) as c:
                assert c.get("/readyz").json()["policy_mode"] == MODES[-1]
                r = c.post("/v1/decide", json=body(deployment="mode"))
                assert r.status_code == 200 and r.json()["policy_mode"] == MODES[-1]
        finally:
            monkeypatch.delenv("POLICY_MODE")
            with TestClient(app):
                pass


# ------------------------------------------------------------------ decide

def test_decide_steady_holds_and_has_schema(client):
    server.state["engine"].reset("default/web")
    r = client.post("/v1/decide", json=body())
    assert r.status_code == 200, r.text
    out = r.json()
    check_schema(out)
    assert out["deployment"] == "web" and out["namespace"] == "default"
    assert out["current_replicas"] == 4
    assert out["telemetry_ok"] is True
    assert out["state"] not in S.FAULT_STATES
    assert not any(a["type"] in ("restart_pod", "rollback") for a in out["actions"])
    assert out["inputs"]["deploy_rows"] == 180 and out["inputs"]["pod_rows"] == 720


def test_decide_minimal_required_fields_only(client):
    req = [{k: r[k] for k in ("ts", "rps", "p95", "error_rate", "cpu_cores_used", "replicas_ready")}
           for r in series()]
    r = client.post("/v1/decide", json={"deployment": "mini", "series": req})
    assert r.status_code == 200, r.text
    out = r.json()
    check_schema(out)
    assert any("no pod_obs" in x for x in out["reasons"])
    assert any("missing optional inputs" in x for x in out["reasons"])


def test_decide_stale_telemetry_is_flagged_and_does_not_scale_down(client):
    rows = series()
    for r in rows[-12:]:
        r["rps"] = None                                             # Prometheus returned nothing for 60 s
    r = client.post("/v1/decide", json=body(deployment="stale", series=rows, now_ts=rows[-1]["ts"] + 30))
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["telemetry_ok"] is False
    assert out["desired_replicas"] >= 4
    assert any("insufficient telemetry" in x for x in out["reasons"])


def test_decide_revision_top_level_is_used(client):
    r = client.post("/v1/decide", json=body(deployment="rev", revision=3))
    assert r.status_code == 200
    assert not any("missing optional inputs" in x and "revision" in x for x in r.json()["reasons"])


@pytest.mark.parametrize("mutate", [
    lambda b: b.pop("series"),
    lambda b: b.update(series=[]),
    lambda b: b.update(min_replicas=5, max_replicas=2),
    lambda b: b.update(max_replicas=0),
    lambda b: b.update(current_replicas=-1),
    lambda b: b.update(unknown_field=1),
    lambda b: b.update(deployment=""),
    lambda b: b.update(now_ts=T0),                                   # older than the newest row
    lambda b: [r.pop("rps") for r in b["series"]],                   # required key missing
    lambda b: b["series"][5].update(error_rate=1.5),
    lambda b: b["series"][5].update(cpu_cores_used=-1),
    lambda b: b["pods"][0].pop("pod"),
])
def test_decide_invalid_input_is_422(client, mutate):
    b = body()
    mutate(b)
    assert client.post("/v1/decide", json=b).status_code == 422


def test_decide_non_finite_is_422(client):
    b = json.dumps(body()).replace('"p95": 0.06', '"p95": Infinity', 1)
    r = client.post("/v1/decide", content=b, headers={"content-type": "application/json"})
    assert r.status_code == 422


def test_decide_extra_dataset_columns_are_ignored(client):
    rows = series()
    for r in rows:
        r["mem_ws_bytes_total"] = 1e9
        r["cpu_cores_proc"] = 0.5
    assert client.post("/v1/decide", json=body(deployment="extra", series=rows)).status_code == 200


def test_decide_is_fast(client):
    b = body(deployment="fast")
    client.post("/v1/decide", json=b)
    t0 = time.perf_counter()
    for _ in range(5):
        assert client.post("/v1/decide", json=b).status_code == 200
    assert (time.perf_counter() - t0) / 5 < 1.0


def test_metrics_count_decisions(client):
    client.post("/v1/decide", json=body(deployment="metrics"))
    txt = client.get("/metrics").text
    assert 'kubeai_requests_total{endpoint="/v1/decide",status="200"}' in txt
    assert "kubeai_decide_latency_seconds_count" in txt
    assert "kubeai_decide_state_total{state=" in txt
    assert "kubeai_decide_actions_total{type=" in txt
    assert 'kubeai_scaling_model_info{model_version="scaling-' in txt
    count = [float(line.split()[-1]) for line in txt.splitlines()
             if line.startswith("kubeai_decide_latency_seconds_count")][0]
    assert count >= 1


# ------------------------------------------------------------------ recorded window (held-out run)

def _jsonable_rows(df):
    return [{k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in r.items()}
            for r in df.to_dict("records")]


@pytest.mark.skipif(not (LAB_DATA / "dataset_deploy.csv.gz").exists(), reason="no loadlab dataset")
def test_decide_recorded_leak_window_does_not_scale_up(client):
    import pandas as pd
    d = pd.read_csv(LAB_DATA / "dataset_deploy.csv.gz")
    p = pd.read_csv(LAB_DATA / "dataset_pods.csv.gz")
    sc = "d-leak-medium-r2"                                          # held out from training
    g = d[(d["scenario"] == sc) & (d["seconds_since_start"] <= 300)].sort_values("ts")
    pg = p[(p["scenario"] == sc) & (p["seconds_since_start"] <= 300)]
    if not len(g):
        pytest.skip("scenario not in dataset")
    cur = int(g["replicas_desired"].iloc[-1])
    pcols = ["ts", "pod", "cpu", "throttle", "mem_ws", "mem_frac_limit", "rps", "ready"]
    server.state["engine"].reset("lab/loadlab")
    r = client.post("/v1/decide", json={
        "deployment": "loadlab", "namespace": "lab", "series": _jsonable_rows(g),   # extra columns ignored
        "pods": _jsonable_rows(pg[pcols]), "current_replicas": cur, "min_replicas": 1, "max_replicas": 6})
    assert r.status_code == 200, r.text
    out = r.json()
    check_schema(out)
    assert out["state"] == S.MEMORY_LEAK
    assert out["desired_replicas"] <= cur
    assert not any(a["type"] == "scale" and a["params"]["to"] > cur for a in out["actions"])


def test_decide_accepts_agent_rollout_ts_and_rejects_non_finite(client):
    r = client.post("/v1/decide", json=body(deployment="own-rollout", agent_rollout_ts=[1.0e9, 1.1e9]))
    assert r.status_code == 200, r.text
    check_schema(r.json())
    bad = client.post("/v1/decide", content=json.dumps(body(deployment="own-rollout", agent_rollout_ts=["nan"])),
                      headers={"content-type": "application/json"})
    assert bad.status_code == 422
