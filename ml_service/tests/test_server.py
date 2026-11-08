import os
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ML_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("MODEL_DIR", str(ML_ROOT / "models" / "LATEST"))

from kubeai_ml import server  # noqa: E402
from kubeai_ml.server import app  # noqa: E402

MIB = 2 ** 20
LIMIT = 900 * MIB
T0 = 1_700_000_000.0

RESPONSE_KEYS = {"pod", "namespace", "model_version", "p_oom", "is_anomaly", "threshold",
                 "horizon_s", "usage_fraction", "trend_bytes_per_s", "time_to_limit_s",
                 "forecast", "reasons"}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def body(values, pod="web-0", limit=LIMIT, step=1.0, **extra):
    return {"pod": pod, "namespace": "default", "limit_bytes": limit,
            "samples": [{"ts": T0 + i * step, "working_set_bytes": v} for i, v in enumerate(values)],
            **extra}


def leak_ramp():
    # 50% -> 95% of the limit over the last 5 minutes, sampled every 15 s.
    return [LIMIT * (0.5 + 0.45 * i / 20) for i in range(21)]


def flat_low():
    return [LIMIT * 0.2] * 21


def test_health_and_ready(client):
    assert client.get("/healthz").json() == {"status": "ok"}
    r = client.get("/readyz")
    assert r.status_code == 200 and r.json()["ready"] is True and r.json()["model_version"]


def test_model_manifest(client):
    m = client.get("/v1/model").json()
    for key in ("model_version", "features", "horizon_s", "threshold", "metrics",
                "trained_at", "data_sha256", "label_definition"):
        assert m[key] is not None
    assert "test" in m["metrics"]


def test_predict_happy_path(client):
    r = client.post("/v1/predict", json=body(leak_ramp(), step=15, restart_count=2,
                                             seconds_since_restart=400))
    assert r.status_code == 200
    out = r.json()
    assert set(out) == RESPONSE_KEYS
    assert out["pod"] == "web-0" and out["namespace"] == "default"
    assert 0.0 <= out["p_oom"] <= 1.0
    assert out["usage_fraction"] == pytest.approx(0.95)
    assert out["trend_bytes_per_s"] > 0 and out["time_to_limit_s"] > 0
    assert [p["in_s"] for p in out["forecast"]] == [15, 30, 60, 120]
    assert any("of limit" in reason for reason in out["reasons"])


def test_leak_ramp_scores_higher_than_flat_low(client):
    ramp = client.post("/v1/predict", json=body(leak_ramp(), step=15)).json()
    flat = client.post("/v1/predict", json=body(flat_low(), step=15)).json()
    assert ramp["p_oom"] > flat["p_oom"]
    assert ramp["is_anomaly"] and not flat["is_anomaly"]


def test_no_limit_falls_back(client):
    out = client.post("/v1/predict", json=body(leak_ramp(), step=15, limit=None)).json()
    assert out["usage_fraction"] is None and out["time_to_limit_s"] is None
    assert out["p_oom"] == 0.0 and out["is_anomaly"] is False
    assert out["trend_bytes_per_s"] > 0
    assert any("no memory limit" in reason for reason in out["reasons"])


@pytest.mark.parametrize("bad", [
    {"samples": [{"ts": T0, "working_set_bytes": 1.0}]},                 # < 2 samples
    {"samples": [{"ts": T0, "working_set_bytes": -1.0},
                 {"ts": T0 + 1, "working_set_bytes": 1.0}]},             # negative bytes
    {"limit_bytes": 0},                                                   # non-positive limit
    {"pod": ""},                                                          # empty pod
    {"restart_count": -1},
])
def test_invalid_input_is_422(client, bad):
    payload = body(flat_low())
    payload.update(bad)
    assert client.post("/v1/predict", json=payload).status_code == 422


def test_missing_pod_is_422(client):
    payload = body(flat_low())
    del payload["pod"]
    assert client.post("/v1/predict", json=payload).status_code == 422


def test_batch(client):
    items = [body(leak_ramp(), pod="leaky", step=15), body(flat_low(), pod="idle", step=15)]
    r = client.post("/v1/predict/batch", json={"items": items})
    assert r.status_code == 200
    results = r.json()["results"]
    assert [x["pod"] for x in results] == ["leaky", "idle"]
    assert results[0]["p_oom"] > results[1]["p_oom"]
    assert client.post("/v1/predict/batch", json={"items": []}).status_code == 422


def test_metrics_endpoint(client):
    client.post("/v1/predict", json=body(flat_low()))
    text = client.get("/metrics").text
    assert 'kubeai_requests_total{endpoint="/v1/predict",status="200"}' in text
    assert "kubeai_predictions_total" in text and "kubeai_anomalies_total" in text
    assert "kubeai_request_latency_seconds_count" in text


def test_predict_is_fast(client):
    """Server-side work for a 600-sample request stays well under 20 ms.

    Timed without the HTTP test client, whose own overhead is ~5 ms per call.
    """
    payload = body([LIMIT * 0.5 + i * MIB for i in range(600)])
    model = server.state["model"]
    start = time.perf_counter()
    for _ in range(20):
        server.run(model, server.PredictRequest.model_validate(payload))
    assert (time.perf_counter() - start) / 20 < 0.02


@pytest.mark.parametrize("frac", [0.5, 0.66, 0.7, 0.8, 0.9, 0.95, 0.99])
def test_flat_high_usage_is_not_flagged(client, frac):
    """Steady pods are not alerted on: is_anomaly needs memory heading for the limit."""
    out = client.post("/v1/predict", json=body([LIMIT * frac] * 40, step=15)).json()
    assert out["is_anomaly"] is False
    assert out["time_to_limit_s"] is None


def test_flat_at_limit_is_flagged(client):
    out = client.post("/v1/predict", json=body([LIMIT] * 40, step=15)).json()
    assert out["is_anomaly"] is True and out["time_to_limit_s"] == 0.0


def test_manifest_has_trend_gate(client):
    m = server.state["model"].manifest
    assert m["alert_rule"]["max_time_to_limit_s"] > 0
