"""DecisionEngine on recorded loadlab windows (all four runs are held out from training).

Uses the shipped bundle (models/scaling/LATEST) and the loadlab dataset; skipped if absent.
"""
from __future__ import annotations

import pandas as pd
import pytest

from kubeai_ml.scaling import states as S
from kubeai_ml.scaling.bundle import LATEST
from kubeai_ml.scaling.data import LAB_DATA
from kubeai_ml.scaling.split import HOLDOUT_SCENARIOS

pytestmark = pytest.mark.skipif(not LATEST.exists() or not (LAB_DATA / "dataset_deploy.csv.gz").exists(),
                                reason="no shipped scaling bundle or no loadlab dataset")

ACTION_TYPES = {"scale", "restart_pod", "rollback", "page", "none"}


@pytest.fixture(scope="module")
def engine():
    from kubeai_ml.scaling.engine import DecisionEngine
    return DecisionEngine()


@pytest.fixture(scope="module")
def data() -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_csv(LAB_DATA / "dataset_deploy.csv.gz"), pd.read_csv(LAB_DATA / "dataset_pods.csv.gz")


def _window(data, scenario: str, at_s: float) -> tuple[list[dict], list[dict], int]:
    """Deployment + pod rows up to ``at_s`` seconds into the run, as an agent would scrape them."""
    d, p = data
    g = d[(d["scenario"] == scenario) & (d["seconds_since_start"] <= at_s)].sort_values("ts")
    pg = p[(p["scenario"] == scenario) & (p["seconds_since_start"] <= at_s)]
    cols = ["ts", "rps", "goodput_rps", "inflight", "p50", "p95", "p99", "error_rate", "cpu_cores_used",
            "cpu_util_vs_request", "throttle_ratio", "mem_ws_max_pod_frac_of_limit", "mem_growth_bytes_per_s",
            "replicas_desired", "replicas_ready", "restarts_total", "oomkills_total", "crashloop_pods",
            "hpa_desired", "pods_not_ready"]
    pcols = ["ts", "pod", "cpu", "throttle", "mem_ws", "mem_frac_limit", "rps", "ready"]
    return g[cols].to_dict("records"), pg[pcols].to_dict("records"), int(g["replicas_desired"].iloc[-1])


def _decide(engine, data, scenario, at_s, current=None, max_r=6):
    assert scenario in HOLDOUT_SCENARIOS
    dep, pods, cur = _window(data, scenario, at_s)
    engine.reset()
    return engine.decide(dep, pods, current if current is not None else cur, 1, max_r), (current or cur)


def _check_schema(out: dict) -> None:
    for k in ("state", "state_probs", "confidence", "desired_replicas", "forecast", "safe_rps_per_pod",
              "slo_risk", "actions", "reasons"):
        assert k in out
    assert out["state"] in S.STATES and set(out["state_probs"]) == set(S.STATES)
    assert abs(sum(out["state_probs"].values()) - 1) < 1e-3
    assert 0 <= out["confidence"] <= 1 and 0 <= out["slo_risk"] <= 1
    assert set(out["forecast"]) == {60, 120, 300}
    assert all(v["p90"] >= v["p50"] >= 0 for v in out["forecast"].values())
    assert all(a["type"] in ACTION_TYPES and {"target", "params", "reason"} <= set(a) for a in out["actions"])


def _no_scale_up(out: dict, current: int) -> None:
    assert out["desired_replicas"] <= current
    assert not any(a["type"] == "scale" and a["params"]["to"] > current for a in out["actions"])


def test_leak_run_is_memory_leak_and_does_not_scale(engine, data):
    out, cur = _decide(engine, data, "d-leak-medium-r2", 300)      # leak started at 120 s
    _check_schema(out)
    assert out["state"] == S.MEMORY_LEAK
    _no_scale_up(out, cur)
    assert {"restart_pod", "page"} <= {a["type"] for a in out["actions"]}


def test_flash_crowd_scales_up(engine, data):
    out, cur = _decide(engine, data, "c-under-fixed1-flash-r2", 240)   # 27 rps onto 1 pod
    _check_schema(out)
    assert cur == 1
    assert out["state"] == S.SCALE_UP
    assert out["desired_replicas"] > cur
    assert any(a["type"] == "scale" and a["params"]["to"] > cur for a in out["actions"])


def test_over_provisioned_scales_down(engine, data):
    out, cur = _decide(engine, data, "c-over-fixed4-diurnal-r2", 300)   # 4 pods, ~4 rps
    _check_schema(out)
    assert cur == 4
    assert out["state"] == S.SCALE_DOWN
    assert out["desired_replicas"] < cur


def test_dependency_latency_does_not_scale(engine, data):
    out, cur = _decide(engine, data, "d-dep-latency-75p600-r2", 300)    # +600 ms on 75 % of calls
    _check_schema(out)
    assert out["state"] == S.DEPENDENCY_LATENCY
    _no_scale_up(out, cur)
    types = {a["type"] for a in out["actions"]}
    assert "page" in types
    assert any(a["type"] == "none" and a["params"].get("scaling") == "blocked" for a in out["actions"])


def test_minimal_input_without_pods(engine, data):
    """Only the required fields, no pod rows: still a valid decision, and the gaps are reported."""
    dep, _, cur = _window(data, "b-hpa-diurnal-sd30-s59-r2", 240)
    req = [{k: r[k] for k in ("ts", "rps", "p95", "error_rate", "cpu_cores_used", "replicas_ready")} for r in dep]
    engine.reset()
    out = engine.decide(req, None, cur, 1, 6)
    _check_schema(out)
    assert any("no pod_obs" in r for r in out["reasons"])
    assert any("missing optional inputs" in r for r in out["reasons"])


def test_missing_required_field_raises(engine):
    with pytest.raises(ValueError, match="required"):
        engine.decide([{"ts": 1.0, "rps": 1.0}], None, 1, 1, 4)


def test_bundle_manifest_records_data_hash_and_split():
    import json
    from kubeai_ml.scaling.bundle import resolve
    m = json.loads((resolve() / "manifest.json").read_text())
    assert len(m["data"]["sha256"]["dataset_deploy.csv.gz"]) == 64
    assert set(m["split"]["holdout_runs"]) == set(HOLDOUT_SCENARIOS)
    assert not set(m["split"]["holdout_runs"]) & set(m["split"]["dev_runs"])
