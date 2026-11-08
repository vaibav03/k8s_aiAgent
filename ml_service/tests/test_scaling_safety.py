"""Engine safety interlocks (review 3) on synthetic telemetry: no recorded data needed, only the
shipped bundle (models/scaling/LATEST)."""
from __future__ import annotations

import numpy as np
import pytest

from kubeai_ml.scaling import states as S
from kubeai_ml.scaling.bundle import LATEST

pytestmark = pytest.mark.skipif(not LATEST.exists(), reason="no shipped scaling bundle")

T0 = 1_700_000_000.0
CPU_REQ = 0.0265


@pytest.fixture()
def engine():
    from kubeai_ml.scaling.engine import DecisionEngine
    return DecisionEngine()


def deploy_rows(n: int = 180, rps: float = 20.0, ready: int = 4, cpr=CPU_REQ, step: float = 5.0, **over):
    """``n`` scrapes of a steady deployment; ``over`` maps field -> value or callable(i)."""
    rows = []
    for i in range(n):
        r = {"ts": T0 + i * step, "rps": rps, "goodput_rps": rps, "inflight": 1.0, "p50": 0.03, "p95": 0.06,
             "p99": 0.09, "error_rate": 0.0, "cpu_cores_used": rps * (cpr(i) if callable(cpr) else cpr) + 0.01 * ready,
             "throttle_ratio": 0.0, "mem_ws_max_pod_frac_of_limit": 0.3, "mem_growth_bytes_per_s": 0.0,
             "replicas_desired": ready, "replicas_ready": ready, "restarts_total": 0, "oomkills_total": 0,
             "crashloop_pods": 0, "hpa_desired": ready, "pods_not_ready": 0}
        for k, v in over.items():
            r[k] = v(i) if callable(v) else v
        r["cpu_util_vs_request"] = r["cpu_cores_used"] / (max(r["replicas_ready"], 1) * 0.25)
        rows.append(r)
    return rows


def pod_rows(n: int = 180, names=("web-abc-0", "web-abc-1", "web-abc-2"), step: float = 5.0, mem=None):
    """Per-pod rows; ``mem(name, i)`` -> working-set bytes."""
    lim = 256 * 2**20
    out = []
    for i in range(n):
        for p in names:
            ws = mem(p, i) if mem else 0.3 * lim
            out.append({"ts": T0 + i * step, "pod": p, "cpu": 0.13, "throttle": 0.0, "mem_ws": ws,
                        "mem_frac_limit": ws / lim, "rps": 20.0 / len(names), "ready": 1})
    return out


def leak(i: int) -> float:
    return 0.3 * 256 * 2**20 + max(0, i - 60) * 5 * 400e3       # +400 KB/s after 5 min


def _types(out):
    return [a["type"] for a in out["actions"]]


def test_leak_restarts_the_growing_pod_not_the_first(engine):
    names = ("web-abc-0", "web-abc-1", "web-abc-2")
    dep = deploy_rows(ready=3, mem_growth_bytes_per_s=lambda i: 400e3 if i > 72 else 0.0,
                      mem_ws_max_pod_frac_of_limit=lambda i: leak(i) / (256 * 2**20))
    pods = pod_rows(names=names, mem=lambda p, i: leak(i) if p == "web-abc-2" else 0.3 * 256 * 2**20)
    out = engine.decide(dep, pods, 3, 1, 6)
    assert out["state"] == S.MEMORY_LEAK
    restarts = [a for a in out["actions"] if a["type"] == "restart_pod"]
    assert restarts and restarts[0]["target"] == "web-abc-2"
    # interlock: the same restart is not repeated 15 s later
    later = [dict(r, ts=r["ts"] + 15) for r in dep]
    pods2 = [dict(r, ts=r["ts"] + 15) for r in pods]
    out2 = engine.decide(later, pods2, 3, 1, 6)
    assert "restart_pod" not in _types(out2)
    assert any("cooldown" in r for r in out2["reasons"])


def test_single_replica_leak_uses_surge_restart(engine):
    dep = deploy_rows(ready=1, rps=8.0, mem_growth_bytes_per_s=lambda i: 400e3 if i > 72 else 0.0,
                      mem_ws_max_pod_frac_of_limit=lambda i: leak(i) / (256 * 2**20))
    pods = pod_rows(names=("web-abc-0",), mem=lambda p, i: leak(i))
    out = engine.decide(dep, pods, 1, 1, 4)
    assert out["state"] == S.MEMORY_LEAK
    r = [a for a in out["actions"] if a["type"] == "restart_pod"]
    assert r and r[0]["params"]["mode"] == "rollout_restart" and r[0]["params"]["max_unavailable"] == 0


def test_metrics_dropout_does_not_scale_down(engine):
    """50 rps on 7 pods, then rps/p95/cpu missing for 4 min: hold, and say why."""
    dep = deploy_rows(n=180, rps=50.0, ready=7)
    for r in dep[132:]:
        for k in ("rps", "goodput_rps", "p95", "error_rate", "cpu_cores_used"):
            r[k] = np.nan
    cur = 7
    for k in range(16):                                    # 4 min of decisions during the dropout
        rows = dep[: 133 + 3 * k]
        out = engine.decide(rows, None, cur, 1, 10, now_ts=rows[-1]["ts"])
        assert out["desired_replicas"] >= cur
        cur = out["desired_replicas"]
    assert out["telemetry_ok"] is False
    assert any("insufficient telemetry" in r for r in out["reasons"])


def test_stale_now_ts_is_reported(engine):
    dep = deploy_rows()
    out = engine.decide(dep, None, 4, 1, 8, now_ts=dep[-1]["ts"] + 120)
    assert out["telemetry_ok"] is False and out["desired_replicas"] >= 4


def test_bounds_validated(engine):
    with pytest.raises(ValueError, match="min_replicas"):
        engine.decide(deploy_rows(), None, 2, 3, 2)


@pytest.mark.parametrize("mode,want", [("predictive", 1), ("hpa_gated", 2)])
def test_flat_traffic_does_not_follow_the_wide_forecast(mode, want):
    """Flat 10 rps on 1 pod for 15 min: the p90 at +120 s is ~3x, but nothing measured backs it.
    PREDICTIVE keeps 1 pod (learned capacity); HPA_GATED (default) follows the CPU-70 % formula:
    0.275 cores on a 0.25 request = 110 % -> ceil(1.1 / 0.7) = 2, never the forecast's 3+."""
    from kubeai_ml.scaling.engine import DecisionEngine
    dep = deploy_rows(n=180, rps=10.0, ready=1)
    out = DecisionEngine(oom_model_dir=None, policy_mode=mode).decide(dep, None, 1, 1, 8)
    assert out["forecast"][120]["p90"] > 15
    assert out["desired_replicas"] == want


def test_bad_deploy_without_rollout_evidence_pages_not_rollback(engine):
    """CPU per request doubles at constant 20 rps on 4 pods, no pod_obs, SLO fine."""
    dep = deploy_rows(ready=4, cpr=lambda i: CPU_REQ * (2.0 if i >= 120 else 1.0))
    out = engine.decide(dep, None, 4, 1, 8)
    assert "rollback" not in _types(out)
    assert out["state"] == S.BAD_DEPLOY
    assert out["desired_replicas"] <= 4                    # SLO fine: no stop-gap scale-up
    assert "page" in _types(out)


def test_bad_deploy_with_revision_change_rolls_back(engine):
    dep = deploy_rows(ready=4, cpr=lambda i: CPU_REQ * (2.0 if i >= 120 else 1.0),
                      revision=lambda i: 7 if i >= 115 else 6)
    out = engine.decide(dep, None, 4, 1, 8)
    assert out["state"] == S.BAD_DEPLOY and "rollback" in _types(out)


def test_crashloop_without_rollout_pages_only(engine):
    dep = deploy_rows(ready=3, crashloop_pods=lambda i: 1 if i >= 160 else 0,
                      replicas_ready=lambda i: 2 if i >= 160 else 3, pods_not_ready=lambda i: 1 if i >= 160 else 0,
                      restarts_total=lambda i: max(0, (i - 150) // 4))
    pods = pod_rows()
    out = engine.decide(dep, pods, 3, 1, 6)
    assert out["state"] == S.CRASHLOOP
    assert "rollback" not in _types(out) and "page" in _types(out)


def test_total_outage_pages_first(engine):
    dep = deploy_rows(ready=3, replicas_ready=lambda i: 0 if i >= 170 else 3,
                      pods_not_ready=lambda i: 3 if i >= 170 else 0, error_rate=lambda i: 1.0 if i >= 170 else 0.0,
                      goodput_rps=lambda i: 0.0 if i >= 170 else 20.0)
    out = engine.decide(dep, None, 3, 1, 6)
    assert out["actions"][0]["type"] == "page" and out["actions"][0]["params"]["severity"] == "critical"
    assert out["desired_replicas"] >= 3


def test_restart_without_pod_name_pages(engine):
    from kubeai_ml.scaling.policy import ScalerMemory
    act, _ = engine._restart(None, "hot pod", mem=ScalerMemory(), ready=3, current=3, now=T0, deployment="web")
    assert act is not None and act["type"] == "page"


# --------------------------------------------------------------------- review fixes (A/B review)
_LIM = 256 * 2**20


def _rollout_pods(n=180, change=118, cpr_jump=120):
    old = ("web-aaa11-x0001", "web-aaa11-x0002", "web-aaa11-x0003", "web-aaa11-x0004")
    new = ("web-bbb22-y0001", "web-bbb22-y0002", "web-bbb22-y0003", "web-bbb22-y0004")
    return [{"ts": T0 + i * 5, "pod": p, "cpu": 0.13 * (2 if i >= cpr_jump else 1), "throttle": 0.0,
             "mem_ws": 0.3 * _LIM, "mem_frac_limit": 0.3, "rps": 5.0, "ready": 1}
            for i in range(n) for p in (old if i < change else new)]


def test_agents_own_rollout_is_not_rollout_evidence():
    """A new ReplicaSet that appears right after the caller's own undo / rollout restart must not
    re-arm `rollback` (undo ping-pong): with agent_rollout_ts the engine pages instead."""
    from kubeai_ml.scaling.engine import DecisionEngine
    dep = deploy_rows(ready=4, cpr=lambda i: CPU_REQ * (2.0 if i >= 120 else 1.0))
    out = DecisionEngine().decide(dep, _rollout_pods(), 4, 1, 8)
    assert out["state"] == S.BAD_DEPLOY and "rollback" in _types(out)
    own = DecisionEngine().decide(dep, _rollout_pods(), 4, 1, 8, agent_rollout_ts=[T0 + 117 * 5])
    assert "rollback" not in _types(own) and "page" in _types(own)
    # an agent rollout long before the new ReplicaSet does not hide it
    old = DecisionEngine().decide(dep, _rollout_pods(), 4, 1, 8, agent_rollout_ts=[T0 + 10 * 5])
    assert "rollback" in _types(old)


def _leak_then_delete(n, growth_until):
    """Pod web-abc-2 leaks 1 MB/s from i=90 and is deleted at i=150 (replacement web-abc-9).
    growth_until: last row index where the deployment-level mem_growth still shows the slope
    (150 = v2 query, the deleted pod drops out at once; 162 = v1 query, ~60 s stale range)."""
    def leak_ws(i):
        return 0.3 * _LIM + max(0, i - 90) * 5 * 1e6
    dep = deploy_rows(n=n, ready=3, mem_growth_bytes_per_s=lambda i: 1e6 if 102 < i < growth_until else 0.0,
                      mem_ws_max_pod_frac_of_limit=lambda i: (leak_ws(i) if i < 150 else 0.3 * _LIM) / _LIM)
    pods = [r for r in pod_rows(n=n, mem=lambda p, i: leak_ws(i) if p == "web-abc-2" else 0.3 * _LIM)
            if not (r["pod"] == "web-abc-2" and r["ts"] >= T0 + 150 * 5)]
    pods += [dict(ts=T0 + i * 5, pod="web-abc-9", cpu=0.13, throttle=0.0, mem_ws=0.2 * _LIM,
                  mem_frac_limit=0.2, rps=6.6, ready=1) for i in range(150, n)]
    return dep, pods


@pytest.mark.parametrize("growth_until,max_latched", [(150, 2), (162, 6)])
def test_leak_state_clears_once_the_deleted_pods_growth_series_ends(growth_until, max_latched):
    """v2 mem_growth (live targets only): after the restart decision the state leaves memory_leak
    within 2 decisions (30 s). The v1 artefact (deleted pod's slope kept ~60 s) latches longer."""
    from kubeai_ml.scaling.engine import DecisionEngine
    e = DecisionEngine()
    states = []
    for n in range(150, 172, 3):                      # one decision per 15 s, first one at the restart
        dep, pods = _leak_then_delete(n, growth_until)
        states.append(e.decide(dep, pods, 3, 1, 6, key="leak")["state"])
    assert states[0] == S.MEMORY_LEAK
    latched = next(i for i, s in enumerate(states[1:], 1) if s != S.MEMORY_LEAK) - 1
    assert latched <= max_latched
    if growth_until == 150:
        assert latched <= 2 and states[3] != S.MEMORY_LEAK
    else:
        assert latched > 2                              # the artefact the v2 query removes
