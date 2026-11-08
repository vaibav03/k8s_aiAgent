"""Policy v2 (FAULT_AWARE / HPA_GATED): signal groups, HPA-style stabilisation and the fault gate.

No artefact needed: a stub capacity (10 safe rps per pod) and synthetic inputs."""
import math

import pytest

from kubeai_ml.scaling import states as S
from kubeai_ml.scaling.policy import Bounds, ScalerMemory, Stabilization, fault_aware_replicas

CPU = 0.026


class StubCap:
    def replicas_needed(self, demand_rps, cpu_per_req_s, max_replicas=50, risk=None):
        return max(1, min(max_replicas, math.ceil(demand_rps * (cpu_per_req_s / CPU) / 10.0 - 1e-9)))


ST = Stabilization(mode="fault_aware", fa_headroom=1.0, fa_cpu_target=0.70, fa_down_window_s=180.0)


def fa(current, *, state=S.HEALTHY, rps=20.0, util=0.5, ready=None, mem=None, ts=0.0, st=ST, cpu=CPU,
       base=CPU, **kw):
    return fault_aware_replicas(StubCap(), current, Bounds(1, 10), st, state=state, now_rps=rps,
                                cpu_per_req_s=cpu, baseline_cpu_per_req_s=base, ts=ts,
                                memory=mem if mem is not None else ScalerMemory(),
                                ready=current if ready is None else ready, cpu_util=util, **kw)


def test_traffic_signal_scales_up_at_once():
    d = fa(2, rps=55.0, util=0.7)                     # 55 rps / 10 per pod -> 6
    assert d.action == "scale_up" and d.replicas == 6 and d.signals["driver"] == "traffic"


def test_compute_signal_is_the_hpa_formula():
    d = fa(2, rps=10.0, util=1.4)                     # ceil(2 x 1.4 / 0.7) = 4
    assert d.replicas == 4 and d.signals["driver"] == "compute"
    assert fa(2, rps=10.0, util=0.75).replicas == 2  # inside the 10 % tolerance band


def test_quality_signal_only_when_pods_are_busy():
    assert fa(3, rps=10.0, util=0.8, slo_breached=True).replicas == 4
    assert fa(3, rps=10.0, util=0.2, slo_breached=True).replicas == 3   # idle pods: not capacity


@pytest.mark.parametrize("state", [S.NOISY_POD, S.MEMORY_LEAK, S.DEPENDENCY_LATENCY, S.ERROR_FAULT, S.CRASHLOOP])
def test_fault_states_never_scale_on_their_symptoms(state):
    d = fa(3, state=state, rps=10.0, util=2.0, slo_breached=True, mem_frac=0.97)
    assert d.replicas == 3 and d.action == "do_not_scale"


def test_real_load_during_a_fault_still_counts_at_baseline_cost():
    d = fa(2, state=S.NOISY_POD, rps=45.0, util=2.0, cpu=3 * CPU)     # hog triples CPU/request
    assert d.replicas == 5                                             # sized at the baseline cost


def test_bad_deploy_stop_gap_only_while_slo_breached():
    assert fa(3, state=S.BAD_DEPLOY, rps=20.0, util=1.6, cpu=2 * CPU).replicas == 3
    d = fa(3, state=S.BAD_DEPLOY, rps=20.0, util=1.6, cpu=2 * CPU, slo_breached=True)
    assert d.replicas == 4 and "stop-gap" in d.reason


def test_cost_per_request_rise_vetoes_compute():
    d = fa(3, rps=20.0, util=1.6, cpu=2 * CPU)        # healthy label, but CPU/request doubled
    assert d.replicas == 4                            # traffic at the new cost: ceil(20 x 2 / 10)
    assert "compute_vetoed" in d.signals


def test_missing_pods_do_not_inflate_compute():
    """A pod is killed: 1 of 2 ready, per-ready-pod util doubles from 0.7 to 1.4 -> HPA keeps 2."""
    assert fa(2, ready=1, rps=10.0, util=1.4).replicas == 2


def test_scale_down_waits_for_the_window_then_follows_hpa():
    mem = ScalerMemory()
    assert fa(6, rps=55.0, util=0.7, mem=mem, ts=0.0).replicas == 6
    for t in range(15, 180, 15):
        assert fa(6, rps=15.0, util=0.25, mem=mem, ts=float(t)).replicas == 6
    d = fa(6, rps=15.0, util=0.25, mem=mem, ts=195.0)
    assert d.action == "scale_down" and d.replicas == 3    # max(traffic 2, ceil(6 x .25 / .7) = 3)


def test_hpa_gated_ignores_traffic_in_load_states():
    st = Stabilization(mode="hpa_gated", fa_cpu_target=0.70)
    assert fa(2, rps=55.0, util=0.7, st=st, mode="hpa_gated").replicas == 2
    assert fa(2, rps=55.0, util=1.4, st=st, mode="hpa_gated").replicas == 4
    assert fa(2, state=S.NOISY_POD, rps=10.0, util=2.0, st=st, mode="hpa_gated").replicas == 2


def test_forecast_is_advisory_unless_prescale_and_rising():
    q = {60: 60.0, 120: 60.0, 300: 60.0}
    d = fa(2, rps=15.0, util=0.5, forecast_q90=q, trend_ratio=2.0)
    assert d.replicas == 2 and "advisory" in d.reason
    st = Stabilization(mode="fault_aware", fa_headroom=1.0, fa_cpu_target=0.70, fa_fc_prescale=True)
    mem = ScalerMemory()
    fa(2, rps=15.0, util=0.5, forecast_q90=q, trend_ratio=2.0, st=st, mem=mem, ts=0.0)
    d = fa(2, rps=15.0, util=0.5, forecast_q90=q, trend_ratio=2.0, st=st, mem=mem, ts=15.0)
    assert d.replicas == 3                            # capped at max(need + 1, 1.5 x need)
    flat = fa(2, rps=15.0, util=0.5, forecast_q90=q, trend_ratio=1.0, st=st, mem=ScalerMemory(), ts=15.0)
    assert flat.replicas == 2


def test_engine_reports_policy_mode_and_signals():
    from kubeai_ml.scaling.bundle import LATEST
    if not LATEST.exists():
        pytest.skip("no shipped scaling bundle")
    from kubeai_ml.scaling.engine import DecisionEngine
    with pytest.raises(ValueError):
        DecisionEngine(oom_model_dir=None, policy_mode="nope")
    rows = [{"ts": 1e9 + 5 * i, "rps": 20.0, "p95": 0.05, "error_rate": 0.0, "cpu_cores_used": 0.55,
             "replicas_ready": 2} for i in range(120)]
    for mode in ("fault_aware", "hpa_gated", "predictive"):
        out = DecisionEngine(oom_model_dir=None, policy_mode=mode).decide(rows, None, 2, 1, 8)
        assert out["policy"]["mode"] == mode
        assert out["desired_replicas"] >= 2
    out = DecisionEngine(oom_model_dir=None, policy_mode="hpa_gated").decide(rows, None, 2, 1, 8)
    assert out["policy"]["signals"]["compute"] == 4           # 0.55 / (2 x 0.25) = 110 % -> ceil(2 x 1.1 / .7)
