"""Happy-path checks for kubeai_ml.scaling forecast / capacity / policy."""
import numpy as np

from kubeai_ml.scaling import forecast as F
from kubeai_ml.scaling.capacity import CapacityModel
from kubeai_ml.scaling.policy import Bounds, ScalerMemory, Stabilization, desired_replicas

# Coefficients of the round-1 fit (models/scaling/capacity.json) so the test needs no artefact.
CAP = CapacityModel(coef=[1.482902, 9.780552, 2.4985, 0.040413], intercept=-9.872411)
CPU = 0.0265


def test_capacity_scales_with_cpu_cost_and_replicas():
    good = CAP.safe_rps_per_pod(CPU, 1)
    bad = CAP.safe_rps_per_pod(3 * CPU, 1)
    assert 12 < good < 18
    assert abs(bad - good / 3) < 0.01
    assert CAP.safe_rps_per_pod(CPU, 3) < good                 # knee moves with replicas
    assert CAP.safe_rps_per_pod(CPU, 1, risk=0.3) > good       # more risk, more rps/pod
    assert CAP.replicas_needed(40, CPU) == 4


def _decide(mem, current, q90, now, state="healthy", ts=0.0, cpu=CPU, **kw):
    return desired_replicas(q90, CAP, current, Bounds(1, 10), Stabilization(), state=state,
                            now_rps=now, cpu_per_req_s=cpu, baseline_cpu_per_req_s=CPU,
                            ts=ts, memory=mem, **kw)


def test_forecast_needs_measured_evidence():
    """A wide p90 on flat traffic adds nothing; a rising trend lets it in after 2 decisions, capped."""
    q90 = {60: 50.0, 120: 55.0, 300: 40.0}
    d = _decide(ScalerMemory(), 2, q90, now=20.0, trend_ratio=1.0, slo_risk=0.01, risk_budget=0.3)
    assert d.action == "hold" and d.replicas == 2
    mem = ScalerMemory()
    first = _decide(mem, 2, q90, now=20.0, trend_ratio=1.5, ts=0.0)
    assert first.replicas == 2                                  # must persist
    second = _decide(mem, 2, q90, now=20.0, trend_ratio=1.5, ts=15.0)
    need_now = CAP.replicas_needed(20.0, CPU)
    assert second.action == "scale_up"
    assert second.replicas == max(need_now + 1, -(-3 * need_now // 2))    # capped at +50 %


def test_measured_need_scales_at_once():
    d = _decide(ScalerMemory(), 1, {60: 50.0, 120: 50.0, 300: 50.0}, now=50.0)
    assert d.action == "scale_up" and d.replicas >= 4


def test_bad_deploy_scales_only_as_a_stop_gap():
    q = {60: 40.0, 120: 40.0, 300: 40.0}
    calm = _decide(ScalerMemory(), 2, q, now=12.0, state="bad_deploy", cpu=3 * CPU, slo_risk=0.02, risk_budget=0.3)
    assert calm.replicas == 2 and calm.action == "do_not_scale"
    hot = _decide(ScalerMemory(), 2, q, now=12.0, state="bad_deploy", cpu=3 * CPU, slo_breached=True)
    assert hot.action == "scale_up" and hot.replicas == CAP.replicas_needed(12.0, 3 * CPU)   # measured, not forecast


def test_scale_down_frozen_without_telemetry():
    mem = ScalerMemory()
    for k in range(30):
        d = _decide(mem, 6, {60: 5.0, 120: 5.0, 300: 5.0}, now=5.0, ts=15.0 * k, allow_down=False)
        assert d.replicas == 6


def test_min_above_max_raises():
    import pytest
    with pytest.raises(ValueError):
        desired_replicas({60: 1.0}, CAP, 2, Bounds(3, 2), Stabilization(), state="healthy", now_rps=1.0,
                         cpu_per_req_s=CPU, ts=0.0, memory=ScalerMemory())


def test_scale_down_slow():
    mem = ScalerMemory()
    reps = []
    cur = 6
    for k in range(30):                               # 7.5 min of low steady load
        d = _decide(mem, cur, {60: 10.0, 120: 10.0, 300: 10.0}, now=10.0, ts=15.0 * k)
        cur = d.replicas
        reps.append(cur)
    assert reps[11] == 6                               # no scale-in before a full 180 s window
    assert all(a - b <= 1 for a, b in zip(reps, reps[1:]))   # one pod at a time
    assert reps[-1] == 1


def test_gate_dependency_latency_does_not_scale():
    d = _decide(ScalerMemory(), 3, {60: 18.0, 120: 18.0, 300: 18.0}, now=18.0, state="dependency_latency")
    assert d.replicas == 3 and d.action == "do_not_scale" and d.remediation == "page_human"


def test_gate_noisy_pod_uses_baseline_cpu():
    # a CPU hog doubles apparent cpu per request; the policy must not add pods for it
    d = _decide(ScalerMemory(), 3, {60: 18.0, 120: 18.0, 300: 18.0}, now=18.0, state="noisy_pod", cpu=2 * CPU)
    assert d.replicas == 3 and d.remediation == "restart_pod"


def test_forecast_features_and_resample():
    ts = np.arange(100, 305, 5.0)
    h = F.resample_history(ts, np.full(len(ts), 10.0), now=300.0)
    assert h.shape == (F.HIST_STEPS + 1,) and np.isnan(h[0]) and h[-1] == 10.0
    X = F.make_features(F.to_z(h)[None, :])
    assert X.shape == (1, len(F.FEATURE_NAMES))
    p = F.PersistenceForecaster()
    Z = F.to_z(np.tile(h, (5, 1)))
    p.fit(Z, {hz: Z[:, -1] for hz in F.HORIZONS_S})
    out = p.predict(Z)
    assert np.allclose(out[(60, 0.5)], 10.0)
