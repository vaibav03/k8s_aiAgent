"""Happy-path checks for the state classifier feature path (synthetic, no dataset needed)."""
import numpy as np
import pandas as pd

from kubeai_ml.scaling import rules, states
from kubeai_ml.scaling.classifier import StateClassifier, capacity_features
from kubeai_ml.scaling.features import FEATURES, build_features, features_at
from kubeai_ml.scaling.labels import SafeCapacity


def _deploy(n: int = 80, step: float = 15.0, **over: float) -> pd.DataFrame:
    ts = 1_000_000 + np.arange(n) * step
    base = dict(rps=18.0, goodput_rps=18.0, inflight=0.0, p50=0.028, p95=0.03, p99=0.035,
                error_rate=0.0, cpu_cores_used=0.47, cpu_util_vs_request=0.63, throttle_ratio=0.04,
                mem_ws_max_pod_frac_of_limit=0.14, mem_growth_bytes_per_s=100.0, replicas_desired=3,
                replicas_ready=3, restarts_total=0, oomkills_total=0, crashloop_pods=0,
                hpa_desired=3, pods_not_ready=0)
    base.update(over)
    return pd.DataFrame({"ts": ts, **{k: np.full(n, v, dtype=float) for k, v in base.items()}})


def test_features_shape_and_cadence_independence():
    f = build_features(_deploy())
    assert list(f.columns) == FEATURES and len(f) == 80
    last = features_at(_deploy(step=5.0, n=240))
    assert abs(last["load_per_pod"] - 6.0) < 1e-9
    assert abs(last["cpr_ratio"] - 1.0) < 1e-6


def test_rules_flag_crashloop_and_leak_and_not_scale():
    cap = SafeCapacity({1: 17.0, 2: 17.0, 3: 13.0}, 13.0)
    crash = capacity_features(build_features(_deploy(crashloop_pods=1, pods_not_ready=1)), cap)
    assert rules.predict(crash)[-1] == states.CRASHLOOP
    leak = capacity_features(build_features(_deploy(mem_growth_bytes_per_s=700_000.0)), cap)
    assert rules.predict(leak)[-1] == states.MEMORY_LEAK
    assert not states.SCALING_HELPS[states.MEMORY_LEAK]


def test_rules_scale_up_on_saturation():
    cap = SafeCapacity({1: 17.0}, 13.0)
    hot = _deploy(replicas_ready=1, replicas_desired=1, hpa_desired=1, p95=10.0, rps=24.0,
                  goodput_rps=3.0, error_rate=0.9, throttle_ratio=1.0, cpu_cores_used=0.5)
    assert StateClassifier("rules", cap).predict(build_features(hot))[-1] == states.SCALE_UP
