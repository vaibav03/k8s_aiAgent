"""Rule-based SRE runbook: what a good on-call engineer checks, in order.

Root causes that scaling cannot fix are checked first (a page that says "scale up" during a
crashloop is the expensive mistake), then load. Thresholds are physical (SLO, limits, leak
rates of tens of KB/s vs a 256 Mi limit) and are documented next to each gate; they are not
fitted to the evaluation folds.

``fault_gate`` returns the fault state or None; ``load_state`` returns healthy / scale_up /
scale_down from the capacity features (see ``classifier.capacity_features``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import states as S

# error_fault also requires the *current* error rate >= this. Added after the round-2 forward
# test showed 22 false pages from the lingering 1 m mean -- one of those runs
# (d-error10-hpa-diurnal-r2) is in today's held-out set, so the held-out error_fault score is
# post-hoc; train_all reports the "blind" score with this set to 0 (review 3).
ERROR_RATE_FLOOR = 0.005


def _v(df: pd.DataFrame, col: str, fill: float = 0.0) -> np.ndarray:
    return df[col].fillna(fill).to_numpy(float) if col in df else np.full(len(df), fill)


def fault_gate(f: pd.DataFrame) -> np.ndarray:
    """Per-row fault state (object array, None where no fault rule fires)."""
    n = len(f)
    out = np.full(n, None, dtype=object)
    have_pods = f["secs_since_rollout"].notna().to_numpy() if "secs_since_rollout" in f else np.zeros(n, bool)
    recent_rollout = _v(f, "secs_since_rollout", 1e9) < 900
    p95 = _v(f, "p95_slo_1m")
    overload = (_v(f, "p95_slo") >= 20) | (_v(f, "throttle_1m") >= 0.9)   # deadline-bound queues

    crash = (_v(f, "crashloop_pods") >= 1) | ((_v(f, "pods_not_ready") >= 1) & (_v(f, "restarts_delta_3m") >= 2))
    # Pod lost: ready dropped while the desired count did not (deficit), no crashloop.
    restart = (_v(f, "ready_drop_1m") >= 1) & (_v(f, "ready_deficit") >= 1)
    # Bad deploy: every pod costs >= 1.8x more CPU per request after a rollout. Without pod
    # names the rollout is unknown: require the jump with no overload (which inflates cpu/req).
    cpr_jump = _v(f, "cpr_ratio", 1.0) >= 1.8
    uniform = _v(f, "pod_cpr_disp_1m", 1.0) < 1.6
    bad = cpr_jump & np.where(have_pods, recent_rollout & uniform, ~overload)
    # Noisy pod: one pod burns far more CPU per request than the median, no rollout.
    noisy = (_v(f, "pod_cpr_disp_1m", 1.0) >= 2.0) & (_v(f, "pod_cpu_disp_1m") >= 0.15) & ~recent_rollout
    # Leak: working set grows >= 50 KB/s (1 m median; a 256 Mi pod OOMs in < 90 min) without
    # a request backlog (queued requests under overload allocate memory too), or an OOMKill.
    # Not on a noisy pod: the hog's forked process also grows memory ~100 KB/s.
    backlog = (_v(f, "inflight_per_pod") >= 10) | (_v(f, "throttle_1m") >= 0.8)
    leak = ((_v(f, "mem_growth_med_1m") >= 50) & ~backlog & ~noisy) | (_v(f, "oomkills_delta_3m") >= 1)
    # Error fault: 5xx >= 1 % while latency is fine (overload 504s come with p95 ~ 10 s), and
    # still erroring now: the 1 m mean lingers ~60 s after the 5xx stop (22 false pages before).
    errors = (_v(f, "fast_error_share") >= 0.01) & (_v(f, "error_rate") >= ERROR_RATE_FLOOR)
    # Dependency latency: p95 over SLO for 30 s straight (a node stall lasts seconds) while
    # the CPU is neither throttled nor near the limit.
    dep = (p95 >= 1.0) & (_v(f, "p95_slo_min_30s") >= 1.0) & (_v(f, "throttle_1m") < 0.3) & (_v(f, "cpu_frac_limit_1m") < 0.6)

    for mask, state in ((crash, S.CRASHLOOP), (leak, S.MEMORY_LEAK), (restart, S.POD_RESTART),
                        (bad, S.BAD_DEPLOY), (noisy, S.NOISY_POD), (errors, S.ERROR_FAULT),
                        (dep, S.DEPENDENCY_LATENCY)):
        out[(out == None) & mask] = state  # noqa: E711
    return out


def load_state(f: pd.DataFrame) -> np.ndarray:
    """healthy / scale_up / scale_down from capacity-normalised load."""
    breach = (_v(f, "p95_slo") >= 1.0) | (_v(f, "error_rate") >= 0.01)
    saturated = (_v(f, "load_frac") >= 0.85) | (_v(f, "throttle") >= 0.5)
    up = (breach & saturated) | (_v(f, "proj_load_frac_1m") >= 1.0)
    down = (_v(f, "down_fit", 9.0) <= 1.0) & ~breach & (_v(f, "replicas_ready") >= 2)
    return np.where(up, S.SCALE_UP, np.where(down, S.SCALE_DOWN, S.HEALTHY)).astype(object)


def predict(f: pd.DataFrame) -> np.ndarray:
    fault = fault_gate(f)
    return np.where(fault != None, fault, load_state(f)).astype(object)  # noqa: E711
