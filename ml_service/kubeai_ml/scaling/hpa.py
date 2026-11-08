"""Kubernetes HPA (autoscaling/v2, CPU Utilization metric) as a simulator policy.

Follows kube-controller-manager's ``replica_calculator.GetResourceReplicas`` and the default
``behavior``:

* usage ratio = mean CPU utilisation of ready pods *with a metric* / target;
* within 10 % tolerance -> keep; otherwise desired = ceil(ratio * pods_with_metrics);
* ``groupPods``: a pod that is not Ready yet has no metrics-server sample -> *missing*; a Ready
  pod whose CPU sample predates readiness + the metric window (inside the 300 s CPU
  initialisation period) -> *unready*. Recompute: on scale-up both count as 0 %; on scale-down
  missing pods count as max(100 %, target) of the request and unready pods are dropped; if that
  flips the direction or lands within tolerance -> keep current;
* scale-up: no stabilisation, rate limit max(+100 %, +4 pods) per 15 s;
* scale-down: stabilisation window (default 300 s; variant 30 s) = max of the recommendations in
  the window, rate limit 100 % per 15 s;
* clamp to [min, max]; sync every 15 s.

The metric the HPA reads is ``state.ms_cpu_util`` (metrics-server view: lagged and windowed by the
simulator, see :class:`~.simulator.SimConfig`).
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import pandas as pd

from .simulator import SimState

TOLERANCE = 0.10


@dataclass
class HPAPolicy:
    target_util: float = 0.70                 # of request (averageUtilization / 100)
    scale_down_window_s: float = 300.0
    period_s: float = 15.0
    phase_s: float = 0.0
    up_percent: float = 100.0
    up_pods: int = 4
    label: str | None = None
    _recs: deque = field(default_factory=deque, repr=False)

    @property
    def name(self) -> str:
        return self.label or f"hpa-{int(round(self.target_util * 100))}-sd{int(self.scale_down_window_s)}"

    def reset(self) -> None:
        self._recs.clear()

    def _raw_desired(self, state: SimState) -> int:
        cur = state.spec_replicas
        lag_window = state.cfg.ms_lag_s + state.cfg.ms_window_s
        ages = state.ready_ages_s
        with_metric = [a for a in ages if a >= lag_window]
        missing = [a for a in ages if a < 0]                  # not Ready yet: no sample at all
        unready = [a for a in ages if 0 <= a < lag_window]    # Ready, sample predates readiness + window
        if not with_metric:
            return cur
        util = state.ms_cpu_util
        ratio = util / self.target_util
        if not missing and not unready:
            if abs(ratio - 1.0) <= TOLERANCE:
                return cur
            return math.ceil(ratio * len(with_metric))
        if ratio > 1.0:   # scale-up: missing and unready count as 0 %
            n_all = len(with_metric) + len(missing) + len(unready)
            new_ratio = util * len(with_metric) / (self.target_util * n_all)
        elif ratio < 1.0:  # scale-down: missing count as max(100 %, target) of request, unready dropped
            n_all = len(with_metric) + len(missing)
            fallback = max(1.0, self.target_util)
            new_ratio = (util * len(with_metric) + fallback * len(missing)) / (self.target_util * n_all)
        else:
            return cur
        if abs(new_ratio - 1.0) <= TOLERANCE or (ratio > 1) != (new_ratio > 1):
            return cur
        new = math.ceil(new_ratio * n_all)
        if (ratio > 1 and new < cur) or (ratio < 1 and new > cur):
            return cur
        return new

    def __call__(self, history: pd.DataFrame, state: SimState) -> int:
        cur = state.spec_replicas
        rec = self._raw_desired(state)
        # scale-down stabilisation: highest recommendation within the window
        self._recs.append((state.t, rec))
        # k8s keeps recommendations strictly *after* now - window (``timestamp.After(cutoff)``)
        while self._recs and self._recs[0][0] <= state.t - self.scale_down_window_s:
            self._recs.popleft()
        if rec < cur:
            rec = min(cur, max(r for _, r in self._recs))
        if rec > cur:
            limit = max(math.floor(cur * (1 + self.up_percent / 100)), cur + self.up_pods)
            rec = min(rec, limit)
        return int(min(max(rec, state.min_replicas), state.max_replicas))
