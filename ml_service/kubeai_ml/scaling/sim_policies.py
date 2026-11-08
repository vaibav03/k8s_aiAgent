"""Policy catalogue for the replay simulator (beyond fixed-N and the stock HPA).

* :func:`tuned_hpa` - HPA whose CPU target matches the *learned* per-pod capacity: the CPU a pod
  burns at the largest per-pod RPS that keeps p95 within a budget (default 200 ms, i.e. 2/3 of the
  SLO), at a reference replica count. The stock 70 % target equals only ~6.7 RPS/pod.
* :class:`ReferencePredictivePolicy` - a small, self-contained forecast + capacity controller used
  as the predictive baseline until ``kubeai_ml.scaling.policy`` (written in parallel) is plugged
  in via :func:`load_agent_policy`. Demand estimate = served rate + backlog growth (so it is not
  blinded by saturation), linear trend over the last 60 s extrapolated over the actuation delay
  (startup + one sync + measurement lag), replicas = smallest n with demand/n <= safe RPS/pod(n).
  Scale-up at once; scale-down only after the lower count has been recommended for
  ``down_hold_s``.
"""
from __future__ import annotations

import importlib
import inspect
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .hpa import HPAPolicy
from .podmodel import PodModel
from .simulator import CallablePolicy, Policy, SimState


def tuned_target_util(model: PodModel, n_ref: int = 4, p95_budget_s: float = 0.2) -> float:
    """CPU utilisation (of request) at the safe per-pod RPS of ``n_ref`` replicas."""
    safe = model.safe_rps_per_pod(n_ref, p95_budget_s)
    return (safe * model.cpu_s_per_req + model.idle_cores_per_pod) / model.cpu_request


def tuned_hpa(model: PodModel, scale_down_window_s: float = 300.0, n_ref: int = 4,
              p95_budget_s: float = 0.2, phase_s: float = 0.0) -> HPAPolicy:
    tgt = tuned_target_util(model, n_ref, p95_budget_s)
    return HPAPolicy(target_util=tgt, scale_down_window_s=scale_down_window_s, phase_s=phase_s,
                     label=f"hpa-tuned-{int(round(tgt * 100))}-sd{int(scale_down_window_s)}")


@dataclass
class ReferencePredictivePolicy:
    model: PodModel
    p95_budget_s: float = 0.2
    trend_window_s: float = 60.0
    obs_lag_s: float = 10.0            # half the 20 s rate window
    down_hold_s: float = 60.0
    headroom: float = 1.0
    period_s: float = 15.0
    phase_s: float = 0.0
    name: str = "predictive-ref"
    _low_since: float | None = field(default=None, repr=False)

    def reset(self) -> None:
        self._low_since = None

    def demand(self, h: pd.DataFrame, step_s: float) -> np.ndarray:
        """Served + backlog growth (req/s) per row; robust to saturation."""
        served = h["goodput_rps"].to_numpy(float)
        err = (h["rps"] - h["goodput_rps"]).to_numpy(float)          # 504s are unmet demand
        infl = h["inflight"].to_numpy(float)
        growth = np.r_[0.0, np.diff(infl)] / step_s
        return np.maximum(served + err + np.maximum(growth, 0.0), 0.0)

    def needed(self, rps: float) -> int:
        for n in range(1, 64):
            if rps * self.headroom <= n * self.model.safe_rps_per_pod(n, self.p95_budget_s):
                return n
        return 64

    def __call__(self, history: pd.DataFrame, state: SimState) -> int:
        d = self.demand(history, state.step_s)
        k = max(2, int(self.trend_window_s / state.step_s))
        y = d[-k:]
        horizon = self.model.startup_delay_s + self.period_s + self.obs_lag_s
        if len(y) >= 3:
            slope = np.polyfit(np.arange(len(y)) * state.step_s, y, 1)[0]
            fc = max(y[-1], y[-1] + max(slope, 0.0) * horizon)
        else:
            fc = float(y[-1])
        want = self.needed(fc)
        cur = state.spec_replicas
        if want >= cur:
            self._low_since = None
            return want
        self._low_since = state.t if self._low_since is None else self._low_since
        if state.t - self._low_since >= self.down_hold_s:
            self._low_since = None
            return max(want, cur - 1)        # step down one at a time
        return cur


@dataclass
class AgentPolicyAdapter:
    """Replays ``kubeai_ml.scaling.policy.desired_replicas`` (forecast + learned capacity +
    stabilisation) inside the simulator.

    ``forecast_fn(history, step_s) -> {horizon_s: q90 rps}`` defaults to a trend extrapolation of
    the saturation-robust demand estimate (placeholder until the trained forecaster is wired in:
    pass ``forecast_fn`` built on ``forecast.forecast_from_samples``). Capacity is the trained
    ``CapacityModel`` (models/scaling/capacity.json). State is always ``healthy`` (traffic-only
    traces); cpu per request is taken from the observed history like the live agent would.
    """
    model: PodModel
    forecast_fn: object = None
    period_s: float = 15.0
    phase_s: float = 0.0
    name: str = "agent-policy"
    _mem: object = field(default=None, repr=False)
    _cap: object = field(default=None, repr=False)

    def reset(self) -> None:
        from .policy import ScalerMemory

        self._mem = ScalerMemory()

    def _capacity(self):
        if self._cap is None:
            from .capacity import CapacityModel
            from .data import MODELS_DIR

            self._cap = CapacityModel.load(MODELS_DIR / "capacity.json")
        return self._cap

    def _forecast(self, h: pd.DataFrame, step_s: float) -> dict[int, float]:
        if self.forecast_fn is not None:
            return dict(self.forecast_fn(h, step_s))
        d = ReferencePredictivePolicy(self.model).demand(h, step_s)
        y = d[-12:]
        slope = np.polyfit(np.arange(len(y)) * step_s, y, 1)[0] if len(y) >= 3 else 0.0
        return {hz: max(0.0, float(y[-1] + slope * hz)) for hz in (60, 120, 300)}

    def __call__(self, history: pd.DataFrame, state: SimState) -> int:
        from .policy import Bounds, Stabilization, desired_replicas

        if self._mem is None:
            self.reset()
        g = history["goodput_rps"].to_numpy(float)[-12:]
        c = history["cpu_cores_used"].to_numpy(float)[-12:]
        ok = g > 1.0
        cpu = float(np.median(c[ok] / g[ok])) if ok.any() else self.model.cpu_s_per_req
        now = float(ReferencePredictivePolicy(self.model).demand(history, state.step_s)[-1])
        dec = desired_replicas(self._forecast(history, state.step_s), self._capacity(),
                               state.spec_replicas, Bounds(state.min_replicas, state.max_replicas),
                               Stabilization(), state="healthy", now_rps=now, cpu_per_req_s=cpu,
                               ts=state.t, memory=self._mem)
        return int(dec.replicas)


def load_agent_policy(model: PodModel) -> Policy | None:
    """Plug in ``kubeai_ml.scaling.policy``: a ``sim_policy(model)`` factory if the module has
    one, else :class:`AgentPolicyAdapter` around ``desired_replicas`` when the trained capacity
    model exists. None if the module or its artefacts are missing (comparison omits it)."""
    try:
        mod = importlib.import_module("kubeai_ml.scaling.policy")
    except Exception:
        return None
    if hasattr(mod, "sim_policy"):
        return mod.sim_policy(model)
    fn = getattr(mod, "desired_replicas", None)
    if fn is None:
        return None
    if len(inspect.signature(fn).parameters) == 2:
        return CallablePolicy(fn=fn, name="agent-policy")
    try:
        ad = AgentPolicyAdapter(model)
        ad._capacity()
        return ad
    except Exception:
        return None


def default_policies(model: PodModel, phase_s: float = 0.0) -> list[Policy]:
    from .simulator import FixedPolicy

    pols: list[Policy] = [FixedPolicy(n) for n in (2, 3, 4, 6)]
    pols += [HPAPolicy(0.70, 300.0, phase_s=phase_s), HPAPolicy(0.70, 30.0, phase_s=phase_s),
             tuned_hpa(model, 300.0, phase_s=phase_s), tuned_hpa(model, 30.0, phase_s=phase_s),
             ReferencePredictivePolicy(model, phase_s=phase_s),
             ReferencePredictivePolicy(model, headroom=1.5, phase_s=phase_s, name="predictive-ref-h1.5")]
    agent = load_agent_policy(model)
    if agent is not None:
        pols.append(agent)
    return pols
