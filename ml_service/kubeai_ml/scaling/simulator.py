"""Discrete-time (5 s) replay simulator for autoscaling policies.

Replays an offered-load trace (:mod:`.traces`) against a calibrated pod model (:mod:`.podmodel`)
and a scaling policy, and reports SLO-violation minutes, pod-minutes (cost), scaling actions, max
replicas and time-to-recover after spikes.

Per 5 s step ``i`` (row time ``t[i]`` = end of the interval, same convention as the dataset):

1. offered load ``lambda`` = mean offered RPS over the interval (from the k6 shape, exact);
2. capacity = sum over pods of the fraction of the interval the pod was Ready * mu(n);
3. a FIFO fluid queue of arrival cohorts serves up to capacity*dt; cohorts older than the 10 s
   deadline time out (504); while the queue head is at the deadline the pods are in congestion
   collapse (only ~20 % of served work is in time, see :mod:`.podmodel`);
   p95 = max(steady-state curve f(rps/pod, n), queue wait + 30 ms);
4. CPU = served * cpu_s_per_req + idle, capped at the pod limits;
5. *observed* signals are derived with the dataset's measurement windows (rps / error_rate / p95
   over 20 s, CPU over 20 s, metrics-server CPU for the HPA lagged by ``ms_lag_s`` and averaged
   over ``ms_window_s``);
6. if a controller sync boundary fell in the interval, the policy is called; a scale-up creates
   pods that become Ready ``startup_delay_s`` later; a scale-down removes pods at once
   (unready / newest first).

Policy plug-in interface
------------------------
A policy is any object with::

    name: str
    period_s: float          # how often it is invoked (HPA 15 s, SRE agent 15 s)
    phase_s: float           # sync offset in [0, period_s) relative to t = 0
    def reset(self) -> None  # clear internal state before a run
    def __call__(self, history: pd.DataFrame, state: SimState) -> int   # desired replicas

``history`` holds the *observable* rows up to and including the current step, with the
dataset_deploy column names (``seconds_since_start, rps, goodput_rps, inflight, p50, p95, p99,
error_rate, cpu_cores_used, cpu_util_vs_request, throttle_ratio, replicas_desired,
replicas_ready, hpa_desired, pods_not_ready``), so a policy written against the live feature
pipeline / dataset can be replayed unchanged. Ground truth (``true_*``) is never passed in.
``state`` carries the controller view (spec replicas, per-pod readiness ages, the metrics-server
CPU utilisation the HPA would read, bounds, the pod model). The returned value is clamped to
[min_replicas, max_replicas]. :class:`CallablePolicy` wraps a plain
``fn(history, state) -> int`` (e.g. ``kubeai_ml.scaling.policy``) into this interface.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np
import pandas as pd

from .data import SLO_ERR, SLO_P95_S
from .podmodel import BASE_P95_S, PodModel
from .traces import Trace

OBS_COLS = ["seconds_since_start", "rps", "goodput_rps", "inflight", "p50", "p95", "p99",
            "error_rate", "cpu_cores_used", "cpu_util_vs_request", "throttle_ratio",
            "replicas_desired", "replicas_ready", "hpa_desired", "pods_not_ready"]


@dataclass
class SimConfig:
    step_s: float = 5.0
    min_replicas: int = 1
    max_replicas: int = 4
    initial_replicas: int = 1
    obs_window_s: float = 20.0      # rate()/histogram_quantile window of the app metrics
    cpu_window_s: float = 20.0      # effective CPU window (max(cAdvisor 45 s, process 20 s))
    ms_lag_s: float = 15.0          # age of the newest metrics-server CPU sample at an HPA sync
    ms_window_s: float = 15.0       # cAdvisor/kubelet rate window behind that sample


@dataclass
class SimState:
    """Controller-visible state handed to a policy at a sync."""
    t: float
    spec_replicas: int
    ready_ages_s: list[float]           # per pod: seconds since Ready (< 0: not ready yet)
    ms_cpu_util: float                  # metrics-server mean CPU util (of request) over ready pods
    min_replicas: int
    max_replicas: int
    step_s: float
    model: PodModel
    cfg: SimConfig
    extra: dict = field(default_factory=dict)

    @property
    def ready(self) -> int:
        return sum(a >= 0 for a in self.ready_ages_s)


class Policy(Protocol):
    name: str
    period_s: float
    phase_s: float

    def reset(self) -> None: ...

    def __call__(self, history: pd.DataFrame, state: SimState) -> int: ...


@dataclass
class CallablePolicy:
    """Adapt ``fn(history, state) -> int`` to :class:`Policy`."""
    fn: Callable[[pd.DataFrame, SimState], int]
    name: str = "callable"
    period_s: float = 15.0
    phase_s: float = 0.0
    reset_fn: Callable[[], None] | None = None

    def reset(self) -> None:
        if self.reset_fn:
            self.reset_fn()

    def __call__(self, history: pd.DataFrame, state: SimState) -> int:
        return int(self.fn(history, state))


@dataclass
class FixedPolicy:
    n: int
    period_s: float = 15.0
    phase_s: float = 0.0

    @property
    def name(self) -> str:
        return f"fixed-{self.n}"

    def reset(self) -> None:
        pass

    def __call__(self, history: pd.DataFrame, state: SimState) -> int:
        return self.n


@dataclass
class _Pod:
    created_t: float
    ready_t: float


@dataclass
class SimResult:
    policy: str
    trace: str
    df: pd.DataFrame        # per step: observable columns + true_* + offered_rps
    actions: list[tuple[float, int, int]]   # (t, from, to)


def _window_mean(x: np.ndarray, i: int, k: int) -> float:
    return float(np.mean(x[max(0, i - k + 1): i + 1]))


def simulate(trace: Trace, policy: Policy, model: PodModel, cfg: SimConfig | None = None
             ) -> SimResult:
    cfg = cfg or SimConfig(step_s=trace.step_s)
    dt = cfg.step_s
    T = len(trace.t)
    t_arr = trace.t.astype(float)
    k_obs = max(1, int(round(cfg.obs_window_s / dt)))
    k_cpu = max(1, int(round(cfg.cpu_window_s / dt)))

    pods: list[_Pod] = [_Pod(-600.0, -600.0) for _ in range(cfg.initial_replicas)]
    spec = cfg.initial_replicas
    backlog = 0.0
    queue: deque = deque()                                  # FIFO cohorts [requests, arrival t]
    policy.reset()

    true = {k: np.zeros(T) for k in ("offered", "served", "dropped", "p95", "cpu", "util_ready",
                                     "n_eff", "running", "backlog", "spec", "ready")}
    obs = {c: np.full(T, np.nan) for c in OBS_COLS}
    actions: list[tuple[float, int, int]] = []

    for i in range(T):
        t = t_arr[i]
        t_start = t - dt
        lam = float(trace.rps[i])
        frac = [min(1.0, max(0.0, (t - p.ready_t) / dt)) for p in pods]
        n_eff = float(sum(frac))
        n_int = max(1, math.ceil(n_eff - 1e-9))
        cap = n_eff * model.mu(n_int)                       # RPS
        arrivals = lam * dt
        b_prev = float(sum(c for c, _ in queue))
        head_wait = b_prev / cap if cap > 0 else 0.0        # FIFO delay seen by new arrivals
        queue.append([arrivals, t - dt / 2])               # cohort: [requests, arrival time]
        errs = 0.0
        if cap <= 0:                                        # no endpoint: refused at once
            good = work = 0.0
            errs += sum(c for c, _ in queue)
            queue.clear()
        else:
            budget = work = cap * dt
            served_all = 0.0
            while queue and budget > 1e-12:
                take = min(queue[0][0], budget)
                budget -= take
                served_all += take
                queue[0][0] -= take
                if queue[0][0] <= 1e-12:
                    queue.popleft()
            work = served_all
            # congestion collapse: the head is (about to be) past the deadline, so most served
            # requests finish late and are answered 504 although their CPU was spent
            if head_wait + dt / 2 > model.deadline_s:
                good = model.collapse_goodput_frac * served_all
                errs += served_all - good
            else:
                good = served_all
            while queue and t - queue[0][1] > model.deadline_s:   # timed out while queued
                errs += queue.popleft()[0]
        backlog = float(sum(c for c, _ in queue))
        dropped = errs
        if cap <= 0:
            p95 = model.deadline_s if arrivals > 0 else BASE_P95_S
        else:
            x = min(lam, cap) / max(n_eff, 1e-9)
            wait = min(model.deadline_s, max(head_wait, backlog / cap))
            p95 = max(model.p95_steady(x, n_int), wait + BASE_P95_S if wait > 0.05 else 0.0)
            p95 = min(p95, model.deadline_s)
        running = len(pods)
        n_ready_now = sum(1 for p in pods if p.ready_t <= t)
        cpu_busy = work / dt * model.cpu_s_per_req
        cpu = min(cpu_busy, n_eff * model.cpu_limit) + model.idle_cores_per_pod * running
        served = good

        true["offered"][i], true["served"][i], true["dropped"][i] = lam, served / dt, dropped / dt
        true["p95"][i], true["cpu"][i], true["n_eff"][i] = p95, cpu, n_eff
        true["util_ready"][i] = (cpu / (n_eff * model.cpu_request)) if n_eff > 0 else 0.0
        true["running"][i], true["backlog"][i] = running, backlog
        true["ready"][i] = n_ready_now

        # ---- observed (dataset-like) signals ------------------------------------------------
        sv = _window_mean(true["served"], i, k_obs)
        dr = _window_mean(true["dropped"], i, k_obs)
        resp = sv + dr
        obs["seconds_since_start"][i] = t
        obs["rps"][i] = resp
        obs["goodput_rps"][i] = sv
        obs["error_rate"][i] = dr / resp if resp > 0 else 0.0
        lo = max(0, i - k_obs + 1)
        w95 = float(np.max(true["p95"][lo:i + 1]))
        if resp > 0 and dr / resp > 0.05:
            w95 = model.deadline_s                          # 504s are recorded at 10 s
        obs["p95"][i] = w95
        obs["p50"][i] = min(w95, max(BASE_P95_S * 0.92, backlog / cap if cap > 0 else w95))
        obs["p99"][i] = w95
        obs["inflight"][i] = min(backlog, lam * model.deadline_s) + lam * BASE_P95_S
        c = _window_mean(true["cpu"], i, k_cpu)
        obs["cpu_cores_used"][i] = c
        obs["cpu_util_vs_request"][i] = c / (max(running, 1) * model.cpu_request)
        obs["throttle_ratio"][i] = float(np.clip((true["util_ready"][i] / 2 - 0.6) / 0.4, 0, 1))
        obs["replicas_desired"][i] = spec
        obs["replicas_ready"][i] = n_ready_now
        obs["hpa_desired"][i] = spec
        obs["pods_not_ready"][i] = running - n_ready_now

        # ---- controller ---------------------------------------------------------------------
        boundary = _sync_boundary(t_start, t, policy.period_s, policy.phase_s)
        if boundary is not None:
            tb = boundary
            ms = _ms_util(t_arr, true["util_ready"], true["n_eff"], i, tb, cfg)
            state = SimState(t=tb, spec_replicas=spec,
                             ready_ages_s=[tb - p.ready_t for p in pods], ms_cpu_util=ms,
                             min_replicas=cfg.min_replicas, max_replicas=cfg.max_replicas,
                             step_s=dt, model=model, cfg=cfg)
            hist = pd.DataFrame({c: obs[c][: i + 1] for c in OBS_COLS})
            desired = int(np.clip(policy(hist, state), cfg.min_replicas, cfg.max_replicas))
            if desired != spec:
                actions.append((tb, spec, desired))
                if desired > spec:
                    pods += [_Pod(tb, tb + model.startup_delay_s) for _ in range(desired - spec)]
                else:
                    pods.sort(key=lambda p: (p.ready_t <= tb, -p.created_t))   # unready, newest first
                    pods = pods[spec - desired:]
                spec = desired
        true["spec"][i] = spec

    df = pd.DataFrame({c: obs[c] for c in OBS_COLS})
    for k, v in true.items():
        df[f"true_{k}"] = v
    df["true_error_rate"] = np.where(df["true_served"] + df["true_dropped"] > 0,
                                     df["true_dropped"] / (df["true_served"] + df["true_dropped"]).clip(lower=1e-9), 0.0)
    return SimResult(policy=policy.name, trace=trace.name, df=df, actions=actions)


def _sync_boundary(t0: float, t1: float, period: float, phase: float) -> float | None:
    """The controller sync time in (t0, t1], if any."""
    k = math.floor((t1 - phase) / period)
    tb = phase + k * period
    return tb if tb > t0 else None


def _ms_util(t_arr: np.ndarray, util: np.ndarray, n_eff: np.ndarray, i: int, tb: float,
             cfg: SimConfig) -> float:
    """metrics-server view at sync time tb: mean ready-pod CPU utilisation over
    [tb - lag - window, tb - lag], from the per-step truth (step k covers (t_k - dt, t_k])."""
    hi, lo = tb - cfg.ms_lag_s, tb - cfg.ms_lag_s - cfg.ms_window_s
    dt = cfg.step_s
    w, acc = 0.0, 0.0
    for k in range(i, -1, -1):
        a, b = t_arr[k] - dt, t_arr[k]
        if b <= lo:
            break
        ov = min(b, hi) - max(a, lo)
        if ov > 0 and n_eff[k] > 0:
            acc += ov * util[k]
            w += ov
    if w == 0:   # window entirely before the trace starts: no load yet
        return 0.0 if hi <= t_arr[0] - dt else float(util[max(0, i)])
    # the part of the window before the trace (no traffic yet) counts as idle
    pre = max(0.0, min(hi, t_arr[0] - dt) - lo)
    return acc / (w + pre)


# ------------------------------------------------------------------------------------------
# Scoring (applies to simulated runs and to recorded dataset runs alike)


def violation_mask(df: pd.DataFrame, p95_col: str = "p95", err_col: str = "error_rate") -> np.ndarray:
    return ((df[p95_col].fillna(0) >= SLO_P95_S) | (df[err_col].fillna(0) >= SLO_ERR)).to_numpy()


def spike_onsets(offered: np.ndarray, t: np.ndarray, ratio: float = 1.4, lookback_s: float = 30.0,
                 refractory_s: float = 90.0, min_rise_rps: float = 5.0) -> list[int]:
    """Indices where the offered load rises >= ``ratio`` x its level ``lookback_s`` earlier."""
    out: list[int] = []
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 5.0
    k = max(1, int(round(lookback_s / dt)))
    for i in range(k, len(offered)):
        prev = offered[i - k]
        if offered[i] >= ratio * max(prev, 1e-9) and offered[i] - prev >= min_rise_rps:
            if not out or t[i] - t[out[-1]] > refractory_s:
                out.append(i)
    return out


def time_to_recover(viol: np.ndarray, t: np.ndarray, onset: int, ok_hold_s: float = 30.0,
                    horizon_s: float = 300.0) -> float:
    """Seconds from a spike onset until the SLO has held for ``ok_hold_s``; 0 if never broken."""
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 5.0
    hold = max(1, int(round(ok_hold_s / dt)))
    end = min(len(t), onset + int(horizon_s / dt))
    if not viol[onset:end].any():
        return 0.0
    last_bad = onset
    for j in range(onset, end):
        if viol[j]:
            last_bad = j
        elif j - last_bad >= hold:
            return float(t[last_bad] - t[onset] + dt)
    return float(t[end - 1] - t[onset] + dt)


def summarize(df: pd.DataFrame, offered: np.ndarray, step_s: float = 5.0,
              replicas_col: str = "replicas_desired", running_col: str | None = None,
              p95_col: str = "p95", err_col: str = "error_rate") -> dict:
    """SLO-violation minutes, pod-minutes, scaling actions, max replicas, TTR after spikes."""
    t = df["seconds_since_start"].to_numpy(float)
    viol = violation_mask(df, p95_col, err_col)
    reps = df[replicas_col].to_numpy(float)
    running = df[running_col].to_numpy(float) if running_col else reps
    onsets = spike_onsets(offered, t)
    ttr = [time_to_recover(viol, t, o) for o in onsets]
    changes = int(np.sum(np.diff(reps) != 0)) if len(reps) > 1 else 0
    events = int(np.sum(np.diff(np.r_[False, viol].astype(int)) == 1))
    return {
        "minutes": round(len(t) * step_s / 60, 2),
        "slo_violation_min": round(float(viol.sum()) * step_s / 60, 3),
        "violation_events": events,
        "pod_minutes": round(float(running.sum()) * step_s / 60, 2),
        "mean_replicas": round(float(reps.mean()), 3),
        "scaling_actions": changes,
        "max_replicas": int(np.nanmax(reps)),
        "n_spikes": len(onsets),
        "ttr_after_spike_max_s": max(ttr) if ttr else None,
        "ttr_after_spike_mean_s": round(float(np.mean(ttr)), 1) if ttr else None,
    }


_ORACLE_CACHE: dict[int, np.ndarray] = {}


def oracle_need(offered: np.ndarray, model: PodModel, max_n: int = 16) -> np.ndarray:
    """Fewest pods that keep the steady-state p95 under the SLO at each step's offered rate
    (perfect foresight, no startup delay): a lower bound on what any policy needs."""
    key = hash((model.to_json(), max_n))
    if key not in _ORACLE_CACHE:
        _ORACLE_CACHE[key] = np.array([n * model.safe_rps_per_pod(n, SLO_P95_S) for n in range(1, max_n + 1)])
    safe = _ORACLE_CACHE[key]
    idx = np.searchsorted(np.maximum.accumulate(safe), np.asarray(offered, float), side="left")
    return np.minimum(idx + 1, max_n)


def summarize_sim(res: SimResult, step_s: float = 5.0, model: PodModel | None = None) -> dict:
    s = summarize(res.df, res.df["true_offered"].to_numpy(), step_s, "true_spec", "true_running")
    if model is not None:
        need = oracle_need(res.df["true_offered"].to_numpy(float), model)
        run = res.df["true_running"].to_numpy(float)
        s["pod_min_over_need"] = round(float(np.maximum(run - need, 0).sum()) * step_s / 60, 2)
        s["oracle_pod_min"] = round(float(need.sum()) * step_s / 60, 2)
    s["scaling_actions"] = len(res.actions)
    tv = violation_mask(res.df, "true_p95", "true_error_rate")
    s["slo_violation_min_true"] = round(float(tv.sum()) * step_s / 60, 3)
    off = res.df["true_offered"].sum()
    s["dropped_pct"] = round(100 * float(res.df["true_dropped"].sum() / off), 3) if off > 0 else 0.0
    return {"policy": res.policy, "trace": res.trace, **s}
