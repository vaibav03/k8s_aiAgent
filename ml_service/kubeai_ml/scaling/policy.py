"""desired_replicas: forecast + learned capacity -> replica count, with a non-load-fault gate.

Semantics (decision every 15 s per deployment):

* Need: smallest N such that demand per pod stays under the learned safe rate,
  ``capacity.replicas_needed(demand, cpu_per_req)``. ``need_now`` uses the *measured* demand.
* The forecast (q90 @ ``up_horizons_s``) may add pods only when current *measurements* back it:
  P(SLO breach now) above the risk budget, or a measured rising trend (``trend_ratio`` >=
  ``trend_min_ratio``). The classifier's scale_up state alone is not evidence: its LightGBM load
  head calls 10.8 rps on one pod "scale_up" on flat traffic, because that is the base rate of
  the lab's flash-crowd run (a schedule prior, review 3). Forecast-driven need is capped at
  max(need_now + 1, ceil(``fc_max_mult`` x need_now)) and must persist for
  ``up_persist`` consecutive decisions. On flat traffic the lab-trained p90 is 2-4x the current
  rate; without this gate the engine parks at the replica cap (review 3).
* Scale up on measured need at once, capped at ``max_up_step`` extra pods (or doubling).
* Scale-down additionally requires max(now, the evidence-backed q90s at ``down_horizons_s``)
  to fit at ``down_headroom``.
* Scale down slow and safe: the target is the max of all scale-down recommendations over
  ``down_window_s`` (Kubernetes-style stabilisation), at most ``max_down_step`` pods per
  ``down_cooldown_s``, and only when the smaller fleet would run below
  ``down_headroom`` x safe capacity (hysteresis, no flapping on the boundary).
* Gate: the classifier's state decides whether load-based scaling applies at all.
  - healthy / scale_up / scale_down: normal behaviour.
  - bad_deploy: scales up only as a stop-gap when the SLO is breached or P(breach) exceeds the
    risk budget, sized from *measured* demand at the new CPU cost (never the forecast); never
    scales down while the state lasts (normal scale-in removes the stop-gap after the rollback).
  - memory_leak, dependency_latency, error_fault, noisy_pod, crashloop, pod_restart: the
    symptom is not load, so the fault itself never adds replicas. Load-driven need still
    counts, computed with the *healthy baseline* CPU cost per request (so a CPU hog cannot
    inflate it) and from the *current* demand only (no speculative forecast-driven scale-up
    while a fault is active); scale-down is frozen while the fault is active.
* Telemetry: with ``allow_down=False`` (stale / missing metrics, outage) scale-down is frozen and
  the forecast is ignored; with ``use_forecast=False`` only measured demand counts.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Mapping, Protocol

from . import states as S

LOAD_STATES = frozenset(S.LOAD_STATES)
NO_SCALE_STATES = frozenset(S.FAULT_STATES) - {S.BAD_DEPLOY}
ALL_STATES = frozenset(S.STATES)

# Short action code per state (the long runbook text is states.REMEDIATION).
REMEDIATION: dict[str, str] = {
    S.HEALTHY: "none",
    S.SCALE_UP: "scale",
    S.SCALE_DOWN: "scale",
    S.MEMORY_LEAK: "restart_pod",          # restart the leaking pod before OOM; page if it recurs
    S.DEPENDENCY_LATENCY: "page_human",    # downstream is slow: scaling won't help
    S.ERROR_FAULT: "page_human",           # 5xx not explained by load
    S.NOISY_POD: "restart_pod",            # delete / reschedule the hot pod
    S.BAD_DEPLOY: "rollback",
    S.CRASHLOOP: "rollback",               # new ReplicaSet cannot start; old one still serves
    S.POD_RESTART: "none",                 # transient: watch, let the Deployment replace it
}


class Capacity(Protocol):
    def replicas_needed(self, demand_rps: float, cpu_per_req_s: float, max_replicas: int = 50,
                        risk: float | None = None) -> int: ...


@dataclass(frozen=True)
class Bounds:
    min_replicas: int = 1
    max_replicas: int = 10


@dataclass(frozen=True)
class Stabilization:
    max_up_step: int = 4                # pods added per decision (or doubling, whichever is larger)
    down_window_s: float = 180.0        # max over recent recommendations before scaling in (>= 180 s)
    down_cooldown_s: float = 60.0       # min time between two scale-downs
    max_down_step: int = 1              # pods removed per scale-down
    down_headroom: float = 0.85         # scale in only if the smaller fleet runs < 85 % of safe
    up_horizons_s: tuple[int, ...] = (60,)   # forecast horizons that can trigger a scale-up
    down_horizons_s: tuple[int, ...] = (60, 120, 300)   # horizons that must fit before scaling in
    trend_min_ratio: float = 1.2        # mean rps last 30 s / mean rps 60-120 s ago that counts as rising
    fc_max_mult: float = 1.5            # forecast-driven need <= max(need_now + 1, ceil(1.5 need_now))
    up_persist: int = 2                 # forecast-driven need must hold for this many decisions
    up_headroom: float = 1.0            # size for demand x this (explicit headroom, swept)


@dataclass
class ScalerMemory:
    """Per-deployment state the caller keeps between decisions."""
    recs: deque[tuple[float, int]] = field(default_factory=deque)
    last_down_ts: float = -math.inf
    first_ts: float | None = None       # first decision seen: no scale-in before a full window
    fc_streak: int = 0                  # consecutive decisions with a forecast-driven need > current
    last_restart_ts: float = -math.inf  # engine-level restart interlock (one pod restart per cooldown)


@dataclass(frozen=True)
class Decision:
    replicas: int
    need_now: int
    need_up: int
    need_down: int
    action: str                 # scale_up | scale_down | hold | do_not_scale
    remediation: str
    reason: str


def desired_replicas(forecast_q90: Mapping[int, float], capacity: Capacity, current: int,
                     bounds: Bounds, stabilization: Stabilization, *, state: str, now_rps: float,
                     cpu_per_req_s: float, baseline_cpu_per_req_s: float | None = None,
                     ts: float, memory: ScalerMemory, slo_risk: float | None = None,
                     risk_budget: float | None = None, trend_ratio: float | None = None,
                     slo_breached: bool = False, allow_down: bool = True,
                     use_forecast: bool = True) -> Decision:
    """Replica count for this decision. ``forecast_q90`` maps horizon seconds -> p90 demand rps."""
    if state not in ALL_STATES:
        raise ValueError(f"unknown state {state!r}")
    if bounds.min_replicas > bounds.max_replicas:
        raise ValueError(f"min_replicas {bounds.min_replicas} > max_replicas {bounds.max_replicas}")
    st, b = stabilization, bounds
    cpu = cpu_per_req_s
    if state in NO_SCALE_STATES and baseline_cpu_per_req_s:
        cpu = baseline_cpu_per_req_s          # a hog / fault must not inflate the load need

    def need(d: float, headroom: float = 1.0) -> int:
        return capacity.replicas_needed(d * st.up_headroom / headroom, cpu, max_replicas=b.max_replicas)

    at_risk = slo_risk is not None and risk_budget is not None and slo_risk > risk_budget
    rising = trend_ratio is not None and trend_ratio >= st.trend_min_ratio
    evidence = at_risk or rising
    fc_on = (use_forecast and allow_down and evidence and state not in NO_SCALE_STATES
             and state != S.BAD_DEPLOY)
    need_now = need(now_rps)
    fc_cap = max(need_now + 1, math.ceil(st.fc_max_mult * need_now))
    up_fc = max([now_rps] + [v for h, v in forecast_q90.items() if h in st.up_horizons_s]) if fc_on else now_rps
    need_fc = min(need(up_fc), fc_cap)
    if need_fc > max(current, need_now):
        memory.fc_streak += 1
    else:
        memory.fc_streak = 0
    fc_driven = need_fc > need_now and memory.fc_streak >= st.up_persist
    need_up, up_demand = (need_fc, up_fc) if fc_driven else (need_now, now_rps)
    if state == S.BAD_DEPLOY and not (slo_breached or at_risk):
        need_up, up_demand = min(need_now, current), now_rps    # stop-gap only while the SLO is at risk
    down_demand = max([now_rps] + [v for h, v in forecast_q90.items() if h in st.down_horizons_s]) if fc_on else now_rps
    need_down = min(need(down_demand, st.down_headroom), max(fc_cap, need(now_rps, st.down_headroom)))

    # stabilisation window of scale-up-style recommendations (recorded on every decision)
    if memory.first_ts is None:
        memory.first_ts = ts
    memory.recs.append((ts, need_up))
    while memory.recs and memory.recs[0][0] < ts - st.down_window_s:
        memory.recs.popleft()

    target = current
    action, reason = "hold", "within capacity"
    if need_up > current:
        step = max(st.max_up_step, current)                  # up to doubling
        target = min(need_up, current + step)
        action = "scale_up"
        src = "q90 forecast" if fc_driven else "measured"
        reason = f"{src} demand {up_demand:.1f} rps needs {need_up} pods"
        if fc_driven:
            why = "P(breach) over budget" if at_risk else f"rising trend x{trend_ratio:.2f}"
            reason += f" (forecast backed by {why}; capped at {fc_cap})"
        if state == S.BAD_DEPLOY:
            reason += " (bad deploy stop-gap while the SLO is at risk: remove after the rollback)"
        elif state in NO_SCALE_STATES:
            reason += f" (load-driven, independent of {state})"
    elif state in NO_SCALE_STATES or state == S.BAD_DEPLOY:
        action = "do_not_scale"
        reason = f"{state}: scaling will not fix it; remediation {REMEDIATION[state]}"
    elif not allow_down:
        reason = "scale-down frozen (insufficient telemetry or outage)"
    else:
        window_max = max(n for _, n in memory.recs)
        candidate = max(window_max, need_down, current - st.max_down_step)
        window_full = ts - memory.first_ts >= st.down_window_s
        if candidate < current and window_full and ts - memory.last_down_ts >= st.down_cooldown_s:
            target = candidate
            action = "scale_down"
            reason = (f"max need over {st.down_window_s:.0f}s is {window_max} pods "
                      f"(demand to +{max(st.down_horizons_s)}s {down_demand:.1f} rps at {st.down_headroom:.0%} headroom)")
    target = max(b.min_replicas, min(b.max_replicas, target))
    if target < current:
        memory.last_down_ts = ts
    if target == current and action in ("scale_up", "scale_down"):
        action, reason = "hold", f"bounded at {target}: " + reason
    if action == "hold" and memory.fc_streak and need_fc > current and not fc_driven:
        reason += f"; forecast needs {need_fc} pods, waiting for it to persist ({memory.fc_streak}/{st.up_persist})"
    return Decision(target, need_now, need_up, need_down, action, REMEDIATION[state], reason)
