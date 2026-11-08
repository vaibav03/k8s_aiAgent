"""SRE decision engine: one call every ~15 s per deployment -> state, replicas, remediation.

    from kubeai_ml.scaling.engine import DecisionEngine
    eng = DecisionEngine()                      # models/scaling/LATEST
    out = eng.decide(deployment_obs, pod_obs, current_replicas=3, min_replicas=1, max_replicas=10)

The engine chains the bundle's models:

1. ``features.features_at`` on the recent history -> state features (train == serve);
2. ``StateClassifier`` (runbook fault gates, then the LightGBM load head) -> state, with
   ``state_probs`` calibrated on leave-run-out predictions (P(true state | predicted state));
3. ``LgbmQuantileForecaster`` on the ``rps`` series -> demand p50 / p90 at +60 / +120 / +300 s;
4. ``CapacityModel`` with the measured CPU cost per request -> safe rps per pod and P(SLO breach);
5. ``policy.desired_replicas`` -> replicas (fast up, slow and safe down, "do NOT scale" gate);
6. the OOM-ahead model (``models/LATEST``, optional) -> per-pod memory risk, which can add a
   ``restart_pod`` action on its own.

Input schema
------------
``deployment_obs``: list of dicts, one per scrape (5-15 s apart, oldest first or any order),
covering at least the last 3 min (15-30 min recommended: the CPU-per-request baseline that
detects a bad deploy looks 2-10 min back). Field names, units and the PromQL that produces
them are the loadlab exporter's (``loadlab/runner/promexport.py``; ``$ns`` = namespace,
``app`` = container name, ``$pod`` = the deployment's pod regex):

==============================  =======  =========================================================
field                           unit     PromQL / source
==============================  =======  =========================================================
ts                     (req.)   unix s   evaluation timestamp
rps                    (req.)   req/s    sum(rate(http_requests_total{namespace="$ns"}[20s]))
p95                    (req.)   s        histogram_quantile(0.95, sum by (le)(rate(http_request_duration_seconds_bucket{namespace="$ns"}[20s])))
error_rate             (req.)   0-1      sum(rate(http_requests_total{namespace="$ns",code=~"5.."}[20s])) / sum(rate(http_requests_total{namespace="$ns"}[20s]))
cpu_cores_used         (req.)   cores    sum(max by (pod)(rate(container_cpu_usage_seconds_total{namespace="$ns",pod=~"$pod",scope="pod"}[45s]) or process CPU))
replicas_ready         (req.)   pods     max(kube_deployment_status_replicas_ready{namespace="$ns",deployment="$deploy"})
goodput_rps                     req/s    rps * (1 - error_rate)  (derived when absent)
inflight                        req      sum(http_inflight_requests{namespace="$ns"})
p50, p99                        s        histogram_quantile(0.50 / 0.99, ...same as p95)
cpu_util_vs_request             0-x      cpu_cores_used / sum(cpu requests of Running pods)  (derived from cpu_request_cores when absent)
throttle_ratio                  0-1      sum(rate(container_cpu_cfs_throttled_periods_total{...}[45s])) / sum(rate(container_cpu_cfs_periods_total{...}[45s]))
mem_ws_max_pod_frac_of_limit    0-1      max(max by (pod)(container_memory_working_set_bytes{...}) / on(pod) group_left() max by (pod)(kube_pod_container_resource_limits{resource="memory"}))
mem_growth_bytes_per_s          B/s      max(deriv(container_memory_working_set_bytes{...}[60s]) and on(pod) (container_memory_working_set_bytes{...} offset 90s))
replicas_desired                pods     max(kube_deployment_spec_replicas{namespace="$ns",deployment="$deploy"})
restarts_total                  count    sum(kube_pod_container_status_restarts_total{namespace="$ns",container="app"})
oomkills_total                  count    restarts whose kube_pod_container_status_last_terminated_reason{reason="OOMKilled"} is set (cumulative)
crashloop_pods                  pods     count(kube_pod_container_status_waiting_reason{reason="CrashLoopBackOff",container="app"} == 1)
hpa_desired                     pods     max(kube_horizontalpodautoscaler_status_desired_replicas{namespace="$ns"})  (replicas_desired when absent)
pods_not_ready                  pods     pods present but kube_pod_status_ready{condition="true"} == 0 (0 when absent)
revision                        int      deployment.kubernetes.io/revision annotation (changes only on a
                                         pod-template rollout; NOT metadata.generation, which also moves
                                         on a scale). Rollout evidence when pod_obs is absent.
==============================  =======  =========================================================

``pod_obs`` (optional, strongly recommended: without it noisy_pod cannot be detected and
bad_deploy loses its rollout check): list of dicts, one per (ts, pod):

==============  =======  ==============================================================================
field           unit     PromQL / source
==============  =======  ==============================================================================
ts, pod                  timestamp, pod name (``<deploy>-<replicaset hash>-<suffix>``: the hash marks rollouts)
cpu             cores    max by (pod)(cAdvisor pod cgroup rate[45s], process CPU rate[20s])
throttle        0-1      per-pod throttled / total CFS periods rate[45s]
mem_ws          bytes    max by (pod)(container_memory_working_set_bytes{...})
mem_frac_limit  0-1      mem_ws / memory limit
rps             req/s    sum by (pod)(rate(http_requests_total[20s]))
ready           0/1      max by (pod)(kube_pod_status_ready{condition="true"})
restarts        count    max by (pod)(kube_pod_container_status_restarts_total)  (optional)
==============  =======  ==============================================================================

Missing optional fields are filled (NaN, or the derivation above); the reasons list says which
inputs were missing. Output: see :meth:`DecisionEngine.decide`.

Safety interlocks (review 3)
----------------------------
* Missing data is unknown, not zero: rows whose ``rps`` is NaN are dropped from demand and from
  the forecast. If the newest valid ``rps`` is older than ``STALE_S`` (``now_ts`` when given,
  else the newest row) or the last ``GAP_LOOKBACK_S`` contain a gap over ``GAP_S``, scale-down is
  frozen, the forecast is ignored and the reasons say "insufficient telemetry".
* Outage (0 ready pods, or >= 50 % 5xx on the last two samples): a critical page first,
  whatever the classifier says; scale-down frozen.
* Pod restarts: at most one per ``RESTART_COOLDOWN_S`` per deployment, never while ready <
  desired, a surge ``rollout_restart`` instead of ``delete_pod`` when only one pod is ready, and
  a page instead of a restart when no concrete pod can be named. The engine is open loop: an
  emitted restart is assumed to be applied.
* Rollback only with rollout evidence (a new ReplicaSet hash in ``pod_obs`` or a ``revision``
  change within 15 min); otherwise bad_deploy is reported as a cost-per-request regression and
  crashloop pages. A crashloop with OOMKills is routed to the memory remediation.
* The leaking pod is the OOM-ahead model's pick only when that model flags it (anomaly or
  p_oom >= ``OOM_POD_MIN_P``); otherwise the pod whose working set grows fastest.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from . import states as S
from .bundle import Bundle
from .classifier import capacity_features
from .data import SLO_ERR, SLO_P95_S
from .features import FEATURES, build_features
from .forecast import HORIZONS_S, forecast_from_samples
from .policy import MODES, Bounds, ScalerMemory, desired_replicas, fault_aware_replicas

REQUIRED_DEPLOY = ("ts", "rps", "p95", "error_rate", "cpu_cores_used", "replicas_ready")
OPTIONAL_DEPLOY = ("goodput_rps", "inflight", "p50", "p99", "cpu_util_vs_request", "throttle_ratio",
                   "mem_ws_max_pod_frac_of_limit", "mem_growth_bytes_per_s", "replicas_desired",
                   "restarts_total", "oomkills_total", "crashloop_pods", "hpa_desired",
                   "pods_not_ready", "revision")
POD_FIELDS = ("ts", "pod", "cpu", "throttle", "mem_ws", "mem_frac_limit", "rps", "ready")
CPR_WINDOW_S = 60.0
CPR_BASELINE = (600.0, 120.0)        # healthy cpu/request baseline: median over [t-600, t-120]
OOM_SAMPLES_MIN = 6
FORECAST_SKIP_S = 30.0               # the first 30 s of the supplied history are not forecast input
STALE_S = 30.0                       # newest valid rps older than 2 scrape intervals -> stale
GAP_S = 60.0                         # a hole this long in the recent history -> unreliable
GAP_LOOKBACK_S = 300.0
RESTART_COOLDOWN_S = 300.0           # one pod restart per deployment per 5 min
OOM_POD_MIN_P = 0.5                  # trust the OOM-ahead model's pod pick above this p_oom
LEAK_POD_MIN_GROWTH = 10e3           # B/s: below this no pod is "the leaking one"
ROLLOUT_RECENT_S = 900.0
# a new ReplicaSet seen within this long after a rollout the caller itself triggered
# (agent_rollout_ts: its own rollout undo / rollout restart) is that rollout, not evidence of
# an external deploy (otherwise the agent's own undo re-arms "rollback" for 15 min)
AGENT_ROLLOUT_ATTRIB_S = 180.0
TREND_NOW_S, TREND_BASE_S = (30.0, (120.0, 60.0))

_EVIDENCE: dict[str, tuple[tuple[str, str, float], ...]] = {
    # state -> (feature, label, scale)
    S.MEMORY_LEAK: (("mem_growth_med_1m", "working-set growth KB/s (1 m median)", 1.0),
                    ("oomkills_delta_3m", "OOMKills in 3 m", 1.0), ("mem_max_frac", "max pod memory / limit", 1.0)),
    S.DEPENDENCY_LATENCY: (("p95_slo_min_30s", "p95 / SLO (30 s min)", 1.0), ("throttle_1m", "CPU throttle (1 m)", 1.0),
                           ("cpu_frac_limit_1m", "CPU / limit (1 m)", 1.0)),
    S.ERROR_FAULT: (("error_rate_1m", "5xx rate (1 m)", 1.0), ("p95_slo_1m", "p95 / SLO (1 m)", 1.0)),
    S.NOISY_POD: (("pod_cpr_disp_1m", "max / median CPU per request across pods", 1.0),
                  ("pod_cpu_disp_1m", "max - median pod CPU (cores)", 1.0)),
    S.BAD_DEPLOY: (("cpr_ratio", "CPU per request vs pre-rollout baseline", 1.0),
                   ("secs_since_rollout", "seconds since rollout", 1.0)),
    S.CRASHLOOP: (("crashloop_pods", "CrashLoopBackOff pods", 1.0), ("restarts_delta_3m", "restarts in 3 m", 1.0)),
    S.POD_RESTART: (("ready_drop_1m", "ready pods lost in 1 m", 1.0), ("ready_deficit", "ready below desired", 1.0)),
    S.SCALE_UP: (("load_frac", "load / learned knee", 1.0), ("p95_slo", "p95 / SLO", 1.0), ("throttle", "CPU throttle", 1.0)),
    S.SCALE_DOWN: (("down_fit", "3 m peak load / capacity of one pod fewer", 1.0),),
    S.HEALTHY: (("load_frac", "load / learned knee", 1.0),),
}


def _frame(rows: pd.DataFrame | Iterable[Mapping[str, Any]] | None) -> pd.DataFrame | None:
    if rows is None:
        return None
    df = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))
    return df if len(df) else None


@dataclass
class DecisionEngine:
    """Stateful per process: keeps the scale-down stabilisation memory per deployment ``key``."""

    bundle_dir: str | Path | None = None
    oom_model_dir: str | Path | None = "auto"
    policy_mode: str | None = None      # predictive | fault_aware | hpa_gated; None = the bundle's
    bundle: Bundle = field(init=False)
    _memory: dict[str, ScalerMemory] = field(init=False, default_factory=dict)
    _oom: Any = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.bundle = Bundle.load(self.bundle_dir)
        self._oom = self._load_oom(self.oom_model_dir)
        if self.policy_mode is not None and self.policy_mode not in MODES:
            raise ValueError(f"policy_mode must be one of {MODES}")

    @property
    def mode(self) -> str:
        """Replica policy in use (``policy_mode`` if set, else the bundle's ``stabilization.mode``)."""
        return self.policy_mode or getattr(self.bundle.policy.stabilization, "mode", "predictive")

    @staticmethod
    def _load_oom(model_dir: str | Path | None) -> Any:
        if model_dir is None:
            return None
        try:
            from ..model import OomModel
            if model_dir == "auto":
                from .data import ML_ROOT
                model_dir = ML_ROOT / "models" / "LATEST"
            return OomModel.load(model_dir)
        except Exception:
            return None

    @property
    def version(self) -> str:
        return str(self.bundle.manifest.get("version", self.bundle.path.name))

    def reset(self, key: str | None = None) -> None:
        if key is None:
            self._memory.clear()
        else:
            self._memory.pop(key, None)

    # ------------------------------------------------------------------ input handling
    def _deploy_frame(self, deployment_obs: Any, current: int) -> tuple[pd.DataFrame, list[str]]:
        d = _frame(deployment_obs)
        if d is None:
            raise ValueError("deployment_obs is empty")
        missing_req = [c for c in REQUIRED_DEPLOY if c not in d]
        if missing_req:
            raise ValueError(f"deployment_obs is missing required fields {missing_req}")
        d = d.sort_values("ts").drop_duplicates("ts", keep="last").reset_index(drop=True)
        d = d[d["ts"] >= d["ts"].iloc[-1] - self.bundle.policy.history_s].reset_index(drop=True)
        missing = [c for c in OPTIONAL_DEPLOY if c not in d]
        num = lambda c: pd.to_numeric(d[c], errors="coerce")  # noqa: E731
        for c in REQUIRED_DEPLOY:
            d[c] = num(c)
        if "goodput_rps" not in d:
            d["goodput_rps"] = d["rps"] * (1 - d["error_rate"].fillna(0))
        if "replicas_desired" not in d:
            d["replicas_desired"] = float(current)
        if "hpa_desired" not in d:
            d["hpa_desired"] = d["replicas_desired"]
        if "cpu_util_vs_request" not in d:
            d["cpu_util_vs_request"] = d["cpu_cores_used"] / (
                d["replicas_ready"].clip(lower=1) * self.bundle.policy.cpu_request_cores)
        for c in ("pods_not_ready", "crashloop_pods"):
            if c not in d:
                d[c] = 0.0
        for c in OPTIONAL_DEPLOY:
            if c not in d:
                d[c] = np.nan
            d[c] = num(c)
        return d, missing

    @staticmethod
    def _pod_frame(pod_obs: Any) -> pd.DataFrame | None:
        p = _frame(pod_obs)
        if p is None:
            return None
        for c in POD_FIELDS:
            if c not in p:
                p[c] = np.nan if c not in ("pod", "ready") else (1.0 if c == "ready" else "")
        return p

    # ------------------------------------------------------------------ helpers
    def _cpu_per_req(self, d: pd.DataFrame) -> tuple[float, float, str]:
        """(current, healthy-baseline) CPU seconds per successful request, and the source."""
        ts = d["ts"].to_numpy(float)
        g = d["goodput_rps"].to_numpy(float)
        ok = (g > 0.5 * d["replicas_ready"].clip(lower=1).to_numpy(float)) & (d["error_rate"].to_numpy(float) < SLO_ERR)
        ratio = np.where(ok, d["cpu_cores_used"].to_numpy(float) / np.maximum(g, 1e-9), np.nan)
        now = ts[-1]
        recent = ratio[(ts >= now - CPR_WINDOW_S) & np.isfinite(ratio)]
        src = "measured (60 s median)"
        if len(recent):
            cur = float(np.median(recent))
        else:
            valid = np.flatnonzero(np.isfinite(ratio))
            if len(valid):
                cur, src = float(np.median(ratio[valid[-12:]])), "last valid measurement (no clean traffic in 60 s)"
            else:
                cur, src = self.bundle.policy.default_cpu_per_req_s, "bundle default (no clean traffic yet)"
        lo, hi = now - CPR_BASELINE[0], now - CPR_BASELINE[1]
        base_vals = ratio[(ts >= lo) & (ts <= hi) & np.isfinite(ratio)]
        base = float(np.median(base_vals)) if len(base_vals) >= 3 else cur
        return cur, base, src

    def _state_probs(self, x: pd.DataFrame, state: str, gate_fired: bool) -> dict[str, float]:
        cal = dict(self.bundle.calibration.get(state, {}))
        if not cal:
            cal = {state: 1.0}
        clf = self.bundle.classifier
        head = clf.models.get("load")
        if not gate_fired and head is not None and clf.mode == "hybrid":
            from .classifier import MODEL_FEATURES
            p = np.asarray(head.booster.predict(x[MODEL_FEATURES]))[0]
            mass = sum(cal.get(s, 0.0) for s in S.LOAD_STATES) or 1.0
            for s, v in zip(head.classes_, p):
                cal[str(s)] = float(mass * v)
        tot = sum(cal.values()) or 1.0
        probs = {s: round(cal.get(s, 0.0) / tot, 4) for s in S.STATES}
        return probs

    def _oom_risk(self, pods: pd.DataFrame | None) -> list[dict[str, Any]]:
        if self._oom is None or pods is None or "mem_ws" not in pods:
            return []
        out = []
        for pod, g in pods.groupby("pod"):
            g = g.dropna(subset=["mem_ws"]).sort_values("ts")
            if len(g) < OOM_SAMPLES_MIN or not str(pod):
                continue
            frac = g["mem_frac_limit"].to_numpy(float)
            limit = float(np.nanmedian(g["mem_ws"].to_numpy(float) / frac)) if np.isfinite(frac).any() else None
            try:
                r = self._oom.predict(str(pod), "", [{"ts": float(t), "working_set_bytes": float(v)}
                                                      for t, v in zip(g["ts"], g["mem_ws"])],
                                      limit_bytes=limit if limit and math.isfinite(limit) else None)
            except Exception:
                continue
            out.append({"pod": str(pod), "p_oom": round(float(r["p_oom"]), 3), "is_anomaly": bool(r["is_anomaly"]),
                        "time_to_limit_s": r["time_to_limit_s"], "usage_fraction": r["usage_fraction"]})
        return sorted(out, key=lambda r: -r["p_oom"])

    def _memory_for(self, key: str, d: pd.DataFrame, cpu: float, max_r: int) -> ScalerMemory:
        """Per-deployment stabilisation memory; warmed from history on the first call so a
        restarted engine does not wait a full window (needs from observed demand)."""
        mem = self._memory.get(key)
        if mem is not None:
            return mem
        mem = ScalerMemory()
        st = self.bundle.policy.stabilization
        mode = self.mode
        window = st.down_window_s if mode == "predictive" else st.fa_down_window_s
        ts = d["ts"].to_numpy(float)
        now = ts[-1]
        if now - ts[0] >= window:
            mem.first_ts = now - window
            period = self.bundle.policy.decision_period_s
            for t in np.arange(now - window + period, now - 1e-6, period):
                w = d[(d["ts"] > t - period) & (d["ts"] <= t)]
                peak = float(w["rps"].max()) if len(w) else float("nan")
                if not np.isfinite(peak):
                    continue
                if mode == "predictive":
                    rec = self.bundle.capacity.replicas_needed(peak, cpu, max_replicas=max_r)
                else:   # v2: the same max() of traffic and compute the live policy records
                    recs = []
                    if mode == "fault_aware":
                        recs.append(self.bundle.capacity.replicas_needed(peak * st.fa_headroom, cpu,
                                                                         max_replicas=max_r))
                    util = float(w["cpu_util_vs_request"].max())
                    rdy = float(w["replicas_ready"].iloc[-1])
                    if (mode == "hpa_gated" or st.fa_use_cpu) and np.isfinite(util) and np.isfinite(rdy) and rdy >= 1:
                        recs.append(max(1, math.ceil(rdy * util / st.fa_cpu_target)))
                    if not recs:
                        continue
                    rec = min(max(recs), max_r)
                mem.recs.append((float(t), rec))
        self._memory[key] = mem
        return mem

    @staticmethod
    def _telemetry(d: pd.DataFrame, eval_ts: float) -> tuple[np.ndarray, list[str]]:
        """Mask of rows with a valid demand sample, and the telemetry problems (empty = ok)."""
        ts = d["ts"].to_numpy(float)
        valid = np.isfinite(d["rps"].to_numpy(float))
        problems: list[str] = []
        if not valid.any():
            return valid, ["no valid rps sample"]
        age = eval_ts - ts[valid][-1]
        if age > STALE_S:
            problems.append(f"newest valid rps sample is {age:.0f}s old")
        recent = ts[valid][ts[valid] >= eval_ts - GAP_LOOKBACK_S]
        edges = np.concatenate([recent, [eval_ts]]) if len(recent) else np.array([eval_ts])
        gap = float(np.max(np.diff(edges))) if len(edges) > 1 else 0.0
        if gap > GAP_S:
            problems.append(f"{gap:.0f}s gap in the last {GAP_LOOKBACK_S:.0f}s of rps samples")
        for c in ("cpu_cores_used", "replicas_ready"):
            if not np.isfinite(d[c].to_numpy(float)[-1]):
                problems.append(f"{c} missing on the newest sample")
        if not np.isfinite(d["p95"].to_numpy(float)[-1]) and d["rps"].to_numpy(float)[-1] > 1.0:
            problems.append("p95 missing on the newest sample while traffic flows")
        return valid, problems

    @staticmethod
    def _trend_ratio(ts: np.ndarray, rps: np.ndarray, now: float) -> float | None:
        cur = rps[(ts > now - TREND_NOW_S) & (ts <= now)]
        base = rps[(ts >= now - TREND_BASE_S[0]) & (ts <= now - TREND_BASE_S[1])]
        if not len(cur) or not len(base):
            return None
        return float(np.mean(cur) / max(np.mean(base), 0.5))

    @staticmethod
    def _rollout_evidence(d: pd.DataFrame, f: pd.Series,
                          agent_rollout_ts: list[float] | None = None) -> tuple[bool, str]:
        since = f.get("secs_since_rollout", np.nan)
        if np.isfinite(since) and since < ROLLOUT_RECENT_S:
            event = float(d["ts"].iloc[-1]) - float(since)
            own = [a for a in (agent_rollout_ts or []) if a - 15.0 <= event <= a + AGENT_ROLLOUT_ATTRIB_S]
            if not own:
                return True, f"new ReplicaSet {since:.0f}s ago"
            # the newest ReplicaSet is the agent's own rollout; fall through to the revision check
        rev = d[["ts", "revision"]].dropna()
        if len(rev):
            recent = rev[rev["ts"] >= float(d["ts"].iloc[-1]) - ROLLOUT_RECENT_S]
            older = rev[rev["ts"] < float(d["ts"].iloc[-1]) - ROLLOUT_RECENT_S]
            vals = set(recent["revision"]) | (set(older["revision"].iloc[-1:]) if len(older) else set())
            if len(vals) > 1:
                return True, "revision changed within 15 min"
        return False, ("no rollout in pod names or revision" if np.isfinite(since) or len(rev)
                       else "no rollout evidence (no pod_obs, no revision input)")

    @staticmethod
    def _outage(d: pd.DataFrame, current: int) -> str | None:
        last = d.iloc[-1]
        r = float(last["replicas_ready"])
        if np.isfinite(r) and r < 0.5 and current >= 1:
            return "0 ready pods"
        er = d["error_rate"].dropna().to_numpy(float)[-2:]
        if len(er) == 2 and (er >= 0.5).all():
            return f"{100 * er[-1]:.0f}% of requests fail"
        return None

    # ------------------------------------------------------------------ main entry point
    def decide(self, deployment_obs: Any, pod_obs: Any = None, current_replicas: int | None = None,
               min_replicas: int = 1, max_replicas: int = 10, *, key: str = "default",
               deployment: str = "deployment", now_ts: float | None = None,
               agent_rollout_ts: list[float] | None = None) -> dict[str, Any]:
        """Decide for the newest observation.

        Returns ``{state, state_probs, confidence, desired_replicas, replicas_needed,
        forecast {60|120|300: {p50, p90}}, safe_rps_per_pod, slo_risk, actions [{type, target,
        params, reason}], reasons [str], scaling_helps, model_version, inputs}`` where action
        ``type`` is one of scale | restart_pod | rollback | page | none.

        ``agent_rollout_ts`` (optional): unix times of rollouts the caller triggered itself; a
        new ReplicaSet appearing within AGENT_ROLLOUT_ATTRIB_S after one is not rollout evidence.

        ``now_ts`` (optional, unix s): wall-clock time of the decision; stale data is detected
        against it (default: the newest row). ``current_replicas`` above ``max_replicas`` is
        clamped to max at once, as the Kubernetes HPA does (max is a hard operator bound).
        """
        if min_replicas > max_replicas:
            raise ValueError(f"min_replicas {min_replicas} > max_replicas {max_replicas}")
        if min_replicas < 0:
            raise ValueError("min_replicas must be >= 0")
        b = self.bundle
        d, missing = self._deploy_frame(deployment_obs, current_replicas or 1)
        pods = self._pod_frame(pod_obs)
        last = d.iloc[-1]
        ready = int(max(1, round(float(last["replicas_ready"]) if np.isfinite(last["replicas_ready"]) else 1)))
        current = int(current_replicas if current_replicas is not None else
                      (last["replicas_desired"] if np.isfinite(last["replicas_desired"]) else ready))
        last_ts = float(last["ts"])
        eval_ts = float(now_ts) if now_ts is not None else last_ts
        if eval_ts < last_ts:
            raise ValueError("now_ts is older than the newest observation")
        valid, telemetry = self._telemetry(d, eval_ts)
        outage = self._outage(d, current)

        # 1-2: state
        f = build_features(d, pods if pods is not None and len(pods) else None)
        x = capacity_features(f.iloc[[-1]][FEATURES], b.classifier.cap)
        from . import rules
        gate = rules.fault_gate(x)[0] if b.classifier.mode != "lgbm" else None
        state = str(b.classifier.predict(f.iloc[[-1]][FEATURES])[0])
        probs = self._state_probs(x, state, gate is not None)

        # 3: forecast on demand (rps counts 504s and fast 5xx: offered load, not goodput).
        # NaN demand is unknown, not zero: only valid samples feed demand and the forecast.
        fts, frps = d["ts"].to_numpy(float)[valid], d["rps"].to_numpy(float)[valid]
        rps_now = float(frps[-1]) if len(frps) else 0.0
        trend = self._trend_ratio(fts, frps, eval_ts) if len(frps) else None
        keep = fts >= d["ts"].iloc[0] + FORECAST_SKIP_S   # same warm-up rule as training (train_forecast.WARMUP_S)
        if keep.any():
            fts, frps = fts[keep], frps[keep]
        if len(frps):
            fc = forecast_from_samples(b.forecaster, fts, frps, eval_ts)
        else:
            fc = {(h, q): 0.0 for h in HORIZONS_S for q in (0.5, 0.9)}
        forecast = {h: {"p50": round(fc[(h, 0.5)], 3), "p90": round(fc[(h, 0.9)], 3)} for h in HORIZONS_S}

        # 4: capacity
        cpu, cpu_base, cpu_src = self._cpu_per_req(d)
        cap = b.capacity
        safe = cap.safe_rps_per_pod(cpu, ready)
        slo_risk = float(cap.p_breach(rps_now / ready, cpu, ready)[0])
        needed = {"now": cap.replicas_needed(rps_now, cpu, max_replicas=max_replicas)}
        needed |= {h: cap.replicas_needed(forecast[h]["p90"], cpu, max_replicas=max_replicas) for h in HORIZONS_S}

        # 5: replicas
        mem = self._memory_for(key, d, cpu, max_replicas)
        p95_last, err_last = float(last["p95"]), float(last["error_rate"])
        breached = (np.isfinite(p95_last) and p95_last >= SLO_P95_S) or (np.isfinite(err_last) and err_last >= SLO_ERR)
        mode = self.mode
        common = dict(state=state, now_rps=rps_now, cpu_per_req_s=cpu, baseline_cpu_per_req_s=cpu_base,
                      ts=eval_ts, memory=mem, slo_risk=slo_risk, risk_budget=cap.risk, trend_ratio=trend,
                      allow_down=not telemetry and outage is None, use_forecast=not telemetry)
        if mode == "predictive":
            dec = desired_replicas({h: forecast[h]["p90"] for h in HORIZONS_S}, cap, current,
                                   Bounds(min_replicas, max_replicas), b.policy.stabilization,
                                   slo_breached=bool(breached), **common)
        else:
            tail = d.tail(2)
            br2 = ((tail["p95"].fillna(0) >= SLO_P95_S) | (tail["error_rate"].fillna(0) >= SLO_ERR)).to_numpy()
            util_v = d["cpu_util_vs_request"].to_numpy(float)
            util_v = util_v[np.isfinite(util_v)]
            mem_v = d["mem_ws_max_pod_frac_of_limit"].to_numpy(float)[-1:]
            # bad_deploy stop-gap keys on the newest sample (as in v1); the request-quality
            # co-signal needs the last two samples breached (one noisy scrape is not a breach)
            dec = fault_aware_replicas(cap, current, Bounds(min_replicas, max_replicas), b.policy.stabilization,
                                       ready=int(round(float(last["replicas_ready"]))) if np.isfinite(last["replicas_ready"]) else 0,
                                       cpu_util=float(util_v[-1]) if len(util_v) and np.isfinite(last["cpu_cores_used"]) else None,
                                       slo_breached=bool(breached) and (state == S.BAD_DEPLOY or (len(br2) == 2 and br2.all())),
                                       mem_frac=float(mem_v[0]) if len(mem_v) and np.isfinite(mem_v[0]) else None,
                                       forecast_q90={h: forecast[h]["p90"] for h in HORIZONS_S},
                                       mode=mode, **common)

        # 6: memory risk (OOM-ahead model)
        oom = self._oom_risk(pods)

        rollout, rollout_why = self._rollout_evidence(d, f.iloc[-1], agent_rollout_ts)
        raw_ready = float(last["replicas_ready"]) if np.isfinite(last["replicas_ready"]) else float(ready)
        actions, reasons = self._actions(state, dec, current, f.iloc[-1], pods, oom, deployment,
                                         mem=mem, ready=int(round(raw_ready)), now=eval_ts,
                                         rollout=(rollout, rollout_why), outage=outage)
        reasons = [self._state_reason(state, probs, x.iloc[0])] + reasons
        if telemetry:
            reasons.insert(0, "insufficient telemetry: " + "; ".join(telemetry)
                           + " (scale-down frozen, forecast ignored, demand from the last valid sample)")
        reasons.append(f"demand {rps_now:.1f} rps now; p90 forecast " + ", ".join(
            f"+{h}s {forecast[h]['p90']:.1f}" for h in HORIZONS_S) + " rps")
        reasons.append(f"capacity: {safe:.1f} rps/pod safe at {ready} ready pods "
                       f"(CPU {cpu * 1000:.1f} ms/request, {cpu_src}; risk budget {cap.risk:.2f}); "
                       f"P(SLO breach now) = {slo_risk:.2f}")
        if missing:
            reasons.append("missing optional inputs (filled): " + ", ".join(missing))
        if pods is None:
            reasons.append("no pod_obs: noisy_pod undetectable, bad_deploy without rollout check")
        return {
            "state": state,
            "state_probs": probs,
            "confidence": probs.get(state, 0.0),
            "scaling_helps": S.SCALING_HELPS[state] or (dec.action == "scale_up"),
            "desired_replicas": int(dec.replicas),
            "replicas_needed": needed,
            "forecast": forecast,
            "safe_rps_per_pod": round(float(safe), 3),
            "slo_risk": round(slo_risk, 4),
            "cpu_per_req_s": round(cpu, 5),
            "actions": actions,
            "reasons": reasons,
            "model_version": self.version,
            "telemetry_ok": not telemetry,
            "policy": {"mode": mode, "action": dec.action,
                       "signals": {k: (v if v is None or isinstance(v, (str, float)) else int(v))
                                   for k, v in dict(dec.signals).items()}},
            "inputs": {"deploy_rows": int(len(d)), "history_s": float(d["ts"].iloc[-1] - d["ts"].iloc[0]),
                       "pod_rows": int(len(pods)) if pods is not None else 0, "current_replicas": current,
                       "ready_replicas": ready},
        }

    # ------------------------------------------------------------------ explanation
    @staticmethod
    def _state_reason(state: str, probs: dict[str, float], x: pd.Series) -> str:
        ev = []
        for feat, label, _ in _EVIDENCE.get(state, ()):
            v = x.get(feat, np.nan)
            if v is not None and np.isfinite(v):
                ev.append(f"{label} {v:.2f}")
        return f"state {state} (p={probs.get(state, 0):.2f})" + (": " + "; ".join(ev) if ev else "")

    @staticmethod
    def _hot_pod(pods: pd.DataFrame | None, kind: str) -> str | None:
        if pods is None or not len(pods):
            return None
        snap = pods[pods["ts"] == pods["ts"].max()]
        snap = snap[snap["ready"].fillna(1) >= 1]
        if not len(snap):
            return None
        if kind == "cpr":
            score = snap["cpu"] / snap["rps"].clip(lower=1.0)
        else:
            score = snap["mem_frac_limit"]
        if score.isna().all():
            return None
        name = str(snap.loc[score.idxmax(), "pod"])
        return name if name and name != "nan" else None

    @classmethod
    def _leaking_pod(cls, pods: pd.DataFrame | None, oom: list[dict[str, Any]]) -> tuple[str | None, str]:
        """The OOM-ahead model's pod when it flags one, else the fastest-growing working set."""
        for r in oom:
            if r["is_anomaly"] or r["p_oom"] >= OOM_POD_MIN_P:
                return r["pod"], f"OOM-ahead model p_oom {r['p_oom']:.2f}"
        if pods is None or not len(pods):
            return None, "no pod_obs"
        now = float(pods["ts"].max())
        live = set(pods.loc[(pods["ts"] == now) & (pods["ready"].fillna(1) >= 1), "pod"].astype(str))
        best, best_g = None, -np.inf
        for pod, g in pods[pods["ts"] >= now - 120.0].groupby("pod"):
            g = g.dropna(subset=["mem_ws"])
            if str(pod) not in live or len(g) < 4 or g["ts"].nunique() < 4:
                continue
            slope = float(np.polyfit(g["ts"].to_numpy(float) - now, g["mem_ws"].to_numpy(float), 1)[0])
            if slope > best_g:
                best, best_g = str(pod), slope
        if best is not None and best_g >= LEAK_POD_MIN_GROWTH:
            return best, f"working set +{best_g / 1e3:.0f} KB/s (fastest of the pods)"
        pod = cls._hot_pod(pods, "mem")
        return pod, "highest memory / limit" if pod else "no pod identified"

    def _restart(self, pod: str | None, why: str, *, mem: Any, ready: int, current: int, now: float,
                 deployment: str) -> tuple[dict[str, Any] | None, str]:
        """A restart action that respects the interlocks, or (None | page, reason)."""
        if pod is None:
            return ({"type": "page", "target": "service owner", "params": {"severity": "high"},
                     "reason": f"{why}; cannot name the pod, so no blind delete"}, "restart skipped: no pod identified")
        if now - mem.last_restart_ts < RESTART_COOLDOWN_S:
            return None, f"restart skipped: cooldown ({now - mem.last_restart_ts:.0f}s since the last restart)"
        if ready < current:
            return None, f"restart skipped: only {ready}/{current} pods ready"
        mem.last_restart_ts = now
        if ready <= 1:
            return ({"type": "restart_pod", "target": deployment,
                     "params": {"mode": "rollout_restart", "max_surge": 1, "max_unavailable": 0, "pod": pod},
                     "reason": f"{why}; single ready pod: surge restart so the service stays up"}, "")
        return ({"type": "restart_pod", "target": pod, "params": {"mode": "delete_pod"}, "reason": why}, "")

    def _actions(self, state: str, dec: Any, current: int, f: pd.Series, pods: pd.DataFrame | None,
                 oom: list[dict[str, Any]], deployment: str, *, mem: Any, ready: int, now: float,
                 rollout: tuple[bool, str], outage: str | None) -> tuple[list[dict[str, Any]], list[str]]:
        acts: list[dict[str, Any]] = []
        reasons: list[str] = []
        if outage:
            acts.append({"type": "page", "target": "on-call", "params": {"severity": "critical"},
                         "reason": f"outage: {outage}"})
        if dec.replicas != current:
            params: dict[str, Any] = {"from": current, "to": int(dec.replicas)}
            if state == S.BAD_DEPLOY and dec.replicas > current:
                params["stop_gap"] = True
            acts.append({"type": "scale", "target": deployment, "params": params, "reason": dec.reason})
        reasons.append(f"policy: {dec.action}: {dec.reason}")
        recent_rollout, rollout_why = rollout
        restart_kw = dict(mem=mem, ready=ready, current=current, now=now, deployment=deployment)

        def restart(pod: str | None, why: str) -> None:
            act, note = self._restart(pod, why, **restart_kw)
            if act is not None:
                acts.append(act)
            if note:
                reasons.append(note)

        oom_rising = float(f.get("oomkills_delta_3m", 0) or 0) >= 1
        if state == S.MEMORY_LEAK:
            pod, how = self._leaking_pod(pods, oom)
            restart(pod, f"working set grows toward the memory limit ({how}); restart before OOM")
            acts.append({"type": "page", "target": "service owner", "params": {"severity": "ticket"},
                         "reason": "memory leak: open a bug; restarts only buy time"})
        elif state == S.NOISY_POD:
            restart(self._hot_pod(pods, "cpr"), "one pod burns far more CPU per request than its peers")
        elif state == S.BAD_DEPLOY:
            if recent_rollout:
                acts.append({"type": "rollback", "target": deployment, "params": {"to": "previous ReplicaSet"},
                             "reason": f"CPU per request x{f.get('cpr_ratio', float('nan')):.2f} the pre-rollout "
                                       f"baseline ({rollout_why})"})
            else:
                acts.append({"type": "page", "target": "service owner", "params": {"severity": "high"},
                             "reason": f"cost-per-request regression x{f.get('cpr_ratio', float('nan')):.2f} with "
                                       f"{rollout_why}: check traffic mix / neighbours / runtime before rolling back"})
        elif state == S.CRASHLOOP:
            if oom_rising:
                acts.append({"type": "page", "target": "service owner", "params": {"severity": "high"},
                             "reason": "pods are OOMKilled in a loop: memory leak or limit too low "
                                       "(raise the limit or fix the leak; a restart will not help)"})
            elif recent_rollout:
                acts.append({"type": "rollback", "target": deployment, "params": {"to": "previous ReplicaSet"},
                             "reason": f"new pods crash-loop after a rollout ({rollout_why})"})
                acts.append({"type": "page", "target": "service owner", "params": {"severity": "high"},
                             "reason": "crashloop after rollout"})
            else:
                acts.append({"type": "page", "target": "service owner", "params": {"severity": "high"},
                             "reason": f"crashloop with {rollout_why}: config, secret or dependency; "
                                       "a rollback would target an unrelated ReplicaSet"})
        elif state == S.ERROR_FAULT:
            if recent_rollout:
                acts.append({"type": "rollback", "target": deployment, "params": {"to": "previous ReplicaSet"},
                             "reason": f"5xx started within 15 min of a rollout ({rollout_why})"})
            acts.append({"type": "page", "target": "service owner", "params": {"severity": "high"},
                         "reason": "5xx not explained by load (latency is fine)"})
        elif state == S.DEPENDENCY_LATENCY:
            acts.append({"type": "page", "target": "owner of the slow dependency", "params": {"severity": "high"},
                         "reason": "p95 over SLO while CPU is idle: a downstream call is slow"})
        if oom and state != S.MEMORY_LEAK:
            for r in oom:
                if r["is_anomaly"]:
                    restart(r["pod"], f"OOM-ahead model: P(OOM within horizon) {r['p_oom']:.2f}")
                    break
        if oom:
            reasons.append("memory risk (OOM-ahead model): " + ", ".join(
                f"{r['pod']} p_oom {r['p_oom']:.2f}" for r in oom[:3]))
        if dec.action == "do_not_scale" or (not S.SCALING_HELPS[state] and state not in S.LOAD_STATES
                                            and dec.replicas <= current):
            acts.append({"type": "none", "target": deployment, "params": {"scaling": "blocked"},
                         "reason": f"do NOT scale: {state} is not a capacity problem"})
        if not acts:
            acts.append({"type": "none", "target": deployment, "params": {}, "reason": dec.reason})
        return acts, reasons
