"""Deployment-level state features, shared by training and serving.

Input: one deployment's time-ordered observations (5-15 s apart) with the loadlab
``dataset_deploy`` column names, plus an optional per-pod snapshot table (``dataset_pods``
columns: ts, pod, cpu, throttle, mem_frac_limit, rps, ready). Output: one feature row per
observation. Training calls :func:`build_features` on a whole run; serving calls
:func:`features_at` on the recent history (>= 15 min is ideal for the cpu-per-request
baseline) and gets the newest row. Every window is time-based (seconds, not row counts), so
the same code works at a 5 s or a 15 s cadence.

Feature groups (names in ``FEATURES``):

* load      demand per ready pod, 1 m mean, 3 m max, 1 m trend; in-flight per pod.
            demand = ``goodput_rps`` (excludes fast 5xx) unless p95 breaches the SLO, then
            ``rps``: under overload goodput *collapses* (deadline 504s are slow errors, not
            served work), so goodput would read "low load" exactly when the pods are drowning.
* efficiency cpu cores per goodput request, its 1 m median and the ratio to a pre-change
            baseline (median over [t-10 m, t-2 m], or the 10 m before the last rollout when
            pod names show one in the last 15 m).  bad_deploy -> ratio ~3.
* SLO       p95/0.3 and p99/0.3 (now, 1 m mean, 1 m max, 30 s min), error rate (now, 1 m, 3 m).
* compute   throttle ratio, CPU vs request, CPU vs limit, latency-vs-cpu decoupling
            (p95 high while the CPU is not throttled: dependency latency).
* memory    growth (KB/s, now and 1/3 m medians), max pod fraction of the limit and its
            3 m delta.  ``mem_growth`` is NaN for a pod's first 90 s, so the 3 m delta of
            the max fraction is the robust leak signal.
* lifecycle restarts / OOMKills over 1 and 3 m, crashloop pods, not-ready pods, ready
            deficit vs desired, ready drop vs the 1 m max, HPA desired minus ready.
* pods      dispersion across ready pods: max-median CPU, throttle, memory fraction, and
            max/median CPU per request (cpu_hog: one hot pod; bad_deploy: every pod hot),
            number of ReplicaSet hashes, seconds since a new hash appeared (rollout).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

ROLLOUT_CAP_S = 1800.0


@dataclass(frozen=True)
class FeatureConfig:
    slo_p95_s: float = 0.300
    cpu_limit_cores: float = 0.5        # per pod
    mem_limit_bytes: float = 256 * 2**20
    min_goodput_for_cpr: float = 1.0    # rps; below this cpu/request is noise
    baseline_window_s: float = 480.0    # cpu/request baseline: median over [t-10m, t-2m]
    baseline_lag_s: float = 120.0
    rollout_lookback_s: float = 900.0   # anchor the baseline before a rollout this recent


DEFAULT_CONFIG = FeatureConfig()

LOAD = ["replicas_ready", "load_per_pod", "goodput_per_pod", "load_per_pod_1m", "load_per_pod_max_3m",
        "load_trend_1m", "goodput_trend_1m", "inflight_per_pod"]
EFFICIENCY = ["cpu_per_req", "cpu_per_req_1m", "cpr_ratio"]
SLO = ["p95_slo", "p95_slo_1m", "p95_slo_max_1m", "p95_slo_min_30s", "p99_slo", "error_rate", "error_rate_1m",
       "error_rate_3m", "fast_error_share"]
COMPUTE = ["throttle", "throttle_1m", "cpu_util_req", "cpu_frac_limit", "cpu_frac_limit_1m",
           "lat_cpu_decoupling"]
MEMORY = ["mem_growth_kbps", "mem_growth_med_1m", "mem_growth_med_3m", "mem_max_frac",
          "mem_frac_delta_3m"]
LIFECYCLE = ["restarts_delta_1m", "restarts_delta_3m", "oomkills_delta_3m", "crashloop_pods",
             "pods_not_ready", "ready_deficit", "ready_drop_1m", "hpa_minus_ready"]
PODS = ["pod_cpu_disp", "pod_cpu_disp_1m", "pod_cpr_disp", "pod_cpr_disp_1m",
        "pod_throttle_disp", "pod_mem_disp", "n_rs_hashes", "secs_since_rollout"]
FEATURES = LOAD + EFFICIENCY + SLO + COMPUTE + MEMORY + LIFECYCLE + PODS


# --------------------------------------------------------------------------- time helpers
def _rolling(values: pd.Series, ts: np.ndarray, window_s: float, how: str) -> np.ndarray:
    """Trailing time window [t - window_s, t] aggregate (NaNs skipped)."""
    s = pd.Series(np.asarray(values, dtype=float), index=pd.to_datetime(ts, unit="s"))
    r = s.rolling(f"{int(window_s)}s", min_periods=1)
    return getattr(r, how)().to_numpy()


def _lag(values: np.ndarray, ts: np.ndarray, lag_s: float, tol_s: float = 10.0) -> np.ndarray:
    """Value at the last observation at or before t - lag_s (NaN if none within tol)."""
    target = ts - lag_s
    idx = np.searchsorted(ts, target, side="right") - 1
    out = np.full(len(ts), np.nan)
    ok = (idx >= 0)
    ok[ok] &= ts[idx[ok]] >= target[ok] - tol_s
    out[ok] = np.asarray(values, dtype=float)[idx[ok]]
    return out


def _safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.asarray(a, dtype=float) / np.asarray(b, dtype=float)
    out[~np.isfinite(out)] = np.nan
    return out


# --------------------------------------------------------------------------- pod features
def _rs_hash(pod: str) -> str:
    parts = pod.rsplit("-", 2)
    return parts[1] if len(parts) == 3 else pod


def pod_snapshot_features(pods: pd.DataFrame) -> pd.DataFrame:
    """Per-ts dispersion and rollout features from the per-pod table (index: ts)."""
    p = pods.copy()
    p["hash"] = p["pod"].astype(str).map(_rs_hash)
    ready = p[(p["ready"] == 1) & p["cpu"].notna()].copy()
    ready["cpr"] = np.where(ready["rps"] >= 1.0, ready["cpu"] / ready["rps"].clip(lower=1e-9),
                            np.nan)
    g = ready.groupby("ts")

    def disp(col: str) -> pd.Series:
        return g[col].max() - g[col].median()

    out = pd.DataFrame({
        "pod_cpu_disp": disp("cpu"),
        "pod_throttle_disp": disp("throttle"),
        "pod_mem_disp": disp("mem_frac_limit"),
        "pod_cpr_disp": g["cpr"].max() / g["cpr"].median(),
    })
    # Rollout events: a ReplicaSet hash that was absent in the previous snapshot appears.
    hashes = p.groupby("ts")["hash"].agg(lambda h: frozenset(h)).sort_index()
    since = np.full(len(hashes), np.nan)
    last_event: float | None = None
    prev: frozenset[str] | None = None
    for i, (t, hs) in enumerate(hashes.items()):
        if prev is not None and prev and (hs - prev):
            last_event = float(t)
        since[i] = (float(t) - last_event) if last_event is not None else ROLLOUT_CAP_S
        prev = hs
    rollout = pd.DataFrame({"n_rs_hashes": hashes.map(len).astype(float),
                            "secs_since_rollout": np.minimum(since, ROLLOUT_CAP_S)},
                           index=hashes.index)
    return rollout.join(out, how="left")


def demand_rps(d: pd.DataFrame, cfg: FeatureConfig = DEFAULT_CONFIG) -> np.ndarray:
    """Offered-load estimate: goodput, or all responses while latency breaches the SLO."""
    slow = d["p95"].fillna(0).to_numpy(float) >= cfg.slo_p95_s
    return np.where(slow, d["rps"].to_numpy(float), d["goodput_rps"].to_numpy(float))


# --------------------------------------------------------------------------- main builder
def build_features(deploy: pd.DataFrame, pods: pd.DataFrame | None = None,
                   cfg: FeatureConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    """Feature rows for one deployment, aligned with ``deploy`` (same index)."""
    d = deploy.sort_values("ts")
    ts = d["ts"].to_numpy(dtype=float)
    col = lambda c: d[c].to_numpy(dtype=float)  # noqa: E731
    f: dict[str, np.ndarray] = {}

    ready = col("replicas_ready")
    ready_c = np.maximum(ready, 1.0)
    goodput = col("goodput_rps")
    demand = demand_rps(d, cfg)
    f["replicas_ready"] = ready
    f["load_per_pod"] = demand / ready_c
    f["goodput_per_pod"] = goodput / ready_c
    f["load_per_pod_1m"] = _rolling(f["load_per_pod"], ts, 60, "mean")
    f["load_per_pod_max_3m"] = _rolling(f["load_per_pod"], ts, 180, "max")
    f["load_trend_1m"] = f["load_per_pod_1m"] - _lag(f["load_per_pod_1m"], ts, 60)
    gp1 = _rolling(demand, ts, 60, "mean")
    f["goodput_trend_1m"] = _safe_div(gp1, _lag(gp1, ts, 60)) - 1.0
    f["inflight_per_pod"] = col("inflight") / ready_c

    cpu = col("cpu_cores_used")
    cpr = np.where(goodput >= cfg.min_goodput_for_cpr, _safe_div(cpu, goodput), np.nan)
    f["cpu_per_req"] = cpr
    f["cpu_per_req_1m"] = _rolling(cpr, ts, 60, "median")
    base = _lag(_rolling(cpr, ts, cfg.baseline_window_s, "median"), ts, cfg.baseline_lag_s,
                tol_s=cfg.baseline_lag_s)

    p95 = np.nan_to_num(col("p95"), nan=0.0) / cfg.slo_p95_s
    err = col("error_rate")
    f["p95_slo"] = p95
    f["p95_slo_1m"] = _rolling(p95, ts, 60, "mean")
    f["p95_slo_max_1m"] = _rolling(p95, ts, 60, "max")
    f["p95_slo_min_30s"] = _rolling(p95, ts, 30, "min")   # sustained, not a 3 s stall
    f["p99_slo"] = np.nan_to_num(col("p99"), nan=0.0) / cfg.slo_p95_s
    f["error_rate"] = err
    f["error_rate_1m"] = _rolling(err, ts, 60, "mean")
    f["error_rate_3m"] = _rolling(err, ts, 180, "mean")
    # Errors while latency is fine are injected/app 5xx; overload 504s come with p95 ~ 10 s.
    f["fast_error_share"] = f["error_rate_1m"] * (f["p95_slo_1m"] < 1.0)

    thr = col("throttle_ratio")
    f["throttle"] = thr
    f["throttle_1m"] = _rolling(thr, ts, 60, "mean")
    f["cpu_util_req"] = col("cpu_util_vs_request")
    f["cpu_frac_limit"] = cpu / (ready_c * cfg.cpu_limit_cores)
    f["cpu_frac_limit_1m"] = _rolling(f["cpu_frac_limit"], ts, 60, "mean")
    f["lat_cpu_decoupling"] = f["p95_slo_1m"] * (1.0 - np.clip(f["throttle_1m"], 0, 1))

    mg = col("mem_growth_bytes_per_s") / 1024.0
    f["mem_growth_kbps"] = mg
    f["mem_growth_med_1m"] = _rolling(mg, ts, 60, "median")
    f["mem_growth_med_3m"] = _rolling(mg, ts, 180, "median")
    mf = col("mem_ws_max_pod_frac_of_limit")
    f["mem_max_frac"] = mf
    f["mem_frac_delta_3m"] = mf - _lag(mf, ts, 180)

    restarts = col("restarts_total")
    f["restarts_delta_1m"] = restarts - _lag(restarts, ts, 60)
    f["restarts_delta_3m"] = restarts - _lag(restarts, ts, 180)
    f["oomkills_delta_3m"] = col("oomkills_total") - _lag(col("oomkills_total"), ts, 180)
    f["crashloop_pods"] = col("crashloop_pods")
    f["pods_not_ready"] = col("pods_not_ready") if "pods_not_ready" in d else np.full(len(d), np.nan)
    f["ready_deficit"] = np.clip(col("replicas_desired") - ready, 0, None)
    f["ready_drop_1m"] = _rolling(ready, ts, 60, "max") - ready
    f["hpa_minus_ready"] = col("hpa_desired") - ready

    for name in PODS:
        f[name] = np.full(len(d), np.nan)
    if pods is not None and len(pods):
        snap = pod_snapshot_features(pods).reindex(ts)
        for name in ("pod_cpu_disp", "pod_cpr_disp", "pod_throttle_disp", "pod_mem_disp",
                     "n_rs_hashes", "secs_since_rollout"):
            f[name] = snap[name].to_numpy(dtype=float)
        f["pod_cpu_disp_1m"] = _rolling(f["pod_cpu_disp"], ts, 60, "mean")
        f["pod_cpr_disp_1m"] = _rolling(f["pod_cpr_disp"], ts, 60, "median")
        base = _anchor_baseline_at_rollout(base, cpr, ts, f["secs_since_rollout"], cfg)

    f["cpr_ratio"] = _safe_div(f["cpu_per_req_1m"], base)
    return pd.DataFrame(f, index=d.index)[FEATURES]


def _anchor_baseline_at_rollout(base: np.ndarray, cpr: np.ndarray, ts: np.ndarray,
                                since: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    """Within ``rollout_lookback_s`` of a rollout, use the median cpu/request before it."""
    out = base.copy()
    recent = np.isfinite(since) & (since < cfg.rollout_lookback_s)
    for i in np.flatnonzero(recent):
        t_roll = ts[i] - since[i]
        win = (ts < t_roll) & (ts >= t_roll - cfg.baseline_window_s - cfg.baseline_lag_s)
        vals = cpr[win]
        vals = vals[np.isfinite(vals)]
        if len(vals) >= 3:
            out[i] = float(np.median(vals))
    return out


def features_at(deploy_rows: pd.DataFrame | Iterable[Mapping[str, float]],
                pod_rows: pd.DataFrame | Iterable[Mapping[str, object]] | None = None,
                cfg: FeatureConfig = DEFAULT_CONFIG) -> dict[str, float]:
    """Serving entry point: features of the newest observation given recent history."""
    d = deploy_rows if isinstance(deploy_rows, pd.DataFrame) else pd.DataFrame(list(deploy_rows))
    p = None
    if pod_rows is not None:
        p = pod_rows if isinstance(pod_rows, pd.DataFrame) else pd.DataFrame(list(pod_rows))
    feats = build_features(d.reset_index(drop=True), p, cfg)
    last = feats.loc[d.reset_index(drop=True)["ts"].idxmax()]
    return {k: float(v) for k, v in last.items()}
