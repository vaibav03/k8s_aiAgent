"""Pod service model for the replay simulator, calibrated from the loadlab dataset.

A deployment of ``n`` ready pods is modelled as a pooled server with throughput ``n * mu(n)`` RPS
(per-pod ceiling ``mu`` shrinks slightly with ``n``: node CPU contention on the 6-CPU minikube) and
two latency regimes:

* below saturation (offered < capacity) the steady-state p95 is an **empirical curve**
  ``p95 = f(rps_per_pod, n)`` (median p95 of stable, fault-free dataset rows, monotone,
  log-linearly interpolated). This is what makes the latency knee move with replicas
  (20/18/16 RPS per pod for 1/2/3 replicas): it is learned, not hard-coded.
* above saturation a **FIFO fluid queue** builds (cohorts of requests with their arrival time):
  queueing delay = age of the head of the queue; requests older than the 10 s server deadline are
  answered 504 and leave the queue; once the head is within one step of the deadline the pods
  enter **congestion collapse**: CPU stays at the limit but only ``collapse_goodput_frac`` (~0.2)
  of the served work finishes in time, the rest also ends as 504 (fitted on the A/C overload rows).

CPU = served RPS * cost-per-request + idle per pod, capped at the pod limit. Everything here is
fitted by :func:`fit_pod_model`; the defaults are the fitted values (2026-11-08 dataset) so the
simulator runs without re-reading the data.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from .data import CPU_LIMIT_CORES, CPU_REQUEST_CORES

DEADLINE_S = 10.0      # REQUEST_TIMEOUT_S of the app: queue wait beyond this -> 504
BASE_P95_S = 0.030     # unloaded p95 (25 ms CPU + overhead)
SAT_P95_S = 0.60       # p95 at rho -> 1 (capacity probe: 0.83 s at 1.03x capacity over 25 s)


@dataclass
class PodModel:
    cpu_s_per_req: float = 0.0262
    idle_cores_per_pod: float = 0.008
    mu_by_n: dict[int, float] = field(default_factory=lambda: {1: 19.4, 2: 17.6, 3: 17.2})
    # n -> (per-pod rps knots, p95 seconds knots), x < mu(n)
    lat_knots: dict[int, tuple[list[float], list[float]]] = field(default_factory=dict)
    startup_delay_s: float = 10.0       # scale decision -> new pod serving traffic
    collapse_goodput_frac: float = 0.22  # goodput / capacity once the queue head is at the deadline
    deadline_s: float = DEADLINE_S
    cpu_limit: float = CPU_LIMIT_CORES
    cpu_request: float = CPU_REQUEST_CORES

    # ---- capacity -------------------------------------------------------------------------
    def mu(self, n: int) -> float:
        """Per-pod throughput ceiling (RPS) with ``n`` ready pods. Beyond the data, the last
        observed per-pod decrement is extrapolated (conservative), floored at 70 % of mu(1)."""
        if n <= 0:
            return 0.0
        ks = sorted(self.mu_by_n)
        if n in self.mu_by_n:
            return self.mu_by_n[n]
        if n < ks[0]:
            return self.mu_by_n[ks[0]]
        slope = (self.mu_by_n[ks[-1]] - self.mu_by_n[ks[0]]) / max(1, ks[-1] - ks[0])
        return max(0.7 * self.mu_by_n[ks[0]], self.mu_by_n[ks[-1]] + slope * (n - ks[-1]))

    def _knots(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        ks = sorted(self.lat_knots)
        k = n if n in self.lat_knots else (ks[-1] if n > ks[-1] else ks[0])
        x, y = self.lat_knots[k]
        return np.asarray(x, float), np.asarray(y, float)

    def p95_steady(self, rps_per_pod: float, n: int) -> float:
        """Steady-state p95 (s) below saturation. Above the last knot it rises log-linearly to
        SAT_P95_S at rho = 1. n beyond the data uses the largest-n curve."""
        if n <= 0:
            return self.deadline_s
        mu = self.mu(n)
        x, y = self._knots(n)
        xs = np.append(x[x < mu * 0.999], mu)
        ys = np.append(y[x < mu * 0.999], max(SAT_P95_S, y.max()))
        v = float(np.exp(np.interp(min(rps_per_pod, mu), xs, np.log(ys))))
        return v

    def safe_rps_per_pod(self, n: int, p95_budget_s: float) -> float:
        """Largest per-pod RPS whose steady-state p95 stays within the budget (for tuning)."""
        grid = np.linspace(0.1, self.mu(n), 400)
        ok = [g for g in grid if self.p95_steady(g, n) <= p95_budget_s]
        return float(ok[-1]) if ok else 0.0

    def to_json(self) -> str:
        d = asdict(self)
        d["mu_by_n"] = {str(k): v for k, v in self.mu_by_n.items()}
        d["lat_knots"] = {str(k): v for k, v in self.lat_knots.items()}
        return json.dumps(d, indent=1)


# ------------------------------------------------------------------------------------------
# Fitting


def _stable_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Fault-free, QA-ok rows whose ready count has been constant for 30 s."""
    d = df[(df["active_fault"] == "none") & (df["qa_flag"] == "ok")].copy()
    d["_rstd"] = d.groupby("run_id")["replicas_ready"].transform(lambda s: s.rolling(6).std())
    return d[(d["_rstd"] == 0) & (d["replicas_ready"] > 0)]


def fit_cpu_cost(df: pd.DataFrame) -> tuple[float, float]:
    """cores = a * goodput + b on unsaturated fault-free rows -> (a s/req, b idle cores/pod)."""
    d = _stable_rows(df)
    d = d[(d["cpu_util_vs_request"] < 1.6) & (d["goodput_rps"] > 3)]
    a, b = np.polyfit(d["goodput_rps"], d["cpu_cores_used"], 1)
    idle = max(0.0, float(b) / float(d["replicas_ready"].median()))
    return float(a), idle


def fit_mu(df: pd.DataFrame) -> dict[int, float]:
    """Per-pod throughput ceiling: median goodput/pod in saturated rows (in-flight > 4/pod)."""
    d = _stable_rows(df)
    d = d[(d["inflight"] > 4 * d["replicas_ready"]) & (d["error_rate"] < 0.5)]
    g = (d["goodput_rps"] / d["replicas_ready"]).groupby(d["replicas_ready"].astype(int))
    stats = g.agg(["median", "size"])
    return {int(n): round(float(r["median"]), 2) for n, r in stats.iterrows() if r["size"] >= 10}


def fit_latency(df: pd.DataFrame, mu: dict[int, float], bin_rps: float = 2.0, min_rows: int = 4
                ) -> dict[int, tuple[list[float], list[float]]]:
    """Empirical p95 vs per-pod RPS for each replica count, from rows where both replicas and the
    20 s rps have been steady (rolling CV < 6 %), so the 20 s p95 window reflects one load level."""
    d = _stable_rows(df)
    d["_cv"] = d.groupby("run_id")["rps"].transform(lambda s: s.rolling(5).std() / s.rolling(5).mean())
    d = d[(d["_cv"] < 0.06) & (d["error_rate"] < 0.01)]
    d["_x"] = d["goodput_rps"] / d["replicas_ready"]
    d["_bin"] = (d["_x"] / bin_rps).round() * bin_rps
    out: dict[int, tuple[list[float], list[float]]] = {}
    mu_any = max(mu.values()) if mu else 18.0
    for n, g in d.groupby(d["replicas_ready"].astype(int)):
        m = mu.get(int(n), min(mu.values()) if mu else mu_any)
        g = g[g["_x"] < 0.97 * m]
        st = g.groupby("_bin")["p95"].agg(["median", "size"])
        st = st[st["size"] >= min_rows]
        if st.empty:
            continue
        xs = [0.0] + [float(x) for x in st.index]
        ys = [BASE_P95_S] + list(np.maximum.accumulate(np.maximum(st["median"].to_numpy(), BASE_P95_S)))
        out[int(n)] = (xs, [float(v) for v in ys])
    return out


def fit_collapse(df: pd.DataFrame, mu: dict[int, float]) -> float:
    """Goodput as a fraction of capacity under deep overload (p95 at the 10 s deadline, error
    rate > 0.5): FIFO serves requests that are about to time out, so most CPU is wasted on
    requests that end as 504 (a-cap-fixed1: CPU at limit, ~2 of 19 RPS good)."""
    d = df[(df["active_fault"] == "none") & (df["error_rate"] > 0.5) & (df["p95"] >= 9.9)
           & (df["replicas_ready"] > 0)]
    m = d["replicas_ready"].astype(int).map(lambda n: mu.get(n, min(mu.values())))
    frac = d["goodput_rps"] / (d["replicas_ready"] * m)
    return float(frac.median()) if len(frac) else 0.22


def measure_startup_delay(df: pd.DataFrame, step_s: float = 5.0) -> dict:
    """Time from a replicas_desired increase to replicas_ready reaching it (kube-state-metrics,
    5 s resolution). The pod-level view (pods.csv) adds ~5 s until the pod takes traffic."""
    lags: list[float] = []
    for _, g in df.groupby("run_id"):
        g = g.sort_values("ts").reset_index(drop=True)
        des, rdy, t = g["replicas_desired"].to_numpy(), g["replicas_ready"].to_numpy(), g["ts"].to_numpy()
        for i in range(1, len(g)):
            if des[i] > des[i - 1]:
                j = i
                while j < len(g) and rdy[j] < des[i]:
                    j += 1
                if j < len(g):
                    lags.append(float(t[j] - t[i]))
    arr = np.asarray(lags)
    return {"n": int(arr.size), "median_s": float(np.median(arr)), "p90_s": float(np.quantile(arr, 0.9)),
            "max_s": float(arr.max()), "values": [float(x) for x in arr]}


def fit_pod_model(df: pd.DataFrame) -> tuple[PodModel, dict]:
    """Fit every PodModel parameter from dataset_deploy rows (+ the runs' events/pods.csv for the
    startup delay); returns (model, fit report)."""
    from .runlogs import pod_startup_delays

    a, idle = fit_cpu_cost(df)
    mu = fit_mu(df)
    lat = fit_latency(df, mu)
    ksm = measure_startup_delay(df)
    ev = np.asarray(pod_startup_delays(sorted(df["run_id"].unique())))
    # Event -> Ready median (~9 s) is the primary measure; kube-state-metrics desired -> ready at
    # 5 s resolution under-reads it (desired itself shows up late), so it is reported only.
    startup = float(np.median(ev)) + 1.0 if ev.size else ksm["median_s"] + 5.0   # +1 s endpoints
    collapse = fit_collapse(df, mu)
    model = PodModel(cpu_s_per_req=round(a, 5), idle_cores_per_pod=round(idle, 4), mu_by_n=mu,
                     lat_knots=lat, startup_delay_s=round(startup, 1),
                     collapse_goodput_frac=round(collapse, 3))
    return model, {"cpu_s_per_req": a, "idle_cores_per_pod": idle, "mu_by_n": mu,
                   "collapse_goodput_frac": collapse,
                   "startup_ksm": {k: v for k, v in ksm.items() if k != "values"},
                   "startup_event_to_ready": {"n": int(ev.size),
                                              "median_s": float(np.median(ev)) if ev.size else None,
                                              "p90_s": float(np.quantile(ev, .9)) if ev.size else None,
                                              "min_s": float(ev.min()) if ev.size else None,
                                              "max_s": float(ev.max()) if ev.size else None},
                   "startup_delay_s": model.startup_delay_s}
