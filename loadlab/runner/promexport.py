"""Export a lab run's Prometheus data (query_range, 5s step) into deploy.csv and pods.csv.

All PromQL is recording-rule free. cAdvisor rows are POD-level cgroups (see monitoring/prometheus.yaml
for why) and are refreshed by the kubelet every ~10-15s, so cAdvisor-derived rates use a 45s window;
app metrics (scraped every 5s) use 30s.

Usage (standalone re-export):  python promexport.py <run_dir>
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

PROM = os.environ.get("PROM_URL", "http://127.0.0.1:9090")
STEP = 5
APP_W = "20s"   # window for app (5s-scraped) counters (4 scrapes; 30s lagged fault ends by ~30s)
CAD_W = "45s"   # window for cAdvisor counters (kubelet housekeeping ~15s)
# Query version v2 (2026-11-08, after the live A/B review). The committed dataset (round 1 + 2)
# was exported with v1: APP_POD_RE "loadlab-.*", kube-state-metrics pod selectors not
# restricted to the pod regex, hpa_desired namespace-wide, and mem_growth_bytes_per_s without
# the live-target join. In a one-Deployment lab namespace v2 selects the same series except
# mem_growth right after a pod deletion (v1 kept a deleted pod's last 60 s slope in the max).
# lib/promDeploy.js builds the same strings (queryVersion "v1" reproduces the old ones).
QUERY_VERSION = "v2"
APP_POD_RE = "loadlab-[a-z0-9]+-[a-z0-9]{5}"   # RS hash + pod suffix: never a sibling Deployment
POD_SCOPE = f',pod=~"{APP_POD_RE}"'
HPA_NAME = "loadlab"


MEM_WARM = "90s"  # mem growth only for pods with a working-set sample >= this long ago (startup ramp != leak)


def _cpu_by_pod(ns: str) -> str:
    """Per-pod CPU cores = max(cAdvisor pod cgroup, app process CPU).
    The pod cgroup can never truly use less CPU than the app process inside it, so max() keeps cAdvisor
    when it is ahead (it includes child processes, e.g. cpu_hog) and falls back to the 5s-fresh process
    counter where cAdvisor lags or has not picked a new pod up yet (~20-30s on this minikube; its first
    rates for a new pod cover the idle start-up). Restricted to pods that are current scrape targets,
    so a deleted pod's lingering cAdvisor rate() (up to CAD_W after deletion) is not double counted
    during rollouts."""
    a = f'namespace="{ns}"'
    c = f'namespace="{ns}",pod=~"{APP_POD_RE}",scope="pod"'
    cad = f'sum by (pod)(rate(container_cpu_usage_seconds_total{{{c}}}[{CAD_W}]))'
    proc = f'sum by (pod)(rate(process_cpu_seconds_total{{{a},job="annotated-pods"}}[{APP_W}]))'
    live = f'group by (pod)(up{{{a},job="annotated-pods"}})'
    both = (f'label_replace({cad}, "src", "cadvisor", "", "") '
            f'or label_replace({proc}, "src", "process", "", "")')
    return f'max by (pod)({both}) and on(pod) {live}'


def deploy_queries(ns: str) -> dict[str, str]:
    a = f'namespace="{ns}"'
    c = f'namespace="{ns}",pod=~"{APP_POD_RE}",scope="pod"'
    k = f'namespace="{ns}",container="app"{POD_SCOPE}'
    cpu = _cpu_by_pod(ns)
    req_running = (f'sum(kube_pod_container_resource_requests{{{k},resource="cpu"}} '
                   f'* on(namespace,pod) group_left() (kube_pod_status_phase{{{a}{POD_SCOPE},phase="Running"}} == bool 1))')
    q = {
        "rps": f'sum(rate(http_requests_total{{{a}}}[{APP_W}]))',
        "inflight": f'sum(http_inflight_requests{{{a}}})',
        "p50": f'histogram_quantile(0.50, sum by (le)(rate(http_request_duration_seconds_bucket{{{a}}}[{APP_W}])))',
        "p95": f'histogram_quantile(0.95, sum by (le)(rate(http_request_duration_seconds_bucket{{{a}}}[{APP_W}])))',
        "p99": f'histogram_quantile(0.99, sum by (le)(rate(http_request_duration_seconds_bucket{{{a}}}[{APP_W}])))',
        "error_rate": (f'sum(rate(http_requests_total{{{a},code=~"5.."}}[{APP_W}])) '
                       f'/ sum(rate(http_requests_total{{{a}}}[{APP_W}]))'),
        "cpu_cores_used": f'sum({cpu})',
        "cpu_util_vs_request": f'sum({cpu}) / {req_running}',
        "throttle_ratio": (f'sum(rate(container_cpu_cfs_throttled_periods_total{{{c}}}[{CAD_W}])) '
                           f'/ sum(rate(container_cpu_cfs_periods_total{{{c}}}[{CAD_W}]))'),
        "mem_ws_bytes_total": f'sum(container_memory_working_set_bytes{{{c}}})',
        "mem_ws_max_pod_frac_of_limit": (
            f'max(max by (pod)(container_memory_working_set_bytes{{{c}}}) '
            f'/ on(pod) group_left() max by (pod)(kube_pod_container_resource_limits{{{k},resource="memory"}}))'),
        # live scrape targets only: a range selector keeps returning a deleted pod's last 60 s
        "mem_growth_bytes_per_s": (f'max((deriv(container_memory_working_set_bytes{{{c}}}[60s]) '
                                   f'and on(pod) (container_memory_working_set_bytes{{{c}}} offset {MEM_WARM})) '
                                   f'and on(pod) group by (pod)(up{{{a},job="annotated-pods"}}))'),
        "replicas_desired": f'max(kube_deployment_spec_replicas{{{a},deployment="loadlab"}})',
        "replicas_ready": f'max(kube_deployment_status_replicas_ready{{{a},deployment="loadlab"}})',
        "restarts_total": f'sum(kube_pod_container_status_restarts_total{{{k}}})',
        "crashloop_pods": f'count(kube_pod_container_status_waiting_reason{{{k},reason="CrashLoopBackOff"}} == 1)',
        "hpa_desired": f'max(kube_horizontalpodautoscaler_status_desired_replicas{{{a},horizontalpodautoscaler="{HPA_NAME}"}})',
        # extra (appended) column: the app's own CPU (prometheus_client process collector, 5s
        # resolution, main process only - excludes the cpu_hog child). Cross-check for cAdvisor.
        "cpu_cores_proc": f'sum(rate(process_cpu_seconds_total{{{a},job="annotated-pods"}}[{APP_W}]))',
        # extra: live pods with no cAdvisor CPU yet (their CPU comes only from the process counter)
        "cpu_fallback_pods": (f'count(sum by (pod)(rate(process_cpu_seconds_total{{{a},job="annotated-pods"}}[{APP_W}])) '
                              f'unless on(pod) sum by (pod)(rate(container_cpu_usage_seconds_total{{{c}}}[{CAD_W}])))'),
    }
    return q


def pod_queries(ns: str) -> dict[str, str]:
    a = f'namespace="{ns}"'
    c = f'namespace="{ns}",pod=~"{APP_POD_RE}",scope="pod"'
    k = f'namespace="{ns}",container="app"{POD_SCOPE}'
    return {
        "cpu": _cpu_by_pod(ns),
        "throttle": (f'sum by (pod)(rate(container_cpu_cfs_throttled_periods_total{{{c}}}[{CAD_W}])) '
                     f'/ sum by (pod)(rate(container_cpu_cfs_periods_total{{{c}}}[{CAD_W}]))'),
        "mem_ws": f'max by (pod)(container_memory_working_set_bytes{{{c}}})',
        "mem_frac_limit": (f'max by (pod)(container_memory_working_set_bytes{{{c}}}) '
                           f'/ on(pod) group_left() max by (pod)(kube_pod_container_resource_limits{{{k},resource="memory"}})'),
        "restarts": f'max by (pod)(kube_pod_container_status_restarts_total{{{k}}})',
        "rps": f'sum by (pod)(rate(http_requests_total{{{a}}}[{APP_W}]))',
        "p95": f'histogram_quantile(0.95, sum by (pod, le)(rate(http_request_duration_seconds_bucket{{{a}}}[{APP_W}])))',
        "errors": f'sum by (pod)(rate(http_requests_total{{{a},code=~"5.."}}[{APP_W}]))',
        "ready": f'max by (pod)(kube_pod_status_ready{{{a},condition="true",pod=~"{APP_POD_RE}"}})',
        "cpu_proc": f'sum by (pod)(rate(process_cpu_seconds_total{{{a},job="annotated-pods"}}[{APP_W}]))',
        "_oom_last": f'max by (pod)(kube_pod_container_status_last_terminated_reason{{{k},reason="OOMKilled"}})',
    }


def prom_range(query: str, start: float, end: float, step: int = STEP) -> list[dict]:
    r = requests.get(f"{PROM}/api/v1/query_range",
                     params={"query": query, "start": start, "end": end, "step": step}, timeout=60)
    r.raise_for_status()
    js = r.json()
    if js.get("status") != "success":
        raise RuntimeError(f"prom error for {query}: {js}")
    return js["data"]["result"]


def prom_instant(query: str, t: float | None = None) -> list[dict]:
    params = {"query": query}
    if t is not None:
        params["time"] = t
    r = requests.get(f"{PROM}/api/v1/query", params=params, timeout=60)
    r.raise_for_status()
    return r.json()["data"]["result"]


def _series_to_s(series: dict, grid: np.ndarray) -> pd.Series:
    vals = {round(float(t)): float(v) for t, v in series["values"]}
    return pd.Series([vals.get(int(g), np.nan) for g in grid], index=grid, dtype="float64")


def _oom_cumulative(restarts: pd.Series, oom_last: pd.Series) -> pd.Series:
    """Count restarts whose termination reason was OOMKilled (KSM 'last_terminated_reason' is
    a state gauge, so we attribute each restart increment to OOM if the flag is set at that step
    or within the next 2 steps, to absorb scrape skew between the two series)."""
    r = restarts.ffill().fillna(0).to_numpy()
    f = oom_last.fillna(0).to_numpy()
    out, total = np.zeros(len(r)), 0.0
    for i in range(len(r)):
        if i > 0 and r[i] > r[i - 1]:
            if f[i:i + 3].max(initial=0) >= 1:
                total += r[i] - r[i - 1]
        out[i] = total
    return pd.Series(out, index=restarts.index)


def export_run(ns: str, start: float, end: float, out_dir: Path, policy: str = "fixed") -> tuple[pd.DataFrame, pd.DataFrame]:
    start = float(int(start) - int(start) % STEP)
    end = float(int(end))
    grid = np.arange(int(start), int(end) + 1, STEP)

    # ---------------- deployment level
    dep = pd.DataFrame(index=grid)
    for col, q in deploy_queries(ns).items():
        res = prom_range(q, start, end)
        dep[col] = _series_to_s(res[0], grid) if res else np.nan

    # ---------------- pod level
    per_pod: dict[str, pd.DataFrame] = {}
    for col, q in pod_queries(ns).items():
        for s in prom_range(q, start, end):
            pod = s["metric"].get("pod")
            if not pod:
                continue
            per_pod.setdefault(pod, pd.DataFrame(index=grid))[col] = _series_to_s(s, grid)
    rows = []
    oom_by_pod = []
    restarts_by_pod = []
    for pod, df in sorted(per_pod.items()):
        for col in ["cpu", "throttle", "mem_ws", "mem_frac_limit", "restarts", "rps", "p95", "errors", "ready", "_oom_last", "cpu_proc"]:
            if col not in df:
                df[col] = np.nan
        df["oomkills"] = _oom_cumulative(df["restarts"], df["_oom_last"])
        oom_by_pod.append(df["oomkills"])
        restarts_by_pod.append(df["restarts"].ffill().fillna(0.0))
        # keep only rows where the pod existed (any signal present)
        alive = df[["cpu", "mem_ws", "restarts", "rps", "ready"]].notna().any(axis=1)
        df = df[alive].drop(columns=["_oom_last"]).copy()
        # counters without samples yet (no request / no 5xx so far) mean zero, not unknown
        df["rps"] = df["rps"].fillna(0.0)
        df["errors"] = df["errors"].fillna(0.0)
        df.insert(0, "pod", pod)
        df.insert(0, "ts", df.index.astype(int))
        rows.append(df)
    pods = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["ts", "pod", "cpu", "throttle", "mem_ws", "mem_frac_limit", "restarts", "rps", "p95",
                 "errors", "ready", "oomkills", "cpu_proc"])

    # ---------------- derived / gap filling (documented in README)
    dep["oomkills_total"] = (pd.concat(oom_by_pod, axis=1).sum(axis=1) if oom_by_pod else 0.0)
    # restarts_total: cumulative over every pod seen in the run (a replaced pod keeps its count), so the
    # column is monotone and diff() gives restarts per step. KSM's raw sum dropped when pods were replaced.
    if restarts_by_pod:
        dep["restarts_total"] = pd.concat(restarts_by_pod, axis=1).sum(axis=1)
    for col in ["rps", "inflight", "crashloop_pods", "restarts_total", "cpu_fallback_pods"]:
        dep[col] = dep[col].fillna(0.0)
    # error_rate is NaN (0/0) only when there is no traffic -> 0
    dep["error_rate"] = dep["error_rate"].where(dep["rps"] > 0, 0.0).fillna(0.0)
    if policy != "hpa_cpu":
        # no HPA object: the "desired" count is just the fixed spec
        dep["hpa_desired"] = dep["hpa_desired"].fillna(dep["replicas_desired"])
    dep.insert(0, "ts", grid.astype(int))
    cols = ["ts", "rps", "inflight", "p50", "p95", "p99", "error_rate", "cpu_cores_used",
            "cpu_util_vs_request", "throttle_ratio", "mem_ws_bytes_total", "mem_ws_max_pod_frac_of_limit",
            "mem_growth_bytes_per_s", "replicas_desired", "replicas_ready", "restarts_total",
            "oomkills_total", "crashloop_pods", "hpa_desired", "cpu_cores_proc", "cpu_fallback_pods"]
    dep = dep[cols]
    # Trim the edges of the window where signals are not yet available (pods just started:
    # cAdvisor needs 2 refreshes, KSM needs to observe the Deployment, no traffic yet -> latency NaN).
    # (mem_growth_bytes_per_s is deliberately NOT here: it is NaN until a pod is MEM_WARM old, and
    # whenever every pod is new, e.g. right after a full rollout.)
    core = ["rps", "p95", "cpu_cores_used", "throttle_ratio", "mem_ws_bytes_total",
            "replicas_desired", "replicas_ready"]
    ok = dep[core].notna().all(axis=1) & (dep["rps"] > 0)
    if ok.any():
        first, last = ok.idxmax(), ok[::-1].idxmax()
        dep = dep.loc[first:last]
        pods = pods[(pods["ts"] >= dep["ts"].iloc[0]) & (pods["ts"] <= dep["ts"].iloc[-1])]
    out_dir.mkdir(parents=True, exist_ok=True)
    dep.to_csv(out_dir / "deploy.csv", index=False)
    pods.to_csv(out_dir / "pods.csv", index=False)
    (out_dir / "queries.json").write_text(json.dumps(
        {"deploy": deploy_queries(ns), "pods": pod_queries(ns), "step_s": STEP}, indent=2))
    return dep, pods


if __name__ == "__main__":
    run_dir = Path(sys.argv[1])
    sc = json.loads((run_dir / "scenario.json").read_text())
    w = sc["_run"]["export_window"]
    d, p = export_run(sc["_run"]["namespace"], w[0], w[1], run_dir, sc["scaling"]["policy"])
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(d.describe().T)
    print(p.describe().T)
