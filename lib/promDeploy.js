// Deployment-level telemetry window for POST /v1/decide.
//
// The PromQL is a verbatim port of loadlab/runner/promexport.py (deploy_queries / pod_queries),
// the exporter that produced the training dataset, so the features the decision engine sees
// live are computed exactly as in training ("train == serve"). The only parameters are the ones
// promexport hard-codes for the lab: namespace, Deployment name (promexport: "loadlab"), the
// Deployment's pod regex (promexport: "loadlab-.*"), the app container name ("app") and the
// Prometheus job of the annotated app pods ("annotated-pods"). With the lab values the strings
// are byte-identical to the ones stored in loadlab/runs/<run>/queries.json (checked by
// test/agent.smoke.mjs).
//
// Query version "v2" (default; loadlab/runner/promexport.py now exports the same strings):
//   - default pod regex `<deployment>-[a-z0-9]+-[a-z0-9]{5}` (ReplicaSet hash + pod suffix; no
//     hyphen in either part), so deployment "web" no longer matches pods of "web-api";
//   - every kube-state-metrics pod-level selector (restarts, OOM reason, CrashLoopBackOff,
//     requests, limits, phase) is restricted to that pod regex, and hpa_desired to the HPA that
//     targets this Deployment (no HPA -> the query is skipped and hpa_desired falls back to
//     replicas_desired, exactly as when the old namespace-wide query returned nothing);
//   - mem_growth_bytes_per_s only counts pods that are still live scrape targets (`up`): a range
//     selector keeps returning a deleted pod's last 60 s, which kept the deleted leaking pod's
//     slope in the max for ~60 s and latched `memory_leak` after the agent's own restart.
// In a namespace with one Deployment (every lab run) v2 selects the same series as v1 except
// for mem_growth after a pod deletion. `queryVersion: "v1"` reproduces the training-time
// strings byte for byte (checked against a run's queries.json by test/scaler.smoke.mjs).
//
// The app-level metrics (http_requests_total, latency histogram, inflight, process CPU) have no
// pod-name join in the lab and are selected by namespace (+ APP_METRIC_MATCHER). With several
// Deployments in one namespace APP_METRIC_MATCHER must be a per-Deployment template such as
// `app="{deployment}"`; otherwise lib/scaler.js refuses to decide (decision_error stage=config).
//
// Each decision cycle runs every query as a query_range over the last HISTORY_S at a 5 s step
// (the dataset's step) and applies promexport's / build_dataset's post-processing:
//   - per-pod cumulative restarts and OOMKills (restart increments with an OOMKilled
//     last-terminated reason within 2 steps), summed into restarts_total / oomkills_total;
//   - crashloop_pods, restarts_total -> 0 when absent; error_rate -> 0 when there is no traffic;
//   - hpa_desired <- replicas_desired when there is no HPA;
//   - goodput_rps = rps * (1 - error_rate); pods_not_ready = pods present but not ready;
//   - pod mem_ws / mem_frac_limit of exactly 0 (cAdvisor before its first refresh) -> missing;
//   - leading steps before the core signals exist are trimmed.
// One deliberate difference for live serving: promexport filled a missing rps/inflight with 0
// (no request yet in a finished run). Live, "no sample" can also mean the app scrape is down,
// so rps/inflight are only zero-filled where the app's inflight gauge exists (the app is being
// scraped); otherwise they stay null and the engine's stale-telemetry interlock fires.

import { requestJson } from "./http.js";

export const STEP_S = 5;
export const APP_W = "20s"; // app (5 s-scraped) counters
export const CAD_W = "45s"; // cAdvisor counters (kubelet housekeeping ~15 s)
export const MEM_WARM = "90s";

const escapeRegex = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
const escapeLabel = (s) => String(s).replace(/\\/g, "\\\\").replace(/"/g, '\\"');

export const defaultPodRegex = (deployment, version = "v2") =>
  version === "v1" ? `${escapeRegex(deployment)}-.*` : `${escapeRegex(deployment)}-[a-z0-9]+-[a-z0-9]{5}`;

export function selectorOpts(namespace, deployment, o = {}) {
  const v1 = o.queryVersion === "v1";
  const podRe = escapeLabel(o.podRegex || defaultPodRegex(deployment, v1 ? "v1" : "v2"));
  return {
    v1,
    ns: escapeLabel(namespace),
    deploy: escapeLabel(deployment),
    podRe,
    container: escapeLabel(o.container || "app"),
    job: escapeLabel(o.job || "annotated-pods"),
    appExtra: o.appMatcher ? `,${o.appMatcher}` : "",
    podScope: v1 ? "" : `,pod=~"${podRe}"`, // kube-state-metrics pod-level selectors
    hpa: o.hpa ? escapeLabel(o.hpa) : null,
  };
}

function cpuByPod(s) {
  const a = `namespace="${s.ns}"${s.appExtra}`;
  const c = `namespace="${s.ns}",pod=~"${s.podRe}",scope="pod"`;
  const cad = `sum by (pod)(rate(container_cpu_usage_seconds_total{${c}}[${CAD_W}]))`;
  const proc = `sum by (pod)(rate(process_cpu_seconds_total{${a},job="${s.job}"}[${APP_W}]))`;
  const live = `group by (pod)(up{${a},job="${s.job}"})`;
  const both = `label_replace(${cad}, "src", "cadvisor", "", "") or label_replace(${proc}, "src", "process", "", "")`;
  return `max by (pod)(${both}) and on(pod) ${live}`;
}

// promexport.deploy_queries
export function deployQueries(namespace, deployment, o = {}) {
  const s = selectorOpts(namespace, deployment, o);
  const a = `namespace="${s.ns}"${s.appExtra}`;
  const ns = `namespace="${s.ns}"`;
  const c = `namespace="${s.ns}",pod=~"${s.podRe}",scope="pod"`;
  const k = `namespace="${s.ns}",container="${s.container}"${s.podScope}`;
  const cpu = cpuByPod(s);
  const reqRunning =
    `sum(kube_pod_container_resource_requests{${k},resource="cpu"} ` +
    `* on(namespace,pod) group_left() (kube_pod_status_phase{${ns}${s.podScope},phase="Running"} == bool 1))`;
  const memGrowth = s.v1
    ? `max(deriv(container_memory_working_set_bytes{${c}}[60s]) ` +
      `and on(pod) (container_memory_working_set_bytes{${c}} offset ${MEM_WARM}))`
    : `max((deriv(container_memory_working_set_bytes{${c}}[60s]) ` +
      `and on(pod) (container_memory_working_set_bytes{${c}} offset ${MEM_WARM})) ` +
      `and on(pod) group by (pod)(up{${a},job="${s.job}"}))`;
  const hpaDesired = s.v1
    ? `max(kube_horizontalpodautoscaler_status_desired_replicas{${ns}})`
    : s.hpa ? `max(kube_horizontalpodautoscaler_status_desired_replicas{${ns},horizontalpodautoscaler="${s.hpa}"})` : null;
  const q = {
    rps: `sum(rate(http_requests_total{${a}}[${APP_W}]))`,
    inflight: `sum(http_inflight_requests{${a}})`,
    p50: `histogram_quantile(0.50, sum by (le)(rate(http_request_duration_seconds_bucket{${a}}[${APP_W}])))`,
    p95: `histogram_quantile(0.95, sum by (le)(rate(http_request_duration_seconds_bucket{${a}}[${APP_W}])))`,
    p99: `histogram_quantile(0.99, sum by (le)(rate(http_request_duration_seconds_bucket{${a}}[${APP_W}])))`,
    error_rate: `sum(rate(http_requests_total{${a},code=~"5.."}[${APP_W}])) / sum(rate(http_requests_total{${a}}[${APP_W}]))`,
    cpu_cores_used: `sum(${cpu})`,
    cpu_util_vs_request: `sum(${cpu}) / ${reqRunning}`,
    throttle_ratio:
      `sum(rate(container_cpu_cfs_throttled_periods_total{${c}}[${CAD_W}])) ` +
      `/ sum(rate(container_cpu_cfs_periods_total{${c}}[${CAD_W}]))`,
    mem_ws_bytes_total: `sum(container_memory_working_set_bytes{${c}})`,
    mem_ws_max_pod_frac_of_limit:
      `max(max by (pod)(container_memory_working_set_bytes{${c}}) ` +
      `/ on(pod) group_left() max by (pod)(kube_pod_container_resource_limits{${k},resource="memory"}))`,
    mem_growth_bytes_per_s: memGrowth,
    replicas_desired: `max(kube_deployment_spec_replicas{${ns},deployment="${s.deploy}"})`,
    replicas_ready: `max(kube_deployment_status_replicas_ready{${ns},deployment="${s.deploy}"})`,
    restarts_total: `sum(kube_pod_container_status_restarts_total{${k}})`,
    crashloop_pods: `count(kube_pod_container_status_waiting_reason{${k},reason="CrashLoopBackOff"} == 1)`,
    hpa_desired: hpaDesired,
    cpu_cores_proc: `sum(rate(process_cpu_seconds_total{${a},job="${s.job}"}[${APP_W}]))`,
    cpu_fallback_pods:
      `count(sum by (pod)(rate(process_cpu_seconds_total{${a},job="${s.job}"}[${APP_W}])) ` +
      `unless on(pod) sum by (pod)(rate(container_cpu_usage_seconds_total{${c}}[${CAD_W}])))`,
  };
  if (q.hpa_desired == null) delete q.hpa_desired; // no HPA targets this Deployment
  return q;
}

// promexport.pod_queries
export function podQueries(namespace, deployment, o = {}) {
  const s = selectorOpts(namespace, deployment, o);
  const a = `namespace="${s.ns}"${s.appExtra}`;
  const ns = `namespace="${s.ns}"`;
  const c = `namespace="${s.ns}",pod=~"${s.podRe}",scope="pod"`;
  const k = `namespace="${s.ns}",container="${s.container}"${s.podScope}`;
  return {
    cpu: cpuByPod(s),
    throttle:
      `sum by (pod)(rate(container_cpu_cfs_throttled_periods_total{${c}}[${CAD_W}])) ` +
      `/ sum by (pod)(rate(container_cpu_cfs_periods_total{${c}}[${CAD_W}]))`,
    mem_ws: `max by (pod)(container_memory_working_set_bytes{${c}})`,
    mem_frac_limit:
      `max by (pod)(container_memory_working_set_bytes{${c}}) ` +
      `/ on(pod) group_left() max by (pod)(kube_pod_container_resource_limits{${k},resource="memory"})`,
    restarts: `max by (pod)(kube_pod_container_status_restarts_total{${k}})`,
    rps: `sum by (pod)(rate(http_requests_total{${a}}[${APP_W}]))`,
    p95: `histogram_quantile(0.95, sum by (pod, le)(rate(http_request_duration_seconds_bucket{${a}}[${APP_W}])))`,
    errors: `sum by (pod)(rate(http_requests_total{${a},code=~"5.."}[${APP_W}]))`,
    ready: `max by (pod)(kube_pod_status_ready{${ns},condition="true",pod=~"${s.podRe}"})`,
    cpu_proc: `sum by (pod)(rate(process_cpu_seconds_total{${a},job="${s.job}"}[${APP_W}]))`,
    _oom_last: `max by (pod)(kube_pod_container_status_last_terminated_reason{${k},reason="OOMKilled"})`,
  };
}

// Fields /v1/decide accepts (the engine's schema); everything else is not sent.
export const DEPLOY_FIELDS = [
  "ts", "rps", "goodput_rps", "inflight", "p50", "p95", "p99", "error_rate", "cpu_cores_used",
  "cpu_util_vs_request", "throttle_ratio", "mem_ws_max_pod_frac_of_limit", "mem_growth_bytes_per_s",
  "replicas_desired", "replicas_ready", "restarts_total", "oomkills_total", "crashloop_pods", "hpa_desired",
  "pods_not_ready", "revision",
];
export const POD_FIELDS = ["ts", "pod", "cpu", "throttle", "mem_ws", "mem_frac_limit", "rps", "ready", "restarts"];
const CORE = ["rps", "p95", "cpu_cores_used", "throttle_ratio", "mem_ws_bytes_total", "replicas_desired", "replicas_ready"];

export async function promRange(cfg, query, start, end, step = STEP_S) {
  const qs = new URLSearchParams({ query, start: String(start), end: String(end), step: String(step) });
  const data = await requestJson(`${cfg.prometheusUrl}/api/v1/query_range?${qs}`, { timeoutMs: cfg.httpTimeoutMs });
  if (data?.status !== "success") throw new Error(`prometheus error: ${data?.error ?? "unknown"}`);
  return data?.data?.result ?? [];
}

const num = (v) => {
  const x = Number(v);
  return Number.isFinite(x) ? x : null; // NaN / +Inf from PromQL -> missing
};

// Series values -> Map(gridTs -> number|null)
function toMap(series) {
  const m = new Map();
  for (const [t, v] of series?.values ?? []) m.set(Math.round(Number(t)), num(v));
  return m;
}

export function gridFor(nowS, historyS, step = STEP_S) {
  const start0 = Math.floor(nowS - historyS);
  const start = start0 - (start0 % step);
  const end = Math.floor(nowS);
  const grid = [];
  for (let t = start; t <= end; t += step) grid.push(t);
  return { start, end, grid };
}

// promexport._oom_cumulative
export function oomCumulative(restarts, oomLast) {
  const r = [];
  let last = null;
  for (const v of restarts) {
    if (v != null) last = v;
    r.push(last ?? 0);
  }
  const f = oomLast.map((v) => v ?? 0);
  const out = [];
  let total = 0;
  for (let i = 0; i < r.length; i++) {
    if (i > 0 && r[i] > r[i - 1] && Math.max(0, ...f.slice(i, i + 3)) >= 1) total += r[i] - r[i - 1];
    out.push(total);
  }
  return out;
}

const ffill0 = (arr) => {
  let last = null;
  return arr.map((v) => {
    if (v != null) last = v;
    return last ?? 0;
  });
};

// Pure: raw query_range results -> {series, pods} in the /v1/decide schema.
// deployRes: {col: [promSeries...]}, podRes: {col: [promSeries...]}
// podMatch (optional RegExp, anchored): defence in depth, pod rows whose name does not belong
// to this Deployment are dropped (a foreign ReplicaSet hash would look like a rollout).
export function buildWindow(grid, deployRes, podRes, { revisionAt, podMatch } = {}) {
  const dep = {};
  for (const [col, res] of Object.entries(deployRes)) {
    const m = res?.length ? toMap(res[0]) : new Map();
    dep[col] = grid.map((t) => (m.has(t) ? m.get(t) : null));
  }

  // per pod
  const perPod = new Map(); // pod -> {col: array}
  for (const [col, res] of Object.entries(podRes)) {
    for (const s of res ?? []) {
      const pod = s?.metric?.pod;
      if (!pod) continue;
      if (podMatch && !podMatch.test(pod)) continue;
      if (!perPod.has(pod)) perPod.set(pod, {});
      const m = toMap(s);
      perPod.get(pod)[col] = grid.map((t) => (m.has(t) ? m.get(t) : null));
    }
  }
  const empty = grid.map(() => null);
  const podRows = [];
  const oomByPod = [];
  const restartsByPod = [];
  for (const pod of [...perPod.keys()].sort()) {
    const d = perPod.get(pod);
    const col = (c) => d[c] ?? empty;
    oomByPod.push(oomCumulative(col("restarts"), col("_oom_last")));
    restartsByPod.push(ffill0(col("restarts")));
    grid.forEach((t, i) => {
      const alive = ["cpu", "mem_ws", "restarts", "rps", "ready"].some((c) => col(c)[i] != null);
      if (!alive) return;
      let memWs = col("mem_ws")[i];
      let memFrac = col("mem_frac_limit")[i];
      if (memWs === 0) memWs = null; // cAdvisor before its first refresh (build_dataset.clean_pods)
      if (memFrac === 0) memFrac = null;
      podRows.push({
        ts: t, pod, cpu: col("cpu")[i], throttle: col("throttle")[i], mem_ws: memWs, mem_frac_limit: memFrac,
        rps: col("rps")[i] ?? 0, ready: col("ready")[i], restarts: col("restarts")[i],
      });
    });
  }

  const sumAcross = (arrs) => grid.map((_, i) => arrs.reduce((acc, a) => acc + (a[i] ?? 0), 0));
  const oomkills = oomByPod.length ? sumAcross(oomByPod) : grid.map(() => 0);
  const restarts = restartsByPod.length ? sumAcross(restartsByPod) : (dep.restarts_total ?? empty).map((v) => v ?? 0);
  const notReady = new Map();
  for (const r of podRows) if (r.ready === 0) notReady.set(r.ts, (notReady.get(r.ts) ?? 0) + 1);

  let rows = grid.map((t, i) => {
    const g = (c) => (dep[c] ? dep[c][i] : null);
    const appUp = g("inflight") != null; // the app's own gauge exists -> app is scraped
    const rps = g("rps") ?? (appUp ? 0 : null);
    let er = g("error_rate");
    if (rps == null) er = null;
    else if (!(rps > 0)) er = 0;
    else er = er ?? 0;
    const repDesired = g("replicas_desired");
    const row = {
      ts: t,
      rps,
      goodput_rps: rps == null || er == null ? null : rps * (1 - er),
      inflight: g("inflight") ?? (appUp ? 0 : null),
      p50: g("p50"), p95: g("p95"), p99: g("p99"),
      error_rate: er,
      cpu_cores_used: g("cpu_cores_used"),
      cpu_util_vs_request: g("cpu_util_vs_request"),
      throttle_ratio: g("throttle_ratio"),
      mem_ws_bytes_total: g("mem_ws_bytes_total"),
      mem_ws_max_pod_frac_of_limit: g("mem_ws_max_pod_frac_of_limit"),
      mem_growth_bytes_per_s: g("mem_growth_bytes_per_s"),
      replicas_desired: repDesired,
      replicas_ready: g("replicas_ready"),
      restarts_total: restarts[i],
      oomkills_total: oomkills[i],
      crashloop_pods: g("crashloop_pods") ?? 0,
      hpa_desired: g("hpa_desired") ?? repDesired,
      pods_not_ready: notReady.get(t) ?? 0,
      revision: revisionAt ? revisionAt(t) : null,
    };
    return row;
  });

  // trim leading rows before the core signals exist (promexport trims both edges of a finished
  // run; live, the trailing edge is kept so stale data reaches the engine's interlock)
  const first = rows.findIndex((r) => CORE.every((c) => r[c] != null) && r.rps > 0);
  if (first > 0) rows = rows.slice(first);
  const t0 = rows.length ? rows[0].ts : Infinity;
  const pods = podRows.filter((p) => p.ts >= t0);
  const series = rows.map((r) => Object.fromEntries(DEPLOY_FIELDS.map((f) => [f, r[f] ?? null])));
  return { series, pods: pods.map((p) => Object.fromEntries(POD_FIELDS.map((f) => [f, p[f] ?? null]))) };
}

const UNUSED_DEPLOY = new Set(["cpu_cores_proc", "cpu_fallback_pods"]);
const UNUSED_POD = new Set(["p95", "errors", "cpu_proc"]);

// Runs every query (bounded concurrency) and builds the window.
// opts.hpa: name of the HPA that targets this Deployment (null: none / unknown).
export async function fetchWindow(cfg, namespace, deployment, { nowS = Date.now() / 1000, revisionAt, hpa = null } = {}) {
  const podRegex = cfg.podRegexFor?.(deployment) || defaultPodRegex(deployment);
  const appMatcher = cfg.appMatcherFor ? cfg.appMatcherFor(deployment) : cfg.appMetricMatcher;
  const o = { podRegex, container: cfg.appContainer, job: cfg.appJob, appMatcher, hpa: hpa && hpa !== "unknown" ? hpa : null };
  let podMatch = null;
  try {
    podMatch = new RegExp(`^(?:${podRegex})$`); // Prometheus regexes are fully anchored
  } catch {
    podMatch = null; // RE2-only syntax: rely on the PromQL filter alone
  }
  const { start, end, grid } = gridFor(nowS, cfg.historyS);
  const dq = deployQueries(namespace, deployment, o);
  const pq = podQueries(namespace, deployment, o);
  // promexport's cross-check columns that the engine does not read are not queried
  const jobs = [
    ...Object.entries(dq).filter(([col]) => !UNUSED_DEPLOY.has(col)).map(([col, q]) => ["d", col, q]),
    ...Object.entries(pq).filter(([col]) => !UNUSED_POD.has(col)).map(([col, q]) => ["p", col, q]),
  ];
  const deployRes = {};
  const podRes = {};
  const conc = Math.max(1, cfg.promConcurrency ?? 6);
  let next = 0;
  async function worker() {
    while (next < jobs.length) {
      const [kind, col, q] = jobs[next++];
      const res = await promRange(cfg, q, start, end);
      (kind === "d" ? deployRes : podRes)[col] = res;
    }
  }
  await Promise.all(Array.from({ length: Math.min(conc, jobs.length) }, worker));
  return { ...buildWindow(grid, deployRes, podRes, { revisionAt, podMatch }), start, end, queries: jobs.length };
}
