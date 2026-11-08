// Smoke test for the deployment-level loop (lib/promDeploy.js + lib/scaler.js):
// fake Prometheus (query_range) + fake ML /v1/decide + fake kube API + fake kubectl.
//
// Run: npm test   (runs agent.smoke.mjs and this file)

import http from "node:http";
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { mkdtemp, writeFile, chmod, readFile, rm } from "node:fs/promises";
import { readFileSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { deployQueries, podQueries, buildWindow, oomCumulative, gridFor, fetchWindow, defaultPodRegex, DEPLOY_FIELDS, POD_FIELDS } from "../lib/promDeploy.js";
import { buildEngineAction, runKubectl } from "../lib/remediation.js";
import { createScaler } from "../lib/scaler.js";
import { summarizeDeployment } from "../lib/kube.js";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
let passed = 0;
const ok = (name) => {
  passed += 1;
  console.log(`  ok - ${name}`);
};

// ---------------------------------------------------------------- unit
console.log("unit (promDeploy / allowlist)");
{
  // Byte-identical to the PromQL that exported the training data (loadlab/runner/promexport.py).
  const run = path.join(ROOT, "loadlab/runs/d-leak-medium-r2-1108-110336-3c8");
  if (existsSync(path.join(run, "queries.json"))) {
    const golden = JSON.parse(readFileSync(path.join(run, "queries.json"), "utf8"));
    const ns = JSON.parse(readFileSync(path.join(run, "scenario.json"), "utf8"))._run.namespace;
    // v1 = the strings that exported the committed training data
    assert.deepEqual(deployQueries(ns, "loadlab", { queryVersion: "v1" }), golden.deploy, "v1 deploy PromQL == training export");
    assert.deepEqual(podQueries(ns, "loadlab", { queryVersion: "v1" }), golden.pods, "v1 pod PromQL == training export");
    ok(`v1 PromQL byte-identical to the training export (${Object.keys(golden.deploy).length + Object.keys(golden.pods).length} queries, run ${path.basename(run)})`);
  } else {
    console.log("  skip - loadlab run with queries.json not present");
  }
  // v2 (default) = what loadlab/runner/promexport.py exports now (fixture generated from it)
  const v2 = JSON.parse(readFileSync(path.join(ROOT, "test/fixtures/promql_v2_loadlab.json"), "utf8"));
  assert.deepEqual(deployQueries(v2.namespace, "loadlab", { hpa: "loadlab" }), v2.deploy, "v2 deploy PromQL == promexport.py");
  assert.deepEqual(podQueries(v2.namespace, "loadlab", { hpa: "loadlab" }), v2.pods, "v2 pod PromQL == promexport.py");
  ok("v2 PromQL byte-identical to the current promexport.py (pod-scoped KSM, HPA-scoped, live-target mem growth)");
  const q = deployQueries("shop", "web.v2", { appMatcher: 'app="web"' });
  assert.match(q.replicas_ready, /deployment="web\.v2"/);
  assert.ok(q.throttle_ratio.includes('pod=~"web\\\\.v2-[a-z0-9]+-[a-z0-9]{5}"'), "deployment name regex-escaped; RS hash + suffix pod regex");
  assert.ok(q.restarts_total.includes('pod=~"web\\\\.v2-[a-z0-9]+-[a-z0-9]{5}"'), "kube-state-metrics selectors scoped to the Deployment's pods");
  assert.ok(!("hpa_desired" in q), "no HPA -> hpa_desired not queried (falls back to replicas_desired)");
  assert.match(deployQueries("shop", "web", { hpa: "web-hpa" }).hpa_desired, /horizontalpodautoscaler="web-hpa"/);
  assert.match(q.rps, /namespace="shop",app="web"/, "APP_METRIC_MATCHER appended to app selectors");
  assert.ok(!q.replicas_ready.includes('app="web"'), "kube-state-metrics selectors untouched");
  ok("query parameters: deployment, escaped pod regex, optional app matcher");
}
{
  assert.deepEqual(oomCumulative([0, 0, 1, 1, 2, null, 3], [0, 0, 1, 0, 0, 0, 0]), [0, 0, 1, 1, 1, 1, 1], "only OOMKilled restarts counted");
  assert.deepEqual(oomCumulative([null, 1, 2], [0, 0, 1]), [0, 1, 2], "flag within the next 2 steps");
  const { grid } = gridFor(1000, 60);
  assert.deepEqual([grid[0], grid.at(-1), grid.length], [940, 1000, 13]);
  ok("oomCumulative (promexport._oom_cumulative) + 5 s grid");
}

{
  // Two Deployments sharing a name prefix in one namespace (web, web-api): default pod regex and
  // the buildWindow pod filter keep web-api's pods (and its ReplicaSet hash) out of web's window.
  const re = new RegExp(`^(?:${defaultPodRegex("web")})$`);
  assert.ok(re.test("web-7f6b5c4d3-zzzzz") && !re.test("web-api-7f6b5c4d3-zzzzz") && !re.test("web-canary-77b8-zzzzz"));
  const g = Array.from({ length: 6 }, (_, i) => 1_800_000_000 + i * 5);
  const ser = (pod) => ({ metric: { pod }, values: g.map((t) => [t, "1"]) });
  const w = buildWindow(g, { rps: [{ metric: {}, values: g.map((t) => [t, "5"]) }] },
    { restarts: [ser("web-7f6b5c4d3-aaaaa"), ser("web-api-5d4c3b2a1-bbbbb"), ser("worker-7f6b5c4d3-zzzzz")], ready: [ser("web-7f6b5c4d3-aaaaa"), ser("web-api-5d4c3b2a1-bbbbb")] },
    { podMatch: re });
  assert.deepEqual([...new Set(w.pods.map((p) => p.pod))], ["web-7f6b5c4d3-aaaaa"], "no foreign pod rows -> no foreign ReplicaSet hash -> no false rollout evidence");
  ok("two Deployments in one namespace: sibling pods excluded by regex and window filter");
}

// synthetic Prometheus results on a grid
const T0 = 1_800_000_000;
const GRID = Array.from({ length: 60 }, (_, i) => T0 + i * 5);
const vals = (f) => GRID.map((t, i) => [t, f(i)]).filter(([, v]) => v !== undefined).map(([t, v]) => [t, String(v)]);
const series = (f, metric = {}) => [{ metric, values: vals(f) }];
function deployRes(over = {}) {
  const base = {
    rps: series((i) => (i < 3 || i === 20 ? undefined : 20)),
    inflight: series(() => 1),
    p50: series(() => 0.03), p95: series(() => 0.06), p99: series(() => 0.09),
    error_rate: series((i) => (i < 3 ? undefined : 0.01)),
    cpu_cores_used: series(() => 0.6), cpu_util_vs_request: series(() => 0.8),
    throttle_ratio: series(() => 0.0), mem_ws_bytes_total: series(() => 3e8),
    mem_ws_max_pod_frac_of_limit: series(() => 0.3), mem_growth_bytes_per_s: series(() => 10),
    replicas_desired: series(() => 3), replicas_ready: series(() => 3),
    restarts_total: series(() => 0), crashloop_pods: [], hpa_desired: [],
  };
  return { ...base, ...over };
}
function podRes() {
  const pods = ["web-abc-1", "web-abc-2", "web-abc-3"];
  const per = (f) => pods.map((p, k) => ({ metric: { pod: p }, values: vals((i) => f(i, k)) }));
  return {
    cpu: per(() => 0.2), throttle: per(() => 0), mem_ws: per((i, k) => (k === 0 && i < 4 ? 0 : 1e8)),
    mem_frac_limit: per((i, k) => (k === 0 && i < 4 ? 0 : 0.3)),
    restarts: per((i, k) => (k === 1 && i >= 30 ? 1 : 0)),
    _oom_last: per((i, k) => (k === 1 && i >= 30 ? 1 : undefined)),
    rps: per(() => 6.6), ready: per((i, k) => (k === 2 && i >= 50 ? 0 : 1)),
  };
}
{
  const w = buildWindow(GRID, deployRes(), podRes(), { revisionAt: (t) => (t >= T0 + 100 ? 2 : null) });
  assert.equal(w.series[0].ts, T0 + 15, "leading rows without traffic trimmed (promexport)");
  const r20 = w.series.find((r) => r.ts === T0 + 100);
  assert.equal(r20.rps, 0, "app is scraped (inflight present): missing rps means no traffic -> 0");
  assert.equal(r20.error_rate, 0, "no traffic -> error_rate 0");
  const last = w.series.at(-1);
  assert.deepEqual(Object.keys(last), DEPLOY_FIELDS);
  assert.equal(last.hpa_desired, 3, "no HPA -> hpa_desired = replicas_desired");
  assert.equal(last.crashloop_pods, 0);
  assert.ok(Math.abs(last.goodput_rps - 19.8) < 1e-9, "goodput = rps * (1 - error_rate)");
  assert.equal(last.restarts_total, 1);
  assert.equal(last.oomkills_total, 1, "restart with OOMKilled reason counted");
  assert.equal(last.pods_not_ready, 1);
  assert.equal(last.revision, 2);
  assert.equal(w.series[0].revision, null);
  assert.deepEqual(Object.keys(w.pods[0]), POD_FIELDS);
  assert.equal(w.pods.find((p) => p.pod === "web-abc-1" && p.ts === T0 + 15).mem_ws, null, "cAdvisor 0 bytes -> missing");

  // app not scraped (no inflight) for the last 4 steps: rps stays null (unknown != 0)
  const gone = deployRes({ rps: series((i) => (i < 56 ? 20 : undefined)), inflight: series((i) => (i < 56 ? 1 : undefined)) });
  const w2 = buildWindow(GRID, gone, podRes());
  assert.equal(w2.series.at(-1).rps, null);
  assert.equal(w2.series.at(-1).error_rate, null);
  assert.equal(w2.series.at(-5).rps, 20);
  // core signals missing at the start -> leading rows trimmed
  const late = deployRes({ cpu_cores_used: series((i) => (i < 10 ? undefined : 0.6)) });
  assert.equal(buildWindow(GRID, late, podRes()).series[0].ts, T0 + 50);
  ok("window: promexport post-processing (fills, OOM/restart counts, not-ready, trims), missing app scrape stays null");
}
{
  const ctx = { namespace: "shop", deployment: "web", current: 3, minReplicas: 2, maxReplicas: 6, pods: ["web-abc-1"] };
  assert.deepEqual(buildEngineAction("scale", { replicas: 5 }, ctx).argv, ["-n", "shop", "scale", "deployment/web", "--current-replicas=3", "--replicas=5"]);
  assert.equal(buildEngineAction("scale", { replicas: 7 }, ctx).ok, false, "above max");
  assert.equal(buildEngineAction("scale", { replicas: 1 }, ctx).ok, false, "below min");
  assert.equal(buildEngineAction("scale", { replicas: 2.5 }, ctx).ok, false);
  const ev = buildEngineAction("delete_pod", { pod: "web-abc-1" }, ctx);
  assert.deepEqual(ev.argv, ["create", "--raw", "/api/v1/namespaces/shop/pods/web-abc-1/eviction", "-f", "-"], "eviction API (honours PDBs), not kubectl delete");
  assert.deepEqual(ev.dryRunArgv, ["create", "--raw", "/api/v1/namespaces/shop/pods/web-abc-1/eviction?dryRun=All", "-f", "-"]);
  assert.deepEqual(JSON.parse(ev.stdin), { apiVersion: "policy/v1", kind: "Eviction", metadata: { name: "web-abc-1", namespace: "shop" } });
  assert.equal(buildEngineAction("delete_pod", { pod: "db-0" }, ctx).ok, false, "not a pod of this deployment");
  assert.equal(buildEngineAction("delete_pod", { pod: "web-abc-1; rm -rf /" }, ctx).ok, false);
  assert.match(buildEngineAction("delete_pod", { pod: "web-abc-1" }, { ...ctx, pods: null }).error, /cannot verify pod ownership/, "unknown ownership -> refused");
  assert.deepEqual(buildEngineAction("rollback", {}, { ...ctx, currentRevision: 7, rollbackTo: 5 }).argv, ["-n", "shop", "rollout", "undo", "deployment/web", "--to-revision=5"]);
  assert.equal(buildEngineAction("rollback", {}, { ...ctx, currentRevision: 7 }).ok, false, "no known-good revision -> refused");
  assert.equal(buildEngineAction("rollback", {}, { ...ctx, currentRevision: 7, rollbackTo: 7 }).ok, false, "target not older than current -> refused");
  const rr = buildEngineAction("rollout_restart", {}, ctx);
  assert.deepEqual(rr.argv, ["-n", "shop", "rollout", "restart", "deployment/web"]);
  assert.equal(rr.dryRunArgv[2], "patch");
  assert.equal(buildEngineAction("delete_namespace", {}, ctx).ok, false);
  const sd = summarizeDeployment({ metadata: { name: "web", annotations: { "deployment.kubernetes.io/revision": "4", "kubeai.io/max-replicas": "8" } }, spec: { replicas: 3, selector: { matchLabels: { app: "web" } } }, status: { readyReplicas: 2 } });
  assert.deepEqual([sd.replicas, sd.readyReplicas, sd.revision, sd.maxReplicas, sd.minReplicas, sd.selector], [3, 2, 4, 8, null, "app=web"]);
  const sx = summarizeDeployment({ metadata: { name: "web" }, spec: { selector: { matchExpressions: [{ key: "app", operator: "In", values: ["web"] }, { key: "canary", operator: "DoesNotExist" }] } } });
  assert.equal(sx.selector, "app in (web),!canary", "matchExpressions-only selectors are kept (ownership can be verified)");
  ok("engine-action allowlist: scale bounds + precondition, own pods only, rollback, restart; deployment summary");
}

// ---------------------------------------------------------------- fake services
const listen = (handler) =>
  new Promise((resolve) => {
    const srv = http.createServer(handler);
    srv.listen(0, "127.0.0.1", () => resolve(srv));
  });
const json = (res, code, body) => {
  res.writeHead(code, { "content-type": "application/json" });
  res.end(JSON.stringify(body));
};
const readBody = (req) =>
  new Promise((resolve) => {
    let b = "";
    req.on("data", (c) => (b += c));
    req.on("end", () => resolve(b));
  });
const port = (s) => s.address().port;

// Fake Prometheus: identifies each query by exact string -> column, answers on the requested grid.
const NS = "shop";
const DEP = "web";
const DQ = Object.fromEntries(Object.entries(deployQueries(NS, DEP)).map(([k, v]) => [v, k]));
const PQ = Object.fromEntries(Object.entries(podQueries(NS, DEP)).map(([k, v]) => [v, k]));
const prom = { queries: [], unknown: [], down: false };
const promSrv = await listen((req, res) => {
  const u = new URL(req.url, "http://x");
  if (u.pathname !== "/api/v1/query_range") return json(res, 404, {});
  if (prom.down) return json(res, 503, { status: "error", error: "down" });
  const q = u.searchParams.get("query");
  const start = Number(u.searchParams.get("start"));
  const end = Number(u.searchParams.get("end"));
  const step = Number(u.searchParams.get("step"));
  prom.queries.push(q);
  const pts = [];
  for (let t = start; t <= end; t += step) pts.push(t);
  const v = (f) => pts.map((t) => [t, String(f(t))]);
  const dcol = DQ[q];
  const pcol = PQ[q];
  if (!dcol && !pcol) prom.unknown.push(q);
  const D = { rps: 30, inflight: 2, p50: 0.03, p95: 0.08, p99: 0.12, error_rate: 0, cpu_cores_used: 0.8, cpu_util_vs_request: 1.07,
    throttle_ratio: 0.05, mem_ws_bytes_total: 3e8, mem_ws_max_pod_frac_of_limit: 0.3, mem_growth_bytes_per_s: 0,
    replicas_desired: 2, replicas_ready: 2, restarts_total: 0 };
  let result = [];
  if (dcol && dcol in D) result = [{ metric: {}, values: v(() => D[dcol]) }];
  if (pcol) {
    const P = { cpu: 0.4, throttle: 0.05, mem_ws: 1e8, mem_frac_limit: 0.3, restarts: 0, rps: 15, ready: 1 };
    if (pcol in P) result = ["web-7f6b5-aaaaa", "web-7f6b5-bbbbb"].map((pod) => ({ metric: { pod }, values: v(() => P[pcol]) }));
  }
  json(res, 200, { status: "success", data: { resultType: "matrix", result } });
});

// Fake ML: the next decisions to return (FIFO), records request bodies.
const ml = { bodies: [], queue: [], fail: 0 };
const decision = (over = {}) => ({
  deployment: DEP, namespace: NS, model_version: "scaling-fake", state: "scale_up", state_probs: {}, confidence: 0.9,
  scaling_helps: true, current_replicas: 2, desired_replicas: 4, replicas_needed: {}, forecast: {}, safe_rps_per_pod: 10,
  slo_risk: 0.4, cpu_per_req_s: 0.026, actions: [{ type: "scale", target: DEP, params: { from: 2, to: 4 }, reason: "load above knee" }],
  reasons: ["state scale_up (p=0.90)", "policy: scale_up"], telemetry_ok: true, inputs: {}, latency_ms: 5, ...over,
});
const mlSrv = await listen(async (req, res) => {
  if (req.url !== "/v1/decide" || req.method !== "POST") return json(res, 404, {});
  const body = JSON.parse(await readBody(req));
  ml.bodies.push(body);
  if (ml.fail > 0) {
    ml.fail -= 1;
    return json(res, 503, { detail: "scaling bundle not loaded" });
  }
  json(res, 200, ml.queue.shift() ?? decision({ state: "healthy", desired_replicas: body.current_replicas, actions: [{ type: "none", target: DEP, params: {}, reason: "within capacity" }] }));
});

// Fake kube API
const kubeState = { replicas: 2, hpas: [], hpaForbidden: false, revision: "3", podsFail: false, noSelector: false, extraDeployment: false };
const kubeSrv = await listen((req, res) => {
  const u = new URL(req.url, "http://x");
  if (u.pathname === `/apis/apps/v1/namespaces/${NS}/deployments/${DEP}`) {
    return json(res, 200, { metadata: { name: DEP, namespace: NS, annotations: { "deployment.kubernetes.io/revision": kubeState.revision, "kubeai.io/max-replicas": "5" } }, spec: { replicas: kubeState.replicas, selector: kubeState.noSelector ? {} : { matchLabels: { app: DEP } } }, status: { readyReplicas: kubeState.replicas } });
  }
  if (u.pathname === `/apis/apps/v1/namespaces/${NS}/deployments`) {
    const items = [{ metadata: { name: DEP, namespace: NS, labels: { tier: "web" } }, spec: { replicas: kubeState.replicas } }];
    if (kubeState.extraDeployment && !u.searchParams.get("labelSelector")) items.push({ metadata: { name: "web-api", namespace: NS }, spec: { replicas: 1 } });
    return json(res, 200, { items });
  }
  if (u.pathname === `/apis/autoscaling/v2/namespaces/${NS}/horizontalpodautoscalers`) {
    if (kubeState.hpaForbidden) return json(res, 403, { message: "forbidden" });
    return json(res, 200, { items: kubeState.hpas });
  }
  if (u.pathname === `/api/v1/namespaces/${NS}/pods`) {
    if (kubeState.podsFail) return json(res, 500, { message: "etcdserver: request timed out" });
    const mk = (name) => ({ metadata: { name, namespace: NS, labels: { app: DEP, "pod-template-hash": "7f6b5" }, ownerReferences: [{ kind: "ReplicaSet", name: `${DEP}-7f6b5` }] }, spec: { containers: [{ name: "app" }] }, status: { phase: "Running" } });
    return json(res, 200, { items: [mk("web-7f6b5-aaaaa"), mk("web-7f6b5-bbbbb")] });
  }
  json(res, 404, { message: `not found ${u.pathname}` });
});

const tmp = await mkdtemp(path.join(tmpdir(), "kubeai-scaler-"));
const marker = path.join(tmp, "kubectl-calls");
const fakeKubectl = path.join(tmp, "kubectl");
const tmpPdb = path.join(tmp, "pdb-blocks");
await writeFile(fakeKubectl, `#!/bin/sh
echo "$@" >> "${marker}"
case " $* " in
  *" rollout restart "*"--dry-run"*) echo "Error: unknown flag: --dry-run" >&2; exit 1 ;;
  *"/eviction"*) if [ -f "${tmpPdb}" ]; then echo "Error from server (TooManyRequests): Cannot evict pod as it would violate the pod's disruption budget." >&2; exit 1; fi ;;
esac
echo "ok"
`);
await chmod(fakeKubectl, 0o755);
const calls = async () => (existsSync(marker) ? (await readFile(marker, "utf8")).trim().split("\n").filter(Boolean) : []);
const resetCalls = () => rm(marker, { force: true });

const baseCfg = {
  namespace: NS, deployments: [DEP], deploySelector: "", decideIntervalS: 15, historyS: 300,
  prometheusUrl: `http://127.0.0.1:${port(promSrv)}`, mlUrl: `http://127.0.0.1:${port(mlSrv)}`, kubeApi: `http://127.0.0.1:${port(kubeSrv)}`,
  kubeTokenFile: path.join(tmp, "no-token"), httpTimeoutMs: 5000, decideTimeoutMs: 5000, promConcurrency: 4,
  kubectl: fakeKubectl, scaleMin: 1, scaleMax: 10, scaleUpCooldownS: 15, scaleDownCooldownS: 60, remediationCooldownS: 300,
  hpaOverride: false, decisionLog: "", appContainer: "app", appJob: "annotated-pods", appMetricMatcher: "", podRegexFor: () => undefined,
};
function mkScaler(over = {}, deps = {}) {
  const cfg = { ...baseCfg, ...over };
  cfg.effectiveActMode = cfg.actMode === "act" && !cfg.allowExec ? "dryrun" : cfg.actMode;
  const lines = [];
  let clock = deps.clock ?? { t: 1_800_000_000_000 };
  const sc = createScaler(cfg, { emit: (l) => lines.push(JSON.parse(l)), now: () => clock.t, ...deps });
  return { sc, lines, clock };
}

// ---------------------------------------------------------------- in-process scaler
console.log("in-process scaler (fake Prometheus + fake ML + fake kube + fake kubectl)");
{
  const logFile = path.join(tmp, "decisions.jsonl");
  const { sc, lines } = mkScaler({ actMode: "observe", allowExec: false, decisionLog: logFile });
  ml.queue.push(decision());
  const [rec] = await sc.cycle();
  assert.equal(prom.unknown.length, 0, `only promexport queries sent: ${prom.unknown[0]}`);
  assert.equal(new Set(prom.queries).size, 24, "19 deploy + 11 pod queries minus 5 unused cross-check columns minus hpa_desired (no HPA)");
  const body = ml.bodies.at(-1);
  assert.equal(body.deployment, DEP);
  assert.equal(body.current_replicas, 2, "current replicas from the Deployment spec");
  assert.equal(body.max_replicas, 5, "kubeai.io/max-replicas annotation");
  assert.equal(body.revision, 3);
  assert.ok(body.series.length >= 50 && body.series.every((r) => Object.keys(r).every((k) => DEPLOY_FIELDS.includes(k))));
  assert.ok(body.pods.length > 0 && body.pods.every((p) => Object.keys(p).every((k) => POD_FIELDS.includes(k))));
  assert.ok(body.now_ts >= body.series.at(-1).ts);
  for (const k of ["timestamp", "state", "desired", "current", "action", "reasons"]) assert.ok(k in rec, `decision record has ${k}`);
  assert.equal(rec.action, "scale:2->4");
  assert.equal(rec.actions[0].outcome, "planned");
  assert.deepEqual(await calls(), [], "observe mode never runs kubectl");
  assert.equal(lines.at(-1).kind, "decision");
  const fileLines = (await readFile(logFile, "utf8")).trim().split("\n").map((l) => JSON.parse(l));
  assert.equal(fileLines.at(-1).desired, 4);
  ok("observe: promexport window -> /v1/decide -> decision JSON on stdout + DECISION_LOG, no kubectl");
}
{
  await resetCalls();
  const { sc } = mkScaler({ actMode: "dryrun", allowExec: false });
  ml.queue.push(decision({ desired_replicas: 9, actions: [{ type: "scale", target: DEP, params: { from: 2, to: 9 }, reason: "spike" }] }));
  const [rec] = await sc.cycle();
  assert.equal(rec.actions[0].clamped_to, 5, "clamped to max");
  assert.equal(rec.actions[0].outcome, "dry_run_ok");
  assert.deepEqual(await calls(), [`-n ${NS} scale deployment/${DEP} --current-replicas=2 --replicas=5 --dry-run=server`]);
  ok("dryrun: kubectl scale --current-replicas=2 --replicas=5 --dry-run=server (clamped to max 5)");
}
{
  await resetCalls();
  kubeState.hpas = [{ metadata: { name: "web-hpa" }, spec: { scaleTargetRef: { kind: "Deployment", name: DEP } } }];
  const { sc, lines } = mkScaler({ actMode: "act", allowExec: true });
  ml.queue.push(decision());
  let [rec] = await sc.cycle();
  assert.equal(rec.hpa, "web-hpa");
  assert.equal(rec.actions[0].outcome, "refused");
  assert.match(rec.actions[0].why, /HPA web-hpa/);
  assert.deepEqual(await calls(), [], "never fights an HPA");
  kubeState.hpas = [];
  kubeState.hpaForbidden = true;
  ml.queue.push(decision());
  [rec] = await sc.cycle();
  assert.equal(rec.actions[0].outcome, "refused", "HPA list unreadable -> refuse");
  kubeState.hpaForbidden = false;
  kubeState.hpas = [{ metadata: { name: "other" }, spec: { scaleTargetRef: { kind: "Deployment", name: "api" } } }];
  ml.queue.push(decision());
  const { sc: sc2 } = mkScaler({ actMode: "act", allowExec: true, hpaOverride: true });
  kubeState.hpas.push({ metadata: { name: "web-hpa" }, spec: { scaleTargetRef: { kind: "Deployment", name: DEP } } });
  [rec] = await sc2.cycle();
  assert.equal(rec.actions[0].outcome, "executed", "HPA_OVERRIDE=true");
  kubeState.hpas = [];
  assert.ok(lines.every((l) => l.kind === "decision"));
  ok("HPA guard: refuses when an HPA targets the deployment or HPAs are unreadable; HPA_OVERRIDE=true scales");
}
{
  await resetCalls();
  const clock = { t: 1_800_000_000_000 };
  const { sc } = mkScaler({ actMode: "act", allowExec: true }, { clock });
  ml.queue.push(decision());
  let [rec] = await sc.cycle();
  assert.equal(rec.actions[0].outcome, "executed");
  assert.deepEqual(await calls(), [`-n ${NS} scale deployment/${DEP} --current-replicas=2 --replicas=4`], "act runs the real scale");
  clock.t += 5000;
  ml.queue.push(decision({ desired_replicas: 5, actions: [{ type: "scale", target: DEP, params: { from: 2, to: 5 }, reason: "more" }] }));
  [rec] = await sc.cycle();
  assert.equal(rec.actions[0].outcome, "skipped");
  assert.match(rec.actions[0].why, /cooldown/);
  clock.t += 20000;
  ml.queue.push(decision({ desired_replicas: 5, actions: [{ type: "scale", target: DEP, params: { from: 2, to: 5 }, reason: "more" }] }));
  [rec] = await sc.cycle();
  assert.equal(rec.actions[0].outcome, "executed", "after the up cooldown");
  clock.t += 20000;
  ml.queue.push(decision({ state: "scale_down", desired_replicas: 1, actions: [{ type: "scale", target: DEP, params: { from: 2, to: 1 }, reason: "idle" }] }));
  [rec] = await sc.cycle();
  assert.match(rec.actions[0].why ?? "", /cooldown .* < 60s/, "scale-down cooldown is longer");
  // act without ALLOW_EXEC degrades to dryrun
  await resetCalls();
  const { sc: sc3 } = mkScaler({ actMode: "act", allowExec: false });
  ml.queue.push(decision());
  [rec] = await sc3.cycle();
  assert.equal(rec.mode, "dryrun");
  assert.equal(rec.actions[0].outcome, "dry_run_ok");
  ok("act (ALLOW_EXEC=true): real scale, own up/down cooldowns; act without ALLOW_EXEC -> dryrun");
}
{
  await resetCalls();
  const { sc, lines } = mkScaler({ actMode: "dryrun", allowExec: false });
  ml.queue.push(decision({
    state: "memory_leak", desired_replicas: 2,
    actions: [
      { type: "restart_pod", target: "web-7f6b5-bbbbb", params: { mode: "delete_pod" }, reason: "leak" },
      { type: "restart_pod", target: "db-0", params: { mode: "delete_pod" }, reason: "not ours" },
      { type: "page", target: "service owner", params: { severity: "ticket" }, reason: "memory leak: open a bug" },
      { type: "none", target: DEP, params: { scaling: "blocked" }, reason: "do NOT scale" },
    ],
  }));
  let [rec] = await sc.cycle();
  assert.equal(rec.action, "restart_pod:web-7f6b5-bbbbb");
  assert.deepEqual(rec.actions.map((a) => a.outcome), ["dry_run_ok", "refused", "paged", "none"]);
  assert.deepEqual(await calls(), [`create --raw /api/v1/namespaces/${NS}/pods/web-7f6b5-bbbbb/eviction?dryRun=All -f -`], "eviction API dry run");
  assert.ok(lines.some((l) => l.level === "PAGE" && /open a bug/.test(l.msg)));
  // revision 2 seen healthy for 4 decisions, then revision 3 (an external rollout) goes bad
  await resetCalls();
  kubeState.revision = "2";
  for (let k = 0; k < 4; k++) await sc.cycle();
  kubeState.revision = "3";
  ml.queue.push(decision({
    state: "bad_deploy", desired_replicas: 2,
    actions: [{ type: "rollback", target: DEP, params: { to: "previous ReplicaSet" }, reason: "cpu/request x2 after rollout" },
      { type: "restart_pod", target: DEP, params: { mode: "rollout_restart", pod: "web-7f6b5-aaaaa" }, reason: "single pod" }],
  }));
  [rec] = await sc.cycle();
  assert.deepEqual(rec.actions.map((a) => a.outcome), ["dry_run_ok", "dry_run_ok"]);
  const c = await calls();
  assert.equal(c[0], `-n ${NS} rollout undo deployment/${DEP} --to-revision=2 --dry-run=server`, "explicit known-good revision");
  assert.match(c[1], /^-n shop patch deployment\/web --type=strategic -p .*restartedAt.* --dry-run=server$/);
  // a fresh agent has seen no healthy older revision: rollback refused and paged
  await resetCalls();
  const { sc: fresh } = mkScaler({ actMode: "dryrun", allowExec: false });
  ml.queue.push(decision({ state: "bad_deploy", actions: [{ type: "rollback", target: DEP, params: {}, reason: "after rollout" }] }));
  [rec] = await fresh.cycle();
  assert.equal(rec.actions[0].outcome, "refused");
  assert.match(rec.actions[0].why, /no known-good earlier revision/);
  assert.equal(rec.actions.at(-1).type, "page");
  assert.deepEqual(await calls(), []);
  ok("remediation via allowlist: evict own pod, refuse foreign pod, rollback --to-revision=<known good> + rollout restart dry runs, page logged");
}
{
  await resetCalls();
  const { sc } = mkScaler({ actMode: "act", allowExec: true });
  ml.queue.push(decision({ state: "memory_leak", actions: [{ type: "restart_pod", target: "web-7f6b5-aaaaa", params: { mode: "delete_pod" }, reason: "leak" }] }));
  await sc.cycle();
  ml.queue.push(decision({ state: "memory_leak", actions: [{ type: "restart_pod", target: "web-7f6b5-bbbbb", params: { mode: "delete_pod" }, reason: "leak" }] }));
  const [rec] = await sc.cycle();
  assert.equal(rec.actions[0].outcome, "skipped");
  assert.match(rec.actions[0].why, /remediation cooldown/);
  assert.equal((await calls()).length, 1, "one pod restart per REMEDIATION_COOLDOWN_S");
  ok("remediation cooldown per deployment");
}
{
  // Undo ping-pong (review fix): one rollback per bad revision; the agent's own undo is not a
  // rollout; a second undo is refused and paged even when the engine keeps asking for it.
  await resetCalls();
  kubeState.revision = "2";
  const clock = { t: 1_900_000_000_000 };
  const { sc } = mkScaler({ actMode: "act", allowExec: true }, { clock });
  for (let k = 0; k < 4; k++) {
    await sc.cycle();
    clock.t += 15000;
  }
  kubeState.revision = "3"; // external deploy (the bad one)
  const rb = () => decision({ state: "error_fault", desired_replicas: 2, actions: [{ type: "rollback", target: DEP, params: {}, reason: "5xx within 15 min of a rollout" }, { type: "page", target: "service owner", params: {}, reason: "5xx" }] });
  ml.queue.push(rb());
  let [rec] = await sc.cycle();
  assert.equal(rec.actions[0].outcome, "executed");
  assert.deepEqual(await calls(), [`-n ${NS} rollout undo deployment/${DEP} --to-revision=2`]);
  kubeState.revision = "4"; // the undo re-creates revision 2's template as revision 4
  for (let k = 0; k < 25; k++) { // 25 x 15 s > ROLLOUT_RECENT_S and > the 300 s remediation cooldown
    clock.t += 15000;
    ml.queue.push(rb());
    [rec] = await sc.cycle();
    assert.notEqual(rec.actions[0].outcome, "executed", `no second undo (cycle ${k})`);
    if (k === 0) {
      assert.equal(ml.bodies.at(-1).revision, 3, "the agent-created revision 4 is reported as the revision before it");
      assert.ok(ml.bodies.at(-1).agent_rollout_ts?.length === 1, "own rollout times sent to /v1/decide");
    }
  }
  assert.equal(rec.actions[0].outcome, "refused");
  assert.match(rec.actions[0].why, /created by this agent/);
  assert.ok(rec.actions.some((a) => a.type === "page" && /rollback refused/.test(a.reason)));
  assert.equal((await calls()).filter((c) => c.includes("rollout undo")).length, 1, "exactly one undo over 6+ minutes");
  ok("rollback: --to-revision=<known good>, own undo not rollout evidence, second undo refused + paged (no ping-pong)");
  kubeState.revision = "3";
}
{
  // our own rollout restart of revision 3 creates revision 4; a later rollback goes to before
  // revision 3's code (2), not to 3 (which would only revert the restartedAt annotation)
  await resetCalls();
  kubeState.revision = "2";
  const clock = { t: 1_950_000_000_000 };
  const { sc } = mkScaler({ actMode: "act", allowExec: true }, { clock });
  const tick = async (d) => { if (d) ml.queue.push(d); clock.t += 15000; return (await sc.cycle())[0]; };
  for (let k = 0; k < 4; k++) await tick();
  kubeState.revision = "3";
  for (let k = 0; k < 4; k++) await tick(); // revision 3 looks healthy at first
  let rec = await tick(decision({ state: "memory_leak", actions: [{ type: "restart_pod", target: DEP, params: { mode: "rollout_restart", pod: "web-7f6b5-aaaaa" }, reason: "single pod leak" }] }));
  assert.equal(rec.actions[0].outcome, "executed");
  kubeState.revision = "4";
  for (let k = 0; k < 21; k++) await tick(); // past the 300 s remediation cooldown
  rec = await tick(decision({ state: "bad_deploy", actions: [{ type: "rollback", target: DEP, params: {}, reason: "cpu/request x2" }] }));
  assert.equal(rec.actions[0].outcome, "executed");
  assert.deepEqual((await calls()).at(-1), `-n ${NS} rollout undo deployment/${DEP} --to-revision=2`);
  kubeState.revision = "3";
  ok("rollback after our own rollout restart targets the revision before the restarted code");
}
{
  // Pod ownership (review fix): no owner-verified pod list -> eviction refused, never Prometheus names
  await resetCalls();
  const leak = () => decision({ state: "memory_leak", actions: [{ type: "restart_pod", target: "web-7f6b5-aaaaa", params: { mode: "delete_pod" }, reason: "leak" }] });
  const { sc } = mkScaler({ actMode: "act", allowExec: true });
  kubeState.podsFail = true;
  ml.queue.push(leak());
  let [rec] = await sc.cycle();
  assert.equal(rec.actions[0].outcome, "refused");
  assert.match(rec.actions[0].why, /cannot verify pod ownership/);
  kubeState.podsFail = false;
  kubeState.noSelector = true;
  ml.queue.push(leak());
  [rec] = await sc.cycle();
  assert.match(rec.actions[0].why, /cannot verify pod ownership/, "empty selector -> refused");
  kubeState.noSelector = false;
  assert.deepEqual(await calls(), []);
  // PDB blocks the eviction (429) -> skipped, surge rollout restart instead
  await writeFile(tmpPdb, "1");
  ml.queue.push(leak());
  [rec] = await sc.cycle();
  await rm(tmpPdb, { force: true });
  assert.equal(rec.actions[0].outcome, "skipped");
  assert.match(rec.actions[0].why, /PodDisruptionBudget/);
  const fb = rec.actions.find((a) => a.params?.fallback_for === "pdb");
  assert.equal(fb?.outcome, "executed");
  const c = await calls();
  assert.equal(c.length, 2);
  assert.match(c[0], /^create --raw \/api\/v1\/namespaces\/shop\/pods\/web-7f6b5-aaaaa\/eviction -f -$/);
  assert.equal(c[1], `-n ${NS} rollout restart deployment/${DEP}`);
  ok("evictions: refused without owner-verified pods (API error / empty selector); PDB 429 -> skipped + surge rollout restart");
}
{
  // Several Deployments in one namespace (review fix): namespace-wide app metrics -> refuse;
  // a per-Deployment APP_METRIC_MATCHER template is expanded per Deployment.
  kubeState.extraDeployment = true;
  const { sc, lines } = mkScaler({ actMode: "observe", allowExec: false });
  const recs = await sc.cycle();
  assert.equal(recs.length, 0);
  assert.equal(lines.at(-1).kind, "decision_error");
  assert.equal(lines.at(-1).stage, "config");
  prom.queries.length = 0;
  const tpl = 'app="{deployment}"';
  const { sc: sc2, lines: l2 } = mkScaler({ actMode: "observe", allowExec: false, appMetricMatcher: tpl, appMatcherPerDeployment: true, appMatcherFor: (d) => tpl.replaceAll("{deployment}", d) });
  await sc2.cycle();
  assert.notEqual(l2.at(-1).stage, "config");
  assert.ok(prom.queries.some((q) => q.includes('http_requests_total{namespace="shop",app="web"}')), "matcher expanded for this Deployment");
  kubeState.extraDeployment = false;
  ok('multi-Deployment namespace: refuses without an APP_METRIC_MATCHER "{deployment}" template; template expanded per Deployment');
}
{
  const { sc, lines } = mkScaler({ actMode: "observe", allowExec: false });
  ml.fail = 1;
  await sc.cycle();
  assert.equal(lines.at(-1).kind, "decision_error");
  assert.equal(lines.at(-1).stage, "ml");
  prom.down = true;
  await sc.cycle();
  assert.equal(lines.at(-1).stage, "prometheus");
  prom.down = false;
  const [rec] = await sc.cycle();
  assert.equal(rec.kind, "decision", "recovers on the next cycle");
  const st = sc.status();
  assert.equal(st.deployments[0].decisions, 1);
  ok("ML down / Prometheus down -> decision_error record, recovered next cycle");
}
{
  // fetchWindow against the fake Prometheus directly: deployment selector discovery path
  const w = await fetchWindow(baseCfg, NS, DEP, { nowS: 1_800_000_000 });
  assert.ok(w.series.length === 61 && w.series.at(-1).ts === 1_800_000_000);
  const { sc } = mkScaler({ actMode: "observe", allowExec: false, deployments: [], deploySelector: "tier=web" });
  const [rec] = await sc.cycle();
  assert.equal(rec.deployment, DEP, "DEPLOY_SELECTOR discovery");
  ok("DEPLOY_SELECTOR discovery + HISTORY_S grid (300 s / 5 s = 61 rows)");
}

// ---------------------------------------------------------------- child process: index.js wiring
console.log("integration (agent child process, deployment loop only)");
{
  await resetCalls();
  const logFile = path.join(tmp, "child-decisions.jsonl");
  const env = {
    PATH: process.env.PATH, HOME: process.env.HOME, PORT: "0",
    PROMETHEUS_URL: baseCfg.prometheusUrl, KUBE_API: baseCfg.kubeApi, ML_URL: baseCfg.mlUrl,
    KUBE_TOKEN_FILE: path.join(tmp, "no-token"), NAMESPACE: NS, KUBECTL: fakeKubectl,
    OOM_LOOP: "false", DEPLOYMENTS: DEP, DECIDE_INTERVAL_S: "1", HISTORY_S: "300", ACT_MODE: "dryrun",
    DECISION_LOG: logFile, GOOGLE_API_KEY: "",
  };
  const child = spawn(process.execPath, ["index.js"], { cwd: ROOT, env, stdio: ["ignore", "pipe", "pipe"] });
  let out = "";
  child.stdout.on("data", (d) => (out += d));
  child.stderr.on("data", (d) => (out += d));
  try {
    const agentPort = await new Promise((resolve, reject) => {
      const to = setTimeout(() => reject(new Error(`agent did not start:\n${out}`)), 10000);
      const iv = setInterval(() => {
        const m = /"msg":"agent listening","port":(\d+)/.exec(out);
        if (m) {
          clearTimeout(to);
          clearInterval(iv);
          resolve(Number(m[1]));
        }
      }, 20);
    });
    ml.queue.push(decision());
    const base = `http://127.0.0.1:${agentPort}`;
    let d;
    const deadline = Date.now() + 15000;
    for (;;) {
      const r = await fetch(`${base}/decisions`);
      d = await r.json();
      if (d.decisions?.length >= 2) break;
      if (Date.now() > deadline) throw new Error(`no decisions: ${JSON.stringify(d)}\n${out}`);
      await new Promise((r2) => setTimeout(r2, 100));
    }
    const st = await (await fetch(`${base}/status`)).json();
    assert.equal(st.scaler.mode, "dryrun");
    assert.equal(st.config.oomLoop, false);
    assert.ok(!("googleApiKey" in st.config));
    assert.ok(d.decisions.some((x) => x.action === "scale:2->4" && x.actions[0].outcome === "dry_run_ok"));
    assert.match(out, /"kind":"decision","timestamp":/, "decision lines on stdout");
    const lines = (await readFile(logFile, "utf8")).trim().split("\n").map((l) => JSON.parse(l));
    assert.ok(lines.length >= 2 && lines.every((l) => l.deployment === DEP));
    assert.equal((await fetch(`${base}/pods/x`)).status, 404, "OOM loop disabled");
    ok(`index.js: DEPLOYMENTS loop every DECIDE_INTERVAL_S, /decisions, /status.scaler, DECISION_LOG (${lines.length} lines)`);
  } finally {
    child.kill("SIGTERM");
  }
}

for (const s of [promSrv, mlSrv, kubeSrv]) s.close();
await rm(tmp, { recursive: true, force: true });
console.log(`\n${passed} checks passed`);
process.exit(0);
