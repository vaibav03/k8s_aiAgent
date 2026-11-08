// Smoke test: fake Prometheus + fake kube API + fake ML service on random
// ports, the real agent (index.js) as a child process with a tiny POLL_MS.
// Plus a few pure unit checks (quantity parser, hysteresis, allowlist).
//
// Run: npm test

import http from "node:http";
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { mkdtemp, writeFile, chmod, readFile, access } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { parseQuantity } from "../lib/quantity.js";
import { createAlertState, updateAlert } from "../lib/alert.js";
import { validateRecommendation, parseRecommendation, runKubectl } from "../lib/remediation.js";
import { buildPredictItem, predictBatch } from "../lib/ml.js";
import { summarizePod } from "../lib/kube.js";
import { buildWorkingSetQuery } from "../lib/prom.js";
import { createAgent } from "../lib/agent.js";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const MI = 2 ** 20;
let passed = 0;
const ok = (name) => {
  passed += 1;
  console.log(`  ok - ${name}`);
};

// ---------------------------------------------------------------- unit checks
console.log("unit");
{
  const cases = [
    ["800Mi", 800 * MI], ["1Gi", 2 ** 30], ["768Mi", 768 * MI], ["512Ki", 512 * 1024], ["1Ti", 2 ** 40],
    ["128M", 128e6], ["1G", 1e9], ["500K", 5e5], ["500k", 5e5], ["805306368", 805306368],
    ["129e6", 129e6], ["1.5Gi", 1.5 * 2 ** 30], ["", null], [null, null], ["abc", null], ["12Mb", null],
  ];
  for (const [inp, want] of cases) assert.equal(parseQuantity(inp), want, `parseQuantity(${inp})`);
  ok("parseQuantity handles Ki/Mi/Gi/Ti, K/M/G, bytes, e-notation, invalid");
}
{
  const s = createAlertState();
  const o = { k: 2, n: 3, cooldownMs: 1000, now: 0 };
  assert.equal(updateAlert(s, true, o).fire, false, "1 anomaly must not fire");
  assert.equal(updateAlert(s, false, o).fire, false);
  const third = updateAlert(s, true, o);
  assert.equal(third.fire, true, "2 of 3 must fire");
  assert.equal(updateAlert(s, true, { ...o, now: 500 }).fire, false, "cooldown suppresses refire");
  assert.equal(updateAlert(s, true, { ...o, now: 1500 }).fire, true, "fires again after cooldown");
  assert.throws(() => updateAlert(s, "1", o), /boolean/, "non-boolean is_anomaly rejected");
  ok("k-of-n hysteresis + cooldown");
}
{
  const ctx = {
    namespace: "default",
    deployment: "finalpod",
    containers: [{ name: "app", memoryLimit: "800Mi" }, { name: "istio-proxy", memoryLimit: null }],
    maxMemoryBytes: 4 * 2 ** 30,
    maxReplicas: 10,
  };
  const good = validateRecommendation({ action: "patch_memory_limit", target: "finalpod", params: { container: "app", memory: "1Gi" } }, ctx);
  assert.equal(good.ok, true);
  assert.deepEqual(good.argv, ["-n", "default", "set", "resources", "deployment/finalpod", "-c", "app", "--limits=memory=1024Mi"]);
  assert.equal(validateRecommendation({ action: "delete_namespace", target: "finalpod" }, ctx).ok, false);
  assert.equal(validateRecommendation({ action: "scale", target: "other", params: { replicas: 2 } }, ctx).ok, false);
  assert.equal(validateRecommendation({ action: "scale", target: "finalpod", params: { replicas: 99 } }, ctx).ok, false);
  assert.equal(validateRecommendation({ action: "patch_memory_limit", target: "finalpod", params: { container: "app", memory: "512Mi" } }, ctx).ok, false);
  assert.equal(validateRecommendation({ action: "patch_memory_limit", target: "finalpod", params: { container: "app", memory: "1Gi; rm -rf /" } }, ctx).ok, false);
  const rr = validateRecommendation({ action: "rollout_restart", target: "deployment/finalpod" }, ctx, new Date("2026-01-01T00:00:00Z"));
  assert.equal(rr.ok, true);
  assert.deepEqual(rr.argv, ["-n", "default", "rollout", "restart", "deployment/finalpod"]);
  assert.deepEqual(rr.dryRunArgv, ["-n", "default", "patch", "deployment/finalpod", "--type=strategic", "-p",
    '{"spec":{"template":{"metadata":{"annotations":{"kubectl.kubernetes.io/restartedAt":"2026-01-01T00:00:00.000Z"}}}}}']);
  // Lower bounds on a new memory limit.
  const noLimit = (ws) => ({ ...ctx, containers: [{ name: "app", memoryLimit: null, workingSetBytes: ws }] });
  const patch = (mem, c) => validateRecommendation({ action: "patch_memory_limit", target: "finalpod", params: { container: "app", memory: mem } }, c);
  assert.equal(patch("1Mi", noLimit(null)).ok, false, "no limit + no working set -> reject");
  assert.equal(patch("1Mi", noLimit(10 * MI)).ok, false, "below 64Mi floor");
  assert.equal(patch("100Mi", noLimit(300 * MI)).ok, false, "below working set x1.2");
  assert.equal(patch("400Mi", noLimit(300 * MI)).ok, true);
  assert.equal(patch("1Gi", { ...ctx, containers: [{ name: "app", memoryLimit: "800Mi", workingSetBytes: 900 * MI }] }).ok, false, "above limit but below working set x1.2");
  assert.equal(validateRecommendation({ action: "none" }, ctx).argv, null);
  assert.deepEqual(parseRecommendation('```json\n{"action":"none"}\n```'), { action: "none" });
  assert.throws(() => parseRecommendation("kubectl delete pod x"));
  ok("LLM recommendation allowlist + argv builder + memory-limit floors");
}
{
  const q = buildWorkingSetQuery("default", ["a.b", "c"]);
  assert.match(q, /^sum by \(pod, container\) \(/);
  assert.match(q, /container_memory_working_set_bytes\{namespace="default",pod=~"a\\\\.b\|c",container!="",container!="POD"\}/);
  ok("PromQL query shape (per container)");
}

// ------------------------------------------------------------- fake services
function mkPod(name, { limits = ["800Mi"], crash = false, restarts = 0 } = {}) {
  return {
    metadata: {
      name,
      namespace: "default",
      labels: { app: "finalpod", "pod-template-hash": "7c649c5fc" },
      ownerReferences: [{ kind: "ReplicaSet", name: "finalpod-7c649c5fc" }],
    },
    spec: { containers: limits.map((l, i) => ({ name: i === 0 ? "app" : `side${i}`, resources: l ? { limits: { memory: l } } : {} })) },
    status: {
      phase: "Running",
      containerStatuses: limits.map((_, i) => ({
        name: i === 0 ? "app" : `side${i}`,
        restartCount: i === 0 ? restarts : 0,
        state: crash && i === 0 ? { waiting: { reason: "CrashLoopBackOff" } } : { running: {} },
        lastState: restarts && i === 0 ? { terminated: { reason: "OOMKilled", finishedAt: new Date(Date.now() - 60000).toISOString() } } : {},
      })),
    },
  };
}

{
  const sp = summarizePod(mkPod("x", { limits: ["800Mi", "128Mi", null], restarts: 3 }));
  assert.deepEqual(sp.containers.map((c) => c.limitBytes), [800 * MI, 128 * MI, null], "per-container limits, never summed");
  assert.deepEqual(sp.containers.map((c) => c.restartCount), [3, 0, 0]);
  assert.equal(sp.restartCount, 3);
  assert.equal(sp.deployment, "finalpod");
  assert.ok(sp.secondsSinceRestart >= 59 && sp.secondsSinceRestart < 120);
  assert.ok(sp.containers[0].secondsSinceRestart >= 59 && sp.containers[1].secondsSinceRestart === null);
  assert.equal(summarizePod(mkPod("y", { limits: ["0"] })).containers[0].limitBytes, null, "limit 0 -> null");
  assert.equal(summarizePod(mkPod("z", { crash: true })).crashLooping, true);
  const item = buildPredictItem({ namespace: "default" }, { name: "p" }, { limitBytes: 0, restartCount: 1, secondsSinceRestart: null }, []);
  assert.equal(item.limit_bytes, null, "non-positive limit sent as null");
  ok("pod summary: per-container limits/restarts, owner deployment, crashloop");
}

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

const seen = { kubeSelectors: [], promQueries: [], mlCalls: 0, mlItems: [] };
const PODS = [
  mkPod("finalpod-7c649c5fc-aaaaa", { limits: ["800Mi"] }),
  mkPod("finalpod-7c649c5fc-bbbbb", { limits: ["1Gi", null] }),
  mkPod("finalpod-7c649c5fc-ccccc", { crash: true, restarts: 5 }),
];
let promTick = 0;

const kubeSrv = await listen((req, res) => {
  const u = new URL(req.url, "http://x");
  if (u.pathname === "/api/v1/namespaces/default/pods") {
    seen.kubeSelectors.push(u.searchParams.get("labelSelector"));
    return json(res, 200, { kind: "PodList", items: PODS });
  }
  json(res, 404, { message: "not found" });
});

const promSrv = await listen((req, res) => {
  const u = new URL(req.url, "http://x");
  if (u.pathname !== "/api/v1/query") return json(res, 404, {});
  seen.promQueries.push(u.searchParams.get("query"));
  promTick += 1;
  const ts = 1_700_000_000 + promTick * 15;
  // pod a climbs toward its limit; pod b is flat; pod c has no data (missing)
  json(res, 200, {
    status: "success",
    data: {
      resultType: "vector",
      result: [
        { metric: { pod: "finalpod-7c649c5fc-aaaaa", container: "app" }, value: [ts, String((600 + promTick * 20) * MI)] },
        { metric: { pod: "finalpod-7c649c5fc-bbbbb", container: "app" }, value: [ts, String(300 * MI)] },
        { metric: { pod: "finalpod-7c649c5fc-bbbbb", container: "side1" }, value: [ts, String(150 * MI)] },
      ],
    },
  });
});

const mlSrv = await listen(async (req, res) => {
  if (req.url !== "/v1/predict/batch" || req.method !== "POST") return json(res, 404, {});
  const body = JSON.parse(await readBody(req));
  seen.mlCalls += 1;
  if (seen.mlCalls === 1) return json(res, 503, { detail: "model not loaded" }); // ML down once
  seen.mlItems.push(...body.items);
  json(res, 200, {
    results: body.items.map((it) => {
      const anomalous = it.pod.endsWith("aaaaa");
      const last = it.samples.at(-1).working_set_bytes;
      return {
        pod: it.pod,
        namespace: it.namespace,
        model_version: "fake-1",
        p_oom: anomalous ? 0.9 : 0.02,
        is_anomaly: anomalous,
        threshold: 0.5,
        horizon_s: 60,
        usage_fraction: it.limit_bytes ? last / it.limit_bytes : null,
        trend_bytes_per_s: 0,
        time_to_limit_s: null,
        forecast: [],
        reasons: [anomalous ? "fake anomaly" : "fake ok"],
      };
    }),
  });
});

const port = (s) => s.address().port;
const tmp = await mkdtemp(path.join(tmpdir(), "kubeai-smoke-"));
const marker = path.join(tmp, "kubectl-called");
const fakeKubectl = path.join(tmp, "kubectl");
// Mimics real kubectl flag parsing for the subcommands we use: `rollout restart`
// has no --dry-run flag (kubectl prints "unknown flag: --dry-run").
await writeFile(fakeKubectl, `#!/bin/sh
echo "$@" >> "${marker}"
case " $* " in
  *" rollout "*"--dry-run"*) echo "Error: unknown flag: --dry-run" >&2; exit 1 ;;
esac
echo "deployment.apps/finalpod updated (server dry run)"
`);
await chmod(fakeKubectl, 0o755);

// -------------------------------------------------- integration: child agent
console.log("integration (agent child process)");
const env = {
  PATH: process.env.PATH,
  HOME: process.env.HOME,
  PORT: "0",
  PROMETHEUS_URL: `http://127.0.0.1:${port(promSrv)}`,
  KUBE_API: `http://127.0.0.1:${port(kubeSrv)}`,
  KUBE_TOKEN_FILE: path.join(tmp, "no-token"),
  ML_URL: `http://127.0.0.1:${port(mlSrv)}`,
  NAMESPACE: "default",
  POD_SELECTOR: "app=finalpod",
  POLL_MS: "100",
  WINDOW: "5",
  ALERT_K: "2",
  ALERT_N: "3",
  COOLDOWN_MS: "600000",
  GOOGLE_API_KEY: "",
  ALLOW_EXEC: "false",
  KUBECTL: fakeKubectl,
};
const child = spawn(process.execPath, ["index.js"], { cwd: ROOT, env, stdio: ["ignore", "pipe", "pipe"] });
let out = "";
child.stdout.on("data", (d) => (out += d));
child.stderr.on("data", (d) => (out += d));

const cleanup = () => {
  child.kill("SIGTERM");
  for (const s of [kubeSrv, promSrv, mlSrv]) s.close();
};

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
  const base = `http://127.0.0.1:${agentPort}`;
  const get = async (p) => {
    const r = await fetch(base + p);
    return { code: r.status, body: await r.json() };
  };

  assert.deepEqual((await get("/healthz")).body, { status: "ok" });
  ok("GET /healthz");

  const A = "finalpod-7c649c5fc-aaaaa";
  const B = "finalpod-7c649c5fc-bbbbb";
  const C = "finalpod-7c649c5fc-ccccc";
  let st;
  const deadline = Date.now() + 15000;
  for (;;) {
    st = (await get("/status")).body;
    const a = st.pods.find((p) => p.pod === A);
    if (a?.alert?.fire_count >= 1 && st.ticks >= 6) break;
    if (Date.now() > deadline) throw new Error(`timed out waiting for alert; status=${JSON.stringify(st, null, 1)}\nlogs:\n${out}`);
    await new Promise((r) => setTimeout(r, 100));
  }

  assert.equal(st.config.llmEnabled, false);
  assert.ok(!("googleApiKey" in st.config), "secret not exposed");
  assert.ok(seen.kubeSelectors.every((s) => s === "app=finalpod"), "label selector used");
  assert.ok(seen.promQueries.every((q) => q.includes("container_memory_working_set_bytes") && q.includes('container!="POD"')));
  assert.equal(seen.promQueries.length >= st.ticks - 1, true, "one PromQL query per tick");
  ok(`discovery + one PromQL query per tick (${seen.promQueries.length} queries / ${st.ticks} ticks)`);

  assert.ok(seen.mlCalls >= 2 && /HTTP 503/.test(out), "ML 503 logged and agent recovered");
  ok("ML service down -> logged, retried next tick");

  const itemA = seen.mlItems.find((i) => i.pod === A);
  assert.equal(itemA.limit_bytes, 800 * MI);
  assert.equal(itemA.namespace, "default");
  assert.ok(itemA.samples.length >= 2 && itemA.samples.every((s, i, arr) => i === 0 || s.ts > arr[i - 1].ts));
  assert.ok(seen.mlItems.every((i) => i.pod !== C), "pod with no metrics is not sent (missing != 0)");
  assert.ok(seen.mlItems.filter((i) => i.pod === A).every((i) => i.samples.length <= 5), "WINDOW respected");
  const itemsB = seen.mlItems.filter((i) => i.pod === B);
  assert.ok(itemsB.some((i) => i.limit_bytes === 2 ** 30) && itemsB.some((i) => i.limit_bytes === null), "one item per container, each with its own limit");
  assert.ok(itemsB.every((i) => i.samples.every((x) => x.working_set_bytes !== 450 * MI)), "container memory never summed");
  ok("predict items: per-container limits, chronological samples, missing pods skipped, window cap");

  const a = st.pods.find((p) => p.pod === A);
  const b = st.pods.find((p) => p.pod === B);
  const c = st.pods.find((p) => p.pod === C);
  assert.equal(a.latest_prediction.model_version, "fake-1");
  assert.equal(a.alert.active, true);
  assert.equal(a.alert.fire_count, 1, "cooldown prevents re-firing");
  assert.ok(a.alert.recent.filter(Boolean).length >= 2);
  assert.equal(a.last_recommendation.action, "none");
  assert.match(a.last_recommendation.reason, /GOOGLE_API_KEY/);
  assert.equal(b.alert.active, false);
  assert.equal(b.alert.fire_count, 0);
  assert.equal(b.limit_bytes, 2 ** 30);
  assert.equal(b.latest_prediction.is_anomaly, false);
  assert.equal(c.crash_looping, true);
  assert.equal(c.samples, 0);
  assert.ok(c.missing_since);
  assert.match(out, /"level":"CRITICAL".*CrashLoopBackOff/);
  assert.equal(b.latest_prediction.container, "app", "pod prediction = most at-risk container");
  assert.deepEqual(b.containers.map((x) => [x.name, x.limit_bytes]), [["app", 2 ** 30], ["side1", null]]);
  ok("/status: predictions, hysteresis alert after k anomalies, healthy pod quiet, crashloop CRITICAL");

  const one = await get(`/pods/${A}`);
  assert.equal(one.code, 200);
  assert.equal(one.body.pod, A);
  assert.equal((await get("/pods/nope")).code, 404);
  ok("GET /pods/:name");

  await assert.rejects(access(marker), "kubectl must not be executed");
  ok("no kubectl exec happened");
} finally {
  cleanup();
}

// ------------------------------- in-process: LLM path with a fake model
console.log("in-process (fake LLM -> allowlist -> kubectl dry-run)");
{
  const fakeLlm = {
    calls: 0,
    async recommend(ctx) {
      this.calls += 1;
      return { action: "patch_memory_limit", target: "finalpod", params: { container: "app", memory: "1Gi" }, reason: `leak in ${ctx.pod}` };
    },
  };
  const results = [true, true];
  let i = 0;
  const cfg = {
    namespace: "default", window: 5, alertK: 2, alertN: 3, cooldownMs: 600000, maxMemoryLimit: "4Gi", maxReplicas: 10,
    allowExec: false, kubectl: fakeKubectl, mlUrl: "", prometheusUrl: "", httpTimeoutMs: 1000, pollMs: 10,
  };
  const ml2 = await listen(async (req, res) => {
    const body = JSON.parse(await readBody(req));
    json(res, 200, { results: body.items.map((it) => ({ pod: it.pod, is_anomaly: results[Math.min(i++, results.length - 1)], p_oom: 0.9 })) });
  });
  const prom2 = await listen((req, res) => {
    promTick += 1;
    json(res, 200, { status: "success", data: { result: [{ metric: { pod: PODS[0].metadata.name, container: "app" }, value: [1_800_000_000 + promTick, String(700 * MI)] }] } });
  });
  cfg.mlUrl = `http://127.0.0.1:${port(ml2)}`;
  cfg.prometheusUrl = `http://127.0.0.1:${port(prom2)}`;
  const agent = createAgent(cfg, { kube: { listPods: async () => [summarizePod(PODS[0])] }, llm: fakeLlm });
  for (let t = 0; t < 4; t++) await agent.tick();
  const p = agent.pod(PODS[0].metadata.name);
  ml2.close();
  prom2.close();
  assert.equal(fakeLlm.calls, 1, "LLM called once per alert (hysteresis + cooldown)");
  assert.equal(p.last_recommendation.valid, true);
  assert.equal(p.last_recommendation.kubectl.dryRun, true);
  const called = (await readFile(marker, "utf8")).trim();
  assert.equal(called, "-n default set resources deployment/finalpod -c app --limits=memory=1024Mi --dry-run=server");
  ok(`validated recommendation executed as server dry-run: kubectl ${called}`);
}

{
  // every allowlisted action's dry-run argv must be accepted by kubectl's flag parser
  const ctx = { namespace: "default", deployment: "finalpod", containers: [{ name: "app", memoryLimit: "800Mi", workingSetBytes: 700 * MI }], maxMemoryBytes: 4 * 2 ** 30, maxReplicas: 10 };
  const recs = [
    { action: "rollout_restart", target: "finalpod" },
    { action: "scale", target: "finalpod", params: { replicas: 2 } },
    { action: "patch_memory_limit", target: "finalpod", params: { container: "app", memory: "1Gi" } },
  ];
  for (const rec of recs) {
    const v = validateRecommendation(rec, ctx);
    const r = await runKubectl({ allowExec: false, kubectl: fakeKubectl }, v.argv, v.dryRunArgv);
    assert.equal(r.ok, true, `${rec.action} dry run failed: ${r.stderr}`);
    assert.equal(r.argv.at(-1), "--dry-run=server");
  }
  const rr = validateRecommendation(recs[0], ctx);
  const real = await runKubectl({ allowExec: true, kubectl: fakeKubectl }, rr.argv, rr.dryRunArgv);
  assert.deepEqual(real.argv, rr.argv, "ALLOW_EXEC=true runs the real rollout restart");
  ok("every action's dry run passes kubectl flag parsing (rollout restart -> patch restartedAt)");
}

{
  // ML client: chunks of <= 500, and a 422 batch is retried item by item.
  const calls = { batch: [], single: 0 };
  const valid = (it) => it.samples.length >= 2 && (it.limit_bytes === null || it.limit_bytes > 0);
  const result = (it) => ({ pod: it.pod, is_anomaly: false, p_oom: 0.1 });
  const ml3 = await listen(async (req, res) => {
    const body = JSON.parse(await readBody(req));
    if (req.url === "/v1/predict/batch") {
      calls.batch.push(body.items.length);
      if (body.items.length > 500 || !body.items.every(valid)) return json(res, 422, { detail: "invalid" });
      return json(res, 200, { results: body.items.map(result) });
    }
    calls.single += 1;
    if (!valid(body)) return json(res, 422, { detail: "invalid" });
    json(res, 200, result(body));
  });
  const smp = [{ ts: 1, working_set_bytes: 1 }, { ts: 2, working_set_bytes: 2 }];
  const items = Array.from({ length: 1020 }, (_, i) => ({ pod: `p${i}`, namespace: "default", samples: smp, limit_bytes: 1e9 }));
  items[700] = { ...items[700], samples: smp.slice(0, 1) }; // one bad item
  const rejected = [];
  const out = await predictBatch({ mlUrl: `http://127.0.0.1:${port(ml3)}`, httpTimeoutMs: 5000, mlBatchMax: 500 }, items, { onItemError: (it) => rejected.push(it.pod) });
  ml3.close();
  assert.deepEqual(calls.batch, [500, 500, 20], "chunks of at most 500");
  assert.equal(calls.single, 500, "only the 422 chunk is retried per item");
  assert.equal(out.length, 1020);
  assert.equal(out[700], null);
  assert.deepEqual(rejected, ["p700"]);
  assert.ok(out.every((r, i) => i === 700 || r.pod === `p${i}`), "results stay aligned with items");
  ok("ML client: 1020 items -> 3 chunks; one invalid item costs only its own prediction");
}

{
  // Sidecar: app at 795/800Mi with a roomy 1Gi sidecar -> scored against 800Mi.
  const pod = mkPod("finalpod-7c649c5fc-sssss", { limits: ["800Mi", "1Gi"] });
  const mlItems = [];
  const ml4 = await listen(async (req, res) => {
    const body = JSON.parse(await readBody(req));
    mlItems.push(...body.items);
    json(res, 200, { results: body.items.map((it) => {
      const uf = it.limit_bytes ? it.samples.at(-1).working_set_bytes / it.limit_bytes : null;
      return { pod: it.pod, is_anomaly: uf !== null && uf > 0.9, p_oom: uf !== null && uf > 0.9 ? 0.9 : 0.0, usage_fraction: uf };
    }) });
  });
  let t = 0;
  const prom4 = await listen((req, res) => {
    t += 1;
    json(res, 200, { status: "success", data: { result: [
      { metric: { pod: pod.metadata.name, container: "app" }, value: [1_900_000_000 + t * 15, String(795 * MI)] },
      { metric: { pod: pod.metadata.name, container: "side1" }, value: [1_900_000_000 + t * 15, String(60 * MI)] },
    ] } });
  });
  const cfg = { namespace: "default", window: 5, alertK: 2, alertN: 3, cooldownMs: 600000, maxMemoryLimit: "4Gi", maxReplicas: 10,
    allowExec: false, kubectl: fakeKubectl, mlUrl: `http://127.0.0.1:${port(ml4)}`, prometheusUrl: `http://127.0.0.1:${port(prom4)}`, httpTimeoutMs: 1000, pollMs: 10, mlBatchMax: 500 };
  const agent = createAgent(cfg, { kube: { listPods: async () => [summarizePod(pod)] }, llm: null });
  for (let k = 0; k < 4; k++) await agent.tick();
  ml4.close();
  prom4.close();
  const v = agent.pod(pod.metadata.name);
  assert.ok(mlItems.some((i) => i.limit_bytes === 800 * MI) && mlItems.some((i) => i.limit_bytes === 2 ** 30));
  assert.equal(v.latest_prediction.container, "app");
  assert.ok(v.latest_prediction.usage_fraction > 0.99);
  assert.equal(v.alert.fire_count, 1, "app container about to be OOM-killed is flagged");
  ok("sidecar with a large limit does not hide an app container at its own limit");
}

console.log(`\n${passed} checks passed`);
process.exit(0);
