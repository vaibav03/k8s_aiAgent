// Offline replay (review fix evidence): re-runs the agent's decisions of an A/B agent run
// against the stored Prometheus data with the v1 (as run live) and v2 (live-target
// mem_growth, pod-scoped KSM) query sets, posting both windows to a local ML service.
// Usage: (cd ml_service && PYTHONPATH=. python -m uvicorn kubeai_ml.server:app --port 18000) &
//        node loadlab/ab/replay_v2.mjs loadlab/ab/runs/c-leak/agent > loadlab/ab/replay_v2_c-leak.jsonl
// Needs the lab Prometheus on 127.0.0.1:9090 (loadlab/monitoring/port-forward.sh) with the
// run's series still in retention. Counterfactual: only the engine's input changes, not the
// cluster's trajectory, so it is valid only up to the first decision that would differ.
import { readFileSync } from "node:fs";
import * as m from "../../lib/promDeploy.js";
const run = process.argv[2];
const cfg = { prometheusUrl: "http://127.0.0.1:9090", httpTimeoutMs: 30000 };
const sc = JSON.parse(readFileSync(`${run}/scenario.json`, "utf8"))._run;
const ns = sc.namespace;
const faults = readFileSync(`${run}/faults.jsonl`, "utf8").trim().split("\n").map(JSON.parse);
const leak = faults.find((f) => f.type === "memory_leak");
const t0 = leak.start_ts;
const decs = readFileSync(`${run}/decisions.jsonl`, "utf8").trim().split("\n").map(JSON.parse).filter((d) => d.kind === "decision" && d.ts >= t0 - 60 && d.ts <= t0 + 300);
async function win(ver, nowS) {
  const o = ver === "v1" ? { queryVersion: "v1" } : {};
  const { start, end, grid } = m.gridFor(nowS, 900);
  const dq = m.deployQueries(ns, "loadlab", o), pq = m.podQueries(ns, "loadlab", o);
  const dr = {}, pr = {};
  await Promise.all([...Object.entries(dq).filter(([k]) => !["cpu_cores_proc", "cpu_fallback_pods"].includes(k)).map(async ([k, q]) => (dr[k] = await m.promRange(cfg, q, start, end))),
    ...Object.entries(pq).filter(([k]) => !["p95", "errors", "cpu_proc"].includes(k)).map(async ([k, q]) => (pr[k] = await m.promRange(cfg, q, start, end)))]);
  return m.buildWindow(grid, dr, pr, {});
}
const tag = Date.now() % 100000;
for (const d of decs) {
  const out = { t: Math.round(d.ts - t0), logged: d.state, action: d.action };
  for (const ver of ["v1", "v2"]) {
    const w = await win(ver, d.ts);
    const body = { deployment: `loadlab-${ver}-${tag}`, namespace: ns, series: w.series, pods: w.pods.length ? w.pods : null, current_replicas: d.current, min_replicas: d.min_replicas, max_replicas: d.max_replicas, now_ts: Math.max(d.ts, w.series.at(-1).ts) };
    const r = await fetch("http://127.0.0.1:18000/v1/decide", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) }).then((x) => x.json());
    const scale = r.actions?.find((a) => a.type === "scale");
    out[ver] = `${r.state}${scale ? ` scale->${scale.params.to}` : ""} mg=${Math.round((w.series.at(-1).mem_growth_bytes_per_s ?? NaN) / 1024)}KB/s`;
  }
  console.log(JSON.stringify(out));
}
