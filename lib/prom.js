// One PromQL instant query per cycle for every watched pod.
//
// Uses container_memory_working_set_bytes (what the kubelet OOM-kills on and
// what metrics-server reports; the training CSV came from metrics-server)
// instead of container_memory_usage_bytes (includes page cache). Kept per
// container (`sum by (pod, container)`), because the kernel OOM-kills each
// container against its own limit: a pod-level sum compared with a summed
// limit hides an app container at its limit behind a roomy sidecar, and an
// unlimited sidecar's memory would inflate usage. Drops the pod-level cgroup
// row (container="") and the pause container ("POD").

import { requestJson } from "./http.js";

const escapeRegex = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
// PromQL string literal escaping (inside double quotes).
const escapeLabel = (s) => s.replace(/\\/g, "\\\\").replace(/"/g, '\\"');

export function buildWorkingSetQuery(namespace, pods) {
  const re = pods.map(escapeRegex).join("|");
  return (
    `sum by (pod, container) (container_memory_working_set_bytes{namespace="${escapeLabel(namespace)}",` +
    `pod=~"${escapeLabel(re)}",container!="",container!="POD"})`
  );
}

// Returns Map<podName, Map<containerName, {ts, working_set_bytes}>>. Pods and
// containers absent from the result are simply absent (missing data is NOT
// turned into 0).
export async function queryWorkingSet(cfg, pods) {
  const out = new Map();
  if (pods.length === 0) return out;
  const q = buildWorkingSetQuery(cfg.namespace, pods);
  const url = `${cfg.prometheusUrl}/api/v1/query?query=${encodeURIComponent(q)}`;
  const data = await requestJson(url, { timeoutMs: cfg.httpTimeoutMs });
  if (data?.status !== "success") throw new Error(`prometheus error: ${data?.error ?? "unknown"}`);
  for (const r of data?.data?.result ?? []) {
    const pod = r?.metric?.pod;
    const container = r?.metric?.container;
    const [ts, v] = r?.value ?? [];
    const bytes = Number(v);
    const t = Number(ts);
    if (!pod || !container || !Number.isFinite(bytes) || !Number.isFinite(t)) continue;
    if (!out.has(pod)) out.set(pod, new Map());
    out.get(pod).set(container, { ts: t, working_set_bytes: bytes });
  }
  return out;
}
