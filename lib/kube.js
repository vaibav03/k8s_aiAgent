// Read-only Kubernetes API access (pod discovery + spec/status extraction).
//
// Out of cluster: KUBE_API=http://localhost:8001 (kubectl proxy), no token.
// In cluster: KUBE_API=https://kubernetes.default.svc, bearer token read from
// the projected service-account token file (re-read every call because it is
// rotated), CA trusted via NODE_EXTRA_CA_CERTS (set in k8s/agent.yaml).

import { readFile } from "node:fs/promises";
import { requestJson } from "./http.js";
import { parseQuantity } from "./quantity.js";

export function createKubeClient(cfg) {
  async function authHeaders() {
    try {
      const token = (await readFile(cfg.kubeTokenFile, "utf8")).trim();
      return token ? { authorization: `Bearer ${token}` } : {};
    } catch {
      return {}; // no service-account token (e.g. local kubectl proxy)
    }
  }

  async function listPods() {
    const qs = cfg.podSelector ? `?labelSelector=${encodeURIComponent(cfg.podSelector)}` : "";
    const url = `${cfg.kubeApi}/api/v1/namespaces/${encodeURIComponent(cfg.namespace)}/pods${qs}`;
    const data = await requestJson(url, { headers: await authHeaders(), timeoutMs: cfg.httpTimeoutMs });
    return (data?.items ?? []).map(summarizePod);
  }

  return { listPods };
}

// A container's own memory limit in bytes; null when unset or not > 0.
// (Never summed across containers: each container is OOM-killed against its
// own limit.)
export function containerLimitBytes(container) {
  const b = parseQuantity(container?.resources?.limits?.memory);
  return b !== null && b > 0 ? b : null;
}

// Owning Deployment name via ReplicaSet ownerReference + pod-template-hash.
export function owningDeployment(pod) {
  const rs = (pod?.metadata?.ownerReferences ?? []).find((o) => o.kind === "ReplicaSet");
  const hash = pod?.metadata?.labels?.["pod-template-hash"];
  if (!rs || !hash || !rs.name.endsWith(`-${hash}`)) return null;
  return rs.name.slice(0, -(hash.length + 1));
}

export function summarizePod(pod, now = Date.now()) {
  const statuses = new Map((pod?.status?.containerStatuses ?? []).map((s) => [s.name, s]));
  let restartCount = null;
  let lastFinished = null;
  let crashLooping = false;
  let lastTerminationReason = null;
  const containers = (pod?.spec?.containers ?? []).map((c) => {
    const s = statuses.get(c.name);
    const rc = s ? s.restartCount ?? 0 : null;
    if (rc !== null) restartCount = (restartCount ?? 0) + rc;
    // Check every container (the old agent only looked at [0], which is the
    // istio sidecar in the training setup).
    const looping = s?.state?.waiting?.reason === "CrashLoopBackOff";
    if (looping) crashLooping = true;
    let finished = null;
    const t = s?.lastState?.terminated;
    if (t?.finishedAt) {
      const ms = Date.parse(t.finishedAt);
      if (Number.isFinite(ms)) {
        finished = ms;
        if (lastFinished === null || ms > lastFinished) {
          lastFinished = ms;
          lastTerminationReason = t.reason ?? null;
        }
      }
    }
    return {
      name: c.name,
      memoryLimit: c?.resources?.limits?.memory ?? null,
      limitBytes: containerLimitBytes(c),
      restartCount: rc,
      secondsSinceRestart: finished === null ? null : Math.max(0, (now - finished) / 1000),
      crashLooping: looping,
    };
  });
  return {
    name: pod?.metadata?.name,
    namespace: pod?.metadata?.namespace,
    phase: pod?.status?.phase ?? null,
    restartCount,
    secondsSinceRestart: lastFinished === null ? null : Math.max(0, (now - lastFinished) / 1000),
    lastTerminationReason,
    crashLooping,
    deployment: owningDeployment(pod),
    containers,
  };
}
