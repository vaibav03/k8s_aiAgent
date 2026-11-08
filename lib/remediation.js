// LLM-assisted remediation with a strict allowlist.
//
// The old agent asked Gemini to "respond ONLY with a shell command" and had a
// (commented-out) exec() of that raw text. Here the model only returns a JSON
// recommendation; we validate it against an allowlist and build the kubectl
// argv ourselves. Commands run via execFile (no shell), are namespace-scoped,
// only target the pod's own owning Deployment, and use --dry-run=server
// unless ALLOW_EXEC=true. `kubectl rollout restart` has no --dry-run flag, so
// its dry run is the equivalent `kubectl patch` of the restartedAt annotation
// (exactly what rollout restart writes) with --dry-run=server.

import { execFile } from "node:child_process";
import { parseQuantity, formatMi } from "./quantity.js";

// "scale" is deliberately NOT here: replica changes belong to the deployment loop (lib/scaler.js),
// which enforces the HPA guard, the kubeai.io/{min,max}-replicas bounds, --current-replicas and
// the scale cooldowns. An LLM-chosen replica count bypassed all of that (and could scale down
// during an incident), so the OOM loop can no longer scale.
export const ACTIONS = ["patch_memory_limit", "rollout_restart", "none"];

// A new memory limit must clear the current limit, the observed working set
// with this much headroom, and an absolute floor.
export const MIN_MEMORY_LIMIT_BYTES = 64 * 2 ** 20;
export const WORKING_SET_SAFETY = 1.2;

const SYSTEM_PROMPT = `You are a Kubernetes SRE assistant. A memory-forecasting model has flagged a pod as likely to be OOM-killed soon.
Choose exactly ONE remediation and reply with STRICT JSON only (no markdown, no prose, no code fences) matching:
{"action": "patch_memory_limit" | "rollout_restart" | "none", "target": "<deployment name>", "params": {...}, "reason": "<one sentence>"}
Params by action:
- patch_memory_limit: {"container": "<container name>", "memory": "<k8s quantity, e.g. 1Gi or 1200Mi>"} (must be larger than the current limit and at least 1.2x the container's current working set)
- rollout_restart: {}
- none: {}
Prefer "none" when evidence is weak. Never invent other actions or targets.`;

export function createLlm(cfg) {
  if (!cfg.googleApiKey) return null;
  let chatPromise = null;
  // Client is created once (lazily, so the module is only loaded when used).
  const getChat = () =>
    (chatPromise ??= import("@langchain/google-genai").then(
      ({ ChatGoogleGenerativeAI }) =>
        new ChatGoogleGenerativeAI({
          model: cfg.geminiModel,
          apiKey: cfg.googleApiKey,
          temperature: 0,
          maxRetries: 2,
        }),
    ));
  return {
    async recommend(context) {
      const chat = await getChat();
      const res = await chat.invoke([
        ["system", SYSTEM_PROMPT],
        ["human", JSON.stringify(context)],
      ]);
      const text = typeof res.content === "string" ? res.content : res.content.map((p) => p.text ?? "").join("");
      return parseRecommendation(text);
    },
  };
}

// Parse a JSON object from model text. Tolerates a ```json fence, nothing else.
export function parseRecommendation(text) {
  let t = String(text ?? "").trim();
  const fence = /^```(?:json)?\s*([\s\S]*?)\s*```$/i.exec(t);
  if (fence) t = fence[1];
  const obj = JSON.parse(t);
  if (!obj || typeof obj !== "object" || Array.isArray(obj)) throw new Error("recommendation is not a JSON object");
  return obj;
}

// ctx: {namespace, deployment, containers:[{name, memoryLimit, workingSetBytes?}], maxMemoryBytes, maxReplicas}
// Returns {ok:true, action, argv|null, dryRunArgv?, summary} or {ok:false, error}.
// dryRunArgv (when set) is what runs instead of argv in dry-run mode.
export function validateRecommendation(rec, ctx, now = new Date()) {
  if (!rec || typeof rec !== "object") return { ok: false, error: "not an object" };
  const action = rec.action;
  if (!ACTIONS.includes(action)) return { ok: false, error: `action "${action}" not in allowlist` };
  if (action === "none") return { ok: true, action, argv: null, summary: "no action" };

  if (!ctx.deployment) return { ok: false, error: "pod has no owning Deployment; only 'none' is allowed" };
  const target = String(rec.target ?? "").replace(/^deployments?(\.apps)?\//, "");
  if (target !== ctx.deployment) {
    return { ok: false, error: `target "${rec.target}" is not this pod's Deployment "${ctx.deployment}"` };
  }
  const params = rec.params && typeof rec.params === "object" ? rec.params : {};
  const base = ["-n", ctx.namespace];
  const dep = `deployment/${ctx.deployment}`;

  if (action === "rollout_restart") {
    const patch = { spec: { template: { metadata: { annotations: { "kubectl.kubernetes.io/restartedAt": now.toISOString() } } } } };
    return {
      ok: true,
      action,
      argv: [...base, "rollout", "restart", dep],
      dryRunArgv: [...base, "patch", dep, "--type=strategic", "-p", JSON.stringify(patch)],
      summary: `rollout restart ${dep}`,
    };
  }

  // patch_memory_limit
  const names = ctx.containers.map((c) => c.name);
  const container = params.container ?? (names.length === 1 ? names[0] : undefined);
  if (!names.includes(container)) return { ok: false, error: `container "${container}" not in pod (${names.join(",")})` };
  const newBytes = parseQuantity(params.memory);
  if (newBytes === null || newBytes <= 0) return { ok: false, error: `invalid memory quantity "${params.memory}"` };
  const c = ctx.containers.find((x) => x.name === container);
  const cur = parseQuantity(c.memoryLimit);
  const ws = typeof c.workingSetBytes === "number" && Number.isFinite(c.workingSetBytes) && c.workingSetBytes > 0 ? c.workingSetBytes : null;
  if (cur === null && ws === null) return { ok: false, error: "container has no current limit and no observed working set; cannot bound the new limit" };
  if (cur !== null && newBytes <= cur) return { ok: false, error: "new memory limit must exceed the current limit" };
  if (ws !== null && newBytes < ws * WORKING_SET_SAFETY) {
    return { ok: false, error: `new memory limit must be at least ${WORKING_SET_SAFETY}x the current working set (${formatMi(ws * WORKING_SET_SAFETY)})` };
  }
  if (newBytes < MIN_MEMORY_LIMIT_BYTES) return { ok: false, error: `new memory limit must be at least ${formatMi(MIN_MEMORY_LIMIT_BYTES)}` };
  if (newBytes > ctx.maxMemoryBytes) return { ok: false, error: "new memory limit exceeds MAX_MEMORY_LIMIT" };
  const q = formatMi(newBytes); // re-render; never pass model text through
  return {
    ok: true,
    action,
    argv: [...base, "set", "resources", dep, "-c", container, `--limits=memory=${q}`],
    summary: `set ${dep} container ${container} memory limit to ${q}`,
  };
}

// opts.stdin: written to kubectl's stdin (eviction body for `create --raw -f -`).
// opts.dryRunInArgv: dryRunArgv already encodes the dry run (e.g. ?dryRun=All on a raw
// API path), so no --dry-run=server flag is appended.
export function runKubectl(cfg, argv, dryRunArgv = null, opts = {}) {
  const full = cfg.allowExec ? [...argv] : opts.dryRunInArgv ? [...(dryRunArgv ?? argv)] : [...(dryRunArgv ?? argv), "--dry-run=server"];
  return new Promise((resolve) => {
    const child = execFile(cfg.kubectl, full, { timeout: 30000, shell: false }, (err, stdout, stderr) => {
      resolve({ argv: full, dryRun: !cfg.allowExec, ok: !err, stdout: String(stdout).trim(), stderr: String(stderr).trim(), error: err?.message ?? null });
    });
    if (opts.stdin != null) {
      child.stdin?.on("error", () => {});
      child.stdin?.end(opts.stdin);
    }
  });
}

// An eviction refused because it would violate a PodDisruptionBudget (HTTP 429).
export function isPdbRefusal(r) {
  const t = `${r?.stderr ?? ""} ${r?.error ?? ""}`;
  return /TooManyRequests|disruption budget|\b429\b/i.test(t);
}

// ------------------------------------------------------------------------
// Decision-engine actions (POST /v1/decide -> lib/scaler.js). Same rules as
// the LLM path: argv is built here from validated fields only (never from
// model text), namespace-scoped, only the decided Deployment or one of its
// own live pods, executed via execFile, --dry-run=server unless act mode.
//
// ctx: {namespace, deployment, current, minReplicas, maxReplicas,
//       pods: [owner-verified live pod names of the Deployment] | null (ownership unknown),
//       currentRevision, rollbackTo}
// Returns {ok:true, kind, argv, dryRunArgv?, stdin?, dryRunInArgv?, summary} or {ok:false, error}.
//
// delete_pod is executed as an EVICTION (POST pods/<name>/eviction via `kubectl create --raw`),
// not `kubectl delete pod`: only the Eviction API honours PodDisruptionBudgets. A 429 (the PDB
// would be violated) is reported by the scaler as skipped and replaced by a surge rollout restart.
export const ENGINE_ACTIONS = ["scale", "delete_pod", "rollout_restart", "rollback"];

export function buildEngineAction(kind, params, ctx, now = new Date()) {
  if (!ENGINE_ACTIONS.includes(kind)) return { ok: false, error: `action "${kind}" not in allowlist` };
  if (!ctx.deployment) return { ok: false, error: "no deployment" };
  const base = ["-n", ctx.namespace];
  const dep = `deployment/${ctx.deployment}`;
  if (kind === "scale") {
    const r = params?.replicas;
    if (!Number.isInteger(r) || r < Math.max(ctx.minReplicas ?? 1, 0) || r > ctx.maxReplicas) {
      return { ok: false, error: `replicas must be an integer in [${ctx.minReplicas}, ${ctx.maxReplicas}]` };
    }
    if (!Number.isInteger(ctx.current) || ctx.current < 0) return { ok: false, error: "current replicas unknown" };
    // --current-replicas: the API server rejects the scale if someone else changed it meanwhile
    return {
      ok: true, kind,
      argv: [...base, "scale", dep, `--current-replicas=${ctx.current}`, `--replicas=${r}`],
      summary: `scale ${dep} ${ctx.current} -> ${r}`,
    };
  }
  if (kind === "delete_pod") {
    const pod = String(params?.pod ?? "");
    // never fall back to Prometheus pod names for a destructive action
    if (!Array.isArray(ctx.pods)) return { ok: false, error: "cannot verify pod ownership (owner-checked pod list unavailable)" };
    if (!pod || !ctx.pods.includes(pod)) {
      return { ok: false, error: `pod "${pod}" is not a live pod of ${dep}` };
    }
    if (!/^[a-z0-9]([-a-z0-9.]*[a-z0-9])?$/.test(pod) || !/^[a-z0-9]([-a-z0-9]*[a-z0-9])?$/.test(ctx.namespace)) {
      return { ok: false, error: "invalid pod or namespace name" };
    }
    const path = `/api/v1/namespaces/${ctx.namespace}/pods/${pod}/eviction`;
    const body = JSON.stringify({ apiVersion: "policy/v1", kind: "Eviction", metadata: { name: pod, namespace: ctx.namespace } });
    return {
      ok: true, kind,
      argv: ["create", "--raw", path, "-f", "-"],
      dryRunArgv: ["create", "--raw", `${path}?dryRun=All`, "-f", "-"],
      dryRunInArgv: true,
      stdin: body,
      summary: `evict pod ${pod} (honours PDBs; ${dep} recreates it)`,
    };
  }
  if (kind === "rollout_restart") {
    return validateRecommendation({ action: "rollout_restart", target: ctx.deployment }, { ...ctx, containers: [] }, now);
  }
  // rollback: only ever emitted by the engine with rollout evidence. Always to an explicit,
  // known-good, OLDER revision: a bare `rollout undo` goes to "the revision before the current
  // one", which after a first undo is the bad revision again (undo ping-pong).
  const to = ctx.rollbackTo;
  if (!Number.isInteger(to) || to < 1) return { ok: false, error: "no known-good earlier revision to roll back to" };
  if (!Number.isInteger(ctx.currentRevision) || to >= ctx.currentRevision) {
    return { ok: false, error: `known-good revision ${to} is not older than the current revision ${ctx.currentRevision}` };
  }
  return { ok: true, kind, argv: [...base, "rollout", "undo", dep, `--to-revision=${to}`], summary: `rollout undo ${dep} --to-revision=${to}` };
}

// Per-Deployment remediation ledger shared by the OOM loop (lib/agent.js) and the deployment
// loop (lib/scaler.js) in one process: both consult it before a restart-type remediation so two
// loops (or N alerting pods) cannot each restart the same Deployment inside the cooldown.
// kind prefixed "dryrun:" / "planned:" = not a real change (dedupes LLM calls only).
export function createRemediationLedger() {
  const m = new Map(); // "ns/deployment" -> [{kind, ts}]
  const key = (ns, d) => `${ns}/${d}`;
  return {
    record(ns, d, kind, ts) {
      const k = key(ns, d);
      const arr = m.get(k) ?? [];
      arr.push({ kind, ts });
      while (arr.length > 20) arr.shift();
      m.set(k, arr);
    },
    // newest ts of any entry (realOnly: only real changes); 0 when none
    last(ns, d, { realOnly = false } = {}) {
      const arr = m.get(key(ns, d)) ?? [];
      let t = 0;
      for (const e of arr) if ((!realOnly || !/^(dryrun|planned):/.test(e.kind)) && e.ts > t) t = e.ts;
      return t;
    },
    entries: (ns, d) => [...(m.get(key(ns, d)) ?? [])],
  };
}
