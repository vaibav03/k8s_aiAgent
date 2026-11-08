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

export const ACTIONS = ["patch_memory_limit", "rollout_restart", "scale", "none"];

// A new memory limit must clear the current limit, the observed working set
// with this much headroom, and an absolute floor.
export const MIN_MEMORY_LIMIT_BYTES = 64 * 2 ** 20;
export const WORKING_SET_SAFETY = 1.2;

const SYSTEM_PROMPT = `You are a Kubernetes SRE assistant. A memory-forecasting model has flagged a pod as likely to be OOM-killed soon.
Choose exactly ONE remediation and reply with STRICT JSON only (no markdown, no prose, no code fences) matching:
{"action": "patch_memory_limit" | "rollout_restart" | "scale" | "none", "target": "<deployment name>", "params": {...}, "reason": "<one sentence>"}
Params by action:
- patch_memory_limit: {"container": "<container name>", "memory": "<k8s quantity, e.g. 1Gi or 1200Mi>"} (must be larger than the current limit and at least 1.2x the container's current working set)
- scale: {"replicas": <integer>}
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

  if (action === "scale") {
    const r = params.replicas;
    if (!Number.isInteger(r) || r < 1 || r > ctx.maxReplicas) {
      return { ok: false, error: `replicas must be an integer in [1, ${ctx.maxReplicas}]` };
    }
    return { ok: true, action, argv: [...base, "scale", dep, `--replicas=${r}`], summary: `scale ${dep} to ${r}` };
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

export function runKubectl(cfg, argv, dryRunArgv = null) {
  const full = cfg.allowExec ? [...argv] : [...(dryRunArgv ?? argv), "--dry-run=server"];
  return new Promise((resolve) => {
    execFile(cfg.kubectl, full, { timeout: 30000, shell: false }, (err, stdout, stderr) => {
      resolve({ argv: full, dryRun: !cfg.allowExec, ok: !err, stdout: String(stdout).trim(), stderr: String(stderr).trim(), error: err?.message ?? null });
    });
  });
}
