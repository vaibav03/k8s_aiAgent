// Deployment-level decision loop: every DECIDE_INTERVAL_S, for each target Deployment,
//   Prometheus (promexport.py PromQL, query_range over HISTORY_S at 5 s) -> window
//   -> POST /v1/decide -> one decision record -> act according to ACT_MODE:
//     observe  log the decision and what would be done; never runs kubectl
//     dryrun   run every allowed action as `kubectl ... --dry-run=server`
//     act      (needs ALLOW_EXEC=true, else it degrades to dryrun) run them for real
//
// Guard rails (enforced here, on top of the engine's own interlocks):
//   - never fight an HPA: if a HorizontalPodAutoscaler targets the Deployment (or the HPA list
//     cannot be read), scale actions are refused unless HPA_OVERRIDE=true;
//   - replicas clamped to [min, max] (kubeai.io/min-replicas / max-replicas annotations, else
//     SCALE_MIN / SCALE_MAX); `kubectl scale --current-replicas` precondition against races;
//   - own scale cooldowns (SCALE_UP_COOLDOWN_S, SCALE_DOWN_COOLDOWN_S since the last scale);
//   - remediations only through the allowlist (lib/remediation.js buildEngineAction): evict one
//     owner-verified live pod of this Deployment (Eviction API, honours PDBs; a PDB refusal falls
//     back to a surge rollout restart; no verified pod list -> refused), rollout restart, rollout
//     undo; one per REMEDIATION_COOLDOWN_S per Deployment and action kind (restart-type kinds
//     also share the cooldown with the OOM loop through the remediation ledger); nothing while
//     the Deployment is paused;
//   - rollback (engine emits it only with rollout evidence) always goes `--to-revision=<the
//     newest older revision this agent saw healthy>`; refused (and paged) when there is none,
//     when the current revision was created by the agent itself (its own undo / rollout
//     restart), or when the agent already rolled back from the current revision: at most one
//     rollback per bad revision, never an undo ping-pong. Revisions and new ReplicaSets the
//     agent created are not counted as rollout evidence (revisionAt maps them to the previous
//     revision; /v1/decide gets agent_rollout_ts);
//   - APP_METRIC_MATCHER without "{deployment}" in a namespace with several Deployments ->
//     refuse to decide (app metrics would be namespace-wide);
//   - "page" actions are logged (level PAGE), never executed.
//
// Every decision is one JSON line on stdout ({"kind":"decision", timestamp, state, desired,
// current, action, reasons, ...}) and, when DECISION_LOG is set, appended to that file too.

import { appendFile } from "node:fs/promises";
import { createKubeClient } from "./kube.js";
import { fetchWindow } from "./promDeploy.js";
import { requestJson } from "./http.js";
import { buildEngineAction, runKubectl, isPdbRefusal, createRemediationLedger } from "./remediation.js";

const MAX_RECENT = 200;
// states in which the current revision counts as known-good (rollback target candidates)
const GOOD_STATES = new Set(["healthy", "scale_up", "scale_down"]);
const GOOD_MIN_DECISIONS = 4; // ~1 min at the default 15 s interval
const AGENT_REV_ATTRIB_S = 180; // a revision change this soon after our own rollout is ours
const RESTART_KINDS = new Set(["delete_pod", "rollout_restart", "rollback"]);

export function createScaler(cfg, deps = {}) {
  const kube = deps.kube ?? createKubeClient(cfg);
  const fetchWin = deps.fetchWindow ?? ((ns, d, opts) => fetchWindow(cfg, ns, d, opts));
  const decide = deps.decide ?? ((body) => requestJson(`${cfg.mlUrl}/v1/decide`, { method: "POST", body, timeoutMs: cfg.decideTimeoutMs }));
  const now = deps.now ?? (() => Date.now());
  const mode = cfg.effectiveActMode;
  const kubectl = deps.runKubectl ?? ((argv, dry, opts) => runKubectl({ allowExec: mode === "act", kubectl: cfg.kubectl }, argv, dry, opts));
  const out = deps.emit ?? ((line) => console.log(line));
  const ledger = deps.ledger ?? createRemediationLedger();

  const per = new Map(); // deployment -> state
  const recent = [];
  const status = { enabled: true, mode, requestedMode: cfg.actMode, cycles: 0, lastCycleAt: null, lastCycleMs: null, errors: {} };
  let timer = null;
  let stopped = true;

  const st = (name) => {
    if (!per.has(name)) {
      per.set(name, {
        name, revisions: [], lastScaleAt: 0, lastRemediation: {}, decisions: 0, last: null, lastError: null,
        goodCount: new Map(), // revision -> decisions in a GOOD_STATES state
        agentRevisions: new Map(), // revision -> "rollback" | "rollout_restart" (created by us)
        agentRolloutTs: [], // when we executed an undo / rollout restart
        pendingAgentRollout: null, // {ts, kind} of our last undo / restart whose revision is not seen yet
        rolledBackFrom: new Set(), // revisions we already rolled back from (one rollback each)
      });
    }
    return per.get(name);
  };

  async function emit(rec) {
    const line = JSON.stringify(rec);
    out(line);
    recent.push(rec);
    while (recent.length > MAX_RECENT) recent.shift();
    if (cfg.decisionLog) {
      try {
        await appendFile(cfg.decisionLog, line + "\n");
      } catch (err) {
        status.errors.decisionLog = err.message;
      }
    }
  }

  async function targets() {
    if (cfg.deployments.length) return cfg.deployments;
    const ds = await kube.listDeployments(cfg.deploySelector);
    return ds.map((d) => d.name).filter(Boolean);
  }

  // revision at time t, with revisions the agent created mapped to the revision before them,
  // so our own undo / rollout restart never shows up as "a rollout happened" to the engine
  function revisionAt(s) {
    return (t) => {
      let rev = null;
      for (const [since, r] of s.revisions) {
        if (since > t) break;
        if (!s.agentRevisions.has(r) || rev == null) rev = r;
      }
      return rev;
    };
  }

  // newest revision older than `current` that this agent saw healthy for >= GOOD_MIN_DECISIONS
  function knownGoodBefore(s, current) {
    let best = null;
    for (const [rev, n] of s.goodCount) {
      if (n >= GOOD_MIN_DECISIONS && rev < current && (best == null || rev > best)) best = rev;
    }
    return best;
  }

  // Resolve the engine's actions into executable steps (or a refusal), in order.
  function plan(dec, s, dep, hpa, podNames, nowS) {
    const pages = [];
    const min = Math.max(dep.minReplicas ?? cfg.scaleMin, 0);
    const max = Math.max(dep.maxReplicas ?? cfg.scaleMax, 1);
    const current = dep.replicas;
    const steps = [];
    for (const a of dec.actions ?? []) {
      const step = { type: a.type, target: a.target, params: a.params ?? {}, reason: a.reason };
      steps.push(step);
      if (a.type === "none") {
        step.outcome = "none";
        continue;
      }
      if (a.type === "page") {
        step.outcome = "paged";
        continue;
      }
      if (dep.paused) {
        step.outcome = "refused";
        step.why = "deployment is paused";
        continue;
      }
      let kind;
      let params = {};
      if (a.type === "scale") {
        const to = Math.min(Math.max(Number(a.params?.to), min), max);
        step.clamped_to = to;
        if (!Number.isInteger(to) || to === current) {
          step.outcome = "skipped";
          step.why = `already at ${current} (engine ${a.params?.to}, bounds [${min}, ${max}])`;
          continue;
        }
        if (hpa && !cfg.hpaOverride) {
          step.outcome = "refused";
          step.why = hpa === "unknown" ? "cannot read HPAs (RBAC?); refusing to scale" : `HPA ${hpa} manages this deployment (HPA_OVERRIDE=false)`;
          continue;
        }
        const up = to > current;
        const since = nowS - s.lastScaleAt;
        const cd = up ? cfg.scaleUpCooldownS : cfg.scaleDownCooldownS;
        if (since < cd) {
          step.outcome = "skipped";
          step.why = `scale cooldown (${since.toFixed(0)}s since last scale < ${cd}s)`;
          continue;
        }
        kind = "scale";
        params = { replicas: to };
      } else if (a.type === "restart_pod") {
        kind = a.params?.mode === "rollout_restart" ? "rollout_restart" : "delete_pod";
        params = { pod: a.target };
      } else if (a.type === "rollback") {
        kind = "rollback";
        const cur = dep.revision;
        let why = null;
        if (cur == null) why = "current revision unknown";
        else if (s.agentRevisions.get(cur) === "rollback") why = `current revision ${cur} was created by this agent's own rollback; not rolling back again`;
        else if (s.rolledBackFrom.has(cur)) why = `already rolled back once from revision ${cur}`;
        if (why) {
          step.outcome = "refused";
          step.why = why;
          pages.push({ type: "page", target: "service owner", params: { severity: "high" }, reason: `rollback refused: ${why}`, outcome: "paged" });
          continue;
        }
        // our own rollout restart of revision N is still N's code: roll back to before N, so the
        // undo never merely reverts the restartedAt annotation
        let code = cur;
        for (let guard = 0; s.agentRevisions.get(code) === "rollout_restart" && guard < 20; guard++) {
          const i = s.revisions.findIndex(([, r]) => r === code);
          if (i <= 0) break;
          code = s.revisions[i - 1][1];
        }
        if (s.rolledBackFrom.has(code)) {
          step.outcome = "refused";
          step.why = `already rolled back once from revision ${code}`;
          pages.push({ type: "page", target: "service owner", params: { severity: "high" }, reason: `rollback refused: ${step.why}`, outcome: "paged" });
          continue;
        }
        params = { currentRevision: cur, rollbackTo: knownGoodBefore(s, code) };
        step.code_revision = code;
      } else {
        step.outcome = "refused";
        step.why = `unknown action type ${a.type}`;
        continue;
      }
      if (kind !== "scale") {
        const last = Math.max(s.lastRemediation[kind] ?? 0, RESTART_KINDS.has(kind) ? ledger.last(cfg.namespace, dep.name, { realOnly: true }) : 0);
        if (nowS - last < cfg.remediationCooldownS) {
          step.outcome = "skipped";
          step.why = `remediation cooldown (${(nowS - last).toFixed(0)}s since last ${kind} < ${cfg.remediationCooldownS}s)`;
          continue;
        }
      }
      const v = buildEngineAction(kind, params, {
        namespace: cfg.namespace, deployment: dep.name, current, minReplicas: min, maxReplicas: max, pods: podNames,
        currentRevision: params.currentRevision, rollbackTo: params.rollbackTo,
      });
      if (!v.ok) {
        step.outcome = "refused";
        step.why = v.error;
        if (kind === "rollback") {
          pages.push({ type: "page", target: "service owner", params: { severity: "high" }, reason: `rollback refused: ${v.error}`, outcome: "paged" });
        }
        continue;
      }
      step.kind = kind;
      step.summary = v.summary;
      step.exec = v;
    }
    return [...steps, ...pages];
  }

  function recordReal(s, kind, nowS, dep, step = {}) {
    s.lastRemediation[kind] = nowS;
    ledger.record(cfg.namespace, dep.name, kind, nowS);
    if (kind === "rollout_restart" || kind === "rollback") {
      s.agentRolloutTs.push(nowS);
      s.pendingAgentRollout = { ts: nowS, kind };
      if (kind === "rollback" && dep.revision != null) s.rolledBackFrom.add(dep.revision);
      if (kind === "rollback" && step.code_revision != null) s.rolledBackFrom.add(step.code_revision);
    }
  }

  async function execute(steps, s, nowS, dep) {
    const extra = [];
    for (const step of steps) {
      const v = step.exec;
      delete step.exec;
      if (!v) continue;
      if (mode === "observe") {
        step.outcome = "planned";
        continue;
      }
      const r = await kubectl(v.argv, v.dryRunArgv, { stdin: v.stdin, dryRunInArgv: v.dryRunInArgv });
      step.argv = r.argv;
      step.outcome = r.ok ? (r.dryRun ? "dry_run_ok" : "executed") : "failed";
      if (!r.ok) step.error = (r.stderr || r.error || "").slice(0, 300);
      if (r.ok && !r.dryRun) {
        if (step.kind === "scale") s.lastScaleAt = nowS;
        else recordReal(s, step.kind, nowS, dep, step);
      }
      // eviction refused by a PodDisruptionBudget (429): skip it, surge rollout restart instead
      if (!r.ok && step.kind === "delete_pod" && isPdbRefusal(r)) {
        step.outcome = "skipped";
        step.why = "eviction refused: PodDisruptionBudget would be violated (429)";
        const fb = { type: "restart_pod", target: step.target, params: { mode: "rollout_restart", fallback_for: "pdb" }, reason: "PDB blocks the eviction; surge rollout restart instead", kind: "rollout_restart" };
        extra.push(fb);
        const last = Math.max(s.lastRemediation.rollout_restart ?? 0, ledger.last(cfg.namespace, dep.name, { realOnly: true }));
        if (nowS - last < cfg.remediationCooldownS) {
          fb.outcome = "skipped";
          fb.why = "remediation cooldown";
          continue;
        }
        const rv = buildEngineAction("rollout_restart", {}, { namespace: cfg.namespace, deployment: dep.name });
        const r2 = await kubectl(rv.argv, rv.dryRunArgv, {});
        fb.summary = rv.summary;
        fb.argv = r2.argv;
        fb.outcome = r2.ok ? (r2.dryRun ? "dry_run_ok" : "executed") : "failed";
        if (!r2.ok) fb.error = (r2.stderr || r2.error || "").slice(0, 300);
        if (r2.ok && !r2.dryRun) recordReal(s, "rollout_restart", nowS, dep);
      }
    }
    steps.push(...extra);
  }

  const primary = (steps) => {
    const s = steps.find((x) => x.type !== "none") ?? steps[0];
    if (!s) return "none";
    if (s.type === "scale") return `scale:${s.params.from}->${s.clamped_to ?? s.params.to}`;
    if (s.type === "restart_pod") return `restart_pod:${s.target}`;
    return s.type;
  };

  async function decideOne(name) {
    const s = st(name);
    const t0 = now();
    const nowS = t0 / 1000;
    const base = { kind: "decision", timestamp: new Date(t0).toISOString(), ts: Math.round(nowS * 1000) / 1000, namespace: cfg.namespace, deployment: name, mode };
    let dep;
    try {
      dep = await kube.getDeployment(name);
    } catch (err) {
      s.lastError = `kube: ${err.message}`;
      await emit({ ...base, kind: "decision_error", stage: "kube", error: err.message });
      return;
    }
    if (dep.revision != null && s.revisions.at(-1)?.[1] !== dep.revision) {
      // a revision change right after our own undo / rollout restart is ours
      if (s.pendingAgentRollout != null && nowS - s.pendingAgentRollout.ts <= AGENT_REV_ATTRIB_S && s.revisions.length) {
        s.agentRevisions.set(dep.revision, s.pendingAgentRollout.kind);
        // a rollout restart of revision N is still revision N's code: one rollback per bad code
        const base = s.revisions.at(-1)[1];
        if (s.rolledBackFrom.has(base)) s.rolledBackFrom.add(dep.revision);
        s.pendingAgentRollout = null;
      }
      s.revisions.push([nowS, dep.revision]);
    }
    // keep the entry before an agent-created revision (it is what that revision maps to)
    while (s.revisions.length > 1 && s.revisions[1][0] < nowS - cfg.historyS && !s.agentRevisions.has(s.revisions[1][1])) s.revisions.shift();
    while (s.agentRolloutTs.length && s.agentRolloutTs[0] < nowS - cfg.historyS) s.agentRolloutTs.shift();

    let hpa = null;
    try {
      const h = (await kube.listHpas()).find((x) => x.kind === "Deployment" && x.target === name);
      hpa = h ? h.name : null;
    } catch {
      hpa = "unknown";
    }

    let win;
    const tp = now();
    try {
      win = await fetchWin(cfg.namespace, name, { nowS, revisionAt: revisionAt(s), hpa });
    } catch (err) {
      s.lastError = `prometheus: ${err.message}`;
      await emit({ ...base, kind: "decision_error", stage: "prometheus", error: err.message, current: dep.replicas });
      return;
    }
    const promMs = now() - tp;
    if (!win.series.length) {
      s.lastError = "no telemetry rows (core signals absent in the whole window)";
      await emit({ ...base, kind: "decision_error", stage: "window", error: s.lastError, current: dep.replicas });
      return;
    }
    const body = {
      deployment: name,
      namespace: cfg.namespace,
      series: win.series,
      pods: win.pods.length ? win.pods : null,
      current_replicas: dep.replicas,
      min_replicas: Math.max(dep.minReplicas ?? cfg.scaleMin, 0),
      max_replicas: Math.max(dep.maxReplicas ?? cfg.scaleMax, 1),
      now_ts: Math.max(nowS, win.series.at(-1).ts),
      revision: revisionAt(s)(nowS) ?? dep.revision,
    };
    // only sent when set (an older ML service with extra="forbid" rejects unknown keys)
    if (s.agentRolloutTs.length) body.agent_rollout_ts = [...s.agentRolloutTs];
    let dec;
    const tm = now();
    try {
      dec = await decide(body);
    } catch (err) {
      s.lastError = `ml: ${err.message}`;
      await emit({ ...base, kind: "decision_error", stage: "ml", error: err.message, current: dep.replicas });
      return;
    }
    const mlMs = now() - tm;

    // owner-verified live pods of this Deployment: the only pods an eviction may target. Never
    // Prometheus names (a regex can match a sibling Deployment); unknown -> evictions refused.
    let podNames = null;
    if (dep.selector) {
      try {
        podNames = (await kube.listPodsBySelector(dep.selector)).filter((p) => p.deployment === name).map((p) => p.name);
      } catch {
        podNames = null;
      }
    }

    const steps = plan(dec, s, dep, hpa, podNames, nowS);
    await execute(steps, s, nowS, dep);
    if (GOOD_STATES.has(dec.state) && dep.revision != null) s.goodCount.set(dep.revision, (s.goodCount.get(dep.revision) ?? 0) + 1);
    s.decisions += 1;
    s.lastError = null;
    const rec = {
      ...base,
      state: dec.state,
      confidence: dec.confidence,
      current: dep.replicas,
      ready: dep.readyReplicas,
      desired: dec.desired_replicas,
      action: primary(steps),
      actions: steps,
      reasons: dec.reasons,
      telemetry_ok: dec.telemetry_ok,
      slo_risk: dec.slo_risk,
      hpa,
      min_replicas: body.min_replicas,
      max_replicas: body.max_replicas,
      model_version: dec.model_version,
      rows: win.series.length,
      pod_rows: win.pods.length,
      latency_ms: { prometheus: promMs, ml: mlMs, total: now() - t0 },
    };
    s.last = rec;
    await emit(rec);
    for (const step of steps) {
      if (step.type === "page") {
        out(JSON.stringify({ t: new Date().toISOString(), level: "PAGE", msg: step.reason, deployment: name, target: step.target, severity: step.params?.severity }));
      }
    }
    return rec;
  }

  async function cycle() {
    const t0 = now();
    status.cycles += 1;
    let names;
    try {
      names = await targets();
      delete status.errors.discovery;
    } catch (err) {
      status.errors.discovery = err.message;
      await emit({ kind: "decision_error", timestamp: new Date().toISOString(), stage: "discovery", error: err.message });
      return [];
    }
    for (const n of per.keys()) if (!names.includes(n)) per.delete(n);
    if (!cfg.appMatcherPerDeployment) {
      // app-level metrics are namespace-wide: refuse when the namespace has several Deployments
      let all = null;
      try {
        all = (await kube.listDeployments("")).map((d) => d.name).filter(Boolean);
      } catch {
        all = null; // cannot tell; the per-Deployment calls will surface API errors
      }
      if (all && all.length > 1) {
        const error = `namespace ${cfg.namespace} has ${all.length} Deployments but APP_METRIC_MATCHER has no "{deployment}" template: app metrics (rps, latency, 5xx) would be namespace-wide; refusing to decide`;
        status.errors.config = error;
        for (const n of names) {
          st(n).lastError = error;
          await emit({ kind: "decision_error", timestamp: new Date().toISOString(), deployment: n, stage: "config", error });
        }
        status.lastCycleMs = now() - t0;
        status.lastCycleAt = new Date().toISOString();
        return [];
      }
      delete status.errors.config;
    }
    const recs = [];
    for (const n of names) {
      try {
        recs.push(await decideOne(n));
      } catch (err) {
        st(n).lastError = err.message;
        await emit({ kind: "decision_error", timestamp: new Date().toISOString(), deployment: n, stage: "internal", error: err.stack ?? String(err) });
      }
    }
    status.lastCycleMs = now() - t0;
    status.lastCycleAt = new Date().toISOString();
    return recs;
  }

  async function loop() {
    if (stopped) return;
    const t0 = Date.now();
    await cycle();
    if (!stopped) timer = setTimeout(loop, Math.max(0, cfg.decideIntervalS * 1000 - (Date.now() - t0)));
  }

  return {
    cycle,
    decideOne,
    start() {
      if (!stopped) return;
      stopped = false;
      loop();
    },
    stop() {
      stopped = true;
      clearTimeout(timer);
    },
    status() {
      return {
        ...status,
        deployments: [...per.values()].map((s) => ({
          name: s.name, decisions: s.decisions, last_error: s.lastError, last: s.last,
          last_scale_at: s.lastScaleAt ? new Date(s.lastScaleAt * 1000).toISOString() : null,
        })),
      };
    },
    recent: (n = 50) => recent.slice(-n),
  };
}
