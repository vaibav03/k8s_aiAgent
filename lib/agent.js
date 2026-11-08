// The monitoring loop: discover pods -> one PromQL query -> per-container ring
// buffers -> ML batch calls (one item per container, scored against that
// container's own limit) -> the most at-risk container is the pod's
// prediction -> k-of-n hysteresis -> (critical | LLM recommendation).
//
// Ticks are strictly sequential: the next tick is scheduled with setTimeout
// only after the current one has finished, so slow Prometheus/ML/LLM calls
// can never cause overlapping ticks (the old setInterval + isRunning flag
// still raced between its two intervals).

import { createKubeClient } from "./kube.js";
import { queryWorkingSet } from "./prom.js";
import { buildPredictItem, predictBatch } from "./ml.js";
import { createAlertState, updateAlert } from "./alert.js";
import { createLlm, validateRecommendation, runKubectl } from "./remediation.js";
import { parseQuantity } from "./quantity.js";

const log = (level, msg, extra) =>
  console.log(JSON.stringify({ t: new Date().toISOString(), level, msg, ...(extra ?? {}) }));

// Order containers by risk: anomalous first, then p_oom, then usage fraction.
function riskKey(r) {
  return (r.is_anomaly ? 2 : 0) + (r.p_oom ?? 0) / 2 + (r.usage_fraction ?? -1) * 1e-3;
}

export function createAgent(cfg, deps = {}) {
  const kube = deps.kube ?? createKubeClient(cfg);
  const llm = deps.llm !== undefined ? deps.llm : createLlm(cfg);
  const kubectl = deps.runKubectl ?? ((argv) => runKubectl(cfg, argv));
  const maxMemoryBytes = parseQuantity(cfg.maxMemoryLimit);

  const pods = new Map(); // name -> per-pod state
  const status = { startedAt: new Date().toISOString(), ticks: 0, lastTickAt: null, lastTickMs: null, errors: {} };
  let timer = null;
  let stopped = true;

  function podState(name) {
    let s = pods.get(name);
    if (!s) {
      s = {
        name,
        info: null,
        series: new Map(), // container name -> chronological samples
        restarts: new Map(), // container name -> last seen restartCount
        lastSampleAt: null,
        missingSince: null,
        prediction: null,
        predictedAt: null,
        alert: createAlertState(),
        lastCriticalAt: 0,
        lastRecommendation: null,
        actions: [],
      };
      pods.set(name, s);
    }
    return s;
  }

  async function handleAlert(s) {
    const info = s.info;
    if (info.crashLooping) {
      // Already crashed: an LLM resource tweak is not the right response.
      log("CRITICAL", "pod already CrashLoopBackOff while anomaly alert active; notify cluster administrator", {
        pod: s.name,
        lastTerminationReason: info.lastTerminationReason,
      });
      s.lastRecommendation = { at: new Date().toISOString(), action: "none", source: "rule", reason: "pod is in CrashLoopBackOff; escalated to administrator" };
      return;
    }

    const ctx = {
      namespace: cfg.namespace,
      deployment: info.deployment,
      containers: info.containers.map((c) => ({ ...c, workingSetBytes: s.series.get(c.name)?.at(-1)?.working_set_bytes ?? null })),
      maxMemoryBytes,
      maxReplicas: cfg.maxReplicas,
    };

    if (!llm) {
      s.lastRecommendation = { at: new Date().toISOString(), action: "none", source: "none", reason: "GOOGLE_API_KEY not set; LLM step skipped" };
      log("WARN", "anomaly alert fired; LLM disabled, recommendation recorded as none", { pod: s.name, p_oom: s.prediction?.p_oom });
      return;
    }

    let rec;
    try {
      rec = await llm.recommend({
        pod: s.name,
        namespace: cfg.namespace,
        deployment: info.deployment,
        containers: info.containers,
        container: s.prediction?.container ?? null,
        limit_bytes: s.prediction?.limit_bytes ?? null,
        restart_count: info.restartCount,
        prediction: s.prediction,
        recent_working_set_bytes: (s.series.get(s.prediction?.container) ?? []).slice(-10).map((x) => x.working_set_bytes),
        constraints: { max_memory_limit: cfg.maxMemoryLimit, max_replicas: cfg.maxReplicas },
      });
    } catch (err) {
      s.lastRecommendation = { at: new Date().toISOString(), action: "none", source: "llm", reason: `LLM failed: ${err.message}` };
      log("ERROR", "LLM recommendation failed", { pod: s.name, error: err.message });
      return;
    }

    const v = validateRecommendation(rec, ctx);
    const record = { at: new Date().toISOString(), source: "llm", raw: rec, action: rec?.action ?? null, reason: rec?.reason ?? null, valid: v.ok };
    if (!v.ok) {
      record.rejected = v.error;
      s.lastRecommendation = record;
      log("WARN", "LLM recommendation rejected by allowlist", { pod: s.name, error: v.error });
      return;
    }
    record.summary = v.summary;
    if (v.argv) {
      const result = await kubectl(v.argv, v.dryRunArgv);
      record.kubectl = result;
      s.actions.push(result);
      if (s.actions.length > 20) s.actions.shift();
      log(result.ok ? "INFO" : "ERROR", `kubectl ${result.dryRun ? "(dry-run=server) " : ""}${v.summary}`, { pod: s.name, ok: result.ok, stderr: result.stderr || undefined });
    }
    s.lastRecommendation = record;
  }

  async function tick() {
    const t0 = Date.now();
    status.ticks += 1;

    // 1) discover pods
    let discovered;
    try {
      discovered = await kube.listPods();
      delete status.errors.kube;
    } catch (err) {
      status.errors.kube = err.message;
      log("ERROR", "kube API pod list failed; retrying next tick", { error: err.message });
      return;
    }
    const live = new Set(discovered.map((p) => p.name));
    for (const name of pods.keys()) if (!live.has(name)) pods.delete(name);

    for (const info of discovered) {
      const s = podState(info.name);
      // A container restart resets its memory; pre-restart samples would fake a
      // trend. (The model is trained on histories cut the same way.)
      for (const c of info.containers) {
        const prev = s.restarts.get(c.name);
        if (prev != null && c.restartCount != null && c.restartCount > prev) {
          log("INFO", "restart detected; clearing sample window", { pod: info.name, container: c.name, restartCount: c.restartCount });
          s.series.delete(c.name);
        }
        if (c.restartCount != null) s.restarts.set(c.name, c.restartCount);
      }
      const names = new Set(info.containers.map((c) => c.name));
      for (const n of s.series.keys()) if (!names.has(n)) s.series.delete(n);
      s.info = info;
      if (info.crashLooping && Date.now() - s.lastCriticalAt >= cfg.cooldownMs) {
        s.lastCriticalAt = Date.now();
        log("CRITICAL", "pod in CrashLoopBackOff; notify cluster administrator", { pod: info.name, lastTerminationReason: info.lastTerminationReason, restartCount: info.restartCount });
      }
    }

    // 2) one PromQL query for all pods
    try {
      const values = await queryWorkingSet(cfg, [...live]);
      delete status.errors.prometheus;
      for (const name of live) {
        const s = pods.get(name);
        const byContainer = values.get(name);
        if (!byContainer || byContainer.size === 0) {
          s.missingSince ??= new Date().toISOString(); // missing stays missing (never 0)
          continue;
        }
        s.missingSince = null;
        for (const c of s.info.containers) {
          const v = byContainer.get(c.name);
          if (!v) continue;
          const buf = s.series.get(c.name) ?? [];
          const last = buf.at(-1);
          if (last && v.ts <= last.ts) continue; // keep chronological order
          buf.push(v);
          while (buf.length > cfg.window) buf.shift();
          s.series.set(c.name, buf);
          s.lastSampleAt = new Date(v.ts * 1000).toISOString();
        }
      }
    } catch (err) {
      status.errors.prometheus = err.message;
      log("ERROR", "prometheus query failed; retrying next tick", { error: err.message });
    }

    // 3) ML batch calls: one item per container with >= 2 samples
    const items = [];
    const owners = [];
    for (const name of live) {
      const s = pods.get(name);
      for (const c of s.info.containers) {
        const buf = s.series.get(c.name);
        if (!buf || buf.length < 2) continue;
        items.push(buildPredictItem(cfg, s.info, c, [...buf]));
        owners.push({ s, container: c });
      }
    }
    if (items.length === 0) return;
    let results;
    try {
      results = await predictBatch(cfg, items, {
        onItemError: (item, err) => log("WARN", "ML service rejected one item (422); skipped", { pod: item.pod, error: err.message }),
      });
      delete status.errors.ml;
    } catch (err) {
      status.errors.ml = err.message;
      log("ERROR", "ML service call failed; retrying next tick", { error: err.message });
      return;
    }

    // 4) per pod, the most at-risk container is the pod's prediction
    const best = new Map(); // pod state -> result
    results.forEach((r, i) => {
      const { s, container } = owners[i];
      if (!r || r.pod !== s.name) return;
      if (typeof r.is_anomaly !== "boolean") {
        log("WARN", "ML result missing boolean is_anomaly; ignored", { pod: s.name, container: container.name });
        return;
      }
      const cand = { ...r, container: container.name, limit_bytes: container.limitBytes };
      const cur = best.get(s);
      if (!cur || riskKey(cand) > riskKey(cur)) best.set(s, cand);
    });

    // 5) decisions
    for (const [s, r] of best) {
      s.prediction = r;
      s.predictedAt = new Date().toISOString();
      const d = updateAlert(s.alert, r.is_anomaly, { k: cfg.alertK, n: cfg.alertN, cooldownMs: cfg.cooldownMs });
      if (d.fire) {
        log("ALERT", `anomaly in ${d.anomalies} of last ${cfg.alertN} predictions`, { pod: s.name, container: r.container, p_oom: r.p_oom, time_to_limit_s: r.time_to_limit_s, reasons: r.reasons });
        try {
          await handleAlert(s);
        } catch (err) {
          log("ERROR", "alert handling failed", { pod: s.name, error: err.message });
        }
      }
    }
    status.lastTickMs = Date.now() - t0;
  }

  async function loop() {
    if (stopped) return;
    try {
      await tick();
    } catch (err) {
      log("ERROR", "tick crashed", { error: err.stack ?? String(err) });
    }
    status.lastTickAt = new Date().toISOString();
    if (!stopped) timer = setTimeout(loop, cfg.pollMs);
  }

  function podView(s) {
    return {
      pod: s.name,
      namespace: cfg.namespace,
      phase: s.info?.phase ?? null,
      deployment: s.info?.deployment ?? null,
      // limit of the container the latest prediction is about (the most at-risk one)
      limit_bytes: s.prediction?.limit_bytes ?? null,
      restart_count: s.info?.restartCount ?? null,
      crash_looping: s.info?.crashLooping ?? false,
      samples: Math.max(0, ...[...s.series.values()].map((b) => b.length)),
      containers: (s.info?.containers ?? []).map((c) => ({
        name: c.name,
        limit_bytes: c.limitBytes,
        restart_count: c.restartCount,
        samples: s.series.get(c.name)?.length ?? 0,
        working_set_bytes: s.series.get(c.name)?.at(-1)?.working_set_bytes ?? null,
      })),
      last_sample_at: s.lastSampleAt,
      missing_since: s.missingSince,
      latest_prediction: s.prediction,
      predicted_at: s.predictedAt,
      alert: {
        active: s.alert.active,
        recent: [...s.alert.history],
        fired_at: s.alert.firedAt ? new Date(s.alert.firedAt).toISOString() : null,
        cooldown_until: s.alert.cooldownUntil ? new Date(s.alert.cooldownUntil).toISOString() : null,
        fire_count: s.alert.fireCount,
      },
      last_recommendation: s.lastRecommendation,
      actions: s.actions,
    };
  }

  return {
    tick,
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
      return { ...status, pods: [...pods.values()].map(podView) };
    },
    pod(name) {
      const s = pods.get(name);
      return s ? podView(s) : null;
    },
  };
}
