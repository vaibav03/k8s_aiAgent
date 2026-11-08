// KubeAI agent. Two loops:
//  1. per pod (lib/agent.js, OOM_LOOP; default on only when the deployment loop is off):
//     watches pods' memory working set, asks the ML service for P(OOM within horizon),
//     applies k-of-n hysteresis, and (optionally) asks Gemini for an allowlisted
//     remediation (memory limit / rollout restart; never scale), gated by ACT_MODE
//     exactly like the deployment loop (observe: no kubectl; dryrun; act);
//  2. per Deployment (lib/scaler.js, DEPLOYMENTS / DEPLOY_SELECTOR): the
//     promexport.py PromQL window -> POST /v1/decide -> state, replicas,
//     remediation, acted on per ACT_MODE (observe | dryrun | act).
// See README.md.
//
// Run: npm start   (.env is loaded by dotenv if present)

import "dotenv/config";
import express from "express";
import { loadConfig, publicConfig } from "./lib/config.js";
import { createAgent } from "./lib/agent.js";
import { createScaler } from "./lib/scaler.js";
import { createRemediationLedger } from "./lib/remediation.js";

const cfg = loadConfig();
// one per-Deployment remediation cooldown shared by both loops
const ledger = createRemediationLedger();
const agent = cfg.oomLoop ? createAgent(cfg, { ledger }) : null;
const scaler = cfg.scalerEnabled ? createScaler(cfg, { ledger }) : null;

const app = express();

app.get("/healthz", (_req, res) => res.json({ status: "ok" }));

app.get("/status", (_req, res) =>
  res.json({
    config: publicConfig(cfg),
    ...(agent ? agent.status() : { pods: [], oomLoop: "disabled" }),
    scaler: scaler ? scaler.status() : { enabled: false },
  }),
);

// The most recent deployment decisions (same records as the JSON log lines).
app.get("/decisions", (req, res) => {
  if (!scaler) return res.status(404).json({ error: "deployment loop disabled (set DEPLOYMENTS or DEPLOY_SELECTOR)" });
  const n = Math.min(Math.max(Number.parseInt(req.query.n, 10) || 50, 1), 200);
  res.json({ decisions: scaler.recent(n) });
});

app.get("/pods/:name", (req, res) => {
  if (!agent) return res.status(404).json({ error: "OOM loop disabled (OOM_LOOP=false)" });
  const p = agent.pod(req.params.name);
  if (!p) return res.status(404).json({ error: `pod ${req.params.name} not watched` });
  res.json(p);
});

const server = app.listen(cfg.port, () => {
  console.log(
    JSON.stringify({
      t: new Date().toISOString(),
      level: "INFO",
      msg: "agent listening",
      port: server.address().port,
      namespace: cfg.namespace,
      selector: cfg.podSelector || "(all pods)",
      ml: cfg.mlUrl,
      llm: Boolean(cfg.googleApiKey),
      allowExec: cfg.allowExec,
      oomLoop: Boolean(agent),
      scaler: scaler
        ? { deployments: cfg.deployments.length ? cfg.deployments : `selector:${cfg.deploySelector}`, actMode: cfg.actMode, effectiveActMode: cfg.effectiveActMode, intervalS: cfg.decideIntervalS, historyS: cfg.historyS, hpaOverride: cfg.hpaOverride, decisionLog: cfg.decisionLog || null }
        : false,
    }),
  );
  if (cfg.actMode === "act" && !cfg.allowExec) {
    console.log(JSON.stringify({ t: new Date().toISOString(), level: "WARN", msg: "ACT_MODE=act without ALLOW_EXEC=true: running as dryrun" }));
  }
  if (agent && scaler) {
    console.log(JSON.stringify({ t: new Date().toISOString(), level: "WARN", msg: "OOM_LOOP=true and the deployment loop both enabled: they share one per-Deployment remediation cooldown; the OOM loop cannot scale" }));
  }
  agent?.start();
  scaler?.start();
});

function shutdown(sig) {
  console.log(JSON.stringify({ t: new Date().toISOString(), level: "INFO", msg: `received ${sig}, shutting down` }));
  agent?.stop();
  scaler?.stop();
  server.close(() => process.exit(0));
  setTimeout(() => process.exit(0), 3000).unref();
}
process.on("SIGTERM", () => shutdown("SIGTERM"));
process.on("SIGINT", () => shutdown("SIGINT"));
