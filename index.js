// KubeAI agent: watches pods' memory working set, asks the ML service for
// P(OOM within horizon), applies k-of-n hysteresis, and (optionally) asks
// Gemini for an allowlisted remediation that is executed as a server-side
// dry run unless ALLOW_EXEC=true. See README.md.
//
// Run: npm start   (.env is loaded by dotenv if present)

import "dotenv/config";
import express from "express";
import { loadConfig, publicConfig } from "./lib/config.js";
import { createAgent } from "./lib/agent.js";

const cfg = loadConfig();
const agent = createAgent(cfg);

const app = express();

app.get("/healthz", (_req, res) => res.json({ status: "ok" }));

app.get("/status", (_req, res) => res.json({ config: publicConfig(cfg), ...agent.status() }));

app.get("/pods/:name", (req, res) => {
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
    }),
  );
  agent.start();
});

function shutdown(sig) {
  console.log(JSON.stringify({ t: new Date().toISOString(), level: "INFO", msg: `received ${sig}, shutting down` }));
  agent.stop();
  server.close(() => process.exit(0));
  setTimeout(() => process.exit(0), 3000).unref();
}
process.on("SIGTERM", () => shutdown("SIGTERM"));
process.on("SIGINT", () => shutdown("SIGINT"));
