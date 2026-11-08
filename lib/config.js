// All configuration comes from the environment (see .env.example).
// Secrets (GOOGLE_API_KEY) are read from env only and never logged.

function int(name, def) {
  const v = process.env[name];
  if (v === undefined || v === "") return def;
  const n = Number.parseInt(v, 10);
  if (!Number.isFinite(n)) throw new Error(`${name} must be an integer, got "${v}"`);
  return n;
}

function bool(name, def) {
  const v = process.env[name];
  if (v === undefined || v === "") return def;
  return /^(1|true|yes|on)$/i.test(v);
}

function str(name, def) {
  const v = process.env[name];
  return v === undefined || v === "" ? def : v;
}

const stripSlash = (u) => u.replace(/\/+$/, "");

export function loadConfig() {
  const cfg = {
    prometheusUrl: stripSlash(str("PROMETHEUS_URL", "http://localhost:9090")),
    kubeApi: stripSlash(str("KUBE_API", "http://localhost:8001")),
    kubeTokenFile: str("KUBE_TOKEN_FILE", "/var/run/secrets/kubernetes.io/serviceaccount/token"),
    namespace: str("NAMESPACE", "default"),
    podSelector: str("POD_SELECTOR", ""),
    mlUrl: stripSlash(str("ML_URL", "http://localhost:8000")),
    pollMs: int("POLL_MS", 15000),
    window: int("WINDOW", 40),
    alertK: int("ALERT_K", 2),
    alertN: int("ALERT_N", 3),
    cooldownMs: int("COOLDOWN_MS", 300000),
    httpTimeoutMs: int("HTTP_TIMEOUT_MS", 10000),
    mlBatchMax: int("ML_BATCH_MAX", 500),
    geminiModel: str("GEMINI_MODEL", "gemini-2.5-flash"),
    googleApiKey: str("GOOGLE_API_KEY", ""),
    allowExec: bool("ALLOW_EXEC", false),
    kubectl: str("KUBECTL", "kubectl"),
    maxMemoryLimit: str("MAX_MEMORY_LIMIT", "4Gi"),
    maxReplicas: int("MAX_REPLICAS", 10),
    port: int("PORT", 3001),


    // Deployment-level decision loop (lib/scaler.js -> POST /v1/decide)
    deployments: str("DEPLOYMENTS", "").split(",").map((s) => s.trim()).filter(Boolean),
    deploySelector: str("DEPLOY_SELECTOR", ""),
    decideIntervalS: int("DECIDE_INTERVAL_S", 15),
    historyS: int("HISTORY_S", 900),
    actMode: str("ACT_MODE", "observe").toLowerCase(),
    scaleMin: int("SCALE_MIN", 1),
    scaleMax: int("SCALE_MAX", int("MAX_REPLICAS", 10)),
    scaleUpCooldownS: int("SCALE_UP_COOLDOWN_S", 15),
    scaleDownCooldownS: int("SCALE_DOWN_COOLDOWN_S", 60),
    remediationCooldownS: int("REMEDIATION_COOLDOWN_S", 300),
    hpaOverride: bool("HPA_OVERRIDE", false),
    decisionLog: str("DECISION_LOG", ""),
    appContainer: str("APP_CONTAINER", "app"),
    appJob: str("APP_JOB", "annotated-pods"),
    // Per-Deployment template, e.g. 'app="{deployment}"'. Without "{deployment}" the app-level
    // metrics are namespace-wide, so the scaler refuses to decide in a namespace that has more
    // than one Deployment (every Deployment would read the namespace's total traffic).
    appMetricMatcher: str("APP_METRIC_MATCHER", ""),
    // default (empty): "{deployment}-<rs hash>-<5 char suffix>" (lib/promDeploy.js), which does
    // not match pods of a sibling Deployment whose name extends this one (web vs web-api)
    podRegexTemplate: str("POD_REGEX", ""),
    promConcurrency: int("PROM_CONCURRENCY", 6),
    decideTimeoutMs: int("DECIDE_TIMEOUT_MS", 20000),
  };
  cfg.scalerEnabled = bool("SCALER", cfg.deployments.length > 0 || Boolean(cfg.deploySelector));
  // Per-pod OOM loop (lib/agent.js). Off by default whenever the deployment loop runs: the
  // deployment loop already handles memory leaks (restart before OOM) with the HPA guard,
  // replica bounds and cooldowns; OOM_LOOP=true turns both on (they then share one
  // per-Deployment remediation cooldown, and the OOM loop cannot scale).
  cfg.oomLoop = bool("OOM_LOOP", !cfg.scalerEnabled);
  cfg.podRegexFor = (d) => (cfg.podRegexTemplate ? cfg.podRegexTemplate.replaceAll("{deployment}", d) : undefined);
  cfg.appMatcherPerDeployment = cfg.appMetricMatcher.includes("{deployment}");
  cfg.appMatcherFor = (d) => cfg.appMetricMatcher.replaceAll("{deployment}", String(d).replace(/[\\"]/g, "\\$&"));
  if (!["observe", "dryrun", "act"].includes(cfg.actMode)) {
    throw new Error(`ACT_MODE must be observe | dryrun | act, got "${cfg.actMode}"`);
  }
  // act needs the explicit exec switch as well; otherwise it degrades to a server-side dry run
  cfg.effectiveActMode = cfg.actMode === "act" && !cfg.allowExec ? "dryrun" : cfg.actMode;
  if (cfg.decideIntervalS < 1) throw new Error("DECIDE_INTERVAL_S must be >= 1");
  if (cfg.historyS < 180) throw new Error("HISTORY_S must be >= 180 (the engine needs >= 3 min of history)");
  if (cfg.scaleMin < 0 || cfg.scaleMax < 1 || cfg.scaleMin > cfg.scaleMax) {
    throw new Error(`invalid SCALE_MIN=${cfg.scaleMin} SCALE_MAX=${cfg.scaleMax}`);
  }
  if (cfg.alertK < 1 || cfg.alertN < cfg.alertK) {
    throw new Error(`invalid hysteresis ALERT_K=${cfg.alertK} ALERT_N=${cfg.alertN} (need 1 <= K <= N)`);
  }
  if (cfg.window < 2) throw new Error("WINDOW must be >= 2");
  if (cfg.mlBatchMax < 1 || cfg.mlBatchMax > 500) throw new Error("ML_BATCH_MAX must be in [1, 500] (the ML service caps a batch at 500)");
  return cfg;
}

// Config safe to expose on /status (no secrets).
export function publicConfig(cfg) {
  const { googleApiKey, podRegexFor, appMatcherFor, ...rest } = cfg;
  return { ...rest, llmEnabled: Boolean(googleApiKey) };
}
