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
  };
  if (cfg.alertK < 1 || cfg.alertN < cfg.alertK) {
    throw new Error(`invalid hysteresis ALERT_K=${cfg.alertK} ALERT_N=${cfg.alertN} (need 1 <= K <= N)`);
  }
  if (cfg.window < 2) throw new Error("WINDOW must be >= 2");
  if (cfg.mlBatchMax < 1 || cfg.mlBatchMax > 500) throw new Error("ML_BATCH_MAX must be in [1, 500] (the ML service caps a batch at 500)");
  return cfg;
}

// Config safe to expose on /status (no secrets).
export function publicConfig(cfg) {
  const { googleApiKey, ...rest } = cfg;
  return { ...rest, llmEnabled: Boolean(googleApiKey) };
}
