// Client for the ML service (ml_service/, FastAPI). Contract:
// POST /v1/predict/batch {"items":[predict body...]} -> {"results":[...]}
// POST /v1/predict <predict body> -> result
//
// The server caps a batch at 500 items and rejects the whole batch with 422
// when any one item is invalid. So items are sent in chunks of at most
// cfg.mlBatchMax, and a chunk that gets a 422 is retried item by item, so
// one bad item costs only its own prediction.

import { requestJson, HttpError } from "./http.js";

export const SERVER_MAX_BATCH = 500;

const positiveOrNull = (x) => (typeof x === "number" && Number.isFinite(x) && x > 0 ? x : null);
const nonNegIntOrNull = (x) => (Number.isInteger(x) && x >= 0 ? x : null);
const nonNegOrNull = (x) => (typeof x === "number" && Number.isFinite(x) && x >= 0 ? x : null);

// One item per container: the container's own samples and its own limit.
export function buildPredictItem(cfg, pod, container, samples) {
  return {
    pod: pod.name,
    namespace: cfg.namespace,
    samples,
    limit_bytes: positiveOrNull(container.limitBytes),
    restart_count: nonNegIntOrNull(container.restartCount),
    seconds_since_restart: nonNegOrNull(container.secondsSinceRestart),
  };
}

// Returns an array aligned with `items`: a result, or null when that item was
// rejected (422). Throws when the service itself is unreachable / failing.
export async function predictBatch(cfg, items, { onItemError } = {}) {
  if (items.length === 0) return [];
  const size = Math.max(1, Math.min(cfg.mlBatchMax ?? SERVER_MAX_BATCH, SERVER_MAX_BATCH));
  const out = [];
  for (let i = 0; i < items.length; i += size) {
    const chunk = items.slice(i, i + size);
    let results;
    try {
      const data = await requestJson(`${cfg.mlUrl}/v1/predict/batch`, {
        method: "POST",
        body: { items: chunk },
        timeoutMs: cfg.httpTimeoutMs,
      });
      results = data?.results;
      if (!Array.isArray(results) || results.length !== chunk.length) {
        throw new Error("ML service returned no results array of the right length");
      }
    } catch (err) {
      if (!(err instanceof HttpError && err.status === 422)) throw err;
      results = [];
      for (const item of chunk) {
        try {
          results.push(await requestJson(`${cfg.mlUrl}/v1/predict`, { method: "POST", body: item, timeoutMs: cfg.httpTimeoutMs }));
        } catch (e) {
          if (!(e instanceof HttpError && e.status === 422)) throw e;
          onItemError?.(item, e);
          results.push(null);
        }
      }
    }
    out.push(...results);
  }
  return out;
}
