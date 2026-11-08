// Per-pod k-of-n hysteresis + cooldown.
//
// The old agent tested an undefined variable (`isAnamoly`, typo of
// `isAnomaly`), so the alert branch could never run. Here the decision is a
// pure function of the ML service's boolean `is_anomaly` field: an alert
// fires when at least K of the last N predictions were anomalous and the pod
// is not in its cooldown window. A single noisy prediction cannot fire.

export function createAlertState() {
  return { history: [], active: false, firedAt: null, cooldownUntil: 0, fireCount: 0 };
}

// Returns { fire: boolean, active: boolean, anomalies: number }.
export function updateAlert(state, isAnomaly, { k, n, cooldownMs, now = Date.now() }) {
  if (typeof isAnomaly !== "boolean") throw new TypeError("is_anomaly must be a boolean");
  state.history.push(isAnomaly);
  while (state.history.length > n) state.history.shift();
  const anomalies = state.history.filter(Boolean).length;
  state.active = anomalies >= k;
  let fire = false;
  if (state.active && now >= state.cooldownUntil) {
    fire = true;
    state.firedAt = now;
    state.cooldownUntil = now + cooldownMs;
    state.fireCount += 1;
  }
  return { fire, active: state.active, anomalies };
}
