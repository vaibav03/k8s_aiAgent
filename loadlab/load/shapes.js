// loadlab traffic shapes for k6 (open-loop: ramping-arrival-rate, independent of latency).
// Env: SHAPE, BASE_RPS, DURATION_S, TARGET_URL, PARAMS_JSON (shape-specific overrides).
// Rates are expressed per minute internally so fractional RPS are representable.
import http from 'k6/http';

const SHAPE = __ENV.SHAPE || 'steady';
const B = parseFloat(__ENV.BASE_RPS || '10');
const D = parseInt(__ENV.DURATION_S || '300', 10);
const URL = __ENV.TARGET_URL || 'http://loadlab/work';
const P = JSON.parse(__ENV.PARAMS_JSON || '{}');

// deterministic PRNG (mulberry32) so a scenario is reproducible
function rng(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// points: [[t_seconds, rps], ...] -> k6 stages (linear interpolation between points)
function toStages(points) {
  const st = [];
  for (let i = 1; i < points.length; i++) {
    const dt = Math.round(points[i][0] - points[i - 1][0]);
    if (dt <= 0) continue;
    st.push({ target: Math.max(0, Math.round(points[i][1] * 60)), duration: `${dt}s` });
  }
  return st;
}

function shapePoints() {
  const pts = [];
  switch (SHAPE) {
    case 'steady': {
      const ramp = P.ramp_s ?? 10;
      pts.push([0, B * 0.2], [ramp, B], [D, B]);
      break;
    }
    case 'step_ramp': { // capacity sweep
      const start = P.start_rps ?? B, step = P.step_rps ?? Math.max(1, B * 0.25);
      const stepS = P.step_s ?? 45, maxR = P.max_rps ?? Infinity, ramp = P.ramp_s ?? 3;
      let t = 0, r = start;
      pts.push([0, start]);
      while (t + stepS <= D + 1e-9) {
        if (t > 0) pts.push([t + ramp, r]); // ramp to new level
        pts.push([t + stepS, r]);           // hold
        t += stepS; r = Math.min(r + step, maxR);
      }
      if (pts[pts.length - 1][0] < D) pts.push([D, pts[pts.length - 1][1]]); // hold to the end
      break;
    }
    case 'diurnal': { // compressed day: sine with noise
      const period = P.period_s ?? 600, amp = P.amplitude ?? 0.6, noise = P.noise ?? 0.08;
      const dt = P.point_s ?? 10, rand = rng(P.seed ?? 42);
      for (let t = 0; t <= D; t += dt) {
        const base = B * (1 - amp * Math.cos((2 * Math.PI * t) / period));
        pts.push([t, Math.max(0.5, base * (1 + noise * (2 * rand() - 1)))]);
      }
      break;
    }
    case 'spike': { // baseline then mult x for spike_s
      const at = P.spike_at_s ?? Math.round(D * 0.35), mult = P.spike_mult ?? 4;
      const dur = P.spike_s ?? 120, ramp = P.ramp_s ?? 10;
      pts.push([0, B], [at, B], [at + ramp, B * mult], [at + ramp + dur, B * mult],
        [at + 2 * ramp + dur, B], [D, B]);
      break;
    }
    case 'flash_crowd': { // sudden sustained mult x
      const at = P.at_s ?? Math.round(D * 0.3), mult = P.mult ?? 3, ramp = P.ramp_s ?? 5;
      pts.push([0, B], [at, B], [at + ramp, B * mult], [D, B * mult]);
      break;
    }
    case 'sawtooth': {
      const period = P.period_s ?? 120, lo = (P.low_mult ?? 0.5) * B, hi = (P.high_mult ?? 2.0) * B;
      pts.push([0, lo]);
      for (let t = 0; t + period <= D; t += period) {
        pts.push([t + period - 1, hi], [t + period, lo]);
      }
      if (pts[pts.length - 1][0] < D) pts.push([D, lo + (hi - lo) * ((D % period) / period)]);
      break;
    }
    default:
      throw new Error(`unknown SHAPE ${SHAPE}`);
  }
  // clip to duration
  return pts.filter((p) => p[0] <= D);
}

const POINTS = shapePoints();
const PEAK = Math.max(...POINTS.map((p) => p[1]));

export const options = {
  discardResponseBodies: true,
  // New TCP connection per request: kube-proxy (L4) balances per connection, so keep-alive
  // connections pin traffic to whichever pods were Ready when they were opened (after an OOM
  // restart the restarted pod would get ~0 traffic forever). Per-request connections give the
  // per-request balancing an L7 proxy/mesh would, which keeps per-pod signals interpretable.
  noConnectionReuse: P.conn_reuse ? false : true,
  summaryTrendStats: ['avg', 'min', 'med', 'p(90)', 'p(95)', 'p(99)', 'max'],
  scenarios: {
    main: {
      executor: 'ramping-arrival-rate',
      startRate: Math.round(POINTS[0][1] * 60),
      timeUnit: '1m',
      preAllocatedVUs: Math.min(Math.ceil(PEAK * 8) + 10, 600), // pre-allocated: on-demand VU start-up lag dropped ~3% of iterations under overload
      maxVUs: Math.min(Math.ceil(PEAK * 13) + 20, 1500), // up to ~12s (client timeout) of queueing at peak
      stages: toStages(POINTS),
      gracefulStop: '10s',
    },
  },
};

export default function () {
  // client timeout > server deadline (REQUEST_TIMEOUT_S=10) so overload surfaces as server-side 504s
  http.get(URL, { timeout: '12s', tags: { name: 'work' } });
}

export function handleSummary(data) {
  data.loadlab = { shape: SHAPE, base_rps: B, duration_s: D, params: P, points: POINTS };
  return { stdout: '\n===K6_SUMMARY_BEGIN===\n' + JSON.stringify(data) + '\n===K6_SUMMARY_END===\n' };
}
