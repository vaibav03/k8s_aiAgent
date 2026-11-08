// Kubernetes resource.Quantity parser (memory-oriented).
// Supports binary suffixes (Ki, Mi, Gi, Ti, Pi, Ei), decimal suffixes
// (k, K, M, G, T, P, E, and the milli suffix "m"), plain numbers and
// e-notation ("129e6"). Returns a Number of bytes, or null when the input is
// empty or not a valid quantity.
//
// The old agent did parseInt(limit.split("M")[0]), which returned 1 for "1Gi",
// NaN for "805306368" and treated "800M" (decimal) as MiB.

const BINARY = { Ki: 2 ** 10, Mi: 2 ** 20, Gi: 2 ** 30, Ti: 2 ** 40, Pi: 2 ** 50, Ei: 2 ** 60 };
const DECIMAL = { m: 1e-3, k: 1e3, K: 1e3, M: 1e6, G: 1e9, T: 1e12, P: 1e15, E: 1e18 };

const RE = /^([+-]?(?:\d+\.?\d*|\.\d+))(?:([eE][+-]?\d+)|(Ki|Mi|Gi|Ti|Pi|Ei|m|k|K|M|G|T|P|E))?$/;

export function parseQuantity(input) {
  if (input === null || input === undefined) return null;
  if (typeof input === "number") return Number.isFinite(input) ? input : null;
  const s = String(input).trim();
  if (!s) return null;
  const m = RE.exec(s);
  if (!m) return null;
  const [, num, exp, suffix] = m;
  let value = Number(num);
  if (exp) value = Number(num + exp);
  else if (suffix) value *= BINARY[suffix] ?? DECIMAL[suffix];
  return Number.isFinite(value) ? value : null;
}

// Render bytes as a canonical Mi quantity (rounded up) for kubectl.
export function formatMi(bytes) {
  return `${Math.ceil(bytes / 2 ** 20)}Mi`;
}
