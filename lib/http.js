// Small fetch wrapper with timeout and useful errors.

export class HttpError extends Error {
  constructor(message, status, body) {
    super(message);
    this.status = status;
    this.body = body;
  }
}

export async function requestJson(url, { method = "GET", headers = {}, body, timeoutMs = 10000 } = {}) {
  const init = { method, headers: { accept: "application/json", ...headers }, signal: AbortSignal.timeout(timeoutMs) };
  if (body !== undefined) {
    init.body = JSON.stringify(body);
    init.headers["content-type"] = "application/json";
  }
  let res;
  try {
    res = await fetch(url, init);
  } catch (err) {
    const why = err?.cause?.code || err?.name || err?.message;
    throw new HttpError(`${method} ${url} failed: ${why}`, 0);
  }
  const text = await res.text();
  if (!res.ok) {
    throw new HttpError(`${method} ${url} -> HTTP ${res.status}: ${text.slice(0, 300)}`, res.status, text);
  }
  try {
    return text ? JSON.parse(text) : null;
  } catch {
    throw new HttpError(`${method} ${url} returned non-JSON body`, res.status, text);
  }
}
