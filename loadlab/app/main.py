"""loadlab: a deliberately simple service whose resource behaviour is *real*.

GET /work burns CPU_MS_PER_REQ ms of real CPU on a small worker pool (WORKERS threads), throttled by
the pod's CFS quota, so when a pod is saturated requests genuinely queue and latency grows. Latency is
timed from arrival (the event loop never runs CPU work), so it includes the queueing delay. Fault switches (POST /chaos) inject the
failure modes that scaling does NOT fix: memory leak, slow dependency, 5xx errors, CPU hog.

Chaos is served both on the main app (:8080/chaos, queued behind work like any request) and on a
tiny admin HTTP server on its own thread (:8081/chaos) so the lab runner can always reach a pod,
even when its event loop is saturated.
"""
from __future__ import annotations

import asyncio
import json
import multiprocessing as mp
import os
import random
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import anyio.to_thread
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest


def _envf(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


if os.environ.get("CRASH_ON_START", "0") == "1":
    # Simulates a bad release that dies during start-up -> CrashLoopBackOff.
    print("CRASH_ON_START=1: simulated fatal start-up error (bad config)", file=sys.stderr, flush=True)
    time.sleep(_envf("CRASH_DELAY_S", 1.0))
    sys.exit(3)

CPU_MS_PER_REQ = _envf("CPU_MS_PER_REQ", 25.0)
ALLOC_KB = int(_envf("ALLOC_KB", 64))
WORKERS = int(_envf("WORKERS", 2))
# Server-side request deadline (like a gateway/upstream timeout). Requests that cannot finish within it
# get a 504, so user-visible timeouts under overload show up in http_requests_total{code="504"} and
# hence in the 5xx error rate (k6's client timeout is set slightly longer, 12s).
REQUEST_TIMEOUT_S = _envf("REQUEST_TIMEOUT_S", 10.0)
POD = os.environ.get("POD_NAME", os.uname().nodename)
VERSION = os.environ.get("APP_VERSION", "v1")

# ---------------------------------------------------------------- chaos state
_lock = threading.Lock()
STATE = {
    "leak_kb_per_s": _envf("LEAK_KB_PER_S", 0),
    "dep_latency_ms": _envf("DEP_LATENCY_MS", 0),
    "dep_fraction": _envf("DEP_FRACTION", 1.0),
    "error_rate": _envf("ERROR_RATE", 0),
    "cpu_hog_millicores": _envf("CPU_HOG_MILLICORES", 0),
}
_leaked: list[bytes] = []
_hog_proc: mp.Process | None = None

# ---------------------------------------------------------------- metrics
BUCKETS = (0.005, 0.01, 0.025, 0.03, 0.04, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4,
           0.5, 0.6, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0)
REQS = Counter("http_requests_total", "Requests to /work", ["code"])
LAT = Histogram("http_request_duration_seconds", "Latency of /work (incl. queueing in the pod)",
                buckets=BUCKETS)
INFLIGHT = Gauge("http_inflight_requests", "In-flight /work requests in this pod")
LEAKED = Gauge("loadlab_leaked_bytes", "Bytes retained by the simulated leak")
CHAOS = Gauge("loadlab_chaos", "Active chaos settings", ["knob"])
for _c in ("200", "500", "504"):  # pre-create series so rate()/error ratio exist from the first scrape
    REQS.labels(_c)


def _publish_chaos() -> None:
    for k, v in STATE.items():
        CHAOS.labels(k).set(v)
    LEAKED.set(sum(len(b) for b in _leaked))


def burn_cpu(ms: float) -> None:
    """Busy-loop until this thread has consumed `ms` of CPU time (wall time stretches under
    CFS throttling or contention, exactly like real compute)."""
    if ms <= 0:
        return
    end = time.thread_time() + ms / 1000.0
    x = 0
    while time.thread_time() < end:
        for i in range(500):
            x += i * i
    return None


def _hog_main(millicores: float) -> None:  # runs in a child process (same cgroup)
    duty = max(0.0, min(millicores / 1000.0, 1.0))
    while True:
        t0 = time.monotonic()
        burn_cpu(100 * duty)
        rest = 0.1 - (time.monotonic() - t0)
        if rest > 0:
            time.sleep(rest)


def _apply_hog(millicores: float) -> None:
    global _hog_proc
    if _hog_proc is not None and _hog_proc.is_alive():
        _hog_proc.terminate()
        _hog_proc.join(timeout=2)
        _hog_proc = None
    if millicores > 0:
        _hog_proc = mp.get_context("fork").Process(target=_hog_main, args=(millicores,), daemon=True)
        _hog_proc.start()


def apply_chaos(body: dict) -> dict:
    with _lock:
        if body.get("reset"):
            body = {"leak_kb_per_s": 0, "dep_latency_ms": 0, "dep_fraction": 1.0, "error_rate": 0,
                    "cpu_hog_millicores": 0, "free_leak": True, **{k: v for k, v in body.items() if k != "reset"}}
        for k in STATE:
            if k in body:
                STATE[k] = float(body[k])
        if body.get("free_leak"):
            _leaked.clear()
        if "cpu_hog_millicores" in body:
            _apply_hog(STATE["cpu_hog_millicores"])
        _publish_chaos()
        return {"pod": POD, **STATE, "leaked_bytes": sum(len(b) for b in _leaked)}


# ---------------------------------------------------------------- background leak
def _leak_loop() -> None:
    tick = 0.2
    while True:
        time.sleep(tick)
        rate = STATE["leak_kb_per_s"]
        if rate > 0:
            n = int(rate * 1024 * tick)
            _leaked.append(b"\x5a" * n)  # non-zero bytes -> pages really resident
            LEAKED.set(sum(len(b) for b in _leaked))


# ---------------------------------------------------------------- admin server (:8081)
class _Admin(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _send(self, code: int, obj: dict) -> None:
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_body(self) -> bytes:
        # The runner POSTs through the API-server pod proxy, which forwards the body chunked.
        if "chunked" in (self.headers.get("Transfer-Encoding") or "").lower():
            out = b""
            while True:
                size = int(self.rfile.readline().split(b";")[0].strip() or b"0", 16)
                if size == 0:
                    while self.rfile.readline() not in (b"\r\n", b"\n", b""):
                        pass  # trailers
                    return out
                out += self.rfile.read(size)
                self.rfile.readline()  # CRLF after each chunk
        return self.rfile.read(int(self.headers.get("Content-Length", 0) or 0))

    def do_GET(self):
        if self.path.startswith("/chaos"):
            self._send(200, {"pod": POD, **STATE, "leaked_bytes": sum(len(b) for b in _leaked)})
        else:
            self._send(200, {"ok": True})

    def do_POST(self):
        if not self.path.startswith("/chaos"):
            return self._send(404, {"error": "not found"})
        try:
            body = json.loads(self._read_body() or b"{}")
            self._send(200, apply_chaos(body))
        except Exception as e:  # noqa: BLE001
            self._send(400, {"error": str(e)})


def _start_admin() -> None:
    srv = ThreadingHTTPServer(("0.0.0.0", int(_envf("ADMIN_PORT", 8081))), _Admin)
    threading.Thread(target=srv.serve_forever, daemon=True, name="admin").start()


# ---------------------------------------------------------------- app
class MetricsMiddleware:
    """Pure-ASGI middleware: times /work from arrival (event loop is never blocked) to response."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("path") != "/work":
            return await self.app(scope, receive, send)
        INFLIGHT.inc()
        t0 = time.perf_counter()
        status = {"code": 500}

        async def _send(msg):
            if msg["type"] == "http.response.start":
                status["code"] = msg["status"]
            await send(msg)

        try:
            await self.app(scope, receive, _send)
        finally:
            LAT.observe(time.perf_counter() - t0)
            REQS.labels(str(status["code"])).inc()
            INFLIGHT.dec()


app = FastAPI(title="loadlab", docs_url=None, redoc_url=None, openapi_url=None)
app.add_middleware(MetricsMiddleware)


@app.on_event("startup")
async def _startup() -> None:
    global _cpu_sem
    _cpu_sem = asyncio.Semaphore(WORKERS)
    anyio.to_thread.current_default_thread_limiter().total_tokens = WORKERS + 2
    threading.Thread(target=_leak_loop, daemon=True, name="leak").start()
    _start_admin()
    if STATE["cpu_hog_millicores"] > 0:
        _apply_hog(STATE["cpu_hog_millicores"])
    _publish_chaos()


_cpu_sem: asyncio.Semaphore | None = None


def _cpu_part() -> None:
    buf = bytearray(b"\x01") * (ALLOC_KB * 1024) if ALLOC_KB > 0 else b""  # transient allocation
    burn_cpu(CPU_MS_PER_REQ)
    del buf


@app.get("/work")
async def work():
    try:
        return await asyncio.wait_for(_work(), timeout=REQUEST_TIMEOUT_S)
    except asyncio.TimeoutError:
        # queued (or waiting on the slow dependency) past the deadline: shed it, as a gateway would
        return JSONResponse({"error": "deadline exceeded", "pod": POD}, status_code=504)


async def _work():
    # CPU work runs on a small worker pool (WORKERS threads, default 2 - plenty to saturate a
    # 500m limit). Requests beyond that wait on the semaphore *inside* the timed region, so the
    # histogram includes real queueing delay, while the event loop (accept/parse/probes) stays
    # responsive. The slow dependency is an async wait outside the CPU pool (no CPU, no slot).
    async with _cpu_sem:
        await anyio.to_thread.run_sync(_cpu_part)
    if STATE["dep_latency_ms"] > 0 and random.random() < STATE["dep_fraction"]:
        await asyncio.sleep(STATE["dep_latency_ms"] / 1000.0)
    if STATE["error_rate"] > 0 and random.random() < STATE["error_rate"]:
        return JSONResponse({"error": "injected failure", "pod": POD}, status_code=500)
    return {"ok": True, "pod": POD, "version": VERSION}


@app.post("/chaos")
async def chaos(req: Request):
    return apply_chaos(await req.json())


@app.get("/chaos")
async def chaos_get():
    return {"pod": POD, **STATE, "leaked_bytes": sum(len(b) for b in _leaked)}


@app.get("/healthz")
async def healthz():
    return {"ok": True}


@app.get("/readyz")
async def readyz():
    return {"ready": True}


@app.get("/metrics")
async def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)
