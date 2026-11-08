"""loadlab scenario runner.

    python runner/lab.py scenarios/smoke.yaml                 # one scenario
    python runner/lab.py --lanes 2 scenarios/a.yaml scenarios/b.yaml ...   # 2 concurrent lanes
    python runner/lab.py --keep scenarios/x.yaml              # keep the namespace for debugging

Per run: fresh namespace lab-<run_id> -> app (+HPA) -> wait ready -> k6 Job (open-loop) -> faults on
schedule (ground truth in faults.jsonl) -> export Prometheus (5s step) -> delete namespace.
Output: runs/<run_id>/{scenario.json, faults.jsonl, deploy.csv, pods.csv, k6_summary.json,
k6.log, events.json, pods_final.json, runner.log, queries.json}
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import copy
import json
import os
import random
import re
import string
import subprocess
import sys
import threading
import time
from pathlib import Path

import requests
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import promexport  # noqa: E402

CTX = os.environ.get("KUBE_CONTEXT", "kubeai-lab")
PROM = promexport.PROM
IMAGE = os.environ.get("LOADLAB_IMAGE", "loadlab:v1")
K6_IMAGE = os.environ.get("K6_IMAGE", "grafana/k6:0.54.0")
CHAOS_FAULTS = {  # fault type -> (chaos keys set at start, reset body at end)
    "memory_leak": (["leak_kb_per_s"], {"leak_kb_per_s": 0}),
    "dep_latency": (["dep_latency_ms", "dep_fraction"], {"dep_latency_ms": 0, "dep_fraction": 1.0}),
    "error_rate": (["error_rate"], {"error_rate": 0}),
    "cpu_hog": (["cpu_hog_millicores"], {"cpu_hog_millicores": 0}),
}
OTHER_FAULTS = {"pod_kill", "bad_deploy", "crashloop"}
RECONCILE_S = 3.0  # how often active chaos faults are re-checked (restarted containers / new pods)
# The node is a docker container capped at 6 CPU (kubelet reports the host's 12, so the scheduler will not
# stop overcommit). With 2 lanes, each lane is capped at LANE_MAX_REPLICAS app pods (x 500m limit = 2 cores)
# so both lanes + 2 k6 + monitoring fit in 6 cores even when both saturate. 0 disables the cap.
LANE_MAX_REPLICAS = int(os.environ.get("LANE_MAX_REPLICAS", "4"))


# ----------------------------------------------------------------------------- helpers
class Log:
    def __init__(self, path: Path, prefix: str):
        self.f = open(path, "a", buffering=1)
        self.prefix = prefix

    def __call__(self, *msg):
        line = f"{time.strftime('%H:%M:%S')} [{self.prefix}] " + " ".join(str(m) for m in msg)
        print(line, flush=True)
        self.f.write(line + "\n")


def kubectl(*args, ns: str | None = None, input: str | None = None, check: bool = True,
            timeout: int = 120) -> subprocess.CompletedProcess:
    cmd = ["kubectl", "--context", CTX] + (["-n", ns] if ns else []) + list(args)
    p = subprocess.run(cmd, input=input, capture_output=True, text=True, timeout=timeout)
    if check and p.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed: {p.stderr.strip()}")
    return p


def kjson(*args, ns=None):
    return json.loads(kubectl(*args, "-o", "json", ns=ns).stdout)


def render(path: Path, **kw) -> list[dict]:
    txt = string.Template(path.read_text()).substitute({k: str(v) for k, v in kw.items()})
    return [d for d in yaml.safe_load_all(txt) if d]


def apply(objs: list[dict], ns: str):
    kubectl("apply", "-f", "-", ns=ns, input=json.dumps({"apiVersion": "v1", "kind": "List", "items": objs}))


def ensure_prometheus(log=print):
    try:
        if requests.get(f"{PROM}/-/ready", timeout=3).ok:
            return
    except requests.RequestException:
        pass
    log("Prometheus not reachable; starting port-forward helper")
    subprocess.Popen([str(ROOT / "monitoring" / "port-forward.sh")], stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL, start_new_session=True)
    for _ in range(30):
        time.sleep(1)
        try:
            if requests.get(f"{PROM}/-/ready", timeout=2).ok:
                return
        except requests.RequestException:
            pass
    raise RuntimeError(f"Prometheus not reachable at {PROM}")


def load_calibration() -> dict:
    p = ROOT / "calibration.json"
    return json.loads(p.read_text()) if p.exists() else {}


def resolve_scenario(sc: dict) -> dict:
    sc = copy.deepcopy(sc)
    sc.setdefault("app", {})
    sc["app"].setdefault("cpu_ms_per_req", 25)
    sc["app"].setdefault("alloc_kb", 64)
    sc.setdefault("faults", [])
    pol = sc.setdefault("scaling", {"policy": "fixed", "replicas": 1})
    if pol["policy"] == "hpa_cpu":
        pol.setdefault("min", 1)
        pol.setdefault("max", 6)
        pol.setdefault("target_cpu", 70)
        pol.setdefault("initial", pol["min"])
    elif pol["policy"] == "fixed":
        pol.setdefault("replicas", 1)
    else:
        raise ValueError(f"unknown scaling policy {pol['policy']}")
    lane_cap = int(os.environ.get("LOADLAB_LANE_CAP", "0") or 0)  # set by main() for --lanes 2
    if lane_cap > 0:
        key = "max" if pol["policy"] == "hpa_cpu" else "replicas"
        if pol[key] > lane_cap:
            pol["_uncapped_" + key] = pol[key]
            pol[key] = lane_cap
            if pol["policy"] == "hpa_cpu":
                pol["min"] = min(pol["min"], lane_cap)
                pol["initial"] = min(pol["initial"], lane_cap)
        pol["_lane_cap"] = lane_cap
    tr = sc["traffic"]
    tr.setdefault("params", {})
    if "base_rps" not in tr:
        # base load expressed in "pods worth of SLO capacity" -> needs calibration.json
        cal = load_calibration()
        cap = cal.get("per_pod_rps_at_slo")
        if not cap:
            raise ValueError("traffic.base_load_pods needs calibration.json (run the capacity probe)")
        tr["base_rps"] = round(float(tr["base_load_pods"]) * cap, 2)
        tr["_capacity_ref_rps_per_pod"] = cap
    for f in sc["faults"]:
        if f["type"] not in CHAOS_FAULTS and f["type"] not in OTHER_FAULTS:
            raise ValueError(f"unknown fault type {f['type']}")
        f.setdefault("target", "one-pod")
        f.setdefault("params", {})
        f.setdefault("end_s", f["start_s"])
    return sc


def make_run_id(sc_id: str) -> str:
    rid = f"{sc_id}-{time.strftime('%m%d-%H%M%S')}-{random.randint(0, 0xfff):03x}"
    rid = re.sub(r"[^a-z0-9-]", "-", rid.lower())
    return rid[:55].strip("-")


# ----------------------------------------------------------------------------- runner
class Run:
    def __init__(self, scenario_path: Path, keep: bool = False):
        raw = yaml.safe_load(scenario_path.read_text())
        self.sc = resolve_scenario(raw)
        self.run_id = make_run_id(self.sc["id"])
        self.ns = f"lab-{self.run_id}"
        self.dir = ROOT / "runs" / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.log = Log(self.dir / "runner.log", self.run_id)
        self.keep = keep
        self.t0: float | None = None
        self.faults_out = open(self.dir / "faults.jsonl", "a", buffering=1)
        self.active: dict[int, dict] = {}
        self.applied: dict[int, dict[str, int]] = {}  # chaos fault idx -> {pod: restartCount it was applied at}
        self.flock = threading.Lock()
        # serialises fault start/end with the reconciler, so a fault is never re-applied after it ended
        self.alock = threading.Lock()
        self.stop = threading.Event()
        self.meta: dict = {"run_id": self.run_id, "namespace": self.ns, "scenario_file": str(scenario_path),
                           "image": IMAGE, "k6_image": K6_IMAGE, "context": CTX}

    # ---- infra
    def deploy(self):
        sc, ns = self.sc, self.ns
        kubectl("create", "namespace", ns)
        kubectl("label", "namespace", ns, "loadlab=run", f"loadlab-scenario={sc['id'][:60]}")
        pol = sc["scaling"]
        replicas = pol["replicas"] if pol["policy"] == "fixed" else pol["initial"]
        objs = render(ROOT / "k8s" / "app.yaml", replicas=replicas, image=IMAGE,
                      cpu_ms_per_req=sc["app"]["cpu_ms_per_req"], alloc_kb=sc["app"]["alloc_kb"])
        for o in objs:  # extra env defaults from the scenario (e.g. ERROR_RATE)
            if o["kind"] == "Deployment":
                env = o["spec"]["template"]["spec"]["containers"][0]["env"]
                for k, v in (sc["app"].get("env") or {}).items():
                    env.append({"name": k, "value": str(v)})
        apply(objs, ns)
        if pol["policy"] == "hpa_cpu":
            hpa = render(ROOT / "k8s" / "hpa.yaml", min_replicas=pol["min"], max_replicas=pol["max"],
                         target_cpu=pol["target_cpu"])
            if pol.get("scale_down_window_s") is not None:
                # optional autoscaling/v2 behavior: shorter scale-down stabilisation so scale-in shows up
                # within a run (default is 300 s). Recorded in scenario.json via the scaling spec.
                for o in hpa:
                    o["spec"]["behavior"] = {"scaleDown": {
                        "stabilizationWindowSeconds": int(pol["scale_down_window_s"])}}
            apply(hpa, ns)
        kubectl("rollout", "status", "deploy/loadlab", "--timeout=180s", ns=ns, timeout=200)
        self.log("app ready:", replicas, "replica(s), policy", pol["policy"])

    def wait_hpa_metrics(self, timeout=240):
        """HPA runs: wait until the HPA has a CPU reading (metrics-server needs 1-2 resolution
        periods for a new pod), so the run starts from a warm, steady autoscaler."""
        if self.sc["scaling"]["policy"] != "hpa_cpu":
            return
        deadline = time.time() + timeout
        while time.time() < deadline:
            st = kjson("get", "hpa", "loadlab", ns=self.ns).get("status", {})
            if st.get("currentMetrics") and any(m.get("resource", {}).get("current", {}).get("averageUtilization")
                                                is not None for m in st["currentMetrics"]):
                self.log("HPA has metrics:", st.get("currentMetrics")[0]["resource"]["current"])
                return
            time.sleep(5)
        self.log("WARN: HPA still has no metrics; continuing")

    def wait_scraped(self, timeout=60):
        """Wait until Prometheus sees the app pods + cAdvisor rows, so the run has a baseline."""
        want = self.sc["scaling"].get("replicas", self.sc["scaling"].get("initial", 1))
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                a = promexport.prom_instant(f'count(up{{namespace="{self.ns}",job="annotated-pods"}} == 1)')
                c = promexport.prom_instant(
                    f'count(container_memory_working_set_bytes{{namespace="{self.ns}",scope="pod",pod=~"loadlab-.*"}})')
                if a and c and float(a[0]["value"][1]) >= want and float(c[0]["value"][1]) >= want:
                    return
            except Exception:  # noqa: BLE001
                pass
            time.sleep(3)
        self.log("WARN: Prometheus has not seen all pods yet; continuing")

    def start_load(self):
        tr, ns = self.sc["traffic"], self.ns
        script = (ROOT / "load" / "shapes.js").read_text()
        cm = {"apiVersion": "v1", "kind": "ConfigMap", "metadata": {"name": "k6-script"},
              "data": {"shapes.js": script}}
        job = render(ROOT / "k8s" / "k6-job.yaml", k6_image=K6_IMAGE, shape=tr["shape"], base_rps=tr["base_rps"],
                     duration_s=self.sc["duration_s"], target_url=f"http://loadlab.{ns}.svc/work",
                     params_json="{}")[0]
        env = job["spec"]["template"]["spec"]["containers"][0]["env"]
        for e in env:
            if e["name"] == "PARAMS_JSON":
                e["value"] = json.dumps(tr.get("params", {}))
        apply([cm, job], ns)
        deadline = time.time() + 180
        while time.time() < deadline:
            pods = kjson("get", "pods", "-l", "job-name=k6", ns=ns)["items"]
            if pods and pods[0]["status"].get("phase") in ("Running", "Succeeded", "Failed"):
                self.t0 = time.time()
                self.k6_pod = pods[0]["metadata"]["name"]
                self.log("k6 started:", tr["shape"], "base_rps", tr["base_rps"], "pod", self.k6_pod)
                return
            time.sleep(1)
        raise RuntimeError("k6 pod did not start")

    # ---- faults
    def app_pods(self, ready_only=True) -> list[str]:
        items = kjson("get", "pods", "-l", "app=loadlab", ns=self.ns)["items"]
        out = []
        for p in items:
            if p["metadata"].get("deletionTimestamp"):
                continue
            conds = {c["type"]: c["status"] for c in p["status"].get("conditions", [])}
            if ready_only and conds.get("Ready") != "True":
                continue
            out.append(p["metadata"]["name"])
        return sorted(out)

    def pod_restarts(self) -> dict[str, int]:
        """Ready app pods -> app container restartCount."""
        out = {}
        for p in kjson("get", "pods", "-l", "app=loadlab", ns=self.ns)["items"]:
            if p["metadata"].get("deletionTimestamp"):
                continue
            conds = {c["type"]: c["status"] for c in p["status"].get("conditions", [])}
            if conds.get("Ready") != "True":
                continue
            cs = [c for c in p["status"].get("containerStatuses", []) if c["name"] == "app"]
            out[p["metadata"]["name"]] = int(cs[0].get("restartCount", 0)) if cs else 0
        return out

    def chaos_many(self, pods: list[str], body: dict) -> list[dict]:
        """Apply a chaos body to several pods in parallel (keeps the per-pod skew to ~1 exec)."""
        def one(p):
            try:
                return {**self.chaos(p, body), "applied_ts": time.time()}
            except Exception as e:  # noqa: BLE001
                return {"pod": p, "error": str(e)[:200], "applied_ts": time.time()}
        if not pods:
            return []
        with cf.ThreadPoolExecutor(max_workers=min(8, len(pods))) as ex:
            return list(ex.map(one, pods))

    def reconcile(self):
        with self.alock:
            self._reconcile()

    def _reconcile(self):
        """Chaos state lives in the container's memory, so an OOM-restarted container (or a new pod from
        the HPA / a rollout) would silently lose an active fault while faults.jsonl still says it is on.
        A real leak or slow dependency does not go away on restart, so re-apply: one-pod faults follow
        their pod across restarts; target=all faults also cover pods created after the fault started.
        Every re-application is recorded in the fault's ground-truth record ("reapplied")."""
        with self.flock:
            todo = [(i, rec) for i, rec in self.active.items() if rec["type"] in CHAOS_FAULTS]
        if not todo:
            return
        cur = self.pod_restarts()
        for i, rec in todo:
            body = {k: rec["params"][k] for k in CHAOS_FAULTS[rec["type"]][0] if k in rec["params"]}
            seen = self.applied.setdefault(i, {})
            cand = cur if rec["target"] == "all" else {p: r for p, r in cur.items() if p in rec["pods"]}
            for pod, rc in cand.items():
                if seen.get(pod) == rc:
                    continue
                try:
                    resp = self.chaos(pod, body)
                except Exception as e:  # noqa: BLE001  (not reachable yet - next tick)
                    self.log(f"reconcile f{i} {pod}: {str(e)[:120]}")
                    continue
                with self.flock:
                    if i not in self.active:
                        return
                    reason = "container_restart" if pod in seen else "new_pod"
                    seen[pod] = rc
                    if pod not in rec["pods"]:
                        rec["pods"].append(pod)
                    rec.setdefault("reapplied", []).append(
                        {"pod": pod, "restart_count": rc, "ts": time.time(), "rel_s": round(time.time() - self.t0, 3),
                         "reason": reason})
                self.log(f"FAULT REAPPLY f{i} {rec['type']} -> {pod} (restartCount={rc})")

    def reconciler(self):
        while not self.stop.wait(RECONCILE_S):
            try:
                self.reconcile()
            except Exception as e:  # noqa: BLE001
                self.log(f"reconcile error: {str(e)[:200]}")

    def chaos(self, pod: str, body: dict) -> dict:
        """POST the chaos body to the pod's admin port through the API-server pod proxy. (The first
        version used `kubectl exec python -c ...`, which started a Python process inside the app's
        500m cgroup on every fault toggle: +12 MB and a CPU burst that showed up as p95 blips.)
        exec stays as a fallback."""
        last = None
        for _ in range(3):
            p = kubectl("create", "--raw", f"/api/v1/namespaces/{self.ns}/pods/{pod}:8081/proxy/chaos",
                        "-f", "-", input=json.dumps(body), check=False, timeout=20)
            if p.returncode == 0:
                try:
                    return json.loads(p.stdout.strip().splitlines()[-1])
                except (ValueError, IndexError):
                    last = f"bad response: {p.stdout[:200]}"
            else:
                last = p.stderr.strip()
            time.sleep(1)
        try:
            return self.chaos_exec(pod, body)
        except RuntimeError as e:
            raise RuntimeError(f"chaos {pod} failed: proxy: {last}; exec: {e}") from None

    def chaos_exec(self, pod: str, body: dict) -> dict:
        code = ("import json,sys,urllib.request as u;"
                "r=u.Request('http://127.0.0.1:8081/chaos',data=sys.argv[1].encode(),"
                "headers={'Content-Type':'application/json'});"
                "print(u.urlopen(r,timeout=10).read().decode())")
        last = None
        for _ in range(2):
            p = kubectl("exec", pod, "-c", "app", "--", "python", "-c", code, json.dumps(body),
                        ns=self.ns, check=False, timeout=30)
            if p.returncode == 0:
                return json.loads(p.stdout.strip().splitlines()[-1])
            last = p.stderr.strip()
            time.sleep(1)
        raise RuntimeError(f"chaos {pod} failed: {last}")

    def fault_start(self, i: int, f: dict):
        typ, params = f["type"], f["params"]
        rec = {"fault_id": f"{self.run_id}-f{i}", "idx": i, "type": typ, "target": f["target"], "params": params,
               "scheduled_start_s": f["start_s"], "scheduled_end_s": f["end_s"], "run_id": self.run_id,
               "namespace": self.ns}
        pods = self.app_pods()
        if f["target"] == "one-pod":
            idx = int(f.get("pod_index", 0)) % max(1, len(pods))
            pods = pods[idx:idx + 1]
        if typ in CHAOS_FAULTS:
            body = {k: params[k] for k in CHAOS_FAULTS[typ][0] if k in params}
            rec["start_requested_ts"] = time.time()
            rec["responses"] = self.chaos_many(pods, body)
            ok = [r["applied_ts"] for r in rec["responses"] if "error" not in r]
            # ground truth: the fault is live from the first pod that actually got it
            rec["start_ts"] = min(ok) if ok else rec["start_requested_ts"]
            rec["pods"] = list(pods)
            rc = self.pod_restarts()
            self.applied[i] = {p: rc.get(p, 0) for p in pods}
        elif typ == "pod_kill":
            victim = pods[:1]
            rec["start_ts"] = time.time()
            kubectl("delete", "pod", *victim, "--wait=false", ns=self.ns)
            rec["pods"] = victim
        elif typ == "bad_deploy":
            rec["start_ts"] = time.time()
            kubectl("set", "env", "deploy/loadlab", "-c", "app",
                    f"CPU_MS_PER_REQ={params.get('cpu_ms_per_req', 3 * self.sc['app']['cpu_ms_per_req'])}",
                    f"APP_VERSION={params.get('version', 'v2-bad')}", ns=self.ns)
            rec["pods"] = "rollout"
        elif typ == "crashloop":
            rec["start_ts"] = time.time()
            kubectl("set", "env", "deploy/loadlab", "-c", "app", "CRASH_ON_START=1",
                    f"APP_VERSION={params.get('version', 'v2-crash')}", ns=self.ns)
            rec["pods"] = "rollout"
        rec["start_rel_s"] = round(rec["start_ts"] - self.t0, 3)
        self.log(f"FAULT START f{i} {typ} target={f['target']} pods={rec['pods']} params={params}")
        if typ == "pod_kill":
            rec["end_ts"], rec["end_rel_s"] = rec["start_ts"], rec["start_rel_s"]
            self.faults_out.write(json.dumps(rec) + "\n")
        else:
            with self.flock:
                self.active[i] = rec

    def fault_end(self, i: int, f: dict):
        with self.flock:
            rec = self.active.pop(i, None)
        if rec is None:
            return
        typ = f["type"]
        rec["end_ts"] = time.time()
        if typ in CHAOS_FAULTS:
            reset = dict(CHAOS_FAULTS[typ][1])
            if typ == "memory_leak" and f["params"].get("free_on_end"):
                reset["free_leak"] = True
            # only pods that still exist (deleted rollout/HPA pods would cost 3 retries each and delay the
            # reset of the live ones); a pod mid-restart may fail - its fresh container has no fault anyway
            live = set(self.app_pods(ready_only=False))
            rec["pods_gone_at_end"] = [p for p in rec["pods"] if p not in live]
            rec["end_responses"] = self.chaos_many([p for p in rec["pods"] if p in live], reset)
            # ground truth: the fault is live until the last pod had it removed
            rec["end_requested_ts"] = rec["end_ts"]
            rec["end_ts"] = max([r["applied_ts"] for r in rec["end_responses"]] or [rec["end_ts"]])
        elif typ == "bad_deploy":
            kubectl("set", "env", "deploy/loadlab", "-c", "app",
                    f"CPU_MS_PER_REQ={self.sc['app']['cpu_ms_per_req']}", "APP_VERSION=v1", ns=self.ns)
            rec["remediation"] = "env reverted (rollout to good version)"
        elif typ == "crashloop":
            kubectl("set", "env", "deploy/loadlab", "-c", "app", "CRASH_ON_START=0", "APP_VERSION=v1", ns=self.ns)
            rec["remediation"] = "env reverted (rollout to good version)"
        rec["end_rel_s"] = round(rec["end_ts"] - self.t0, 3)
        self.log(f"FAULT END   f{i} {typ}")
        self.faults_out.write(json.dumps(rec) + "\n")

    def scheduler(self):
        events = []
        for i, f in enumerate(self.sc["faults"]):
            events.append((f["start_s"], 0, i, "start"))
            if f["type"] != "pod_kill":
                events.append((f["end_s"], 1, i, "end"))
        events.sort()
        for at, _, i, kind in events:
            while not self.stop.is_set() and time.time() < self.t0 + at:
                time.sleep(0.1)
            if self.stop.is_set():
                return
            try:
                with self.alock:
                    (self.fault_start if kind == "start" else self.fault_end)(i, self.sc["faults"][i])
            except Exception as e:  # noqa: BLE001
                self.log(f"ERROR applying fault f{i} {kind}: {e}")

    def wait_load_done(self):
        deadline = self.t0 + self.sc["duration_s"] + 120
        while time.time() < deadline:
            st = kjson("get", "job", "k6", ns=self.ns)["status"]
            if st.get("succeeded") or st.get("failed"):
                return "succeeded" if st.get("succeeded") else "failed"
            time.sleep(3)
        return "timeout"

    def collect(self):
        p = kubectl("logs", f"job/k6", ns=self.ns, check=False, timeout=60)
        (self.dir / "k6.log").write_text(p.stdout + p.stderr)
        m = re.search(r"===K6_SUMMARY_BEGIN===\n(.*)\n===K6_SUMMARY_END===", p.stdout, re.S)
        if m:
            (self.dir / "k6_summary.json").write_text(json.dumps(json.loads(m.group(1)), indent=1))
        else:
            self.log("WARN: no k6 summary found in logs")
        ev = kubectl("get", "events", "-o", "json", ns=self.ns, check=False)
        (self.dir / "events.json").write_text(ev.stdout)
        pods = kubectl("get", "pods", "-o", "json", ns=self.ns, check=False)
        (self.dir / "pods_final.json").write_text(pods.stdout)

    def execute(self) -> Path:
        sc = self.sc
        ensure_prometheus(self.log)
        self.log("scenario", sc["id"], "-> namespace", self.ns)
        t_created = time.time()
        sched = None
        status = "error"
        try:
            self.deploy()
            self.wait_scraped()
            self.wait_hpa_metrics()
            time.sleep(10)  # short idle baseline in the exported window
            self.start_load()
            sched = threading.Thread(target=self.scheduler, daemon=True)
            sched.start()
            threading.Thread(target=self.reconciler, daemon=True).start()
            status = self.wait_load_done()
            self.log("load finished:", status)
        except BaseException as e:
            self.log("RUN FAILED:", repr(e))
            self.stop.set()
            try:
                self.collect()
            except Exception:  # noqa: BLE001
                pass
            if not self.keep:
                kubectl("delete", "namespace", self.ns, "--wait=false", check=False)
            raise
        finally:
            self.stop.set()
            if sched:
                sched.join(timeout=5)
            for i in list(self.active):
                try:
                    with self.alock:
                        self.fault_end(i, sc["faults"][i])
                except Exception as e:  # noqa: BLE001
                    self.log("ERROR ending fault", i, e)
            try:
                self.collect()
            except Exception as e:  # noqa: BLE001
                self.log("collect error", e)
        t_end = time.time()
        time.sleep(20)  # let the last scrapes land
        window = [self.t0 - 30 if self.t0 else t_created, t_end + 10]
        meta = {**self.meta, "t_created": t_created, "t0": self.t0, "t_end": t_end, "status": status,
                "export_window": window, "step_s": promexport.STEP}
        (self.dir / "scenario.json").write_text(json.dumps({**sc, "_run": meta}, indent=2))
        try:
            dep, pods = promexport.export_run(self.ns, window[0], window[1], self.dir, sc["scaling"]["policy"])
            self.log(f"exported deploy.csv ({len(dep)} rows) pods.csv ({len(pods)} rows) -> {self.dir}")
        finally:  # never leak a namespace, even if the export fails (re-export later: promexport.py <run_dir>)
            if not self.keep:
                kubectl("delete", "namespace", self.ns, "--wait=false", check=False)
                self.log("namespace deleted")
        return self.dir


def run_one(path: str, keep: bool) -> str:
    return str(Run(Path(path), keep=keep).execute())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenarios", nargs="+")
    ap.add_argument("--lanes", type=int, default=1, help="concurrent scenarios (max 2 on the 6-CPU lab)")
    ap.add_argument("--keep", action="store_true", help="do not delete the namespace")
    a = ap.parse_args()
    if a.lanes <= 1 or len(a.scenarios) == 1:
        for s in a.scenarios:
            print(run_one(s, a.keep))
        return
    # lanes: each scenario in its own process (isolated logs / failures), app pods capped per lane
    if LANE_MAX_REPLICAS > 0:
        os.environ["LOADLAB_LANE_CAP"] = str(LANE_MAX_REPLICAS)
        print(f"lanes: capping each lane at {LANE_MAX_REPLICAS} app replicas (LANE_MAX_REPLICAS)")
    with cf.ThreadPoolExecutor(max_workers=min(a.lanes, 2)) as ex:
        futs = {ex.submit(subprocess.run, [sys.executable, __file__] + (["--keep"] if a.keep else []) + [s]): s
                for s in a.scenarios}
        for fut in cf.as_completed(futs):
            print(futs[fut], "->", "ok" if fut.result().returncode == 0 else "FAILED")


if __name__ == "__main__":
    main()
