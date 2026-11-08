"""Run the full scenario matrix (scenarios/matrix.yaml) on 2 lanes.

    python runner/matrix.py                       # run everything not yet done
    python runner/matrix.py --only d-leak-oom     # (re-)run selected ids
    python runner/matrix.py --list                # show plan / state

* Each matrix entry is written to scenarios/matrix/<id>.yaml and executed by `lab.py <file>` in its own
  process, with LOADLAB_LANE_CAP=LANE_MAX_REPLICAS (4) exactly like `lab.py --lanes 2`.
* Scheduling: 2 lanes pull from one queue in matrix order; a `heavy: true` run never starts while the
  other lane is running a heavy run (deep overload / connection churn interferes across lanes).
* Validation after each run: status succeeded, deploy.csv rows >= 80 % of duration/5s, no all-NaN core
  column, one faults.jsonl record per scheduled fault. A failed run is moved to runs/_failed/ and retried
  once.
* State: runs/_matrix/state.json (id -> run_id/status), log: runs/_matrix/matrix.log.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MATRIX = ROOT / "scenarios" / "matrix.yaml"
SCDIR = ROOT / "scenarios" / "matrix"
MDIR = ROOT / "runs" / "_matrix"
LANE_CAP = os.environ.get("LANE_MAX_REPLICAS", "4")
CORE = ["rps", "p95", "error_rate", "cpu_cores_used", "mem_ws_bytes_total", "replicas_ready"]

lock = threading.Lock()


def log(*m):
    line = f"{time.strftime('%H:%M:%S')} " + " ".join(str(x) for x in m)
    print(line, flush=True)
    with open(MDIR / "matrix.log", "a") as f:
        f.write(line + "\n")


def load_state() -> dict:
    p = MDIR / "state.json"
    return json.loads(p.read_text()) if p.exists() else {}


def save_state(st: dict):
    (MDIR / "state.json").write_text(json.dumps(st, indent=1))


def validate(run_dir: Path, sc: dict) -> list[str]:
    errs = []
    sj = run_dir / "scenario.json"
    if not sj.exists():
        return ["no scenario.json"]
    meta = json.loads(sj.read_text()).get("_run", {})
    if meta.get("status") != "succeeded":
        errs.append(f"k6 status {meta.get('status')}")
    dp = run_dir / "deploy.csv"
    if not dp.exists():
        return errs + ["no deploy.csv"]
    df = pd.read_csv(dp)
    need = 0.8 * sc["duration_s"] / 5
    if len(df) < need:
        errs.append(f"deploy.csv {len(df)} rows < {need:.0f}")
    for c in CORE:
        if c not in df or df[c].isna().all():
            errs.append(f"column {c} all NaN")
    if not (run_dir / "pods.csv").exists():
        errs.append("no pods.csv")
    nf = len(sc.get("faults", []))
    fl = run_dir / "faults.jsonl"
    got = len([x for x in fl.read_text().splitlines() if x.strip()]) if fl.exists() else 0
    if got != nf:
        errs.append(f"faults.jsonl {got} records != {nf} scheduled")
    if "ERROR applying fault" in (run_dir / "runner.log").read_text() if (run_dir / "runner.log").exists() else "":
        errs.append("fault application error in runner.log")
    return errs


def run_entry(sc: dict, attempt: int) -> tuple[str | None, list[str]]:
    path = SCDIR / f"{sc['id']}.yaml"
    out = MDIR / f"{sc['id']}.attempt{attempt}.log"
    env = {**os.environ, "LOADLAB_LANE_CAP": LANE_CAP}
    with open(out, "w") as f:
        p = subprocess.run([sys.executable, str(HERE / "lab.py"), str(path)], stdout=f, stderr=subprocess.STDOUT,
                           env=env, timeout=sc["duration_s"] + 1200)
    lines = [x.strip() for x in out.read_text().splitlines() if x.strip()]
    run_dir = Path(lines[-1]) if lines and lines[-1].startswith(str(ROOT / "runs")) else None
    if run_dir is None:  # failed before printing; find newest dir for this id
        cands = sorted((ROOT / "runs").glob(f"{sc['id']}-*"), key=lambda d: d.stat().st_mtime)
        run_dir = cands[-1] if cands else None
    if p.returncode != 0:
        return (str(run_dir) if run_dir else None), [f"lab.py exit {p.returncode}"]
    return str(run_dir), validate(run_dir, sc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--lanes", type=int, default=2)
    a = ap.parse_args()
    MDIR.mkdir(parents=True, exist_ok=True)
    SCDIR.mkdir(parents=True, exist_ok=True)
    entries = yaml.safe_load(MATRIX.read_text())["runs"]
    for sc in entries:
        (SCDIR / f"{sc['id']}.yaml").write_text(
            f"# generated from scenarios/matrix.yaml by runner/matrix.py - edit matrix.yaml instead\n"
            + yaml.safe_dump(sc, sort_keys=False))
    st = load_state()
    if a.list:
        for sc in entries:
            print(f"{sc['id']:28s} {sc['group']} heavy={sc['heavy']!s:5s} {sc['duration_s']:4d}s ->",
                  st.get(sc["id"], {}).get("status", "pending"))
        return
    todo = [sc for sc in entries if (sc["id"] in a.only if a.only else st.get(sc["id"], {}).get("status") != "ok")]
    log(f"matrix start: {len(todo)} runs, lanes={a.lanes}, lane cap={LANE_CAP}, "
        f"est lane-s={sum(s['duration_s'] + 80 for s in todo)}")
    running: dict[int, dict] = {}

    def pick(lane: int):
        with lock:
            other_heavy = any(r["heavy"] for ln, r in running.items() if ln != lane)
            for sc in todo:
                if sc["heavy"] and other_heavy:
                    continue
                todo.remove(sc)
                running[lane] = sc
                return sc
            return None if not todo else "wait"

    def lane_loop(lane: int):
        while True:
            sc = pick(lane)
            if sc is None:
                return
            if sc == "wait":
                time.sleep(5)
                continue
            for attempt in (1, 2):
                log(f"lane{lane} START {sc['id']} (attempt {attempt}, {sc['duration_s']}s, heavy={sc['heavy']})")
                t = time.time()
                try:
                    rd, errs = run_entry(sc, attempt)
                except Exception as e:  # noqa: BLE001
                    rd, errs = None, [repr(e)[:300]]
                dur = time.time() - t
                with lock:
                    st = load_state()
                    st[sc["id"]] = {"run_id": Path(rd).name if rd else None, "status": "ok" if not errs else "failed",
                                    "errors": errs, "attempt": attempt, "wall_s": round(dur), "lane": lane,
                                    "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
                    save_state(st)
                log(f"lane{lane} END   {sc['id']} -> {Path(rd).name if rd else None} "
                    f"{'OK' if not errs else 'FAILED ' + '; '.join(errs)} ({dur:.0f}s)")
                if not errs:
                    break
                if rd and Path(rd).exists():
                    dst = ROOT / "runs" / "_failed" / Path(rd).name
                    dst.parent.mkdir(exist_ok=True)
                    Path(rd).rename(dst)
                    log(f"lane{lane} moved failed run to {dst}")
            with lock:
                running.pop(lane, None)

    ths = [threading.Thread(target=lane_loop, args=(i,)) for i in range(a.lanes)]
    for i, t in enumerate(ths):
        if i:
            time.sleep(20)  # stagger namespace creation
        t.start()
    for t in ths:
        t.join()
    log("matrix done:", json.dumps({k: v["status"] for k, v in load_state().items()}))


if __name__ == "__main__":
    main()
