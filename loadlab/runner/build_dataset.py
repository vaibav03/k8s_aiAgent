"""Consolidate matrix runs into data/dataset_deploy.csv.gz + data/dataset_pods.csv.gz (+ data/summary.json).

    python runner/build_dataset.py            # status-ok runs from every round's matrix state file

Rounds (STATE_FILES): round 1 = runs/_matrix/state.json (scenarios/matrix.yaml),
round 2 = runs/_matrix/state_matrix_round2.json (scenarios/matrix_round2.yaml).

Added columns (both files):
  run_id, scenario, group, policy, shape, base_rps, seconds_since_start (ts - t0, t0 = k6 pod Running),
  active_fault      '+'-joined fault types whose ground-truth window contains ts ('none' if none)
  fault_target_pod  '|'-joined per active fault: 'all', 'rollout' or ';'-joined target pod names
  fault_ids         '|'-joined fault_id(s) from faults.jsonl
pods file only:
  pod_is_fault_target  1 if this pod is targeted by an active fault (target=all -> every pod;
                       rollout faults -> pods of the new ReplicaSet, i.e. pod-template-hash differs from
                       the baseline one; one-pod faults -> the pod(s) in faults.jsonl `pods`)
  pod_active_fault     '+'-joined active faults that target this pod ('none' otherwise)
both files (QA pass): qa_flag  'ok' | 'node_glitch' (known unlabelled node-wide interference) |
                     'post_fault_tail' (labelled none, <=30 s after a fault label ends; signals still trail)
deploy file (QA pass): goodput_rps = rps * (1 - error_rate); pods_not_ready = pods present but not ready
pods file (QA pass): mem_ws / mem_frac_limit == 0 (cAdvisor before first refresh) -> NaN
both files (round 2): round  1 | 2 -- collection round; appended as the LAST column so every earlier
                     column keeps its position (backward compatible)

Fault windows: chaos faults [start_ts, end_ts] (actual apply/reset times); bad_deploy / crashloop
[start_ts, end_ts] (rollout start -> rollback start); pod_kill is instantaneous and is labelled for
POD_KILL_WINDOW_S after the delete (replacement pod scheduling + readiness).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "data"
POD_KILL_WINDOW_S = 30.0
POST_FAULT_TAIL_S = 30.0
# Known node-wide interference that hit both lanes at once (DATASET_CARD caveat 2; found by
# runner/qa_dataset.py): unix-ts ranges, inclusive. Rows inside get qa_flag = node_glitch.
NODE_GLITCH_WINDOWS = [(1794151443, 1794151457)]  # 2026-11-08 08:24:03-08:24:17 MST (3 rows per lane)
# collection round -> matrix state file (runner/matrix.py writes one per matrix yaml)
STATE_FILES = {1: ROOT / "runs" / "_matrix" / "state.json",
               2: ROOT / "runs" / "_matrix" / "state_matrix_round2.json"}


def clean_pods(p: pd.DataFrame) -> pd.DataFrame:
    """cAdvisor reports a working set of exactly 0 for a container before its first stats refresh.
    A running Python process never has 0 bytes, so these are missing values, not measurements."""
    for c in ("mem_ws", "mem_frac_limit"):
        p.loc[p[c] == 0, c] = np.nan
    return p


def add_derived(d: pd.DataFrame, p: pd.DataFrame) -> pd.DataFrame:
    """Deploy-level columns derived from existing signals (added by the data QA pass)."""
    # rps counts every completed response, including fast 5xx under deep overload -> goodput
    d["goodput_rps"] = d["rps"] * (1 - d["error_rate"])
    # replicas_ready does not drop during a stuck rollout (old ReplicaSet keeps serving; surge pod
    # is the one that is not ready). Count pods present but not ready at each step.
    nr = p[p["ready"] == 0].groupby("ts").size()
    d["pods_not_ready"] = d["ts"].map(nr).fillna(0).astype(int)
    return d


def qa_flags(df: pd.DataFrame, faults: list[dict]) -> list[str]:
    out = []
    for ts, af in zip(df["ts"].to_numpy(), df["active_fault"].to_numpy()):
        if any(a <= ts <= b for a, b in NODE_GLITCH_WINDOWS):
            out.append("node_glitch")
        elif af == "none" and any(0 < ts - r["label_end_ts"] <= POST_FAULT_TAIL_S for r in faults):
            out.append("post_fault_tail")
        else:
            out.append("ok")
    return out


def rs_hash(pod: str) -> str:
    parts = pod.split("-")
    return parts[-2] if len(parts) >= 3 else ""


def load_faults(run_dir: Path) -> list[dict]:
    recs = []
    p = run_dir / "faults.jsonl"
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r["type"] == "pod_kill":
                    r["label_end_ts"] = r["start_ts"] + POD_KILL_WINDOW_S
                else:
                    r["label_end_ts"] = r["end_ts"]
                recs.append(r)
    return sorted(recs, key=lambda r: r["start_ts"])


def target_str(r: dict) -> str:
    if r["target"] == "all":
        return "all"
    if r.get("pods") == "rollout":
        return "rollout"
    return ";".join(r.get("pods") or [])


def label_deploy(df: pd.DataFrame, faults: list[dict]) -> pd.DataFrame:
    af, tp, ids = [], [], []
    for ts in df["ts"].to_numpy():
        act = [r for r in faults if r["start_ts"] <= ts <= r["label_end_ts"]]
        af.append("+".join(r["type"] for r in act) or "none")
        tp.append("|".join(target_str(r) for r in act))
        ids.append("|".join(r["fault_id"] for r in act))
    df["active_fault"], df["fault_target_pod"], df["fault_ids"] = af, tp, ids
    return df


def label_pods(df: pd.DataFrame, faults: list[dict]) -> pd.DataFrame:
    base_hash = rs_hash(df.sort_values("ts")["pod"].iloc[0]) if len(df) else ""
    af, tp, ids, tgt, paf = [], [], [], [], []
    for ts, pod in zip(df["ts"].to_numpy(), df["pod"].to_numpy()):
        act = [r for r in faults if r["start_ts"] <= ts <= r["label_end_ts"]]
        af.append("+".join(r["type"] for r in act) or "none")
        tp.append("|".join(target_str(r) for r in act))
        ids.append("|".join(r["fault_id"] for r in act))
        mine = []
        for r in act:
            if r["target"] == "all" and r["type"] != "pod_kill":
                mine.append(r)
            elif r.get("pods") == "rollout":
                if rs_hash(pod) != base_hash:
                    mine.append(r)
            elif pod in (r.get("pods") or []):
                mine.append(r)
        tgt.append(int(bool(mine)))
        paf.append("+".join(r["type"] for r in mine) or "none")
    df["active_fault"], df["fault_target_pod"], df["fault_ids"] = af, tp, ids
    df["pod_is_fault_target"], df["pod_active_fault"] = tgt, paf
    return df


def main():
    entries = []
    for rnd, sf in STATE_FILES.items():
        if sf.exists():
            entries += [(rnd, sid, st) for sid, st in json.loads(sf.read_text()).items()]
    OUT.mkdir(exist_ok=True)
    deps, pods, runs, fault_rows = [], [], [], []
    for rnd, sid, st in entries:
        if st.get("status") != "ok":
            print("skip", sid, st.get("status"), file=sys.stderr)
            continue
        rd = ROOT / "runs" / st["run_id"]
        sc = json.loads((rd / "scenario.json").read_text())
        meta = sc["_run"]
        t0 = meta["t0"]
        faults = load_faults(rd)
        common = {"run_id": st["run_id"], "scenario": sc["id"], "group": sc.get("group", ""),
                  "policy": sc["scaling"]["policy"], "shape": sc["traffic"]["shape"],
                  "base_rps": sc["traffic"]["base_rps"]}
        d = pd.read_csv(rd / "deploy.csv")
        p = clean_pods(pd.read_csv(rd / "pods.csv"))
        d = add_derived(d, p)
        for df in (d, p):
            for k, v in reversed(list(common.items())):
                df.insert(0, k, v)
            df.insert(len(common), "seconds_since_start", (df["ts"] - t0).round(1))
        d, p = label_deploy(d, faults), label_pods(p, faults)
        d["qa_flag"], p["qa_flag"] = qa_flags(d, faults), qa_flags(p, faults)
        d["round"], p["round"] = rnd, rnd
        deps.append(d)
        pods.append(p)
        for r in faults:
            fault_rows.append({"run_id": st["run_id"], "scenario": sc["id"], "round": rnd, "fault_id": r["fault_id"],
                               "type": r["type"], "target": target_str(r), "start_rel_s": r["start_rel_s"],
                               "end_rel_s": r.get("end_rel_s"),
                               "duration_s": round(r.get("end_ts", r["start_ts"]) - r["start_ts"], 2),
                               "label_window_s": round(r["label_end_ts"] - r["start_ts"], 2),
                               "params": json.dumps(r.get("params", {})),
                               "reapplied": len(r.get("reapplied", []))})
        k6 = {}
        if (rd / "k6_summary.json").exists():
            m = json.loads((rd / "k6_summary.json").read_text())["metrics"]
            k6 = {"k6_reqs": m.get("http_reqs", {}).get("values", {}).get("count"),
                  "k6_failed_rate": m.get("http_req_failed", {}).get("values", {}).get("rate"),
                  "k6_p95_ms": m.get("http_req_duration", {}).get("values", {}).get("p(95)"),
                  "k6_dropped": m.get("dropped_iterations", {}).get("values", {}).get("count", 0)}
        runs.append({**common, "round": rnd, "duration_s": sc["duration_s"], "lane_cap": sc["scaling"].get("_lane_cap"),
                     "scaling": json.dumps({k: v for k, v in sc["scaling"].items()}),
                     "deploy_rows": len(d), "pod_rows": len(p), "faults": len(faults),
                     "max_replicas_ready": float(d["replicas_ready"].max()),
                     "p95_max_s": float(d["p95"].max()), "slo_violation_rows": int(((d["p95"] > 0.3) | (d["error_rate"] > 0.01)).sum()),
                     **k6})
    D = pd.concat(deps, ignore_index=True)
    P = pd.concat(pods, ignore_index=True)
    D.to_csv(OUT / "dataset_deploy.csv.gz", index=False, compression="gzip")
    P.to_csv(OUT / "dataset_pods.csv.gz", index=False, compression="gzip")
    R = pd.DataFrame(runs)
    F = pd.DataFrame(fault_rows)
    R.to_csv(OUT / "runs_index.csv", index=False)
    F.to_csv(OUT / "faults_index.csv", index=False)
    # fault time totals
    ft = {}
    for typ, g in F.groupby("type"):
        ft[typ] = {"count": int(len(g)), "minutes": round(g["duration_s"].sum() / 60, 2),
                   "label_minutes": round(g["label_window_s"].sum() / 60, 2)}
    rows_by_fault = D["active_fault"].value_counts().to_dict()
    summary = {"runs": len(R), "deploy_rows": len(D), "pod_rows": len(P),
               "deploy_columns": list(D.columns), "pod_columns": list(P.columns),
               "fault_totals": ft, "deploy_rows_by_active_fault": rows_by_fault,
               "deploy_rows_by_group": D["group"].value_counts().to_dict(),
               "runs_by_round": R["round"].value_counts().sort_index().to_dict(),
               "deploy_rows_by_round": D["round"].value_counts().sort_index().to_dict(),
               "deploy_rows_by_active_fault_and_round": {f"{k[0]}|r{k[1]}": int(v) for k, v in
                                                         D.groupby(["active_fault", "round"]).size().items()},
               "fault_injections_by_type_and_round": {f"{k[0]}|r{k[1]}": int(v) for k, v in
                                                      F.groupby(["type", "round"]).size().items()},
               "nan_share_deploy": {c: round(float(D[c].isna().mean()), 4) for c in D.columns if D[c].isna().any()},
               "nan_share_pods": {c: round(float(P[c].isna().mean()), 4) for c in P.columns if P[c].isna().any()},
               "deploy_rows_by_qa_flag": D["qa_flag"].value_counts().to_dict(),
               "slo_violation_rows": int(((D["p95"] > 0.3) | (D["error_rate"] > 0.01)).sum())}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps({k: v for k, v in summary.items() if "columns" not in k}, indent=1, default=str))


if __name__ == "__main__":
    main()
