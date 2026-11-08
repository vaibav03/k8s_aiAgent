"""What the engine would have done during each recorded fault, next to what the HPA actually did.

Open-loop replay on the recorded data: every 15 s the engine sees the deployment and pod rows
up to that moment (as live Prometheus would give them) and the replica count the lab actually
had; its recommendation is recorded, not applied. Inside each fault window we report

* engine: the dominant state, the share of decisions that recommend adding pods (useless for a
  non-load fault), the share that block scaling (``none`` action with ``scaling: blocked``);
* HPA (actual data, ``hpa_cpu`` runs): scale-up events during the fault, the peak replica count
  vs the count at fault start, the extra pod-minutes, and the SLO-violation minutes before and
  after its first scale-up (did adding pods help?).
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from . import states as S
from .engine import DecisionEngine

PERIOD_S = 15.0
HISTORY_S = 1800.0


def _spec(runs_index: pd.DataFrame, run_id: str) -> dict[str, Any]:
    r = runs_index[runs_index["run_id"] == run_id].iloc[0]
    sc = json.loads(r["scaling"]) if isinstance(r["scaling"], str) else {}
    # the engine may recommend anything in [1, lab cap], also on fixed-size runs
    return {"policy": r["policy"], "min": 1, "max": int(sc.get("max", sc.get("_lane_cap", 4)) or 4)}


def replay_run(engine: DecisionEngine, deploy: pd.DataFrame, pods: pd.DataFrame | None,
               min_r: int, max_r: int) -> pd.DataFrame:
    """Engine decisions every 15 s over one recorded run (open loop)."""
    engine.reset()
    d = deploy.sort_values("ts").reset_index(drop=True)
    t0 = float(d["ts"].iloc[0])
    out = []
    for t in np.arange(t0 + 60.0, float(d["ts"].iloc[-1]) + 1e-6, PERIOD_S):
        h = d[(d["ts"] <= t) & (d["ts"] > t - HISTORY_S)]
        p = pods[(pods["ts"] <= t) & (pods["ts"] > t - HISTORY_S)] if pods is not None else None
        cur = int(h["replicas_desired"].iloc[-1])
        r = engine.decide(h, p, cur, min_r, max(max_r, cur))
        out.append({"ts": t, "secs": float(h["seconds_since_start"].iloc[-1]), "current": cur,
                    "state": r["state"], "confidence": r["confidence"], "desired": r["desired_replicas"],
                    "blocked": any(a["type"] == "none" and a["params"].get("scaling") == "blocked"
                                   for a in r["actions"]),
                    "action_types": "+".join(sorted({a["type"] for a in r["actions"]})),
                    "restart_targets": "+".join(str(a["target"]) for a in r["actions"] if a["type"] == "restart_pod"),
                    "rollback": any(a["type"] == "rollback" for a in r["actions"])})
    return pd.DataFrame(out)


def _viol(g: pd.DataFrame) -> np.ndarray:
    return ((g["p95"].fillna(0) >= 0.3) | (g["error_rate"].fillna(0) >= 0.01)).to_numpy()


def fault_table(engine: DecisionEngine, deploy: pd.DataFrame, pods: pd.DataFrame, faults: pd.DataFrame,
                runs_index: pd.DataFrame, holdout: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, decisions = [], []
    pods_by_run = dict(tuple(pods.groupby("run_id")))
    for run_id, fr in faults.groupby("run_id"):
        g = deploy[deploy["run_id"] == run_id].sort_values("ts")
        if g.empty:
            continue
        sp = _spec(runs_index, run_id)
        dec = replay_run(engine, g, pods_by_run.get(run_id), sp["min"], sp["max"])
        dec["run_id"] = run_id
        decisions.append(dec)
        for f in fr.itertuples():
            end = f.start_rel_s + max(float(f.label_window_s), 30.0)
            w = dec[(dec["secs"] >= f.start_rel_s) & (dec["secs"] <= end)]
            gw = g[(g["seconds_since_start"] >= f.start_rel_s) & (g["seconds_since_start"] <= end)]
            want = S.FAULT_TO_STATE[f.type]
            r0 = float(g[g["seconds_since_start"] <= f.start_rel_s]["replicas_desired"].iloc[-1]) \
                if (g["seconds_since_start"] <= f.start_rel_s).any() else float(gw["replicas_desired"].iloc[0])
            rep = gw["replicas_desired"].to_numpy(float)
            ups = int(np.sum(np.diff(np.r_[r0, rep]) > 0))
            first_up = gw[gw["replicas_desired"] > r0]["seconds_since_start"]
            v = _viol(gw)
            if len(first_up):
                before = gw["seconds_since_start"] < first_up.iloc[0]
                viol_before = float(v[before.to_numpy()].mean()) if before.any() else float("nan")
                viol_after = float(v[~before.to_numpy()].mean())
            else:
                viol_before, viol_after = float(v.mean()) if len(v) else float("nan"), float("nan")
            rows.append({
                "run": g["scenario"].iloc[0], "split": "holdout" if g["scenario"].iloc[0] in holdout else "dev",
                "fault": f.type, "expected_state": want, "lab_policy": sp["policy"],
                "decisions": len(w),
                "engine_top_state": w["state"].mode().iloc[0] if len(w) else None,
                "engine_state_recall": round(float((w["state"] == want).mean()), 3) if len(w) else None,
                "engine_upscale_pct": round(100 * float((w["desired"] > w["current"]).mean()), 1) if len(w) else None,
                "engine_max_extra_pods": int((w["desired"] - w["current"]).clip(lower=0).max()) if len(w) else 0,
                "engine_blocked_pct": round(100 * float(w["blocked"].mean()), 1) if len(w) else None,
                "engine_restart_actions": int((w["restart_targets"] != "").sum()) if len(w) else 0,
                "engine_restart_on_target_pct": (round(100 * float(w.loc[w["restart_targets"] != "", "restart_targets"]
                                                                    .str.contains(str(f.target), regex=False).mean()), 1)
                                                 if len(w) and (w["restart_targets"] != "").any()
                                                 and isinstance(f.target, str) and f.target else None),
                "engine_rollback_pct": round(100 * float(w["rollback"].mean()), 1) if len(w) else None,
                "actual_replicas_at_start": r0,
                "actual_peak_replicas": float(rep.max()) if len(rep) else r0,
                "actual_scale_ups": ups,
                "actual_extra_pod_min": round(float(np.clip(rep - r0, 0, None).sum()) * 5 / 60, 2),
                "slo_viol_min": round(float(v.sum()) * 5 / 60, 2),
                "viol_frac_before_first_upscale": None if np.isnan(viol_before) else round(viol_before, 3),
                "viol_frac_after_first_upscale": None if np.isnan(viol_after) else round(viol_after, 3),
            })
    return pd.DataFrame(rows), pd.concat(decisions, ignore_index=True) if decisions else pd.DataFrame()
