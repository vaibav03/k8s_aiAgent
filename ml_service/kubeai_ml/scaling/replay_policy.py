"""Counterfactual replay of replica policies on the shaped runs (B, C, E).

    python -m kubeai_ml.scaling.replay_policy      # needs train_forecast + train_capacity outputs

Two replays: round 1 with the leave-run-out forecasts (plus the outer conformal correction
the shipped model uses), and the round-2 holdout with forecasts from the round-1 model, where
neither the forecaster, its p90 calibration nor the capacity model saw the run.

This is a model-based replay, not a measurement: the replica count changes, the observed
demand does not. Every 15 s a policy picks a replica count; a scale-up becomes effective
30 s later (pod start + readiness), a scale-down immediately. Each interval is scored
against an oracle that does NOT use the learned model: the smallest N whose per-pod demand
is under the measured SLO step of the capacity runs (last SLO-ok step: 18 / 16 / 14 rps/pod
for 1 / 2 / >=3 pods, at the healthy 26 ms/request), scaled by the run's CPU cost.

Policies:
    learned+forecast  policy.desired_replicas with the leave-run-out LightGBM p90 forecast
    learned reactive  same policy and capacity, forecast replaced by current demand
    hpa70 (sim)       ceil(demand * cpu_per_req / (0.7 * 0.25)), 5 min scale-down window
    fixed18 reactive  ceil(demand / 18), same stabilisation as hpa70
    hpa observed      replicas_ready actually observed in the run (capped at 4 by the lab)
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .capacity import CapacityModel, robust_cpu_per_req
from .data import (CPU_REQUEST_CORES, HOLDOUT_ROUNDS, HPA_TARGET_UTIL, MODELS_DIR, PLOTS_DIR,
                   TRAIN_ROUNDS, load_deploy)
from .policy import Bounds, ScalerMemory, Stabilization, desired_replicas

DECISION_S = 15
READY_DELAY_S = 30
BOUNDS = Bounds(1, 10)
KNEE = {1: 18.0, 2: 16.0, 3: 14.0}          # last SLO-ok step (rps/pod) at 26.5 ms/request
FAULT_TO_STATE = {"none": "healthy", "dep_latency": "dependency_latency", "memory_leak": "memory_leak",
                  "error_rate": "error_fault", "cpu_hog": "noisy_pod", "bad_deploy": "bad_deploy",
                  "crashloop": "crashloop", "pod_kill": "pod_restart"}


def oracle_need(demand: float, cpu: float, nmax: int = 10) -> int:
    for n in range(1, nmax + 1):
        if demand / n <= KNEE.get(n, 14.0) * 0.0265 / cpu:
            return n
    return nmax


def _state(fault: str) -> str:
    first = fault.split("+")[0]
    if first == "dep_latency" or "dep_latency" in fault:
        return "dependency_latency"
    return FAULT_TO_STATE.get(first, "healthy")


def simulate(run: pd.DataFrame, fc: pd.DataFrame, cap: CapacityModel,
             stab: Stabilization = Stabilization()) -> pd.DataFrame:
    cpu = robust_cpu_per_req(run.cpu_cores_used, run.goodput_rps, run.error_rate,
                             run.replicas_ready).bfill().to_numpy()
    t = run.seconds_since_start.to_numpy()
    dur = float(run.duration_s.iloc[0])
    q90 = {h: dict(zip(g.t.round(1), g.q90)) for h, g in fc.groupby("h")}
    idx = [i for i in range(len(run)) if t[i] >= 60 and t[i] <= dur and i % 3 == 0]
    start = int(run.replicas_ready.iloc[idx[0]])
    hpa_stab = Stabilization(max_up_step=4, down_window_s=300, down_cooldown_s=15, max_down_step=10)
    names = ["learned+forecast", "learned reactive", "hpa70 (sim)", "fixed18 reactive"]
    mem = {n: ScalerMemory() for n in names}
    cur = {n: start for n in names}
    pending = {n: [] for n in names}               # (effective_ts, replicas)
    eff = {n: start for n in names}
    out = []
    for i in idx:
        now = float(run.rps.iloc[i])
        key = round(t[i], 1)
        fq = {h: float(q90[h].get(key, np.nan)) for h in q90}
        fq = {h: v for h, v in fq.items() if not math.isnan(v)}
        state = _state(run.active_fault.iloc[i])
        for n in names:
            if n.startswith("learned"):
                d = desired_replicas(fq if n == "learned+forecast" else {}, cap, cur[n], BOUNDS, stab,
                                     state=state, now_rps=now, cpu_per_req_s=cpu[i],
                                     baseline_cpu_per_req_s=0.0265, ts=t[i], memory=mem[n]).replicas
            else:
                raw = (math.ceil(now * cpu[i] / (HPA_TARGET_UTIL * CPU_REQUEST_CORES)) if n.startswith("hpa")
                       else math.ceil(now / 18.0))
                raw = max(1, min(BOUNDS.max_replicas, raw))
                m = mem[n]
                m.recs.append((t[i], raw))
                while m.recs[0][0] < t[i] - hpa_stab.down_window_s:
                    m.recs.popleft()
                d = raw if raw >= cur[n] else max(r for _, r in m.recs)
            if d > cur[n]:
                pending[n].append((t[i] + READY_DELAY_S, d))
            elif d < cur[n]:
                pending[n] = [p for p in pending[n] if p[1] <= d]
                eff[n] = min(eff[n], d)
            cur[n] = d
        for n in names:
            ready = [p for p in pending[n] if p[0] <= t[i]]
            if ready:
                eff[n] = max(eff[n], max(r for _, r in ready))
                pending[n] = [p for p in pending[n] if p[0] > t[i]]
        # demand over the next decision interval (the interval this decision must cover)
        nxt = run.rps.iloc[i:i + 3].max()
        need = oracle_need(nxt, cpu[i])
        rec = {"scenario": run.scenario.iloc[0], "t": t[i], "demand": nxt, "oracle": need,
               "hpa observed": int(run.replicas_ready.iloc[i])}
        rec.update({n: eff[n] for n in names})
        out.append(rec)
    return pd.DataFrame(out)


POLICIES = ["learned+forecast", "learned reactive", "hpa70 (sim)", "fixed18 reactive", "hpa observed"]


def run_all(df: pd.DataFrame, fc: pd.DataFrame, cap: CapacityModel, stab: Stabilization) -> pd.DataFrame:
    sims = []
    for run_id, run in df[df.group.isin(["B", "C", "E"])].groupby("run_id"):
        sims.append(simulate(run.reset_index(drop=True), fc[fc.run_id == run_id], cap, stab))
    return pd.concat(sims, ignore_index=True)


def score(S: pd.DataFrame, pols: list[str]) -> pd.DataFrame:
    rows = []
    for p in pols:
        under = S[p] < S.oracle
        rows.append({"policy": p, "under_provisioned_pct": 100 * under.mean(),
                     "pod_deficit_sum": (S.oracle - S[p]).clip(lower=0).sum(),
                     "mean_replicas": S[p].mean(), "over_vs_oracle_mean": (S[p] - S.oracle).clip(lower=0).mean(),
                     "replica_changes": int((S.groupby("scenario")[p].diff().fillna(0) != 0).sum())})
    return pd.DataFrame(rows)


def sweep(df: pd.DataFrame, fc: pd.DataFrame, cap: CapacityModel) -> pd.DataFrame:
    """Sensitivity of the learned policy to risk, scale-up horizons and scale-down headroom."""
    from dataclasses import replace
    rows = []
    for risk in (0.05, 0.10, 0.20):
        for ups in ((60,), (60, 120)):
            for downs in ((60,), (60, 120), (60, 120, 300)):
                c = replace(cap, risk=risk)
                st = Stabilization(up_horizons_s=ups, down_horizons_s=downs)
                S = run_all(df, fc, c, st)
                r = score(S, ["learned+forecast", "learned reactive"])
                for _, x in r.iterrows():
                    if x["policy"] == "learned reactive" and (ups, downs) != ((60,), (60,)):
                        continue
                    rows.append({"risk": risk, "up_h": "+".join(map(str, ups)),
                                 "down_h": "+".join(map(str, downs)), **x.to_dict()})
    return pd.DataFrame(rows)


def load_forecasts(kind: str) -> pd.DataFrame:
    """LightGBM forecasts: 'loro' (round-1 leave-run-out, outer correction applied) or 'holdout'."""
    import json
    fdir = MODELS_DIR / "forecast"
    if kind == "holdout":
        fc = pd.read_csv(fdir / "holdout_predictions_rps.csv.gz")
        return fc[fc.method == "lgbm_quantile"]
    fc = pd.read_csv(fdir / "loro_predictions_rps.csv.gz")
    fc = fc[fc.method == "lgbm_quantile"].copy()
    extra = json.loads((fdir / "loro_metrics_rps.json").read_text()).get("q90_outer_correction", {})
    for h, v in extra.items():
        sel = fc.h == int(h)
        fc.loc[sel, "q90"] = np.expm1(np.log1p(fc.loc[sel, "q90"]) + v)
    return fc


def report(S: pd.DataFrame, title: str) -> pd.DataFrame:
    R = score(S, POLICIES)
    print(f"== {title}: {S.scenario.nunique()} runs, {len(S)} decisions, oracle mean {S.oracle.mean():.2f} pods")
    print(R.round(3).to_string(index=False))
    per = S.groupby("scenario").apply(lambda g: pd.Series({p: 100 * (g[p] < g.oracle).mean() for p in POLICIES}),
                                      include_groups=False)
    print(per.round(1).to_string())
    return R


def main() -> None:
    import sys
    pd.set_option("display.width", 200)
    cap = CapacityModel.load(MODELS_DIR / "capacity.json")
    df = load_deploy(TRAIN_ROUNDS)
    fc = load_forecasts("loro")
    if "--sweep" in sys.argv:
        W = sweep(df, fc, cap)
        print(W.round(2).to_string(index=False))
        W.to_csv(MODELS_DIR / "policy_sweep.csv", index=False)
        return
    S = run_all(df, fc, cap, Stabilization())
    R = report(S, f"rounds {TRAIN_ROUNDS}, leave-run-out forecasts")
    R.to_csv(MODELS_DIR / "policy_replay.csv", index=False)
    _plot(S, ["b-hpa-spike-b08", "b-hpa-saw-b10", "c-under-fixed2-spike", "b-hpa-diurnal-s7"], "policy_replay.png")
    dh = load_deploy(HOLDOUT_ROUNDS)
    if dh.group.isin(["B", "C", "E"]).any():
        SH = run_all(dh, load_forecasts("holdout"), cap, Stabilization())
        RH = report(SH, f"holdout rounds {HOLDOUT_ROUNDS}, round-1 models")
        RH.to_csv(MODELS_DIR / "policy_replay_holdout.csv", index=False)
        _plot(SH, sorted(SH.scenario.unique())[:4], "policy_replay_holdout.png")


def _plot(S: pd.DataFrame, scen: list[str], fname: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(scen), 1, figsize=(10, 2.6 * len(scen)))
    for ax, sc in zip(axes, scen):
        g = S[S.scenario == sc]
        ax2 = ax.twinx()
        ax2.plot(g.t, g.demand, color="k", lw=0.8, alpha=0.4)
        ax2.set_ylabel("demand rps", fontsize=8)
        ax.step(g.t, g.oracle, where="post", color="k", lw=1.6, label="oracle need")
        for p, c in (("learned+forecast", "tab:blue"), ("learned reactive", "tab:cyan"),
                     ("hpa70 (sim)", "tab:purple"), ("hpa observed", "tab:gray")):
            ax.step(g.t, g[p], where="post", color=c, lw=1, label=p)
        ax.set_title(sc, fontsize=9)
        ax.set_ylabel("effective replicas", fontsize=8)
    axes[0].legend(fontsize=7, ncol=5)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / fname, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
