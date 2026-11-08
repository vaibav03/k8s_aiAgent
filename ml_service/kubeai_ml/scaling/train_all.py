"""One reproducible pipeline: evaluate, train and ship the scaling bundle (rounds 1 + 2).

    cd ml_service && PYTHONPATH=. python -m kubeai_ml.scaling.train_all [--jobs 6] [--no-sim]

Protocol (``split.py``): 11 whole runs are held out (round-2 runs whose class has a round-1
twin) and never used to fit, select or tune anything. On the 33 development runs:

1. labels: ground-truth faults + capacity-derived load states (knee learned on dev runs);
2. state classifier: leave-one-run-out (GroupKFold, one run per fold; capacity re-learned in
   every fold) for rules / hybrid / LightGBM / two-stage plus HPA-rule and majority baselines;
   rules vs hybrid picked by settled macro-F1 (ties -> rules); calibration P(true | predicted)
   from the out-of-fold predictions; final fit on all dev rows;
3. forecaster: leave-one-run-out over the traffic-shaped dev runs (training = all other dev
   runs) for persistence / linear trend / damped Holt / LightGBM quantile; outer conformal p90
   correction from those residuals; final fit on all dev runs;
4. capacity: leave-one-run-out logistic vs load-only / monotone LightGBM / throttle nowcast, and
   as a rule vs fixed 18 rps/pod and HPA 70 %; final fit on all dev rows;
5. simulator: pod model and metrics-server timing fitted on dev runs only; policy parameters
   (risk, scale-up horizons) tuned on dev traces only;
6. held-out evaluation of every component and of the closed-loop policies, and an open-loop
   replay of the engine on every recorded fault run next to what the HPA actually did.

Writes ``models/scaling/<version>/`` (see ``bundle.py``) and ``models/scaling/LATEST``.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import forecast as F
from . import states as S
from . import train_capacity as TC
from . import train_classifier as TCL
from . import train_forecast as TF
from .bundle import LATEST, PolicyParams, write_json
from .capacity import CapacityModel, design_matrix
from .classifier import MODEL_FEATURES, MODES, StateClassifier
from .data import LAB_DATA, ML_ROOT, MODELS_DIR, load_deploy
from .features import FEATURES, build_features
from .labels import SAFE_BREACH_RATE, TARGET_UTIL, learn_safe_capacity, make_labels
from .split import HOLDOUT_SCENARIOS, SHAPES_WITH_DYNAMICS, data_provenance, is_holdout, is_shaped

CAL_ALPHA = 0.5          # Laplace smoothing of the state calibration rows


# ---------------------------------------------------------------------------------- data
def load_labelled() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Any]:
    d = pd.read_csv(LAB_DATA / "dataset_deploy.csv.gz")
    d["active_fault"] = d["active_fault"].fillna("none")
    d = d.sort_values(["run_id", "ts"]).reset_index(drop=True)
    missing = set(HOLDOUT_SCENARIOS) - set(d["scenario"])
    if missing:
        raise SystemExit(f"held-out runs missing from the dataset: {sorted(missing)}")
    pods = pd.read_csv(LAB_DATA / "dataset_pods.csv.gz")
    faults = pd.read_csv(LAB_DATA / "faults_index.csv")
    by_run = dict(tuple(pods.groupby("run_id")))
    feats = pd.concat([build_features(g, by_run.get(r)) for r, g in d.groupby("run_id")]).loc[d.index]
    hold = is_holdout(d["scenario"])
    cap_lab = learn_safe_capacity(d[~hold])           # the label knee never sees held-out rows
    lab = make_labels(d, cap=cap_lab, pods=pods, faults=faults)
    d["state"], d["use"], d["residual_hog"] = lab["state"], lab["use"], lab["residual_hog"]
    d["settled"] = TCL._settled(d)
    return d, feats, faults, pods, cap_lab


# ---------------------------------------------------------------------------------- classifier
def hpa_rule_states(f: pd.DataFrame, target: float = 0.70, tol: float = 0.10) -> np.ndarray:
    """Baseline: what a CPU-70 % HPA 'thinks' (scale up/down/hold), blind to faults."""
    ratio = f["cpu_util_req"].fillna(0).to_numpy(float) / target
    n = np.maximum(f["replicas_ready"].fillna(1).to_numpy(float), 1)
    want = np.ceil(ratio * n)
    up = (ratio > 1 + tol) & (want > n)
    down = (ratio < 1 - tol) & (want < n)
    return np.where(up, S.SCALE_UP, np.where(down, S.SCALE_DOWN, S.HEALTHY)).astype(object)


def calibration(y: np.ndarray, p: np.ndarray) -> dict[str, dict[str, float]]:
    """P(true state | predicted state) with Laplace smoothing over the states seen as truth."""
    ct = pd.crosstab(pd.Series(p, name="pred"), pd.Series(y, name="true"))
    truths = list(ct.columns)
    out: dict[str, dict[str, float]] = {}
    for s in S.STATES:
        if s in ct.index:
            row = ct.loc[s].astype(float) + CAL_ALPHA
            out[s] = {t: round(float(v / row.sum()), 4) for t, v in row.items()}
        else:                                           # never predicted in CV: uninformative
            out[s] = {s: 0.5, **{t: round(0.5 / max(1, len(truths) - 1), 4) for t in truths if t != s}}
    return out


def blind_holdout(hd: pd.DataFrame, fh: pd.DataFrame, fd: pd.DataFrame, dev: pd.DataFrame, cap: Any,
                  chosen: str, faults: pd.DataFrame) -> dict[str, Any]:
    """Held-out score with the two decisions that were made while held-out runs were visible
    undone (review 3): the error_fault ``error_rate`` floor (rules.ERROR_RATE_FLOOR) and the
    residual cpu_hog mask on held-out rows. This is the blind figure; the shipped figure is
    post-hoc for error_fault and noisy_pod."""
    from . import rules
    out: dict[str, Any] = {}
    use = dev["use"].to_numpy(bool)
    variants = {"no_error_floor": (0.0, False), "hog_rows_kept": (rules.ERROR_RATE_FLOOR, True),
                "blind_both": (0.0, True)}
    saved = rules.ERROR_RATE_FLOOR
    try:
        for name, (floor, keep_hog) in variants.items():
            rules.ERROR_RATE_FLOOR = floor
            clf = StateClassifier(mode=chosen, cap=cap).fit(fd[use], dev.loc[use, "state"].to_numpy())
            h = hd.copy()
            if keep_hog:
                h["use"] = h["use"] | (h["residual_hog"] & (h["qa_flag"] == "ok"))
            p = pd.DataFrame({chosen: clf.predict(fh)}, index=h.index, dtype=object)
            r = TCL.evaluate(h, p, faults)[chosen]
            out[name] = {k: r[k] for k in ("macro_f1_strict", "macro_f1_settled", "accuracy_settled",
                                           "per_class_settled", "classes_settled")}
            out[name]["rows_scored_settled"] = int((h["use"] & h["settled"]).sum())
    finally:
        rules.ERROR_RATE_FLOOR = saved
    return out


def classifier_stage(d: pd.DataFrame, feats: pd.DataFrame, faults: pd.DataFrame, pods: pd.DataFrame,
                     out: Path) -> dict[str, Any]:
    hold = is_holdout(d["scenario"])
    dev, fd = d[~hold], feats[~hold]
    t0 = time.time()

    def relabel(cap: Any) -> pd.Series:
        return make_labels(dev, cap=cap, pods=pods, faults=faults)["state"]

    preds, y_fold = TCL.cross_validate(dev, fd, relabel=relabel)
    dev_loro = dev.assign(state=y_fold)
    dev_loro["settled"] = TCL._settled(dev_loro)
    preds["hpa_cpu70_rule"] = hpa_rule_states(fd)
    majority = dev.loc[dev["use"], "state"].mode().iloc[0]
    preds["majority"] = majority
    res = TCL.evaluate(dev_loro, preds, faults)
    chosen = max(TCL.CANDIDATES, key=lambda m: (res[m]["macro_f1_settled"], m == "rules"))
    print(f"[classifier] LORO {dev['run_id'].nunique()} folds in {time.time() - t0:.0f}s; chosen {chosen}")

    use = dev["use"].to_numpy(bool)
    cap = learn_safe_capacity(dev)
    hd, fh = d[hold], feats[hold]
    hpreds = pd.DataFrame(index=hd.index, dtype=object)
    shipped = None
    for mode in MODES:
        clf = StateClassifier(mode=mode, cap=cap).fit(fd[use], dev.loc[use, "state"].to_numpy())
        hpreds[mode] = clf.predict(fh)
        if mode == chosen:
            shipped = clf
    hpreds["hpa_cpu70_rule"] = hpa_rule_states(fh)
    hpreds["majority"] = majority
    hres = TCL.evaluate(hd, hpreds, faults)
    cal = calibration(dev_loro.loc[use, "state"].to_numpy(), preds.loc[dev.index[use], chosen].to_numpy())
    blind = blind_holdout(hd, fh, fd, dev, cap, chosen, faults)
    print(f"[classifier] held-out settled macro-F1 shipped {hres[chosen]['macro_f1_settled']}, "
          f"blind {blind['blind_both']['macro_f1_settled']}")
    assert shipped is not None
    shipped.save(out / "classifier", {"trained_on_rows": int(use.sum()), "runs": int(dev["run_id"].nunique()),
                                      "loro_macro_f1_settled": res[chosen]["macro_f1_settled"],
                                      "holdout_macro_f1_settled": hres[chosen]["macro_f1_settled"]})
    write_json(out / "classifier" / "calibration.json", cal)
    return {"chosen": chosen, "capacity_knees": cap.to_dict(),
            "label_counts_dev": dev.loc[dev["use"], "state"].value_counts().to_dict(),
            "label_counts_holdout": hd.loc[hd["use"], "state"].value_counts().to_dict(),
            "masked_rows": int((~d["use"]).sum()),
            "loro_label_changes_vs_global_knee": int((y_fold != dev["state"]).sum()),
            "residual_hog_rows": {"dev": int(dev["residual_hog"].sum()), "holdout": int(hd["residual_hog"].sum())},
            "loro": res, "holdout": hres, "holdout_blind": blind}


# ---------------------------------------------------------------------------------- forecaster
def forecast_stage(out: Path, n_jobs: int) -> dict[str, Any]:
    from joblib import Parallel, delayed
    t0 = time.time()
    df = load_deploy()
    shape = pd.read_csv(LAB_DATA / "runs_index.csv").set_index("run_id")["shape"]
    o = TF.build_origins(df, "rps")
    o["shape"] = o["run_id"].map(shape)
    hold = is_holdout(o["scenario"])
    o_dev, o_hold = o[~hold], o[hold & is_shaped(o["shape"])]
    holt = TF.holt_centers(o)
    test_runs = o_dev.loc[is_shaped(o_dev["shape"]), "run_id"].unique()
    parts = Parallel(n_jobs=n_jobs)(delayed(TF.fit_predict)(o_dev[o_dev["run_id"] != r], o_dev[o_dev["run_id"] == r], holt)
                                    for r in test_runs)
    P_all = pd.concat(parts, ignore_index=True)
    P = P_all[P_all["y"].notna()]
    extra = TF.outer_q90_correction(P_all)
    model = TF.fit_lgbm_with_margin(o_dev[o_dev["t"] >= 30])
    for h, v in extra.items():
        model.q90_margin[h] += v
    model.save(out / "forecast", info={"runs": int(o_dev["run_id"].nunique()), "holdout_excluded": True,
                                       "q90_outer_correction": extra, "target": "rps (demand)"})
    H_all = TF.fit_predict(o_dev, o_hold, holt, q90_extra=extra)
    te = o_hold[o_hold["eval"]]                       # score the shipped booster itself
    ph = model.predict(TF._Z(te))
    for h in F.HORIZONS_S:
        m = (H_all["method"] == "lgbm_quantile") & (H_all["h"] == h)
        H_all.loc[m, "q50"], H_all.loc[m, "q90"] = ph[(h, 0.5)], ph[(h, 0.9)]
    H = H_all[H_all["y"].notna()]
    print(f"[forecast] LORO over {len(test_runs)} shaped dev runs + holdout in {time.time() - t0:.0f}s")
    return {"target": "rps", "loro_runs": sorted(map(str, test_runs)),
            "holdout_runs": sorted(o_hold["scenario"].unique()),
            "loro": TF.summarize(P).round(4).to_dict("records"),
            "loro_by_group": TF.summarize_groups(P).round(4).to_dict("records"),
            "holdout": TF.summarize(H).round(4).to_dict("records"),
            "q90_outer_correction": extra, "q90_margin_total": model.q90_margin,
            "flat_traffic_probe": TF.flat_probe(model)}


# ---------------------------------------------------------------------------------- capacity
def capacity_stage(out: Path) -> tuple[dict[str, Any], CapacityModel, float]:
    df = load_deploy()
    hold = is_holdout(df["scenario"])
    m = TC.training_rows(df[~hold])
    P = TC.loro(m)
    mh = TC.training_rows(df[hold])
    H = TC.fit_predict(m, mh)
    lr = TC._fit_logit(design_matrix(m.u.to_numpy(), m.n.to_numpy()), m.breach)
    model = CapacityModel(coef=lr.coef_[0].round(6).tolist(), intercept=round(float(lr.intercept_[0]), 6),
                          n_max=float(m.n.max()), risk=TC.RISK)
    model.save(out / "capacity.json")
    safe = [{"replicas": n, **{f"safe_rps_per_pod@{int(c * 1000)}ms": round(model.safe_rps_per_pod(c, n), 2)
                               for c in (0.0265, 0.05, 0.078)}} for n in range(1, 7)]
    cpu_default = float(m.loc[m.active_fault == "none", "cpu_per_req_robust"].median())
    print(f"[capacity] {len(m)} dev rows ({m.run_id.nunique()} runs), {len(mh)} held-out rows")
    return ({"train_rows": int(len(m)), "train_runs": int(m.run_id.nunique()),
             "breach_rate": round(float(m.breach.mean()), 4), "holdout_rows": int(len(mh)),
             "coef": model.coef, "intercept": model.intercept, "n_max": model.n_max,
             "loro_models": TC.score(P).round(4).to_dict("records"),
             "loro_rules": TC.rule_table(P).round(4).to_dict("records"),
             "holdout_models": TC.score(H).round(4).to_dict("records"),
             "holdout_rules": TC.rule_table(H).round(4).to_dict("records"),
             "breach_rows_by_run": {"dev": TC.breach_by_run(P), "holdout": TC.breach_by_run(H)},
             "safe_table": safe, "healthy_cpu_per_req_s": round(cpu_default, 5)}, model, cpu_default)


# ---------------------------------------------------------------------------------- specs
def feature_spec() -> dict[str, Any]:
    from .engine import OPTIONAL_DEPLOY, POD_FIELDS, REQUIRED_DEPLOY
    return {"engine_input": {"deployment_obs": {"required": list(REQUIRED_DEPLOY), "optional": list(OPTIONAL_DEPLOY)},
                             "pod_obs": list(POD_FIELDS), "cadence_s": "5-15", "min_history_s": 180,
                             "recommended_history_s": 1800,
                             "promql": "loadlab/runner/promexport.py deploy_queries / pod_queries "
                                       "(documented field by field in kubeai_ml/scaling/engine.py)"},
            "state_features": MODEL_FEATURES, "state_base_features": FEATURES,
            "forecast": {"features": F.FEATURE_NAMES, "target": "log1p(rps) delta", "step_s": F.STEP_S,
                         "history_steps": F.HIST_STEPS, "horizons_s": list(F.HORIZONS_S), "quantiles": list(F.QUANTILES)},
            "capacity": {"features": ["u", "u^2", "ln N", "u ln N"],
                         "u": "rps_per_pod * cpu_s_per_successful_request / cpu_limit_cores"}}


def label_rules(cap_lab: Any) -> dict[str, Any]:
    return {
        "slo": {"p95_s": 0.300, "error_rate": 0.01},
        "fault_rows": "active_fault -> state (states.FAULT_TO_STATE); overlaps by states.FAULT_PRIORITY",
        "label_capacity_knee_rps_per_pod": cap_lab.to_dict(),
        "scale_up": "fault-free and ((SLO breached and (demand/pod >= 0.85 c(r) or throttle >= 0.5)) "
                    "or peak demand over (t, t+60s] / ready >= c(r))",
        "scale_down": f"fault-free, SLO ok, ready >= 2, peak demand over [t-60s, t+180s] <= {TARGET_UTIL} * "
                      "c(r-1) * (r-1), sustained over the trailing 60 s",
        "healthy": "otherwise",
        "masked": "qa_flag != ok; cpu_hog that outlived its logged end (labels.residual_hog_mask)",
        "demand": "goodput_rps, or rps while p95 >= SLO (overload 504s are offered load)",
        "knee_rule": f"decision stump per replica count on fault-free ok rows (SAFE_BREACH_RATE {SAFE_BREACH_RATE})",
        "fault_gates_in_priority_order": {
            "crashloop": "crashloop_pods >= 1, or (pods_not_ready >= 1 and restarts_delta_3m >= 2)",
            "memory_leak": "(mem_growth_med_1m >= 50 KB/s and no backlog (inflight/pod < 10, throttle_1m < 0.8) "
                           "and not noisy) or oomkills_delta_3m >= 1",
            "pod_restart": "ready_drop_1m >= 1 and ready_deficit >= 1",
            "bad_deploy": "cpr_ratio >= 1.8 and (with pods: rollout < 900 s ago and pod_cpr_disp_1m < 1.6; "
                          "without pods: not overloaded)",
            "noisy_pod": "pod_cpr_disp_1m >= 2.0 and pod_cpu_disp_1m >= 0.15 cores and no rollout < 900 s",
            "error_fault": "fast_error_share >= 0.01 and error_rate >= 0.005",
            "dependency_latency": "p95_slo_1m >= 1 and p95_slo_min_30s >= 1 and throttle_1m < 0.3 and "
                                  "cpu_frac_limit_1m < 0.6"},
        "settled_scoring": f"drop rows within {TCL.SETTLE_S:.0f} s after a label change",
    }


# ---------------------------------------------------------------------------------- simulator
def sim_stage(bundle: Path, params: PolicyParams, n_jobs: int) -> tuple[dict[str, Any], PolicyParams]:
    from . import eval_policies as EP
    from . import policy_eval as PE
    from .podmodel import fit_pod_model
    from .traces import run_trace

    t0 = time.time()
    df = load_deploy()
    hold = is_holdout(df["scenario"])
    pm, fit_report = fit_pod_model(df[~hold])
    runs_dev = EP.b_runs(df[~hold])
    _, cfg, grid = EP.fit_ms_timing(runs_dev, pm)   # lag 0 is the grid floor: see ms_timing.boundary_fit
    runs_hold = EP.b_runs(df[hold])
    val = pd.DataFrame([{k: v for k, v in EP.validate_hpa_run(r, pm, cfg).items() if k != "_res"}
                        for r in runs_dev + runs_hold])
    val["split"] = np.where(val["scenario"].isin(HOLDOUT_SCENARIOS), "holdout", "dev")
    write_json(bundle / "sim_calibration.json", {"pod_model": json.loads(pm.to_json()), "ms_lag_s": cfg.ms_lag_s,
                                                 "ms_window_s": cfg.ms_window_s, "fit_on": "development runs only",
                                                 "fit_report": fit_report})
    print(f"[sim] calibrated in {time.time() - t0:.0f}s: lag {cfg.ms_lag_s} window {cfg.ms_window_s}")

    shape = pd.read_csv(LAB_DATA / "runs_index.csv").set_index("run_id")["shape"]
    traces: dict[str, list] = {"dev": [], "holdout": []}
    for rid, g in df.groupby("run_id"):
        if shape[rid] in SHAPES_WITH_DYNAMICS:
            traces["holdout" if g["scenario"].iloc[0] in HOLDOUT_SCENARIOS else "dev"].append(run_trace(rid, g))
    syn = PE.synthetic_traces()

    sw, chosen = PE.sweep(traces["dev"] + syn, pm, cfg, str(bundle), params, n_jobs=n_jobs)
    print(f"[sim] policy sweep (dev traces) in {time.time() - t0:.0f}s -> risk {chosen.risk} "
          f"up {chosen.stabilization.up_horizons_s} headroom {chosen.stabilization.up_headroom} "
          f"window {chosen.stabilization.down_window_s}")
    warm = [PE.warm_start(t) for t in traces["holdout"]]
    cmp = pd.concat([PE.compare(traces["dev"], pm, cfg, str(bundle), chosen, split="dev", n_jobs=n_jobs),
                     PE.compare(syn, pm, cfg, str(bundle), chosen, split="synthetic", n_jobs=n_jobs),
                     PE.compare(traces["holdout"], pm, cfg, str(bundle), chosen, split="holdout", n_jobs=n_jobs),
                     PE.compare(warm, pm, cfg, str(bundle), chosen, split="holdout_warm", n_jobs=n_jobs)])
    print(f"[sim] comparison done in {time.time() - t0:.0f}s")
    agg = lambda v: {k: (round(float(v[k].mean()), 3) if k != "viol_min_actual" else None)  # noqa: E731
                     for k in ("rep_mae_1s", "rep_exact_rows_pct", "rescale_timing_err_s", "rescale_timing_signed_s",
                               "p95_log_mae")}
    by_round = {f"r{rd}": grid.sort_values([f"r{rd}_rep_mae", "ms_lag_s"]).iloc[0][["ms_lag_s", "ms_window_s"]].to_dict()
                for rd in (1, 2) if grid[f"r{rd}_rep_mae"].notna().any()}
    out = {"ms_timing": {"lag_s": cfg.ms_lag_s, "window_s": cfg.ms_window_s, "lag_grid": list(EP.LAG_GRID),
                         "boundary_fit": cfg.ms_lag_s in (min(EP.LAG_GRID), max(EP.LAG_GRID)),
                         "best_by_round": by_round},
           "pod_model": {"cpu_s_per_req": pm.cpu_s_per_req, "mu_by_n": pm.mu_by_n, "startup_delay_s": pm.startup_delay_s,
                         "collapse_goodput_frac": pm.collapse_goodput_frac},
           "hpa_replay_validation": {
               s: {**agg(v), "viol_min_actual": round(float(v["viol_min_actual"].sum()), 2),
                   "viol_min_sim": round(float(v["viol_min_sim"].sum()), 2), "runs": len(v),
                   "per_run": v[["scenario", "rescales_actual", "rescales_sim", "rescale_timing_signed_s",
                                 "viol_min_actual", "viol_min_sim"]].round(3).to_dict("records")}
               for s, v in val.groupby("split")},
           "viol_min_actual_over_sim": round(float(val["viol_min_actual"].sum() / max(val["viol_min_sim"].sum(), 1e-9)), 3),
           "policy_sweep_dev": sw.to_dict("records"),
           "traces": {k: [t.name for t in v] for k, v in traces.items()} | {"synthetic": [t.name for t in syn],
                                                                             "holdout_warm": [t.name for t in warm]},
           "per_trace": cmp.round(3).to_dict("records")}
    for split in ("dev", "synthetic", "holdout", "holdout_warm"):
        tab = PE.table(cmp[cmp["split"] == split])
        out[f"table_{split}"] = tab.to_dict("records")
        out[f"cost_matched_{split}"] = PE.cost_matched(tab)
    out["table_dev_plus_synthetic"] = PE.table(cmp[cmp["split"] != "holdout"]).to_dict("records")
    return out, chosen


def fault_stage(bundle: Path) -> dict[str, Any]:
    from .engine import DecisionEngine
    from .fault_eval import fault_table
    t0 = time.time()
    d = pd.read_csv(LAB_DATA / "dataset_deploy.csv.gz")
    pods = pd.read_csv(LAB_DATA / "dataset_pods.csv.gz")
    faults = pd.read_csv(LAB_DATA / "faults_index.csv")
    runs = pd.read_csv(LAB_DATA / "runs_index.csv")
    tab, dec = fault_table(DecisionEngine(bundle), d, pods, faults, runs, set(HOLDOUT_SCENARIOS))
    dec.to_csv(bundle / "fault_replay_decisions.csv.gz", index=False)
    print(f"[faults] engine replay on {tab['run'].nunique()} fault runs in {time.time() - t0:.0f}s")
    return {"per_fault": tab.to_dict("records")}


# ---------------------------------------------------------------------------------- report
def _pick(records: list[dict], **kw: Any) -> list[dict]:
    return [r for r in records if all(r.get(k) == v for k, v in kw.items())]


def headline(cls: dict, fc: dict, cap: dict, sim: dict | None) -> dict[str, Any]:
    def cls_row(r: dict) -> dict:
        return {k: r[k] for k in ("macro_f1_strict", "macro_f1_settled", "accuracy_settled",
                                  "false_scale_up_on_fault_rows", "false_page_rate_on_load_rows", "missed_scale_up")}
    h: dict[str, Any] = {
        "classifier": {"chosen": cls["chosen"],
                       "loro": {m: cls_row(r) for m, r in cls["loro"].items()},
                       "holdout": {m: cls_row(r) for m, r in cls["holdout"].items()},
                       "holdout_blind": {k: {kk: v[kk] for kk in ("macro_f1_strict", "macro_f1_settled")}
                                         for k, v in cls["holdout_blind"].items()},
                       "note": "holdout = shipped rules, post-hoc for error_fault (error_rate floor) and the "
                               "residual-hog mask; holdout_blind.blind_both undoes both"},
        "forecast_mean_pinball_pct": {
            split: {f"{r['method']}@{r['h']}": round(r["rel_pinball_mean_pct"], 2) for r in fc[split]}
            for split in ("loro", "holdout")},
        "forecast_p90_coverage": {split: {f"{r['method']}@{r['h']}": round(r["cov90"], 3) for r in fc[split]
                                          if r["method"] == "lgbm_quantile"} for split in ("loro", "holdout")},
        "capacity": {"loro": cap["loro_models"], "holdout": cap["holdout_models"],
                     "rules_holdout": cap["holdout_rules"], "breach_rows_by_run": cap["breach_rows_by_run"]},
        "forecast_flat_traffic_probe": fc["flat_traffic_probe"],
    }
    if sim:
        h["policy"] = {s: sim[f"table_{s}"] for s in ("dev", "holdout", "holdout_warm")}
        h["policy_cost_matched"] = {s: sim[f"cost_matched_{s}"] for s in ("dev", "synthetic", "holdout", "holdout_warm")}
    return h


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ML_ROOT, text=True).strip()
    except Exception:
        return None


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--no-sim", action="store_true", help="skip simulator + fault replay (fast)")
    ap.add_argument("--no-latest", action="store_true", help="do not move models/scaling/LATEST")
    ap.add_argument("--version", default=None)
    args = ap.parse_args(argv)
    t0 = time.time()
    version = args.version or f"scaling-{datetime.now(timezone.utc):%Y%m%d%H%M%S}"
    out = MODELS_DIR / version
    out.mkdir(parents=True, exist_ok=True)
    prov = data_provenance()

    d, feats, faults, _pods, cap_lab = load_labelled()
    hold = is_holdout(d["scenario"])
    print(f"[data] {len(d)} rows, {d['run_id'].nunique()} runs ({d.loc[hold, 'run_id'].nunique()} held out)")
    cls = classifier_stage(d, feats, faults, _pods, out)
    fc = forecast_stage(out, args.jobs)
    cap, _cap_model, cpu_default = capacity_stage(out)

    params = PolicyParams(risk=TC.RISK, default_cpu_per_req_s=round(cpu_default, 5))
    write_json(out / "policy.json", params.to_dict())
    write_json(out / "feature_spec.json", feature_spec())
    write_json(out / "label_rules.json", label_rules(cap_lab))
    manifest: dict[str, Any] = {
        "version": version, "created_at": datetime.now(timezone.utc).isoformat(), "git_commit": _git_commit(),
        "data": {"path": str(LAB_DATA.relative_to(ML_ROOT.parent)), "sha256": prov,
                 "rows": int(len(d)), "runs": int(d["run_id"].nunique()), "rounds": sorted(int(r) for r in d["round"].unique())},
        "split": {"holdout_runs": list(HOLDOUT_SCENARIOS),
                  "dev_runs": sorted(d.loc[~hold, "scenario"].unique()),
                  "rule": "whole runs; round-2 runs with a round-1 class twin; never used to fit, select or tune"},
        "components": {"classifier": f"{cls['chosen']} (classifier/)", "forecaster": "LightGBM quantile (forecast/)",
                       "capacity": "logistic P(breach | u, ln N) (capacity.json)", "policy": "policy.json",
                       "oom_model": "models/LATEST (unchanged, optional memory-risk input)"},
        "status": "partial (simulation pending)",
    }
    write_json(out / "manifest.json", manifest)

    sim = faults_out = None
    if not args.no_sim:
        sim, chosen = sim_stage(out, params, args.jobs)
        write_json(out / "policy.json", chosen.to_dict())
        faults_out = fault_stage(out)

    metrics = {"classifier": cls, "forecast": fc, "capacity": cap, "simulation": sim, "fault_runs": faults_out}
    write_json(out / "metrics.json", metrics)
    manifest["status"] = "complete" if sim else "complete (no simulation)"
    manifest["policy"] = json.loads((out / "policy.json").read_text())
    manifest["headline_metrics_with_baselines"] = headline(cls, fc, cap, sim)
    manifest["runtime_s"] = round(time.time() - t0, 1)
    write_json(out / "manifest.json", manifest)
    if not args.no_latest:
        LATEST.write_text(version + "\n")
    print(f"[done] {out} in {time.time() - t0:.0f}s")
    return out


if __name__ == "__main__":
    main()
