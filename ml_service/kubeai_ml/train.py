"""Train, evaluate and package the OOM-risk model.

    python -m kubeai_ml.train                 # train + write models/<version>/ and models/LATEST
    python -m kubeai_ml.train --no-old-agent  # skip the TensorFlow baseline

Label: on rows where the pod is Running, y = 1 if the pod crashes
(OOMKilled / CrashLoopBackOff, or a restart-count increment) within the next
H seconds. Split: time-ordered on crash-event boundaries (60/20/20 of events),
no shuffling. Threshold and calibration come from the validation split only;
every number in the manifest is on the untouched test split.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)

from . import features as F
from .data import MIB, build_examples, label, load_k8s_metrics, load_nirmal
from .model import OomModel

ROOT = Path(__file__).resolve().parent.parent
AGENT_DIR = ROOT.parent


def old_artifact(name):
    """The legacy agent files live in legacy/ (or the repo root on older checkouts)."""
    for path in (AGENT_DIR / "legacy" / name, AGENT_DIR / name):
        if path.exists():
            return path
    raise FileNotFoundError(name)
LOOKBACK_S = 300  # an alert counts toward an event if raised in the 300 s before it

BASE_FEATURES = list(F.FRACTION_FEATURES)
RESTART_FEATURES = BASE_FEATURES + ["seconds_since_restart"]


def monotone(feature):
    if feature.startswith("frac_std"):
        return 0
    if feature.startswith("frac_"):
        return 1          # more usage / faster growth -> more risk
    if feature in ("headroom", "time_to_limit_s"):
        return -1
    return 0


LGB_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=15,
                  min_child_samples=20, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.8, reg_lambda=1.0, verbose=-1, seed=7,
                  deterministic=True, force_row_wise=True)
NUM_ROUNDS = 200
# Trend-gate candidates for is_anomaly (max time_to_limit_s); picked on validation.
GATE_CANDIDATES_S = (300, 600, 900, 1800)


# ---------------------------------------------------------------- metrics

def best_f1_threshold(y, score):
    """Threshold on `score` that maximises F1 (ties -> highest threshold)."""
    best_t, best_f = 1.0, -1.0
    for t in np.unique(score):
        f = f1_score(y, score >= t, zero_division=0)
        if f > best_f + 1e-12 or (abs(f - best_f) <= 1e-12 and t > best_t):
            best_t, best_f = float(t), f
    return best_t


def event_stats(t, alert, next_event_in_s, events):
    """Event recall, lead time and false alarms per hour of Running time."""
    leads, observable = [], 0
    for e in events:
        pts = (t >= e - LOOKBACK_S) & (t < e)
        if not pts.any():
            continue
        observable += 1
        if alert[pts].any():
            leads.append(float(e - t[pts][alert[pts]].min()))
    far = next_event_in_s > LOOKBACK_S
    hours = far.sum() * F.STEP_S / 3600
    return {
        "events": observable,
        "events_caught": len(leads),
        "event_recall": len(leads) / observable if observable else None,
        "mean_lead_time_s": float(np.mean(leads)) if leads else None,
        "median_lead_time_s": float(np.median(leads)) if leads else None,
        "false_alarms_per_hour": float(alert[far].sum() / hours) if hours else None,
    }


def classification_report(y, score, alert, ex, events):
    has_both = 0 < y.sum() < len(y)
    out = {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "pr_auc": float(average_precision_score(y, score)) if has_both else None,
        "roc_auc": float(roc_auc_score(y, score)) if has_both else None,
        "f1": float(f1_score(y, alert, zero_division=0)),
        "precision": float(precision_score(y, alert, zero_division=0)),
        "recall": float(recall_score(y, alert, zero_division=0)),
    }
    out.update(event_stats(ex["t"].to_numpy(), alert, ex["next_event_in_s"].to_numpy(), events))
    return out


def rounded(obj, nd=4):
    if isinstance(obj, dict):
        return {k: rounded(v, nd) for k, v in obj.items()}
    if isinstance(obj, list):
        return [rounded(v, nd) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return None if not np.isfinite(obj) else round(float(obj), nd)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


# ---------------------------------------------------------------- old agent

class OldAgent:
    """The previous LSTM(10 x 1s samples) -> LightGBM(1 feature) pair."""

    def __init__(self):
        import os
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        import joblib
        import tensorflow as tf
        self.lstm = tf.keras.models.load_model(old_artifact("hybrid_lstm_model.keras"), compile=False)
        # Trusted: the repo's own legacy artifact, loaded only for this offline baseline.
        self.gbm = joblib.load(old_artifact("lightgbm_anomaly.pkl"))

    def run(self, ex):
        windows = np.stack([np.pad(w, (10 - len(w), 0), mode="edge") if len(w) < 10 else w[-10:]
                            for w in ex["last10_frac"]])
        pred = self.lstm.predict(windows.reshape(-1, 10, 1), verbose=0).ravel()
        score = self.gbm.predict_proba(pred.reshape(-1, 1))[:, 1]
        flag = self.gbm.predict(pred.reshape(-1, 1)).astype(bool)
        return pred, score, flag


# ---------------------------------------------------------------- forecast

def forecast_mae(ex, limit_bytes, damping, old_pred=None):
    """MAE (MiB) at each horizon for persistence, robust trend and old LSTM."""
    out = {}
    for h in F.FORECAST_HORIZONS_S:
        target = ex[f"future_bytes_{h}s"].to_numpy()
        ok = np.isfinite(target)
        last = ex["last_bytes"].to_numpy()
        trend = np.clip(last + damping * ex["trend_bytes_per_s"].to_numpy() * h, 0, limit_bytes)
        row = {
            "n": int(ok.sum()),
            "persistence": float(np.mean(np.abs(last[ok] - target[ok])) / MIB),
            "trend": float(np.mean(np.abs(trend[ok] - target[ok])) / MIB),
        }
        if old_pred is not None:
            # The old LSTM predicts one step ahead; it is the only forecast it gives.
            row["old_lstm"] = float(np.mean(np.abs(old_pred[ok] * limit_bytes - target[ok])) / MIB)
        out[f"{h}s"] = row
    return out


def pick_damping(ex, limit_bytes):
    """Choose trend damping on validation; 0 means plain persistence."""
    scores = {}
    for d in (0.0, 0.25, 0.5, 0.75, 1.0):
        mae = forecast_mae(ex, limit_bytes, d)
        scores[d] = np.mean([m["trend"] for m in mae.values()])
    return min(scores, key=scores.get), scores


# ---------------------------------------------------------------- training

def split_masks(t, events):
    """60/20/20 of crash events, cut right after an event (pod is down then)."""
    n = len(events)
    cut1, cut2 = events[int(round(0.6 * n)) - 1], events[int(round(0.8 * n)) - 1]
    return t <= cut1, (t > cut1) & (t <= cut2), t > cut2, (cut1, cut2)


def ttl_raw(d):
    """time_to_limit_raw_s as floats (NaN = memory not rising)."""
    return pd.to_numeric(d["time_to_limit_raw_s"], errors="coerce").to_numpy(dtype=float)


def pick_threshold_and_gate(y, p, ttl):
    """Best validation F1 over (gate, threshold). Ties -> tighter gate."""
    best = None
    for gate in GATE_CANDIDATES_S:
        passes = np.isfinite(ttl) & (ttl <= gate)
        score = np.where(passes, p, -1.0)
        thr = best_f1_threshold(y, score)
        thr = max(thr, 0.0)
        f = f1_score(y, (score >= thr) & passes, zero_division=0)
        if best is None or f > best[2] + 1e-12:
            best = (thr, gate, f)
    return best[0], best[1]


def fit_one(train, val, y_train, y_val, feature_names):
    pos = y_train.sum()
    params = dict(LGB_PARAMS, scale_pos_weight=float((len(y_train) - pos) / pos),
                  monotone_constraints=[monotone(f) for f in feature_names])
    booster = lgb.train(params, lgb.Dataset(train[feature_names].to_numpy(), y_train,
                                            feature_name=feature_names), NUM_ROUNDS)
    raw_val = booster.predict(val[feature_names].to_numpy())
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw_val, y_val)
    calibrator = {"x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist()}
    return booster, calibrator


def evaluate_horizon(H, ex, ex_train, events, masks, feature_names, cross, old, ex_raw=None):
    tr, va, te, (cut1, cut2) = masks
    train = ex_train[ex_train["t"] <= cut1]
    val, test = ex[va], ex[te]
    y_tr, y_va, y_te = label(train, H), label(val, H), label(test, H)
    ev_val = events[(events > cut1) & (events <= cut2)]
    ev_test = events[events > cut2]

    booster, calibrator = fit_one(train, val, y_tr, y_va, feature_names)
    model = OomModel(booster, calibrator, {"features": feature_names, "threshold": 0.5,
                                           "horizon_s": H, "model_version": "candidate"})
    p_val = model.p_oom_batch(val[feature_names].to_numpy())
    p_test = model.p_oom_batch(test[feature_names].to_numpy())
    thr, gate = pick_threshold_and_gate(y_va, p_val, ttl_raw(val))
    model.threshold, model.max_ttl_s = thr, float(gate)
    a_val, a_test = model.alerts(p_val, ttl_raw(val)), model.alerts(p_test, ttl_raw(test))

    # Ablation: the same model with no trend gate (threshold re-tuned on val).
    thr_ng = best_f1_threshold(y_va, p_val)

    res = {"horizon_s": H, "threshold": thr, "max_time_to_limit_s": gate,
           "train": {"n": int(len(train)), "positives": int(y_tr.sum())},
           "val": classification_report(y_va, p_val, a_val, val, ev_val),
           "test": classification_report(y_te, p_test, a_test, test, ev_test),
           "baselines": {},
           "no_trend_gate": {
               "threshold": thr_ng,
               "val": classification_report(y_va, p_val, p_val >= thr_ng, val, ev_val),
               "test": classification_report(y_te, p_test, p_test >= thr_ng, test, ev_test)}}

    # The shipped model fed raw ~1 s collector rows instead of agent-style 15 s ticks.
    if ex_raw is not None:
        tr_r = ex_raw["t"].to_numpy()
        res["raw_input"] = {}
        for name, mk, evs in (("val", (tr_r > cut1) & (tr_r <= cut2), ev_val),
                              ("test", tr_r > cut2, ev_test)):
            d = ex_raw[mk]
            p = model.p_oom_batch(d[feature_names].to_numpy())
            res["raw_input"][name] = classification_report(
                label(d, H), p, model.alerts(p, ttl_raw(d)), d, evs)

    # (a) usage-fraction rule and a time-to-limit rule, thresholds tuned on val.
    rules = {
        "usage_fraction_rule": lambda d: d["frac_last"].to_numpy(),
        "time_to_limit_rule": lambda d: -d["time_to_limit_s"].to_numpy(),
    }
    rule_thr = {}
    for name, score_fn in rules.items():
        thr_r = best_f1_threshold(y_va, score_fn(val))
        rule_thr[name] = thr_r
        s = score_fn(test)
        res["baselines"][name] = dict(
            threshold=thr_r if name == "usage_fraction_rule" else -thr_r,
            **classification_report(y_te, s, s >= thr_r, test, ev_test))

    # (b) the old agent pair, scored against the new label.
    if old is not None:
        _, s, flag = old.run(test)
        res["baselines"]["old_agent_lstm_lgbm"] = classification_report(y_te, s, flag, test, ev_test)

    # Cross-dataset: same model, same val threshold, true limit.
    cx, cev = cross
    y_cx = label(cx, H)
    p_cx = model.p_oom_batch(cx[feature_names].to_numpy())
    res["cross_dataset"] = {
        "model": classification_report(y_cx, p_cx, model.alerts(p_cx, ttl_raw(cx)), cx, cev),
        "model_no_trend_gate": classification_report(y_cx, p_cx, p_cx >= thr_ng, cx, cev)}
    for name, score_fn in rules.items():
        s = score_fn(cx)
        res["cross_dataset"][name] = classification_report(y_cx, s, s >= rule_thr[name], cx, cev)
    if old is not None:
        _, s, flag = old.run(cx)
        res["cross_dataset"]["old_agent_lstm_lgbm"] = classification_report(y_cx, s, flag, cx, cev)
    return res, booster, calibrator


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=str(ROOT / "data/k8s_metrics.csv"))
    ap.add_argument("--cross-data", default=str(ROOT / "data/nirmal_dataset_1.csv"))
    ap.add_argument("--out", default=str(ROOT / "models"))
    ap.add_argument("--horizons", type=int, nargs="+", default=[30, 60, 120])
    ap.add_argument("--horizon", type=int, default=None,
                    help="horizon to ship; default: best validation F1 among --horizons "
                         "(ties -> longer horizon, i.e. more lead time)")
    ap.add_argument("--with-restart-feature", action="store_true",
                    help="add seconds_since_restart to the model features")
    ap.add_argument("--train-stride", type=int, default=5,
                    help="seconds between training examples (val/test always use 15)")
    ap.add_argument("--no-old-agent", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="evaluate only, write nothing")
    args = ap.parse_args(argv)

    feature_names = RESTART_FEATURES if args.with_restart_feature else BASE_FEATURES
    series = load_k8s_metrics(args.data)
    cross_series = load_nirmal(args.cross_data)
    ex, events = build_examples(series, stride=F.STEP_S)              # agent-style 15 s ticks
    ex_train = build_examples(series, stride=args.train_stride)[0]
    ex_raw = build_examples(series, stride=F.STEP_S, sampler="raw")[0]  # raw ~1 s rows, for comparison
    cross = build_examples(cross_series, stride=F.STEP_S)
    masks = split_masks(ex["t"].to_numpy(), events)
    old = None if args.no_old_agent else OldAgent()

    results, bundles = {}, {}
    for H in sorted(set(args.horizons) | ({args.horizon} if args.horizon else set())):
        res, booster, calibrator = evaluate_horizon(H, ex, ex_train, events, masks,
                                                    feature_names, cross, old, ex_raw)
        results[H], bundles[H] = res, (booster, calibrator)
        print(f"H={H:>3}s thr={res['threshold']:.3f} gate={res['max_time_to_limit_s']}s "
              f"val F1={res['val']['f1']:.3f} PR-AUC={res['val']['pr_auc']:.3f} | "
              f"test F1={res['test']['f1']:.3f} PR-AUC={res['test']['pr_auc']:.3f} "
              f"P={res['test']['precision']:.3f} R={res['test']['recall']:.3f} "
              f"lead={res['test']['mean_lead_time_s']} | "
              + " ".join(f"{k}: F1={v['f1']:.3f} PR-AUC={v['pr_auc']:.3f}"
                         for k, v in res["baselines"].items()))

    # Forecast: damping picked on val, reported on test and cross-dataset.
    tr, va, te, (cut1, cut2) = masks
    damping, damping_scores = pick_damping(ex[va], series.limit_bytes)
    old_te = old.run(ex[te])[0] if old else None
    old_cx = old.run(cross[0])[0] if old else None
    forecast = {
        "method": "persistence" if damping == 0 else "theil-sen trend extrapolation",
        "damping": damping,
        "trend_window_s": F.TREND_WINDOW_S,
        "val_mean_mae_mib_by_damping": damping_scores,
        "test_mae_mib": forecast_mae(ex[te], series.limit_bytes, damping, old_te),
        "cross_dataset_mae_mib": forecast_mae(cross[0], cross_series.limit_bytes, damping, old_cx),
    }
    print("forecast damping", damping, json.dumps(rounded(forecast["test_mae_mib"], 2)))

    H = args.horizon or max(results, key=lambda h: (round(results[h]["val"]["f1"], 3), h))
    chosen = results[H]
    print(f"shipping horizon {H}s (validation F1 {chosen['val']['f1']:.3f})")
    trained_at = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    version = f"oom-gbm-h{H}-{trained_at:%Y%m%d%H%M%S}"
    iso = lambda x: dt.datetime.fromtimestamp(x, dt.timezone.utc).isoformat()
    manifest = rounded({
        "model_version": version,
        "trained_at": trained_at.isoformat(),
        "features": feature_names,
        "monotone_constraints": {f: monotone(f) for f in feature_names},
        "horizon_s": H,
        "horizon_rule": ("best validation F1 among the evaluated horizons, ties to the longer one"
                         if args.horizon is None else "set on the command line"),
        "threshold": chosen["threshold"],
        "threshold_rule": ("max F1 on the validation split (calibrated probability), chosen "
                           "jointly with alert_rule.max_time_to_limit_s"),
        "alert_rule": {
            "is_anomaly": "p_oom >= threshold AND time_to_limit_s <= max_time_to_limit_s",
            "max_time_to_limit_s": chosen["max_time_to_limit_s"],
            "candidates_s": list(GATE_CANDIDATES_S),
            "why": ("the training data has almost no pods that stay flat at high usage, so "
                    "p_oom alone flags steady pods above ~2/3 of the limit; alerts also need "
                    "memory to be on course to reach the limit"),
        },
        "label_definition": (f"On rows where the pod is Running, y=1 if within the next {H}s the pod "
                             "becomes OOMKilled or CrashLoopBackOff, or its restart count increments "
                             "(markers within 60s merged into one crash). Non-Running rows are not "
                             "scored."),
        "feature_definition": {"step_s": F.STEP_S, "windows_s": list(F.WINDOWS_S),
                               "history_s": F.HISTORY_S, "trend_window_s": F.TREND_WINDOW_S,
                               "ttl_cap_s": F.TTL_CAP_S},
        "data": {"train_file": series.name, "data_sha256": series.sha256,
                 "cross_file": cross_series.name, "cross_sha256": cross_series.sha256,
                 "limit_bytes": series.limit_bytes, "limit_note": series.limit_note,
                 "cross_limit_note": cross_series.limit_note,
                 "crash_events": len(events)},
        "data_sha256": series.sha256,
        "split": {"method": "time-ordered, cut after crash events (60/20/20 of events)",
                  "train_until": iso(cut1), "val_until": iso(cut2),
                  "train_stride_s": args.train_stride, "eval_stride_s": F.STEP_S,
                  "history": "agent-style 15 s ticks, cut at container restarts"},
        "lightgbm": dict(LGB_PARAMS, num_boost_round=NUM_ROUNDS,
                         imbalance="scale_pos_weight = neg/pos (only correction)",
                         calibration="isotonic on validation"),
        "metrics": {"val": chosen["val"], "test": chosen["test"]},
        "input_shape": {"metrics": "agent-style: one sample every 15 s (instant query = newest "
                                   "scrape), 40-sample window, cleared at container restart",
                        "raw_input": chosen.get("raw_input")},
        "no_trend_gate": chosen["no_trend_gate"],
        "baselines": chosen["baselines"],
        "cross_dataset": chosen["cross_dataset"],
        "horizon_comparison": {str(h): r for h, r in results.items()},
        "forecast": forecast,
    })

    if args.dry_run:
        print(json.dumps(manifest["metrics"]["test"], indent=1))
        return manifest

    out = Path(args.out) / version
    out.mkdir(parents=True, exist_ok=True)
    booster, calibrator = bundles[H]
    booster.save_model(str(out / "model.txt"))
    (out / "calibrator.json").write_text(json.dumps(calibrator))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (Path(args.out) / "LATEST").write_text(version + "\n")
    print("wrote", out)
    return manifest


if __name__ == "__main__":
    main()
