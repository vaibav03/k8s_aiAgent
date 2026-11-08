"""Leave-one-run-out evaluation of the state classifiers; fits and saves the chosen one.

    PYTHONPATH=. python -m kubeai_ml.scaling.train_classifier [--no-save]

For each held-out run: capacity is re-learned on the training runs only, every model is fit
on the training runs' ``qa_flag == ok`` rows and scored on the held-out run's ok rows.
Metrics: macro-F1 / per-class precision & recall / confusion (strict, and "settled" which
drops rows within ``SETTLE_S`` after a label change: labels trail signals by up to ~25 s),
detection delay per injected fault, false ``scale_up`` on fault rows (the costly mistake) and
false pages (fault state predicted on a fault-free row). Outputs ``classifier_cv.json`` next
to this file and the chosen model (rules vs hybrid, by settled macro-F1) under
``models/scaling/state_classifier``.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from . import states as S
from .classifier import MODES, StateClassifier
from .data import LAB_DATA, MODELS_DIR
from .features import build_features
from .labels import learn_safe_capacity, make_labels

SETTLE_S = 30.0
OUT_JSON = Path(__file__).resolve().parent / "classifier_cv.json"
CANDIDATES = ("rules", "hybrid")   # shipped = best settled macro-F1; ties go to rules


def load_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Deploy rows with features + labels, and the faults index. Works on appended rounds."""
    d = pd.read_csv(LAB_DATA / "dataset_deploy.csv.gz")
    d["active_fault"] = d["active_fault"].fillna("none")
    d = d.sort_values(["run_id", "ts"]).reset_index(drop=True)
    pods = pd.read_csv(LAB_DATA / "dataset_pods.csv.gz")
    by_run = dict(tuple(pods.groupby("run_id")))
    feats = pd.concat([build_features(g, by_run.get(r)) for r, g in d.groupby("run_id")]).loc[d.index]
    faults = pd.read_csv(LAB_DATA / "faults_index.csv")
    lab = make_labels(d, pods=pods, faults=faults)
    d["state"], d["use"] = lab["state"], lab["use"]
    d["settled"] = _settled(d)
    return d, feats, faults


def _settled(d: pd.DataFrame) -> np.ndarray:
    """False on rows within SETTLE_S after a label change -- except inside a label segment that
    is itself no longer than SETTLE_S (a 30 s pod_kill window would otherwise never be scored)."""
    out = np.ones(len(d), dtype=bool)
    for _, g in d.groupby("run_id"):
        g = g.sort_values("ts")
        pos = d.index.get_indexer(g.index)
        st, ts = g["state"].to_numpy(), g["ts"].to_numpy(float)
        change = np.flatnonzero(st[1:] != st[:-1]) + 1
        ends = np.r_[ts[change[1:]] if len(change) > 1 else [], np.inf] if len(change) else []
        for k, i in enumerate(change):
            t, end = ts[i], ends[k]
            if end - t <= SETTLE_S + 1e-6:          # short segment: score it whole
                continue
            out[pos[(ts >= t) & (ts < t + SETTLE_S)]] = False
    return out


def cross_validate(d: pd.DataFrame, feats: pd.DataFrame,
                   relabel: Callable[[object], pd.Series] | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Leave-one-run-out predictions, and the labels each fold scored against.

    ``relabel(cap)`` -> state per row of ``d``; when given, every fold re-derives the load labels
    from its own capacity knee (learned without the test run), so neither the training labels
    nor the test labels depend on the test run's data (review 3)."""
    preds = pd.DataFrame(index=d.index, columns=list(MODES), dtype=object)
    y = d["state"].copy()
    for run in d["run_id"].unique():
        test = d["run_id"] == run
        train = ~test & d["use"]
        cap = learn_safe_capacity(d[~test])
        state = relabel(cap) if relabel is not None else d["state"]
        y.loc[test] = state.loc[test]
        for mode in MODES:
            clf = StateClassifier(mode=mode, cap=cap).fit(feats[train], state.loc[train].to_numpy())
            preds.loc[test, mode] = clf.predict(feats[test])
    return preds, y


def round_holdout(d: pd.DataFrame, feats: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series] | None:
    """Forward test: fit capacity + models on earlier rounds only, predict the newest round.

    This is the honest test of the runbook thresholds (they were written after seeing round 1
    only). Returns (predictions on the newest round, mask of those rows) or None for one round."""
    if "round" not in d or d["round"].nunique() < 2:
        return None
    test = d["round"] == d["round"].max()
    train = ~test & d["use"]
    cap = learn_safe_capacity(d[~test])
    preds = pd.DataFrame(index=d.index[test], columns=list(MODES), dtype=object)
    for mode in MODES:
        clf = StateClassifier(mode=mode, cap=cap).fit(feats[train], d.loc[train, "state"].to_numpy())
        preds[mode] = clf.predict(feats[test])
    return preds, test


def _class_report(y: np.ndarray, p: np.ndarray) -> dict[str, dict[str, float]]:
    labels = [s for s in S.STATES if s in set(y)]
    pr, rc, f1, n = precision_recall_fscore_support(y, p, labels=labels, zero_division=0)
    return {s: {"precision": round(float(a), 3), "recall": round(float(b), 3),
                "f1": round(float(c), 3), "n": int(k)} for s, a, b, c, k in zip(labels, pr, rc, f1, n)}


def detection_delays(d: pd.DataFrame, pred: pd.Series, faults: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    for f in faults.itertuples():
        g = d[d["run_id"] == f.run_id]
        if g.empty:
            continue
        want = S.FAULT_TO_STATE[f.type]
        end = f.start_rel_s + max(float(f.label_window_s), 30.0) + SETTLE_S
        win = g[(g["seconds_since_start"] >= f.start_rel_s) & (g["seconds_since_start"] <= end)]
        hit = win[pred.loc[win.index] == want]
        any_fault = win[pred.loc[win.index].isin(S.FAULT_STATES)]
        rows.append({"fault_id": f.fault_id, "type": f.type,
                     "delay_s": float(hit["seconds_since_start"].iloc[0] - f.start_rel_s) if len(hit) else None,
                     "any_fault_alert_s": float(any_fault["seconds_since_start"].iloc[0] - f.start_rel_s) if len(any_fault) else None,
                     "row_recall": round(float((pred.loc[win.index] == want).mean()), 3)})
    return rows


def evaluate(d: pd.DataFrame, preds: pd.DataFrame, faults: pd.DataFrame) -> dict[str, object]:
    res: dict[str, object] = {}
    is_fault = d["state"].isin(S.FAULT_STATES)
    for mode in preds.columns:
        out: dict[str, object] = {}
        for name, m in (("strict", d["use"]), ("settled", d["use"] & d["settled"])):
            y, p = d.loc[m, "state"].to_numpy(), preds.loc[m, mode].to_numpy()
            labels = [s for s in S.STATES if s in set(y)]
            out[f"macro_f1_{name}"] = round(float(f1_score(y, p, labels=labels, average="macro", zero_division=0)), 3)
            out[f"accuracy_{name}"] = round(float((y == p).mean()), 3)
            out[f"classes_{name}"] = len(labels)
        ms = d["use"]
        out["per_class_strict"] = _class_report(d.loc[ms, "state"].to_numpy(), preds.loc[ms, mode].to_numpy())
        m = d["use"] & d["settled"]
        y, p = d.loc[m, "state"].to_numpy(), preds.loc[m, mode].to_numpy()
        out["per_class_settled"] = _class_report(y, p)
        labels = [s for s in S.STATES if s in set(y) | set(p)]
        out["confusion_settled"] = {"labels": labels,
                                    "matrix": confusion_matrix(y, p, labels=labels).tolist()}
        fm = d["use"] & is_fault
        fm_noe = fm & ~d["scenario"].str.startswith("e-spike")
        out["false_scale_up_on_fault_rows"] = round(float((preds.loc[fm, mode] == S.SCALE_UP).mean()), 3)
        out["false_scale_up_on_fault_rows_excl_spike_overlap"] = round(float((preds.loc[fm_noe, mode] == S.SCALE_UP).mean()), 3)
        out["false_scale_up_by_fault"] = (preds.loc[fm, mode] == S.SCALE_UP).groupby(d.loc[fm, "state"]).mean().round(3).to_dict()
        lm = d["use"] & d["settled"] & ~is_fault
        out["false_page_rate_on_load_rows"] = round(float(preds.loc[lm, mode].isin(S.FAULT_STATES).mean()), 3)
        out["missed_scale_up"] = round(float((preds.loc[lm & (d["state"] == S.SCALE_UP), mode] != S.SCALE_UP).mean()), 3)
        out["detection"] = detection_delays(d, preds[mode], faults)
        res[mode] = out
    return res


def print_summary(res: dict[str, object]) -> None:
    print("| model | macro-F1 strict | macro-F1 settled | acc settled | false scale_up on faults | false pages | missed scale_up |")
    print("|---|---|---|---|---|---|---|")
    for mode, r in res.items():
        print(f"| {mode} | {r['macro_f1_strict']} | {r['macro_f1_settled']} | {r['accuracy_settled']} | "
              f"{r['false_scale_up_on_fault_rows']} | {r['false_page_rate_on_load_rows']} | {r['missed_scale_up']} |")
    for mode, r in res.items():
        print(f"\n## {mode}\n| class | P | R | F1 | n |\n|---|---|---|---|---|")
        for c, v in r["per_class_settled"].items():
            print(f"| {c} | {v['precision']} | {v['recall']} | {v['f1']} | {v['n']} |")
        print("\n| fault | delay s | any-fault alert s | row recall |\n|---|---|---|---|")
        for x in r["detection"]:
            print(f"| {x['fault_id'].split('-1108')[0]} {x['type']} | {x['delay_s']} | {x['any_fault_alert_s']} | {x['row_recall']} |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    d, feats, faults = load_frames()
    cap = learn_safe_capacity(d)
    print(f"rows {len(d)}, runs {d['run_id'].nunique()}, capacity knees {cap.table} fallback {cap.fallback}")
    print(d.loc[d["use"], "state"].value_counts().to_string())
    preds, _ = cross_validate(d, feats)
    res = evaluate(d, preds, faults)
    print_summary(res)
    payload: dict[str, object] = {"capacity": cap.to_dict(), "label_counts":
                                  d.loc[d["use"], "state"].value_counts().to_dict(),
                                  "masked_rows": int((~d["use"]).sum()), "results": res}
    fwd = round_holdout(d, feats)
    if fwd is not None:
        fp, test = fwd
        newest = int(d["round"].max())
        print(f"\n# forward test: train rounds < {newest}, test round {newest} "
              f"({d.loc[test, 'run_id'].nunique()} runs)")
        fres = evaluate(d[test], fp, faults)
        print_summary(fres)
        payload["round_holdout"] = {"test_round": newest, "results": fres}
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=float))
    chosen = max(CANDIDATES, key=lambda m: (res[m]["macro_f1_settled"], m == "rules"))
    print(f"chosen: {chosen}")
    if not args.no_save:
        clf = StateClassifier(mode=chosen, cap=cap).fit(feats[d["use"]], d.loc[d["use"], "state"].to_numpy())
        out = MODELS_DIR / "state_classifier"
        clf.save(out, {"trained_on_rows": int(d["use"].sum()), "runs": int(d["run_id"].nunique()),
                       "cv_macro_f1_settled": res[chosen]["macro_f1_settled"]})
        print(f"saved {out}")
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
