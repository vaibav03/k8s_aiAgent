"""Fit and evaluate the capacity model; compare with fixed 18 rps/pod and HPA's 70 % CPU rule.

    python -m kubeai_ml.scaling.train_capacity      # eval + plots + models/scaling/capacity.json

Fit on TRAIN_ROUNDS (round 1); HOLDOUT_ROUNDS (round 2) are scored out of sample.
Training rows: qa_flag == ok, active_fault in {none, bad_deploy}, traffic flowing. Label:
breach = p95 >= 300 ms or error_rate >= 1 %. Evaluation: leave-one-run-out over every run
that has eligible rows, metrics pooled over held-out rows.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from .capacity import CapacityModel, design_matrix, robust_cpu_per_req, utilisation
from .data import (CPU_REQUEST_CORES, HOLDOUT_ROUNDS, HPA_TARGET_UTIL, MODELS_DIR, PLOTS_DIR,
                   TRAIN_ROUNDS, load_deploy)

RISK = 0.10
RULE_RISKS = (0.1, 0.2, 0.3)     # the policy sweep picks one of these; report them all
FIXED_RPS_PER_POD = 18.0
C_REG = 10.0   # LORO log-loss: C=1 0.128, C=10 0.125, C=100 0.125


def training_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["cpu_per_req_robust"] = df.groupby("run_id", group_keys=False).apply(
        lambda g: robust_cpu_per_req(g.cpu_cores_used, g.goodput_rps, g.error_rate, g.replicas_ready),
        include_groups=False)
    m = df[(df.qa_flag == "ok") & df.active_fault.isin(["none", "bad_deploy"])
           & (df.rps_per_pod > 0.5) & df.cpu_per_req_robust.notna()
           & (df.seconds_since_start <= df.duration_s)].copy()     # k6 still sending
    m["u"] = utilisation(m.rps_per_pod, m.cpu_per_req_robust)
    m["n"] = m.replicas_ready.clip(lower=1)
    return m


def _fit_logit(X: np.ndarray, y: np.ndarray) -> LogisticRegression:
    return LogisticRegression(C=C_REG, max_iter=5000).fit(X, y)


def _load_only_X(m: pd.DataFrame) -> np.ndarray:
    # same functional form, but load per pod in place of utilisation: ignores CPU cost per request
    return design_matrix(m.rps_per_pod.to_numpy() / 18.0, m.n.to_numpy())


def _nowcast_X(m: pd.DataFrame) -> np.ndarray:
    return np.column_stack([design_matrix(m.u.to_numpy(), m.n.to_numpy()), m.throttle_ratio.to_numpy()])


def fit_predict(tr: pd.DataFrame, te: pd.DataFrame) -> pd.DataFrame:
    """Fit every candidate model on ``tr``; return ``te`` with one ``p_<model>`` column each."""
    import lightgbm as lgb
    res = te[["run_id", "scenario", "active_fault", "breach", "u", "n", "rps_per_pod",
              "cpu_per_req_robust", "cpu_util_vs_request", "throttle_ratio"]].copy()
    lr = _fit_logit(design_matrix(tr.u.to_numpy(), tr.n.to_numpy()), tr.breach)
    res["p_util"] = lr.predict_proba(design_matrix(te.u.to_numpy(), te.n.to_numpy()))[:, 1]
    lo = _fit_logit(_load_only_X(tr), tr.breach)
    res["p_load_only"] = lo.predict_proba(_load_only_X(te))[:, 1]
    nc = _fit_logit(_nowcast_X(tr), tr.breach)
    res["p_nowcast_throttle"] = nc.predict_proba(_nowcast_X(te))[:, 1]
    gb = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=7, min_child_samples=20,
                            monotone_constraints=[1, 1], verbose=-1, n_jobs=1)
    gb.fit(tr[["u", "n"]], tr.breach)
    res["p_lgbm_mono"] = gb.predict_proba(te[["u", "n"]])[:, 1]
    return res


def loro(m: pd.DataFrame) -> pd.DataFrame:
    """Out-of-run predictions of every candidate model."""
    out = []
    for run in m.run_id.unique():
        tr, te = m[m.run_id != run], m[m.run_id == run]
        if tr.breach.nunique() < 2:
            continue
        out.append(fit_predict(tr, te))
    return pd.concat(out)


def score(P: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in [c for c in P.columns if c.startswith("p_")]:
        p = P[col].clip(1e-6, 1 - 1e-6)
        bd = P.active_fault == "bad_deploy"
        rows.append({"model": col[2:], "logloss": log_loss(P.breach, p), "brier": brier_score_loss(P.breach, p),
                     "auc": roc_auc_score(P.breach, p) if P.breach.nunique() > 1 else np.nan,
                     "bad_deploy_mean_p": P.loc[bd, col].mean(), "bad_deploy_breach_rate": P.loc[bd, "breach"].mean()})
    return pd.DataFrame(rows)


def breach_by_run(P: pd.DataFrame) -> dict[str, int]:
    """Breach rows per run: how many independent runs carry the evaluation."""
    c = P.groupby("run_id")["breach"].sum().astype(int)
    return {str(k): int(v) for k, v in c[c > 0].sort_values(ascending=False).items()}


def rule_table(P: pd.DataFrame) -> pd.DataFrame:
    """Each rule as a capacity: 'load above capacity' is its predicted breach."""
    hpa_cap = HPA_TARGET_UTIL * CPU_REQUEST_CORES / P.cpu_per_req_robust
    rules = {
        f"fixed {FIXED_RPS_PER_POD:.0f} rps/pod": P.rps_per_pod > FIXED_RPS_PER_POD,
        "HPA 70% CPU of request": P.rps_per_pod > hpa_cap,
        **{f"learned, risk {r}": P.p_util > r for r in RULE_RISKS},
    }
    rows = []
    for name, pred in rules.items():
        y = P.breach.astype(bool)
        bd = P.active_fault == "bad_deploy"
        rows.append({"rule": name, "recall": (pred & y).sum() / y.sum(),
                     "precision": (pred & y).sum() / max(pred.sum(), 1),
                     "false_alarm_rate": (pred & ~y).sum() / (~y).sum(),
                     "bad_deploy_recall": (pred & y & bd).sum() / max((y & bd).sum(), 1)})
    return pd.DataFrame(rows)


def replay_cost(df: pd.DataFrame, model: CapacityModel) -> pd.DataFrame:
    """Replicas each rule would ask for at the observed demand (rps) in groups B and C."""
    rows = []
    for _, g in df[df.group.isin(["B", "C"])].groupby("run_id"):
        cpr = robust_cpu_per_req(g.cpu_cores_used, g.goodput_rps, g.error_rate, g.replicas_ready).bfill().to_numpy()
        d = g.rps.to_numpy()
        ours = np.array([model.replicas_needed(x, c) for x, c in zip(d, cpr)])
        fixed = np.ceil(d / FIXED_RPS_PER_POD).clip(min=1)
        hpa = np.ceil(d * cpr / (HPA_TARGET_UTIL * CPU_REQUEST_CORES)).clip(min=1)
        rows.append({"scenario": g.scenario.iloc[0], "mean_demand_rps": d.mean(),
                     "actual_replicas": g.replicas_ready.mean(), "learned": ours.mean(),
                     "fixed18": fixed.mean(), "hpa70": hpa.mean()})
    r = pd.DataFrame(rows)
    tot = r[["actual_replicas", "learned", "fixed18", "hpa70"]].mean()
    return pd.concat([r, pd.DataFrame([{"scenario": "MEAN", **tot.to_dict()}])], ignore_index=True)


def plot(m: pd.DataFrame, model: CapacityModel) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    base = m[m.active_fault == "none"]
    x = np.linspace(0, 24, 200)
    for n, c in zip(range(1, 5), ["tab:blue", "tab:orange", "tab:green", "tab:red"]):
        cpr = 0.0265
        ax.plot(x, model.p_breach(x, cpr, n), color=c, label=f"N={n} (model, 26.5 ms/req)")
        g = base[base.n.clip(upper=4) == n]
        b = pd.cut(g.rps_per_pod, np.arange(0, 26, 2))
        e = g.groupby(b, observed=True).breach.agg(["mean", "count"])
        e = e[e["count"] >= 5]
        ax.scatter([iv.mid for iv in e.index], e["mean"], color=c, s=14)
    ax.axhline(RISK, color="k", ls="--", lw=0.8)
    ax.axvline(FIXED_RPS_PER_POD, color="gray", ls=":", lw=1)
    ax.axvline(HPA_TARGET_UTIL * CPU_REQUEST_CORES / 0.026, color="purple", ls=":", lw=1)
    ax.text(FIXED_RPS_PER_POD + 0.2, 0.5, "fixed 18", fontsize=8, color="gray")
    ax.text(6.9, 0.5, "HPA 70%", fontsize=8, color="purple")
    ax.set_xlabel("offered rps per ready pod")
    ax.set_ylabel("P(SLO breach)")
    ax.set_title("capacity: model curves vs empirical breach rate (dots)", fontsize=9)
    ax.legend(fontsize=7)

    ax = axes[1]
    cprs = np.linspace(0.015, 0.09, 60)
    for n, c in zip(range(1, 5), ["tab:blue", "tab:orange", "tab:green", "tab:red"]):
        ax.plot(cprs * 1000, [model.safe_rps_per_pod(cp, n) for cp in cprs], color=c, label=f"learned N={n}")
    ax.plot(cprs * 1000, HPA_TARGET_UTIL * CPU_REQUEST_CORES / cprs, color="purple", ls="--", label="HPA 70% equiv")
    ax.axhline(FIXED_RPS_PER_POD, color="gray", ls=":", label="fixed 18")
    ax.axvline(26.5, color="k", lw=0.5)
    ax.axvline(78, color="k", lw=0.5)
    ax.text(27, 1, "good build", fontsize=7)
    ax.text(70, 1, "bad deploy", fontsize=7)
    ax.set_xlabel("CPU ms per request")
    ax.set_ylabel(f"safe rps per pod (risk {RISK})")
    ax.set_title("safe_rps_per_pod(cpu_per_req, replicas)", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "capacity.png", dpi=120)
    plt.close(fig)


def main() -> None:
    df = load_deploy(TRAIN_ROUNDS)
    m = training_rows(df)
    print(f"rows {len(m)}, breach rate {m.breach.mean():.3f}, runs {m.run_id.nunique()}")
    P = loro(m)
    S = score(P)
    R = rule_table(P)
    pd.set_option("display.width", 200)
    print("== leave-one-run-out (train rounds)")
    print(S.round(3).to_string(index=False))
    print(R.round(3).to_string(index=False))

    # out-of-sample: fit on every train-round row, score later rounds as collected
    hd = load_deploy(HOLDOUT_ROUNDS)
    H = fit_predict(m, training_rows(hd)) if len(hd) else pd.DataFrame()
    SH = score(H) if len(H) else pd.DataFrame()
    RH = rule_table(H) if len(H) else pd.DataFrame()
    if len(H):
        print(f"== holdout rounds {HOLDOUT_ROUNDS}: rows {len(H)}, runs {H.run_id.nunique()}, "
              f"breach rate {H.breach.mean():.3f}")
        print(SH.round(3).to_string(index=False))
        print(RH.round(3).to_string(index=False))

    lr = _fit_logit(design_matrix(m.u.to_numpy(), m.n.to_numpy()), m.breach)
    model = CapacityModel(coef=lr.coef_[0].round(6).tolist(), intercept=round(float(lr.intercept_[0]), 6),
                          n_max=float(m.n.max()), risk=RISK)
    safe = pd.DataFrame([{"replicas": n, **{f"safe_rps_per_pod@{int(c * 1000)}ms": model.safe_rps_per_pod(c, n)
                                              for c in (0.0265, 0.05, 0.078)},
                          "safe_u": model.safe_utilisation(n)} for n in range(1, 7)])
    print(safe.round(2).to_string(index=False))
    C = replay_cost(load_deploy(), model)
    print(C.round(2).to_string(index=False))
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot(m, model)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODELS_DIR / "capacity.json")
    (MODELS_DIR / "capacity_metrics.json").write_text(json.dumps({
        "train_rounds": list(TRAIN_ROUNDS), "holdout_rounds": list(HOLDOUT_ROUNDS),
        "loro_models": S.round(4).to_dict("records"), "rules": R.round(4).to_dict("records"),
        "holdout_models": SH.round(4).to_dict("records"), "holdout_rules": RH.round(4).to_dict("records"),
        "safe_table": safe.round(3).to_dict("records"), "replay_cost": C.round(3).to_dict("records")}, indent=1))
    print("saved", MODELS_DIR / "capacity.json", model)


if __name__ == "__main__":
    main()
