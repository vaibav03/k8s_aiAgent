"""Leave-run-out evaluation of the demand forecasters, then fit and save the chosen one.

    python -m kubeai_ml.scaling.train_forecast            # eval + plots + models/scaling/forecast/
    python -m kubeai_ml.scaling.train_forecast --target goodput_rps   # ablation
    python -m kubeai_ml.scaling.train_forecast --final-all-rounds     # ship a model fit on every round

Protocol (1) leave-run-out inside TRAIN_ROUNDS (round 1): each shaped run (groups B, C, E;
13 runs) is held out in turn; the model is trained on all 23 other round-1 runs (incl. A step
ramps and D steady runs). (2) Out of sample: fit on every round-1 run, score the shaped runs
of HOLDOUT_ROUNDS (round 2: new seeds, shapes and HPA settings) exactly as collected. Origins every 15 s (the
decision cadence) from t >= 60 s; a target counts only while k6 is still sending
(t+h <= duration_s). Baselines get p50/p90 from residual quantiles on the training runs; the
LightGBM p90 gets an additive conformal margin from 5-fold grouped out-of-fold residuals
inside the training runs. Metrics pooled over all held-out origins.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from . import forecast as F
from .data import HOLDOUT_ROUNDS, MODELS_DIR, PLOTS_DIR, SHAPED_GROUPS, TRAIN_ROUNDS, load_deploy

EVAL_STRIDE = 3          # 15 s
MIN_HISTORY_S = 60
WARMUP_S = 30            # k6 ramps from 0 in the first seconds of every run (a lab artefact)


def build_origins(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """One row per (run, origin): history matrix, targets per horizon, metadata."""
    recs = []
    for run_id, g in df.groupby("run_id", sort=False):
        z = F.to_z(g[target].to_numpy())
        t = g["seconds_since_start"].to_numpy()
        hz = np.where(t >= WARMUP_S, z, np.nan)        # history never shows the k6 start ramp
        dur = float(g["duration_s"].iloc[0])
        pad = np.concatenate([np.full(F.HIST_STEPS, np.nan), hz])
        for i in range(len(z)):
            if t[i] > dur:
                break
            ys = {}
            for h in F.HORIZONS_S:
                j = i + h // F.STEP_S
                ys[h] = z[j] if j < len(z) and t[j] <= dur else np.nan
            recs.append({"run_id": run_id, "scenario": g["scenario"].iloc[0], "group": g["group"].iloc[0],
                         "round": int(g["round"].iloc[0]),
                         "i": i, "t": t[i], "hist": pad[i:i + F.HIST_STEPS + 1],
                         **{f"y{h}": ys[h] for h in F.HORIZONS_S}})
    o = pd.DataFrame(recs)
    o["eval"] = (o["t"] >= MIN_HISTORY_S) & (o["i"] % EVAL_STRIDE == 0)
    return o


def _Z(o: pd.DataFrame) -> np.ndarray:
    return np.stack(o["hist"].to_numpy())


def _Y(o: pd.DataFrame) -> dict[int, np.ndarray]:
    return {h: o[f"y{h}"].to_numpy(float) for h in F.HORIZONS_S}


def pinball(y: np.ndarray, q: np.ndarray, tau: float) -> np.ndarray:
    d = y - q
    return np.maximum(tau * d, (tau - 1) * d)


def fit_lgbm_with_margin(o_train: pd.DataFrame) -> F.LgbmQuantileForecaster:
    """Fit on all training origins; p90 margin from grouped out-of-fold residuals."""
    Z, Y = _Z(o_train), _Y(o_train)
    groups = o_train["run_id"].to_numpy()
    oof = {h: np.full(len(o_train), np.nan) for h in F.HORIZONS_S}
    for tr, va in GroupKFold(n_splits=5).split(Z, groups=groups):
        m = F.LgbmQuantileForecaster().fit(Z[tr], {h: Y[h][tr] for h in F.HORIZONS_S})
        pz = m.predict_z(Z[va])
        for h in F.HORIZONS_S:
            oof[h][va] = pz[(h, 0.9)]
    model = F.LgbmQuantileForecaster().fit(Z, Y)
    for h in F.HORIZONS_S:
        r = Y[h] - oof[h]
        r = r[~np.isnan(r)]
        model.q90_margin[h] = float(max(0.0, np.quantile(r, 0.9)))
    return model


def fit_predict(tr: pd.DataFrame, te: pd.DataFrame, holt_cache: dict[int, dict[int, float]] | None,
                q90_extra: dict[int, float] | None = None) -> pd.DataFrame:
    """Fit every forecaster on origins ``tr``; long-format predictions for eval origins of ``te``.

    ``q90_extra`` (z units per horizon) is added to the LightGBM p90 conformal margin.
    """
    te = te[te["eval"]]
    tr_eval = tr[tr["eval"]]
    Zte = _Z(te)
    models: dict[str, dict] = {}
    for base in (F.PersistenceForecaster(), F.TrendForecaster()):
        base.fit(_Z(tr_eval), _Y(tr_eval))
        models[base.name] = base.predict(Zte)
    if holt_cache is not None:
        holt = F.HoltForecaster()
        ctr = {h: np.array([holt_cache[k][h] for k in tr_eval.index]) for h in F.HORIZONS_S}
        holt.rq.fit(ctr, _Y(tr_eval))
        cte = {h: np.array([holt_cache[k][h] for k in te.index]) for h in F.HORIZONS_S}
        models[holt.name] = holt.rq.apply(cte)
    lgbm = fit_lgbm_with_margin(tr[tr["t"] >= 30])
    for h, extra in (q90_extra or {}).items():
        lgbm.q90_margin[h] += extra
    models[lgbm.name] = lgbm.predict(Zte)
    preds = []
    for name, p in models.items():
        for h in F.HORIZONS_S:
            preds.append(pd.DataFrame({
                "run_id": te["run_id"].values, "scenario": te["scenario"].values,
                "group": te["group"].values, "round": te["round"].values,
                "t": te["t"].values, "method": name, "h": h,
                "now": F.from_z(Zte[:, -1]), "y": F.from_z(te[f"y{h}"].values),
                "q50": p[(h, 0.5)], "q90": p[(h, 0.9)]}))
    return pd.concat(preds, ignore_index=True)


def evaluate(o: pd.DataFrame, holt_cache: dict[int, dict[int, float]] | None) -> pd.DataFrame:
    """Leave-run-out over the shaped runs of ``o``. All origins kept (y may be NaN past the end
    of traffic); metrics drop NaN targets."""
    from joblib import Parallel, delayed
    test_runs = o.loc[o["group"].isin(SHAPED_GROUPS), "run_id"].unique()
    parts = Parallel(n_jobs=6)(delayed(fit_predict)(o[o["run_id"] != run], o[o["run_id"] == run], holt_cache)
                               for run in test_runs)
    return pd.concat(parts, ignore_index=True)


def summarize(P: pd.DataFrame) -> pd.DataFrame:
    def agg(g: pd.DataFrame) -> pd.Series:
        pb50 = pinball(g.y, g.q50, 0.5).mean()
        pb90 = pinball(g.y, g.q90, 0.9).mean()
        surge = g.y > 1.3 * g.now
        return pd.Series({
            "pinball50": pb50, "pinball90": pb90,
            "rel_pinball_mean_pct": 100 * (pb50 + pb90) / 2 / g.y.mean(),
            "mae50": (g.y - g.q50).abs().mean(),
            "cov90": (g.y <= g.q90).mean(),
            "cov90_surge": (g.y[surge] <= g.q90[surge]).mean() if surge.any() else np.nan,
            "n_surge": int(surge.sum()),
            "q90_over_y": (g.q90 / g.y.clip(lower=1)).median(),
            "n": len(g)})
    return P.groupby(["h", "method"]).apply(agg, include_groups=False).reset_index()


def outer_q90_correction(P: pd.DataFrame, level: float = 0.9) -> dict[int, float]:
    """Second conformal stage: extra p90 margin (z units) so leave-run-out coverage reaches ``level``.

    The inner margin comes from 5-fold grouped residuals of ~23 runs, which under-covers on a new
    run (LORO coverage 0.85-0.88). Residuals of the *outer* leave-run-out predictions are the
    honest calibration set for a never-seen run; the correction is checked on the round-2 holdout.
    """
    g = P[(P["method"] == "lgbm_quantile") & P["y"].notna()]
    out = {}
    for h in F.HORIZONS_S:
        x = g[g["h"] == h]
        r = np.log1p(x["y"]) - np.log1p(x["q90"])
        out[h] = float(max(0.0, np.quantile(r, level)))
    return out


FLAT_LEVELS = (4.0, 8.0, 10.8, 14.4, 20.0, 27.0, 40.0, 60.0)


def flat_probe(model: F.LgbmQuantileForecaster, levels: tuple[float, ...] = FLAT_LEVELS,
               seeds: int = 40, rate_window_s: float = 20.0) -> list[dict[str, float]]:
    """p90 / current rate on flat traffic: perfectly flat, and Poisson arrivals seen through a
    rate[20s] window every 5 s (what the dataset's ``rps`` is). Measures the forecaster's prior
    on traffic with no dynamics: a level-invariant, unbiased model would give a flat line."""
    rows = []
    for lam in levels:
        rec: dict[str, float] = {"level_rps": lam}
        flat = F.to_z(np.full((1, F.HIST_STEPS + 1), lam))
        p = model.predict(flat)
        rng = np.random.default_rng(0)
        k = int(rate_window_s / F.STEP_S)
        counts = rng.poisson(lam * F.STEP_S, size=(seeds, F.HIST_STEPS + k))
        rate = np.stack([np.convolve(c, np.ones(k), "valid") for c in counts]) / rate_window_s
        pn = model.predict(F.to_z(rate[:, -(F.HIST_STEPS + 1):]))
        for h in F.HORIZONS_S:
            rec[f"flat_p90_x_{h}"] = round(float(p[(h, 0.9)][0] / lam), 3)
            rec[f"poisson_p90_x_{h}"] = round(float(np.median(pn[(h, 0.9)] / lam)), 3)
            rec[f"poisson_p50_x_{h}"] = round(float(np.median(pn[(h, 0.5)] / lam)), 3)
        rows.append(rec)
    return rows


def summarize_groups(P: pd.DataFrame) -> pd.DataFrame:
    return P.groupby(["group", "h", "method"]).apply(
        lambda g: pd.Series({"rel_pinball_mean_pct": 100 * (pinball(g.y, g.q50, .5).mean()
                                                             + pinball(g.y, g.q90, .9).mean()) / 2 / g.y.mean(),
                             "cov90": (g.y <= g.q90).mean(), "n": len(g)}), include_groups=False).reset_index()


def holt_centers(o: pd.DataFrame) -> dict[int, dict[int, float]]:
    sel = o[o["eval"]]
    c = F.HoltForecaster().center(_Z(sel))
    return {idx: {h: float(c[h][k]) for h in F.HORIZONS_S} for k, idx in enumerate(sel.index)}


def plot(P: pd.DataFrame, S: pd.DataFrame, target: str, runs: list[str] | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = list(S["method"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    w = 0.8 / len(methods)
    for k, m in enumerate(methods):
        s = S[S.method == m]
        x = np.arange(len(F.HORIZONS_S)) + k * w
        axes[0].bar(x, s.rel_pinball_mean_pct, w, label=m)
        axes[1].bar(x, s.cov90, w, label=m)
    for ax, title in zip(axes, ["mean pinball (p50,p90) as % of mean demand", "p90 coverage (target 0.90)"]):
        ax.set_xticks(np.arange(len(F.HORIZONS_S)) + 0.4 - w / 2, [f"+{h}s" for h in F.HORIZONS_S])
        ax.set_title(title, fontsize=10)
    axes[1].axhline(0.9, color="k", lw=0.8, ls="--")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"forecast_scores_{target}.png", dpi=120)
    plt.close(fig)

    runs = runs or ["b-hpa-spike-b08", "b-hpa-saw-b10", "b-hpa-diurnal-s7", "c-under-fixed2-spike"]
    fig, axes = plt.subplots(len(runs), 1, figsize=(10, 2.6 * len(runs)), sharex=False)
    h = 120
    for ax, sc in zip(axes, runs):
        for m, c in (("persistence", "tab:gray"), ("lgbm_quantile", "tab:blue")):
            g = P[(P.scenario == sc) & (P.method == m) & (P.h == h)].sort_values("t")
            if g.empty:
                continue
            tt = g.t + h
            if m == "persistence":
                ax.plot(tt, g.y, color="k", lw=1.6, label=f"actual {target.removesuffix('_holdout')}")
            ax.plot(tt, g.q50, color=c, lw=1, label=f"{m} p50")
            ax.fill_between(tt, g.q50, g.q90, color=c, alpha=0.2, label=f"{m} p50-p90")
        ax.set_title(f"{sc}: forecast made at t for t+{h}s (plotted at target time)", fontsize=9)
        ax.set_ylabel("rps")
    axes[0].legend(fontsize=7, ncol=3)
    axes[-1].set_xlabel("seconds since start (target time)")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"forecast_examples_{target}.png", dpi=120)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="rps", choices=["rps", "goodput_rps"])
    ap.add_argument("--no-holt", action="store_true")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--final-all-rounds", action="store_true",
                    help="fit the shipped model on every round (default: TRAIN_ROUNDS only)")
    args = ap.parse_args()

    df = load_deploy()
    o_all = build_origins(df, args.target)
    o = o_all[o_all["round"].isin(TRAIN_ROUNDS)]
    o_hold = o_all[o_all["round"].isin(HOLDOUT_ROUNDS) & o_all["group"].isin(SHAPED_GROUPS)]
    holt = None if args.no_holt else holt_centers(pd.concat([o, o_hold]))
    pd.set_option("display.width", 200)
    out = MODELS_DIR / "forecast"
    out.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    P_all = evaluate(o, holt)
    P = P_all[~P_all["y"].isna()]
    S, S_group = summarize(P), summarize_groups(P)
    print(f"== leave-run-out, rounds {TRAIN_ROUNDS}")
    print(S.round(3).to_string(index=False))
    print(S_group.round(3).to_string(index=False))
    plot(P, S, args.target)
    P_all.to_csv(out / f"loro_predictions_{args.target}.csv.gz", index=False)

    extra = outer_q90_correction(P_all)
    print("outer conformal p90 correction (z units):", {h: round(v, 4) for h, v in extra.items()})
    metrics = {"train_rounds": list(TRAIN_ROUNDS), "overall": S.round(4).to_dict("records"),
               "by_group": S_group.round(4).to_dict("records"), "q90_outer_correction": extra}
    if len(o_hold):
        H_all = fit_predict(o, o_hold, holt, q90_extra=extra)
        H = H_all[~H_all["y"].isna()]
        SH, SH_group = summarize(H), summarize_groups(H)
        print(f"== holdout rounds {HOLDOUT_ROUNDS} (fit on {TRAIN_ROUNDS}): runs {sorted(o_hold.scenario.unique())}")
        print(SH.round(3).to_string(index=False))
        print(SH_group.round(3).to_string(index=False))
        plot(H, SH, f"{args.target}_holdout", runs=sorted(o_hold.scenario.unique())[:4])
        H_all.to_csv(out / f"holdout_predictions_{args.target}.csv.gz", index=False)
        metrics |= {"holdout_rounds": list(HOLDOUT_ROUNDS), "holdout_runs": sorted(o_hold.run_id.unique()),
                    "holdout_overall": SH.round(4).to_dict("records"),
                    "holdout_by_group": SH_group.round(4).to_dict("records")}
    (out / f"loro_metrics_{args.target}.json").write_text(json.dumps(metrics, indent=1))

    if not args.no_save and args.target == "rps":
        fit_o = o_all if args.final_all_rounds else o
        model = fit_lgbm_with_margin(fit_o[fit_o["t"] >= 30])
        for h, v in extra.items():
            model.q90_margin[h] += v
        model.save(out, info={"rounds": sorted(int(r) for r in fit_o["round"].unique()),
                              "runs": int(fit_o["run_id"].nunique()), "q90_outer_correction": extra})
        print("saved", out, "rounds", sorted(fit_o["round"].unique()), "q90 margins", model.q90_margin)


if __name__ == "__main__":
    main()
