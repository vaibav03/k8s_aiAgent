"""Calibrate + validate the replay simulator, then compare autoscaling policies offline.

    python -m kubeai_ml.scaling.eval_policies            # everything (~1 min)

Steps
1. fit the pod model on dataset_deploy (cpu/request, per-pod ceiling, latency curve, startup);
2. fit the metrics-server lag/window (the only HPA timing knobs) on 4 round-1 B runs, test on
   the other 4;
3. validate: replay HPA(cpu 70 %) on the 8 round-1 B runs (default 300 s scale-down window) and
   on the round-2 B runs (30 s scale-down window: the only runs with scale-in), and compare
   simulated vs actual replicas (event-exact and per 5 s row), rescale timing, SLO-violation
   minutes, pod-minutes, p95 and CPU; replay fixed-N on the A/C runs (pod model alone, no
   controller). Round-2 runs are a pure hold-out: nothing is fitted on them (``--fit-rounds``);
4. compare policies on recorded traces (B, C, E load) and longer synthetic traces, with the lab's
   max of 4 replicas and an uncapped max of 8 (pod model extrapolated beyond 4).

Writes models/scaling/sim_calibration.json, kubeai_ml/scaling/sim_results/*.csv|json and
kubeai_ml/scaling/plots/sim_*.png.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .data import MODELS_DIR, PKG_DIR, PLOTS_DIR, load_deploy
from .hpa import HPAPolicy
from .podmodel import PodModel, fit_pod_model
from .data import LAB_DATA
from .runlogs import hpa_events, rescales_from_rows
from .sim_policies import default_policies
from .simulator import FixedPolicy, SimConfig, simulate, summarize, summarize_sim, violation_mask
from .traces import Trace, run_trace, synthetic_trace

RESULTS_DIR = PKG_DIR / "sim_results"
TRAIN_B = ("b-hpa-diurnal-s7", "b-hpa-spike-b08", "b-hpa-flash-b06", "b-hpa-saw-b10")
LAG_GRID = (0.0, 5.0, 10.0, 15.0, 20.0, 25.0)
WIN_GRID = (5.0, 10.0, 15.0, 20.0, 30.0)


# ------------------------------------------------------------------------------------------
# helpers


def _step_fn(events: list[tuple[float, int]], horizon_s: float, init: int) -> np.ndarray:
    """Replica count per second from (time, new size) events."""
    t = np.arange(0, int(horizon_s) + 1)
    r = np.full(t.shape, float(init))
    for te, n in sorted(events):
        r[t >= te] = n
    return r


def _first_reach(events: list[tuple[float, int]], n: int) -> float | None:
    for te, k in sorted(events):
        if k >= n:
            return te
    return None


@dataclass
class BRun:
    """One recorded HPA run with the HPA spec it ran under."""
    rid: str
    rows: pd.DataFrame
    trace: Trace
    round: int
    scale_down_window_s: float
    initial: int
    min_replicas: int
    max_replicas: int

    def __iter__(self):            # (rid, rows, trace) unpacking, kept for older callers
        return iter((self.rid, self.rows, self.trace))


def _scaling_specs() -> dict[str, dict]:
    idx = pd.read_csv(LAB_DATA / "runs_index.csv")
    return {r.run_id: {**json.loads(r.scaling), "round": int(r["round"]) if "round" in idx else 1}
            for _, r in idx.iterrows()}


def b_runs(df: pd.DataFrame, rounds: tuple[int, ...] = (1, 2)) -> list[BRun]:
    """HPA runs of group B (no faults, start at the HPA min)."""
    specs = _scaling_specs()
    out = []
    for rid, g in df[df["group"] == "B"].groupby("run_id"):
        sp = specs[rid]
        if sp["round"] not in rounds:
            continue
        out.append(BRun(rid, g.sort_values("ts"), run_trace(rid, g), sp["round"],
                        float(sp.get("scale_down_window_s", 300)), int(sp.get("initial", sp["min"])),
                        int(sp["min"]), int(sp["max"])))
    return sorted(out, key=lambda b: (b.round, b.trace.name))


def _match_events(actual: list[tuple[float, int]], sim: list[tuple[float, int]],
                  max_dt_s: float = 120.0) -> tuple[int, int, float, list[float]]:
    """Greedy in-order matching of rescale events with the same new size within ``max_dt_s``.
    Returns (matched, unmatched actual + unmatched sim, mean |dt| of matches, |dt| list)."""
    used: set[int] = set()
    dts: list[float] = []
    for ta, na in actual:
        best, bj = None, None
        for j, (ts, ns) in enumerate(sim):
            if j in used or ns != na or abs(ts - ta) > max_dt_s:
                continue
            if best is None or abs(ts - ta) < best:
                best, bj = abs(ts - ta), j
        if bj is not None:
            used.add(bj)
            dts.append(best)
    unmatched = (len(actual) - len(dts)) + (len(sim) - len(used))
    return len(dts), unmatched, float(np.mean(dts)) if dts else float("nan"), dts


# ------------------------------------------------------------------------------------------
# validation


def validate_hpa_run(run: BRun, model: PodModel, cfg: SimConfig,
                     phase_s: float | None = None) -> dict:
    rid, rows, tr = run.rid, run.rows, run.trace
    ev = hpa_events(rid)
    phase = ev.phase_s if phase_s is None else phase_s
    cfg = replace(cfg, initial_replicas=run.initial, min_replicas=run.min_replicas,
                  max_replicas=run.max_replicas)
    res = simulate(tr, HPAPolicy(0.70, run.scale_down_window_s, phase_s=phase), model, cfg)
    sim_ev = [(t, b) for t, _, b in res.actions]
    act_ev = rescales_from_rows(rows, ev)        # complete even when k8s deduplicated events
    horizon = float(tr.t[-1])
    act_s, sim_s = _step_fn(act_ev, horizon, run.initial), _step_fn(sim_ev, horizon, run.initial)
    timing, signed = [], []
    for _, n in act_ev:
        if n <= run.initial:
            continue
        a, s = _first_reach(act_ev, n), _first_reach(sim_ev, n)
        timing.append(abs(a - s) if (a is not None and s is not None) else np.nan)
        if a is not None and s is not None:
            signed.append(s - a)                 # < 0: the simulated HPA scales up early
    n_match, n_unmatched, dt_match, _ = _match_events(act_ev, sim_ev)
    a_down = [(t, n) for (t, n), prev in zip(act_ev, [run.initial] + [k for _, k in act_ev]) if n < prev]
    s_down = [(t, b) for t, a0, b in res.actions if b < a0]
    _, down_unmatched, down_dt, _ = _match_events(a_down, s_down)
    a_sum = summarize(rows, tr.rps)
    s_sum = summarize_sim(res)
    sim_rep = res.df["true_spec"].to_numpy()
    act_rep = rows["replicas_desired"].to_numpy()
    p95a, p95s = rows["p95"].to_numpy(float), res.df["p95"].to_numpy(float)
    ok = np.isfinite(p95a)
    return {
        "scenario": tr.name, "phase_s": round(phase, 1),
        "round": run.round, "sd_window_s": run.scale_down_window_s,
        "rescales_actual": " ".join(f"{t:.0f}s->{n}" for t, n in act_ev),
        "rescales_sim": " ".join(f"{t:.0f}s->{n}" for t, n in sim_ev),
        "rep_mae_1s": float(np.abs(act_s - sim_s).mean()),
        "rep_mae_rows": float(np.abs(act_rep - sim_rep).mean()),
        "rep_exact_rows_pct": float(100 * np.mean(act_rep == sim_rep)),
        "rescale_timing_err_s": float(np.nanmean(timing)) if timing else 0.0,
        "rescale_timing_signed_s": float(np.mean(signed)) if signed else 0.0,
        "missed_or_extra_levels": int(sum(np.isnan(timing))) + abs(len(set(n for _, n in sim_ev))
                                                                    - len(set(n for _, n in act_ev))),
        "rescales_n_actual": len(act_ev), "rescales_n_sim": len(sim_ev),
        "rescales_matched": n_match, "rescales_unmatched": n_unmatched,
        "rescale_match_dt_s": dt_match,
        "scale_downs_actual": len(a_down), "scale_downs_sim": len(s_down),
        "scale_down_unmatched": down_unmatched, "scale_down_dt_s": down_dt,
        "final_rep_actual": int(act_rep[-1]), "final_rep_sim": int(sim_rep[-1]),
        "viol_min_actual": a_sum["slo_violation_min"], "viol_min_sim": s_sum["slo_violation_min"],
        "pod_min_actual": a_sum["pod_minutes"], "pod_min_sim": s_sum["pod_minutes"],
        "viol_rows_agree_pct": float(100 * np.mean(violation_mask(rows) == violation_mask(res.df))),
        "p95_log_mae": float(np.mean(np.abs(np.log(p95a[ok]) - np.log(p95s[ok])))),
        "cpu_util_mae": float(np.mean(np.abs(rows["cpu_util_vs_request"].to_numpy() -
                                             res.df["cpu_util_vs_request"].to_numpy()))),
        "_res": res,
    }


def fit_ms_timing(runs: list[BRun], model: PodModel) -> tuple[SimConfig, SimConfig, pd.DataFrame]:
    """Grid-fit metrics-server lag/window (replica MAE, 1 s resolution).

    Returns (cfg fitted on the 4 round-1 TRAIN_B runs only -- used for the strict hold-out
    numbers --, cfg fitted on every B run of every round -- the shipped calibration --, grid)."""
    rows = []
    for L in LAG_GRID:
        for W in WIN_GRID:
            cfg = SimConfig(ms_lag_s=L, ms_window_s=W)
            m = {r.trace.name: (r.round, validate_hpa_run(r, model, cfg)["rep_mae_1s"]) for r in runs}
            rows.append({"ms_lag_s": L, "ms_window_s": W,
                         "train_rep_mae": float(np.mean([v for k, (_, v) in m.items() if k in TRAIN_B])),
                         "r1_rep_mae": float(np.mean([v for _, (rd, v) in m.items() if rd == 1])),
                         "r2_rep_mae": float(np.mean([v for _, (rd, v) in m.items() if rd == 2]))
                         if any(rd == 2 for rd, _ in m.values()) else np.nan,
                         "all_rep_mae": float(np.mean([v for _, v in m.values()]))})
    grid = pd.DataFrame(rows)

    def best(col: str) -> SimConfig:
        b = grid.sort_values([col, "ms_lag_s"]).iloc[0]
        return SimConfig(ms_lag_s=float(b.ms_lag_s), ms_window_s=float(b.ms_window_s))

    return best("train_rep_mae"), best("all_rep_mae"), grid.sort_values("all_rep_mae")


def validate_fixed(df: pd.DataFrame, model: PodModel) -> pd.DataFrame:
    """Fixed-N replay of the A (capacity) and C (wrong size) runs: pod model only."""
    out = []
    for rid, g in df[df["group"].isin(["A", "C"])].groupby("run_id"):
        g = g.sort_values("ts")
        n = int(g["replicas_desired"].median())
        tr = run_trace(rid, g)
        res = simulate(tr, FixedPolicy(n), model, SimConfig(initial_replicas=n, min_replicas=n,
                                                            max_replicas=n))
        a, s = summarize(g, tr.rps), summarize_sim(res)
        p95a, p95s = g["p95"].to_numpy(float), res.df["p95"].to_numpy(float)
        ok = np.isfinite(p95a)
        ga, gs = g["goodput_rps"].to_numpy(float), res.df["goodput_rps"].to_numpy(float)
        out.append({"scenario": tr.name, "round": int(g["round"].iloc[0]), "replicas": n,
                    "viol_min_actual": a["slo_violation_min"], "viol_min_sim": s["slo_violation_min"],
                    "viol_rows_agree_pct": float(100 * np.mean(violation_mask(g) == violation_mask(res.df))),
                    "p95_log_mae": float(np.mean(np.abs(np.log(p95a[ok]) - np.log(p95s[ok])))),
                    "goodput_mape": float(np.mean(np.abs(ga - gs)[ga > 1] / ga[ga > 1])),
                    "err_rate_mae": float(np.mean(np.abs(g["error_rate"].to_numpy() -
                                                         res.df["error_rate"].to_numpy())))})
    return pd.DataFrame(out)


# ------------------------------------------------------------------------------------------
# comparison


SYNTHETIC = [
    ("syn-diurnal-30m", "diurnal", 27.0, 1800, {"period_s": 900, "amplitude": 0.6, "seed": 7}),
    ("syn-spike-then-quiet", "spike", 9.0, 1500, {"spike_at_s": 300, "spike_mult": 5, "spike_s": 120}),
    ("syn-flash-4x", "flash_crowd", 10.0, 900, {"at_s": 300, "mult": 4}),
    ("syn-saw-4m", "sawtooth", 20.0, 1200, {"period_s": 240}),
]


def comparison_traces(df: pd.DataFrame) -> list[Trace]:
    trs = []
    for rid, g in df[df["group"].isin(["B", "C", "E"])].groupby("run_id"):
        trs.append(run_trace(rid, g))
    trs += [synthetic_trace(*s) for s in SYNTHETIC]
    return trs


def compare(traces: list[Trace], model: PodModel, cfg: SimConfig, max_replicas: int) -> pd.DataFrame:
    rows = []
    for tr in traces:
        for pol in default_policies(model):
            init = pol.n if isinstance(pol, FixedPolicy) else 1
            if isinstance(pol, FixedPolicy) and pol.n > max_replicas:
                continue
            c = replace(cfg, max_replicas=max_replicas, initial_replicas=init)
            rows.append({**summarize_sim(simulate(tr, pol, model, c)), "max_cap": max_replicas})
    return pd.DataFrame(rows)


def compare_table(cmp: pd.DataFrame) -> pd.DataFrame:
    agg = cmp.groupby("policy").agg(
        traces=("trace", "size"), slo_violation_min=("slo_violation_min", "sum"),
        slo_violation_min_true=("slo_violation_min_true", "sum"), pod_minutes=("pod_minutes", "sum"),
        scaling_actions=("scaling_actions", "sum"), max_replicas=("max_replicas", "max"),
        ttr_after_spike_max_s=("ttr_after_spike_max_s", "max"),
        ttr_after_spike_mean_s=("ttr_after_spike_mean_s", "mean"), dropped_pct=("dropped_pct", "mean"))
    return agg.sort_values(["slo_violation_min", "pod_minutes"]).round(2)


# ------------------------------------------------------------------------------------------
# plots


def plot_validation(vals: list[dict], runs) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nrow = (len(vals) + 1) // 2
    fig, axes = plt.subplots(nrow, 2, figsize=(14, 3.2 * nrow + 0.6), sharex=False, squeeze=False)
    for ax in axes.ravel()[len(vals):]:
        ax.set_visible(False)
    for ax, v, (rid, g, tr) in zip(axes.ravel(), vals, runs):
        res = v["_res"]
        t = g["seconds_since_start"].to_numpy()
        ax.step(t, g["replicas_desired"], where="post", color="k", lw=2, label="actual replicas")
        ax.step(res.df["seconds_since_start"], res.df["true_spec"], where="post", color="C1", lw=1.5,
                ls="--", label="sim replicas")
        ax.set_ylim(0, 5)
        ax.set_ylabel("replicas")
        ax2 = ax.twinx()
        ax2.plot(t, g["p95"], color="C0", alpha=.6, label="actual p95")
        ax2.plot(res.df["seconds_since_start"], res.df["p95"], color="C3", alpha=.6, label="sim p95")
        ax2.plot(t, tr.rps / 60, color="0.6", lw=.8, label="offered RPS / 60")
        ax2.axhline(0.3, color="r", lw=.5, ls=":")
        ax2.set_yscale("log")
        ax2.set_ylim(0.02, 12)
        ax.set_title(f"{tr.name} (sd {v['sd_window_s']:.0f}s): rep MAE {v['rep_mae_1s']:.2f}, "
                     f"viol {v['viol_min_actual']:.2f}"
                     f"/{v['viol_min_sim']:.2f} min (act/sim)", fontsize=9)
    h1, l1 = axes[0, 0].get_legend_handles_labels()
    h2, l2 = axes[0, 0].get_figure().axes[-1].get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc="lower center", ncol=5)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(PLOTS_DIR / "sim_validation_hpa_B.png", dpi=110)
    plt.close(fig)


def plot_tradeoff(tab4: pd.DataFrame, tab8: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, tab, cap in zip(axes, (tab4, tab8), (4, 8)):
        ax.scatter(tab["pod_minutes"], tab["slo_violation_min"])
        for name, r in tab.iterrows():
            ax.annotate(name, (r["pod_minutes"], r["slo_violation_min"]), fontsize=7)
        ax.set_xlabel("pod-minutes (all traces)")
        ax.set_ylabel("SLO-violation minutes (all traces)")
        ax.set_title(f"policy trade-off, max replicas {cap}")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "sim_policy_tradeoff.png", dpi=110)
    plt.close(fig)


# ------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--fit-rounds", default="1",
                    help="dataset rounds used to fit the pod model (default 1: round 2 is held out)")
    args = ap.parse_args(argv)
    fit_rounds = tuple(int(x) for x in args.fit_rounds.split(","))
    RESULTS_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_deploy()
    model, fit_report = fit_pod_model(df[df["round"].isin(fit_rounds)])
    all_runs = b_runs(df)
    runs = [r for r in all_runs if r.round == 1]          # round-1 B: sd 300 s, train/test split
    runs_r2 = [r for r in all_runs if r.round == 2]       # round-2 B: sd 30 s, scale-in, hold-out
    cfg_strict, cfg, grid = fit_ms_timing(runs + runs_r2, model)
    grid.to_csv(RESULTS_DIR / "ms_timing_grid.csv", index=False)
    (MODELS_DIR / "sim_calibration.json").write_text(json.dumps(
        {"pod_model": json.loads(model.to_json()), "ms_lag_s": cfg.ms_lag_s,
         "ms_window_s": cfg.ms_window_s, "ms_timing_fitted_on": "all B runs (rounds 1+2)",
         "ms_timing_strict": {"ms_lag_s": cfg_strict.ms_lag_s, "ms_window_s": cfg_strict.ms_window_s,
                              "fitted_on": list(TRAIN_B)},
         "fit_rounds": list(fit_rounds), "fit_report": fit_report}, indent=1, default=float))

    # 0) strict protocol: timing fitted on the 4 round-1 train runs, pod model on round 1 only;
    #    round-1 test runs and every round-2 run are held out.
    strict = pd.DataFrame([{k: v for k, v in validate_hpa_run(r, model, cfg_strict).items() if k != "_res"}
                           for r in runs + runs_r2])
    strict["split"] = np.select([strict["scenario"].isin(TRAIN_B), strict["round"] == 1],
                                ["train", "test"], "holdout_r2_sd30")
    strict.to_csv(RESULTS_DIR / "validation_hpa_B_strict.csv", index=False)

    # 1) HPA replay on B runs, events-derived sync phase
    vals = [validate_hpa_run(r, model, cfg) for r in runs + runs_r2]
    val_all = pd.DataFrame([{k: v for k, v in x.items() if k != "_res"} for x in vals])
    val_all["split"] = np.select([val_all["scenario"].isin(TRAIN_B), val_all["round"] == 1],
                                 ["r1_train", "r1_test"], "r2_sd30")   # labels only: timing fitted on all
    val_all.to_csv(RESULTS_DIR / "validation_hpa_B.csv", index=False)
    val, val_r2 = val_all[val_all["round"] == 1], val_all[val_all["round"] == 2]
    # 2) same, phase unknown: mean over phases 0/5/10 s
    ph = []
    for r in runs:
        vs = [validate_hpa_run(r, model, cfg, phase_s=p) for p in (0.0, 5.0, 10.0)]
        ph.append({"scenario": r.trace.name, "rep_mae_1s": np.mean([v["rep_mae_1s"] for v in vs]),
                   "viol_min_sim": np.mean([v["viol_min_sim"] for v in vs]),
                   "viol_min_actual": vs[0]["viol_min_actual"]})
    val_ph = pd.DataFrame(ph)
    # 3) first-principles timing (15 s lag, 15 s window) for reference
    v15 = pd.DataFrame([{k: v for k, v in validate_hpa_run(r, model, SimConfig()).items()
                         if k != "_res"} for r in runs])
    fixed = validate_fixed(df, model)
    fixed.to_csv(RESULTS_DIR / "validation_fixed_AC.csv", index=False)

    def agg(v: pd.DataFrame) -> dict:
        return {"rep_mae_1s": round(float(v["rep_mae_1s"].mean()), 3),
                "rep_mae_rows": round(float(v["rep_mae_rows"].mean()), 3) if "rep_mae_rows" in v else None,
                "rep_exact_rows_pct": round(float(v["rep_exact_rows_pct"].mean()), 1) if "rep_exact_rows_pct" in v else None,
                "rescale_timing_err_s": round(float(v["rescale_timing_err_s"].mean()), 1) if "rescale_timing_err_s" in v else None,
                "viol_min_actual": round(float(v["viol_min_actual"].sum()), 3),
                "viol_min_sim": round(float(v["viol_min_sim"].sum()), 3),
                "viol_min_abs_err_per_run": round(float((v["viol_min_sim"] - v["viol_min_actual"]).abs().mean()), 3),
                "pod_min_err_pct": round(float(100 * ((v["pod_min_sim"] - v["pod_min_actual"]).abs()
                                                      / v["pod_min_actual"]).mean()), 2) if "pod_min_sim" in v else None,
                "p95_log_mae": round(float(v["p95_log_mae"].mean()), 3) if "p95_log_mae" in v else None,
                "cpu_util_mae": round(float(v["cpu_util_mae"].mean()), 3) if "cpu_util_mae" in v else None,
                **({"rescales_actual": int(v["rescales_n_actual"].sum()),
                    "rescales_sim": int(v["rescales_n_sim"].sum()),
                    "rescales_matched": int(v["rescales_matched"].sum()),
                    "rescale_match_dt_s": round(float(np.nanmean(v["rescale_match_dt_s"])), 1),
                    "scale_downs_actual": int(v["scale_downs_actual"].sum()),
                    "scale_downs_sim": int(v["scale_downs_sim"].sum()),
                    "scale_down_unmatched": int(v["scale_down_unmatched"].sum()),
                    "scale_down_dt_s": (round(float(np.nanmean(v["scale_down_dt_s"])), 1)
                                        if v["scale_down_dt_s"].notna().any() else None)}
                   if "rescales_n_actual" in v else {})}

    def agg_fixed(f: pd.DataFrame) -> dict:
        return {"runs": len(f), "viol_min_actual": round(float(f["viol_min_actual"].sum()), 3),
                "viol_min_sim": round(float(f["viol_min_sim"].sum()), 3),
                "viol_rows_agree_pct": round(float(f["viol_rows_agree_pct"].mean()), 1),
                "p95_log_mae": round(float(f["p95_log_mae"].mean()), 3),
                "goodput_mape": round(float(f["goodput_mape"].mean()), 3)}

    summary = {
        "ms_timing": {"ms_lag_s": cfg.ms_lag_s, "ms_window_s": cfg.ms_window_s,
                      "fitted_on": "all B runs, rounds 1+2"},
        "strict_holdout": {
            "ms_timing": {"ms_lag_s": cfg_strict.ms_lag_s, "ms_window_s": cfg_strict.ms_window_s,
                          "fitted_on": list(TRAIN_B)},
            "hpa_B_r1_train": agg(strict[strict.split == "train"]),
            "hpa_B_r1_test": agg(strict[strict.split == "test"]),
            "hpa_B_r2_sd30": agg(strict[strict.split == "holdout_r2_sd30"]) if (strict["round"] == 2).any() else None},
        "pod_model": {"cpu_s_per_req": model.cpu_s_per_req, "mu_by_n": model.mu_by_n,
                      "startup_delay_s": model.startup_delay_s},
        "hpa_B_all": agg(val), "hpa_B_train": agg(val[val.split == "r1_train"]),
        "hpa_B_test": agg(val[val.split == "r1_test"]),
        "hpa_B_phase_unknown": agg(val_ph), "hpa_B_first_principles_15_15": agg(v15),
        "hpa_B_r2_sd30": agg(val_r2) if len(val_r2) else None,
        "fixed_AC": agg_fixed(fixed[fixed["round"] == 1]),
        "fixed_AC_r2_holdout": agg_fixed(fixed[fixed["round"] == 2]) if (fixed["round"] == 2).any() else None,
        "fit_rounds": list(fit_rounds),
    }

    # 4) policy comparison
    traces = comparison_traces(df)
    cmp4, cmp8 = compare(traces, model, cfg, 4), compare(traces, model, cfg, 8)
    pd.concat([cmp4, cmp8]).to_csv(RESULTS_DIR / "policy_comparison_per_trace.csv", index=False)
    tab4, tab8 = compare_table(cmp4), compare_table(cmp8)
    tab4.to_csv(RESULTS_DIR / "policy_comparison_max4.csv")
    tab8.to_csv(RESULTS_DIR / "policy_comparison_max8.csv")
    summary["n_comparison_traces"] = len(traces)
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=1, default=float))

    if not args.no_plots:
        plot_validation(vals, runs + runs_r2)
        plot_tradeoff(tab4, tab8)

    pd.set_option("display.width", 220)
    print(val_all.drop(columns=["rescales_actual", "rescales_sim"]).round(3).to_string())
    print(json.dumps(summary, indent=1, default=float))
    print(fixed.round(3).to_string())
    print("\n== policies, max 4 ==\n", tab4.to_string())
    print("\n== policies, max 8 ==\n", tab8.to_string())
    return summary


if __name__ == "__main__":
    main()
