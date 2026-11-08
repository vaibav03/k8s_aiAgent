"""Closed-loop policy evaluation in the replay simulator, with the shipped DecisionEngine as the
PREDICTIVE policy (forecast q90 + learned capacity + classifier gate, exactly the serving code).

Policies compared on every traffic-shaped run (development and held-out runs reported apart) and
on four longer synthetic shapes:

* ``fixed-2/3/4``             fixed replica counts;
* ``hpa-70-sd300``            stock HPA, CPU 70 % of request, default 300 s scale-down window;
* ``hpa-70-sd30``             the same with a 30 s scale-down window;
* ``hpa-tuned-<t>-sd30``      CPU target raised to the learned safe rate (``sim_policies.tuned_hpa``);
* ``hpa-50/35/25-sd300``      lower CPU targets: candidates for a cost-matched baseline;
* ``predictive``              :class:`EnginePolicy` around :class:`engine.DecisionEngine`.

Besides violation minutes and pod-minutes every run reports ``violation_events`` (independent
breach episodes) and ``pod_min_over_need`` (pods running above the perfect-foresight oracle
need of the pod model). :func:`cost_matched` puts PREDICTIVE next to the baseline whose
pod-minutes are closest to its own. Flat and noisy-flat synthetic traces, and held-out traces
with 600 s of warm-up at their base load (``warm_start``), check that the engine does not buy
"zero violations" by parking at the replica cap (review 3).

Faults are not simulated (traces carry traffic only); the gate is judged on recorded fault runs
(:mod:`.fault_eval`). Here we count how often the gate fired anyway (``gate_false_fire_pct``).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd

from . import states as S
from .bundle import PolicyParams
from .engine import DecisionEngine
from .hpa import HPAPolicy
from .podmodel import PodModel
from .sim_policies import tuned_hpa
from .simulator import FixedPolicy, Policy, SimConfig, SimState, simulate, summarize_sim
from .traces import Trace, synthetic_trace

SYNTHETIC = [
    ("syn-diurnal-30m", "diurnal", 27.0, 1800, {"period_s": 900, "amplitude": 0.6, "seed": 7}),
    ("syn-spike-then-quiet", "spike", 9.0, 1500, {"spike_at_s": 300, "spike_mult": 5, "spike_s": 120}),
    ("syn-flash-4x", "flash_crowd", 10.0, 900, {"at_s": 300, "mult": 4}),
    ("syn-saw-4m", "sawtooth", 20.0, 1200, {"period_s": 240}),
    ("syn-flat-10.8", "steady", 10.8, 900, {"ramp_s": 10}),
    ("syn-flat-40", "steady", 40.0, 900, {"ramp_s": 10}),
    ("syn-noisy-flat-30", "diurnal", 30.0, 1800, {"amplitude": 0.0, "noise": 0.35, "point_s": 10, "seed": 11}),
]
WARM_S = 600.0


@dataclass
class EnginePolicy:
    """The serving DecisionEngine as a simulator policy (history -> deployment_obs dicts)."""
    engine: DecisionEngine
    name: str = "predictive"
    period_s: float = 15.0
    phase_s: float = 0.0
    states: list[str] = field(default_factory=list)

    mode: str | None = None              # engine policy_mode for this run (None = the engine's)

    def reset(self) -> None:
        self.engine.reset()
        self.states = []

    def __call__(self, history: pd.DataFrame, state: SimState) -> int:
        h = history.assign(ts=history["seconds_since_start"])
        if self.mode is not None:
            self.engine.policy_mode = self.mode
        out = self.engine.decide(h, None, state.spec_replicas, state.min_replicas, state.max_replicas)
        self.states.append(out["state"])
        return int(out["desired_replicas"])


def synthetic_traces() -> list[Trace]:
    return [synthetic_trace(*s) for s in SYNTHETIC]


def warm_start(trace: Trace, pre_s: float = WARM_S) -> Trace:
    """``trace`` preceded by ``pre_s`` of its base load (median offered rate over t in [30, 90] s),
    so a policy starts from a steady state instead of an empty history at t = 0."""
    t, r = np.asarray(trace.t, float), np.asarray(trace.rps, float)
    sel = r[(t >= 30) & (t <= 90)]
    base = float(np.median(sel)) if len(sel) else float(r[r > 0][0])
    n_pre = int(round(pre_s / trace.step_s))
    head = trace.step_s * np.arange(1, n_pre + 1)
    rps = np.concatenate([np.full(n_pre, base), np.where((t < 30) & (r < base), base, r)])
    return Trace(name=f"{trace.name}+warm{int(pre_s)}", t=np.concatenate([head, t + pre_s]), rps=rps,
                 step_s=trace.step_s, meta={**(trace.meta or {}), "warm_start_s": pre_s, "base_rps": base})


def policies(model: PodModel, engine: DecisionEngine) -> list[Policy]:
    return [FixedPolicy(2), FixedPolicy(3), FixedPolicy(4),
            HPAPolicy(0.70, 300.0), HPAPolicy(0.70, 30.0), tuned_hpa(model, 30.0),
            HPAPolicy(0.50, 300.0), HPAPolicy(0.35, 300.0), HPAPolicy(0.25, 300.0),
            EnginePolicy(engine, mode="predictive")]


def run_one(trace: Trace, pol: Policy, model: PodModel, cfg: SimConfig, max_replicas: int) -> dict[str, Any]:
    init = pol.n if isinstance(pol, FixedPolicy) else 1
    c = replace(cfg, max_replicas=max(max_replicas, init), min_replicas=1, initial_replicas=init)
    res = simulate(trace, pol, model, c)
    row = {**summarize_sim(res, model=model), "max_cap": max_replicas}
    if isinstance(pol, EnginePolicy):
        st = pol.states
        row["gate_false_fire_pct"] = round(100 * float(np.mean([s not in S.LOAD_STATES for s in st])), 2) if st else 0.0
    return row


def _trace_rows(trace: Trace, model: PodModel, cfg: SimConfig, bundle_dir: str, params: PolicyParams,
                caps: tuple[int, ...], split: str, only_engine: bool) -> list[dict[str, Any]]:
    engine = DecisionEngine(bundle_dir, oom_model_dir=None)
    set_policy(engine, params)
    rows = []
    for cap in caps:
        pols = [EnginePolicy(engine, mode="predictive")] if only_engine else policies(model, engine)
        for pol in pols:
            if isinstance(pol, FixedPolicy) and pol.n > cap:
                continue
            rows.append({**run_one(trace, pol, model, cfg, cap), "split": split})
    return rows


def compare(traces: list[Trace], model: PodModel, cfg: SimConfig, bundle_dir: str, params: PolicyParams,
            caps: tuple[int, ...] = (4, 8), split: str = "", only_engine: bool = False,
            n_jobs: int = 6) -> pd.DataFrame:
    """Every policy on every trace (parallel over traces; each worker loads the bundle)."""
    from joblib import Parallel, delayed
    parts = Parallel(n_jobs=n_jobs)(delayed(_trace_rows)(tr, model, cfg, str(bundle_dir), params, caps, split,
                                                         only_engine) for tr in traces)
    return pd.DataFrame([r for p in parts for r in p])


def table(cmp: pd.DataFrame) -> pd.DataFrame:
    agg = cmp.groupby(["max_cap", "policy"]).agg(
        traces=("trace", "size"), slo_violation_min=("slo_violation_min", "sum"),
        violation_events=("violation_events", "sum"),
        pod_minutes=("pod_minutes", "sum"), pod_min_over_need=("pod_min_over_need", "sum"),
        oracle_pod_min=("oracle_pod_min", "sum"), scaling_actions=("scaling_actions", "sum"),
        max_replicas=("max_replicas", "max"), ttr_after_spike_max_s=("ttr_after_spike_max_s", "max"),
        dropped_pct=("dropped_pct", "mean"))
    if "gate_false_fire_pct" in cmp:
        agg["gate_false_fire_pct"] = cmp.groupby(["max_cap", "policy"])["gate_false_fire_pct"].mean()
    return agg.reset_index().sort_values(["max_cap", "slo_violation_min", "pod_minutes"]).round(2)


def cost_matched(tab: pd.DataFrame) -> list[dict[str, Any]]:
    """Per cap: PREDICTIVE next to the non-predictive policy with the closest pod-minutes."""
    out = []
    for cap, g in tab.groupby("max_cap"):
        pr = g[g["policy"] == "predictive"]
        if pr.empty:
            continue
        p = pr.iloc[0]
        others = g[g["policy"] != "predictive"].assign(gap=lambda x: (x["pod_minutes"] - p["pod_minutes"]).abs())
        if others.empty:
            continue
        o = others.sort_values("gap").iloc[0]
        out.append({"max_cap": int(cap), "predictive_viol_min": float(p["slo_violation_min"]),
                    "predictive_pod_min": float(p["pod_minutes"]), "baseline": str(o["policy"]),
                    "baseline_viol_min": float(o["slo_violation_min"]), "baseline_pod_min": float(o["pod_minutes"])})
    return out


# Scale-down windows start at 180 s: a 60 s window flapped on noisy traffic (review 3).
# The down-veto horizons (60+120 vs 60+120+300) changed nothing once the forecast needs
# measured evidence, so that dimension became an explicit scale-up headroom.
SWEEP = [(r, h, hr, w) for r in (0.10, 0.20, 0.30) for h in ((60,), (60, 120))
         for hr in (1.0, 1.3) for w in (180.0, 300.0)]


def sweep(traces: list[Trace], model: PodModel, cfg: SimConfig, bundle_dir: str, base: PolicyParams,
          cap: int = 8, n_jobs: int = 6) -> tuple[pd.DataFrame, PolicyParams]:
    """Tune risk, scale-up horizons, scale-up headroom and the scale-down window on
    development traces only (+ the synthetic shapes, incl. flat and noisy-flat traffic).

    Rule: fewest pod-minutes among settings whose SLO-violation minutes do not exceed the
    stock-HPA-with-30 s-scale-in baseline on the same traces; if none qualifies, fewest
    violation minutes."""
    hpa = sum(run_one(tr, HPAPolicy(0.70, 30.0), model, cfg, cap)["slo_violation_min"] for tr in traces)
    rows, cands = [], {}
    for risk, hz, hr, w in SWEEP:
        p = replace(base, risk=risk, stabilization=replace(base.stabilization, up_horizons_s=hz,
                                                           up_headroom=hr, down_window_s=w))
        key = (risk, "+".join(map(str, hz)), hr, w)
        cands[key] = p
        res = compare(traces, model, cfg, bundle_dir, p, caps=(cap,), only_engine=True, n_jobs=n_jobs)
        rows.append({"risk": risk, "up_horizons_s": key[1], "up_headroom": hr, "down_window_s": w,
                     "slo_violation_min": round(float(res["slo_violation_min"].sum()), 3),
                     "pod_minutes": round(float(res["pod_minutes"].sum()), 1),
                     "pod_min_over_need": round(float(res["pod_min_over_need"].sum()), 1),
                     "scaling_actions": int(res["scaling_actions"].sum()),
                     "hpa_70_sd30_violation_min": round(hpa, 3)})
    sw = pd.DataFrame(rows)
    ok = sw[sw["slo_violation_min"] <= hpa + 1e-9]
    best = (ok.sort_values("pod_minutes") if len(ok) else sw.sort_values(["slo_violation_min", "pod_minutes"])).iloc[0]
    key = (float(best["risk"]), str(best["up_horizons_s"]), float(best["up_headroom"]), float(best["down_window_s"]))
    sw["chosen"] = [(r.risk, r.up_horizons_s, r.up_headroom, r.down_window_s) == key for r in sw.itertuples()]
    return sw, cands[key]


def set_policy(engine: DecisionEngine, p: PolicyParams) -> None:
    engine.bundle.policy = p
    engine.bundle.capacity.risk = p.risk
    engine.reset()


# ==========================================================================================
# Policy v2: FAULT_AWARE and HPA_GATED candidates vs a dense HPA frontier (cost-matched)
# ==========================================================================================
#
#     cd ml_service && PYTHONPATH=. python -m kubeai_ml.scaling.policy_eval      # ~10-20 min
#
# Writes kubeai_ml/scaling/sim_results/policy_v2_*.csv|json. Tuning (``V2_SWEEP``) uses dev +
# synthetic traces only; held-out (cold and 600 s warm start) are scored once with the picks.

HPA_FRONTIER = [(t, w) for t in (0.25, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90, 1.00)
                for w in (30.0, 300.0)]


@dataclass(frozen=True)
class Candidate:
    name: str
    mode: str                                   # predictive | fault_aware | hpa_gated
    over: tuple[tuple[str, Any], ...] = ()      # Stabilization overrides

    def params(self, base: PolicyParams) -> PolicyParams:
        return replace(base, stabilization=replace(base.stabilization, mode=self.mode, **dict(self.over)))


def _fa(name: str, **kw: Any) -> Candidate:
    return Candidate(name, "fault_aware", tuple(sorted(kw.items())))


def v2_sweep_candidates() -> list[Candidate]:
    """FAULT_AWARE grid (traffic headroom x compute target x down window x forecast pre-scale)
    plus HPA_GATED with 30 / 180 s windows and the v1 PREDICTIVE policy for reference."""
    out = [Candidate("predictive", "predictive")]
    for w in (30.0, 180.0):
        out.append(Candidate(f"hpa_gated-70-sd{int(w)}", "hpa_gated",
                             (("fa_cpu_target", 0.70), ("fa_down_window_s", w))))
    for hr in (1.0, 1.3, 1.6, 2.0, 2.5):
        for cpu in (None, 0.70, 0.85, 1.0):
            for w in (30.0, 180.0):
                for fc in (False, True):
                    if fc and cpu is not None:
                        continue        # forecast pre-scale only probed on the traffic-only variants
                    kw: dict[str, Any] = {"fa_headroom": hr, "fa_down_window_s": w, "fa_fc_prescale": fc}
                    if cpu is None:
                        kw["fa_use_cpu"] = False
                    else:
                        kw["fa_cpu_target"] = cpu
                    tag = f"fa-h{hr:g}-{'cpuoff' if cpu is None else f'cpu{int(cpu * 100)}'}-sd{int(w)}{'-fc' if fc else ''}"
                    out.append(_fa(tag, **kw))
    return out


def _hpa_baselines() -> list[Policy]:
    return [HPAPolicy(t, w) for t, w in HPA_FRONTIER] + [FixedPolicy(2), FixedPolicy(3), FixedPolicy(4)]


def _v2_trace_rows(trace: Trace, model: PodModel, cfg: SimConfig, bundle_dir: str, base: PolicyParams,
                   cands: list[Candidate], caps: tuple[int, ...], split: str, baselines: bool) -> list[dict[str, Any]]:
    engine = DecisionEngine(bundle_dir, oom_model_dir=None)
    rows = []
    for cap in caps:
        for c in cands:
            set_policy(engine, c.params(base))
            engine.policy_mode = c.mode
            pol = EnginePolicy(engine, name=c.name, mode=c.mode)
            rows.append({**run_one(trace, pol, model, cfg, cap), "split": split, "family": c.mode})
        if baselines:
            for pol in _hpa_baselines():
                if isinstance(pol, FixedPolicy) and pol.n > cap:
                    continue
                rows.append({**run_one(trace, pol, model, cfg, cap), "split": split,
                             "family": "fixed" if isinstance(pol, FixedPolicy) else "hpa"})
    return rows


def compare_v2(traces: list[Trace], model: PodModel, cfg: SimConfig, bundle_dir: str, base: PolicyParams,
               cands: list[Candidate], caps: tuple[int, ...] = (4, 8), split: str = "",
               baselines: bool = True, n_jobs: int = 6) -> pd.DataFrame:
    from joblib import Parallel, delayed
    parts = Parallel(n_jobs=n_jobs)(delayed(_v2_trace_rows)(tr, model, cfg, str(bundle_dir), base, cands, caps,
                                                            split, baselines) for tr in traces)
    return pd.DataFrame([r for p in parts for r in p])


def table_v2(cmp: pd.DataFrame) -> pd.DataFrame:
    fam = cmp.groupby(["max_cap", "policy"])["family"].first()
    tab = table(cmp).set_index(["max_cap", "policy"])
    tab["family"] = fam
    return tab.reset_index()


def hpa_frontier_interp(tab_cap: pd.DataFrame, pod_min: float) -> float:
    """Violation minutes of the HPA Pareto frontier at ``pod_min``, linearly interpolated between
    its neighbouring Pareto points (a CPU target between two swept ones); nan below its range."""
    h = tab_cap[tab_cap["family"] == "hpa"].sort_values(["pod_minutes", "slo_violation_min"])
    pts, best = [], float("inf")
    for r in h.itertuples():
        if r.slo_violation_min < best - 1e-9:
            best = r.slo_violation_min
            pts.append((r.pod_minutes, r.slo_violation_min))
    if not pts or pod_min < pts[0][0] - 1e-9:
        return float("nan")
    xs, ys = zip(*pts)
    return float(np.interp(pod_min, xs, ys))


def hpa_frontier_viol(tab_cap: pd.DataFrame, pod_min: float) -> tuple[float, str]:
    """Fewest violation minutes any HPA configuration achieves at no more pod-minutes than
    ``pod_min`` (the HPA Pareto frontier evaluated at equal cost; inf if every HPA is costlier)."""
    h = tab_cap[(tab_cap["family"] == "hpa") & (tab_cap["pod_minutes"] <= pod_min + 1e-9)]
    if h.empty:
        return float("inf"), "-"
    b = h.sort_values(["slo_violation_min", "pod_minutes"]).iloc[0]
    return float(b["slo_violation_min"]), str(b["policy"])


def cost_matched_v2(tab: pd.DataFrame, policies_: list[str]) -> pd.DataFrame:
    """Per cap and candidate: (a) the HPA / fixed policy with the closest pod-minutes and (b) the
    best HPA not costlier than the candidate. ``delta_frontier`` < 0 means the candidate beats
    every HPA configuration that costs no more than it does."""
    out = []
    for cap, g in tab.groupby("max_cap"):
        base = g[g["family"].isin(["hpa", "fixed"])]
        for name in policies_:
            r = g[g["policy"] == name]
            if r.empty:
                continue
            r = r.iloc[0]
            o = base.assign(gap=(base["pod_minutes"] - r["pod_minutes"]).abs()).sort_values("gap").iloc[0]
            fv, fp = hpa_frontier_viol(g, float(r["pod_minutes"]))
            out.append({"max_cap": int(cap), "policy": name, "viol_min": float(r["slo_violation_min"]),
                        "pod_min": float(r["pod_minutes"]), "actions": int(r["scaling_actions"]),
                        "closest_baseline": str(o["policy"]), "closest_viol_min": float(o["slo_violation_min"]),
                        "closest_pod_min": float(o["pod_minutes"]),
                        "best_hpa_not_costlier": fp, "best_hpa_viol_min": fv,
                        "delta_frontier": (round(float(r["slo_violation_min"]) - fv, 3) if np.isfinite(fv) else None),
                        "hpa_frontier_interp_viol_min": round(hpa_frontier_interp(g, float(r["pod_minutes"])), 3)})
    return pd.DataFrame(out)


def pick_v2(dev_tab: pd.DataFrame, cap: int = 4) -> dict[str, str]:
    """Pick per family on dev + synthetic only. Rule: smallest gap to the *interpolated* HPA
    Pareto frontier at equal pod-minutes (violation minutes above it); ties -> fewer pod-minutes.
    Candidates cheaper than every HPA configuration are compared against the cheapest HPA."""
    g = dev_tab[dev_tab["max_cap"] == cap]
    picks = {}
    for fam in ("fault_aware", "hpa_gated"):
        f = g[g["family"] == fam].copy()
        if f.empty:
            continue
        gaps = []
        for r in f.itertuples():
            fv = hpa_frontier_interp(g, r.pod_minutes)
            if not np.isfinite(fv):
                fv = float(g[g["family"] == "hpa"].sort_values("pod_minutes").iloc[0]["slo_violation_min"])
            gaps.append(r.slo_violation_min - fv)
        f["gap"] = gaps
        picks[fam] = str(f.sort_values(["gap", "pod_minutes"]).iloc[0]["policy"])
    return picks


# ------------------------------------------------------------------------------ fault replay
def hpa_open_loop(deploy: pd.DataFrame, target: float = 0.70, tol: float = 0.10) -> pd.Series:
    """What a CPU-70 % HPA would recommend on each recorded row (open loop, no stabilisation)."""
    util = deploy["cpu_util_vs_request"].astype(float)
    ready = deploy["replicas_ready"].astype(float).clip(lower=1)
    cur = deploy["replicas_desired"].astype(float)
    ratio = util / target
    rec = np.where((ratio - 1).abs() <= tol, cur, np.ceil(ratio * ready))
    return pd.Series(np.where(np.isfinite(rec), rec, cur), index=deploy.index)


def fault_replay_v2(bundle_dir: str | None, base: PolicyParams | None, cands: list[Candidate],
                    n_jobs: int = 6) -> pd.DataFrame:
    """Open-loop replay of each candidate on every recorded fault window (``fault_eval``):
    share of decisions that add pods, wasted pod-minutes (extra pods recommended x 15 s), and the
    same numbers for an open-loop CPU-70 % HPA on the recorded telemetry."""
    from joblib import Parallel, delayed
    from .data import LAB_DATA
    faults = pd.read_csv(LAB_DATA / "faults_index.csv")
    runs = list(faults["run_id"].unique())
    parts = Parallel(n_jobs=n_jobs)(delayed(_fault_rows)(bundle_dir, base, cands, rid) for rid in runs)
    return pd.DataFrame([r for p in parts for r in p])


def _fault_rows(bundle_dir: str | None, base: PolicyParams | None, cands: list[Candidate], run_id: str
                ) -> list[dict[str, Any]]:
    from .data import LAB_DATA
    from .fault_eval import PERIOD_S, _spec, replay_run
    from .split import HOLDOUT_SCENARIOS
    d = pd.read_csv(LAB_DATA / "dataset_deploy.csv.gz")
    d = d[d["run_id"] == run_id].sort_values("ts").reset_index(drop=True)
    pods = pd.read_csv(LAB_DATA / "dataset_pods.csv.gz")
    pods = pods[pods["run_id"] == run_id]
    fr = pd.read_csv(LAB_DATA / "faults_index.csv")
    fr = fr[fr["run_id"] == run_id]
    sp = _spec(pd.read_csv(LAB_DATA / "runs_index.csv"), run_id)
    engine = DecisionEngine(bundle_dir, oom_model_dir=None)
    bp = base or engine.bundle.policy
    scen = str(d["scenario"].iloc[0])
    split = "holdout" if scen in HOLDOUT_SCENARIOS else "dev"
    hpa = d.assign(hpa_rec=hpa_open_loop(d))
    rows = []
    decs = {}
    for c in cands:
        set_policy(engine, c.params(bp))
        engine.policy_mode = c.mode
        decs[c.name] = replay_run(engine, d, pods if len(pods) else None, sp["min"], sp["max"])
    for f in fr.itertuples():
        end = f.start_rel_s + max(float(f.label_window_s), 30.0)
        want = S.FAULT_TO_STATE[f.type]
        hw = hpa[(hpa["seconds_since_start"] >= f.start_rel_s) & (hpa["seconds_since_start"] <= end)]
        # sample the HPA rule at the same 15 s cadence as the engine
        hw = hw.iloc[::int(PERIOD_S // 5)] if len(hw) else hw
        h_extra = (hw["hpa_rec"].clip(upper=sp["max"]) - hw["replicas_desired"]).clip(lower=0)
        pre_lo = f.start_rel_s - min(120.0, end - f.start_rel_s)
        base_row = {"run": scen, "split": split, "fault": f.type, "expected_state": want,
                    "hpa70_upscale_pct": round(100 * float((h_extra > 0).mean()), 1) if len(hw) else None,
                    "hpa70_wasted_pod_min": round(float(h_extra.sum()) * PERIOD_S / 60, 2)}
        for name, dec in decs.items():
            w = dec[(dec["secs"] >= f.start_rel_s) & (dec["secs"] <= end)]
            extra = (w["desired"] - w["current"]).clip(lower=0)
            pw = dec[(dec["secs"] >= pre_lo) & (dec["secs"] < f.start_rel_s)]
            rows.append({**base_row, "policy": name, "decisions": len(w),
                         "state_recall": round(float((w["state"] == want).mean()), 3) if len(w) else None,
                         "upscale_pct": round(100 * float((extra > 0).mean()), 1) if len(w) else None,
                         # same policy just before the fault: open-loop replay against the lab's
                         # replica count confounds "fault scale-up" with a different operating point
                         "pre_fault_upscale_pct": round(100 * float((pw["desired"] > pw["current"]).mean()), 1)
                         if len(pw) else None,
                         "wasted_pod_min": round(float(extra.sum()) * PERIOD_S / 60, 2),
                         "blocked_pct": round(100 * float(w["blocked"].mean()), 1) if len(w) else None})
    return rows


def fault_summary(fr: pd.DataFrame) -> pd.DataFrame:
    """Per policy and split; pod_kill (harmless, Deployment replaces the pod) reported apart."""
    g = fr.assign(kind=np.where(fr["fault"] == "pod_kill", "pod_kill", "fault"))
    agg = g.groupby(["policy", "split", "kind"]).agg(
        windows=("run", "size"), upscale_pct=("upscale_pct", "mean"), wasted_pod_min=("wasted_pod_min", "sum"),
        pre_fault_upscale_pct=("pre_fault_upscale_pct", "mean"),
        hpa70_upscale_pct=("hpa70_upscale_pct", "mean"), hpa70_wasted_pod_min=("hpa70_wasted_pod_min", "sum"),
        blocked_pct=("blocked_pct", "mean"))
    return agg.reset_index().round(2)


def main_v2(n_jobs: int = 6, out_dir: str | None = None, quick: bool = False) -> dict[str, Any]:
    import json
    import time
    from .bundle import LATEST
    from .data import load_deploy
    from .eval_policies import RESULTS_DIR
    from .podmodel import fit_pod_model
    from .split import HOLDOUT_SCENARIOS, is_holdout
    from .data import LAB_DATA
    from .traces import run_trace

    t0 = time.time()
    bundle = LATEST.parent / LATEST.read_text().strip() if LATEST.is_file() else LATEST
    calib = json.loads((bundle / "sim_calibration.json").read_text())
    df = load_deploy()
    pm, _ = fit_pod_model(df[~is_holdout(df["scenario"])])
    cfg = SimConfig(ms_lag_s=float(calib["ms_lag_s"]), ms_window_s=float(calib["ms_window_s"]))
    base = DecisionEngine(bundle, oom_model_dir=None).bundle.policy
    shape = pd.read_csv(LAB_DATA / "runs_index.csv").set_index("run_id")["shape"]
    traces: dict[str, list[Trace]] = {"dev": [], "holdout": []}
    for rid, g in df.groupby("run_id"):
        if shape[rid] in ("diurnal", "spike", "flash_crowd", "sawtooth"):
            traces["holdout" if g["scenario"].iloc[0] in HOLDOUT_SCENARIOS else "dev"].append(run_trace(rid, g))
    syn = synthetic_traces()
    out_p = RESULTS_DIR if out_dir is None else __import__("pathlib").Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    # 1. sweep on dev + synthetic (tuning set)
    cands = v2_sweep_candidates()
    if quick:
        cands = cands[:3] + [c for c in cands if c.name in ("fa-h1.3-cpu70-sd30", "fa-h1.3-cpuoff-sd180")]
    tune = compare_v2(traces["dev"] + syn, pm, cfg, str(bundle), base, cands, caps=(4,), split="tune", n_jobs=n_jobs)
    tune.to_csv(out_p / "policy_v2_sweep_per_trace.csv", index=False)
    tune_tab = table_v2(tune)
    tune_tab.to_csv(out_p / "policy_v2_sweep.csv", index=False)
    picks = pick_v2(tune_tab, cap=4)
    print(f"[v2] sweep {len(cands)} candidates in {time.time() - t0:.0f}s -> {picks}", flush=True)

    # 2. score picks + reference candidates on every split, caps 4 and 8
    by_name = {c.name: c for c in cands}
    final_names = list(dict.fromkeys(["predictive", picks.get("fault_aware"), picks.get("hpa_gated"),
                                      "hpa_gated-70-sd30", "hpa_gated-70-sd180",
                                      "fa-h1.3-cpuoff-sd180", "fa-h1.3-cpuoff-sd180-fc",
                                      "fa-h1.3-cpu70-sd30"]))
    final = [by_name[n] for n in final_names if n in by_name]
    warm = [warm_start(t) for t in traces["holdout"]]
    splits = {"dev": traces["dev"], "synthetic": syn, "holdout": traces["holdout"], "holdout_warm": warm}
    cmp = pd.concat([compare_v2(v, pm, cfg, str(bundle), base, final, split=k, n_jobs=n_jobs)
                     for k, v in splits.items()])
    cmp.to_csv(out_p / "policy_v2_per_trace.csv", index=False)
    res: dict[str, Any] = {"picks": picks, "candidates": [c.name for c in final],
                           "params": {c.name: dict(c.over) | {"mode": c.mode} for c in final}}
    for k in splits:
        tab = table_v2(cmp[cmp["split"] == k])
        tab.to_csv(out_p / f"policy_v2_table_{k}.csv", index=False)
        res[f"table_{k}"] = tab.to_dict("records")
        cm = cost_matched_v2(tab, [c.name for c in final])
        cm.to_csv(out_p / f"policy_v2_cost_matched_{k}.csv", index=False)
        res[f"cost_matched_{k}"] = cm.to_dict("records")
    print(f"[v2] scored {len(final)} candidates on 4 splits in {time.time() - t0:.0f}s", flush=True)

    # 3. fault replay (open loop on recorded fault runs)
    fr = fault_replay_v2(str(bundle), base, final, n_jobs=n_jobs)
    fr.to_csv(out_p / "policy_v2_fault_replay.csv", index=False)
    fs = fault_summary(fr)
    fs.to_csv(out_p / "policy_v2_fault_summary.csv", index=False)
    res["fault_summary"] = fs.to_dict("records")
    print(f"[v2] fault replay in {time.time() - t0:.0f}s", flush=True)
    (out_p / "policy_v2_results.json").write_text(json.dumps(res, indent=1, default=str))
    return res


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--out", default=None)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    main_v2(a.jobs, a.out, a.quick)
