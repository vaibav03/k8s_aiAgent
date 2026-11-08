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

    def reset(self) -> None:
        self.engine.reset()
        self.states = []

    def __call__(self, history: pd.DataFrame, state: SimState) -> int:
        h = history.assign(ts=history["seconds_since_start"])
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
            EnginePolicy(engine)]


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
        pols = [EnginePolicy(engine)] if only_engine else policies(model, engine)
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
