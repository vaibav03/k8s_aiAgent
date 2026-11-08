"""State labels for the classifier: ground-truth faults + capacity-derived load states.

Rules (applied per run, on 5 s rows; future windows are allowed because these are labels):

1. **Fault rows** (``active_fault != "none"``) take the state of the fault
   (``states.FAULT_TO_STATE``). Overlapping faults resolve by ``states.FAULT_PRIORITY``. In
   ``e-spike-dep-latency`` the traffic spike overlaps the dependency fault; the row is
   ``dependency_latency`` because that is the action-relevant root cause (scaling hides it).
2. **Learned capacity knee** ``c(r)``: per replica count, the per-pod demand threshold (1 RPS
   grid) that best separates SLO-ok from breached fault-free ``qa_flag == ok`` rows (a decision
   stump, robust to the transient HPA-lag breaches). Demand = goodput, or rps while p95
   breaches (overload 504s are offered load; see ``features.demand_rps``). A replica count whose
   rows above the stump are not mostly breached never reached its knee and inherits the most
   conservative learned knee (capacity falls as replicas are added: CFS throttling starts
   earlier). Values are logged by ``train_classifier`` (EXPLORE_CLASSIFIER.md).
3. **scale_up** (fault-free row) if either
   a. the SLO is breached now *because of load*: (p95 >= 300 ms or err >= 1 %) and
      (demand/pod >= 0.85 c(r) or throttle >= 0.5); or
   b. a breach is coming at the current size: peak demand over (t, t+60 s] / ready >= c(r).
4. **scale_down** (fault-free, SLO ok, ready >= 2) if the peak demand over [t-60 s, t+180 s]
   fits ``ready-1`` pods at ``TARGET_UTIL`` (75 %) of ``c(ready-1)``, and that held for every
   row of the trailing 60 s (sustained, not a dip).
5. Otherwise **healthy**.

``qa_flag != ok`` rows (node glitch, post-fault tails) are returned with ``use=False``: excluded
from training and scoring. So are rows where a fault is physically still active after its logged
end (``residual_hog_mask``: in every cpu_hog run of rounds 1 and 2 the hog process survives the
chaos reset and the target pod keeps burning ~0.45-0.5 core until the run ends, so the "none"
label there is wrong; detected from the pod table, no run names), and rows listed in the manual
override ``KNOWN_LABEL_ISSUES``. See EXPLORE_CLASSIFIER.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import states as S
from .features import demand_rps

SLO_P95_S = 0.300
SLO_ERR = 0.01
TARGET_UTIL = 0.75
SAFE_BREACH_RATE = 0.10
MIN_BIN_ROWS = 3

RESIDUAL_HOG_CORES = 0.15     # target pod CPU above the median pod after the hog's logged end

# Manual override: (scenario, seconds_since_start from which rows are unusable, reason).
# Empty since the cpu_hog survival is detected from data (``residual_hog_mask``).
KNOWN_LABEL_ISSUES: tuple[tuple[str, float, str], ...] = ()


@dataclass
class SafeCapacity:
    """Learned SLO-safe goodput per pod as a function of ready replicas."""

    table: dict[int, float] = field(default_factory=dict)
    fallback: float = 14.0

    def at(self, replicas: float | np.ndarray) -> np.ndarray:
        r = np.atleast_1d(np.asarray(replicas, dtype=float))
        out = np.full(r.shape, self.fallback)
        for k, v in self.table.items():
            out[r == k] = v
        return out

    def to_dict(self) -> dict[str, object]:
        return {"table": {str(k): v for k, v in self.table.items()}, "fallback": self.fallback}


def _knee(load: np.ndarray, breach: np.ndarray) -> float | None:
    """Decision stump: the load threshold that best separates SLO-ok from breached rows.

    Returns None when the rows above the best threshold are not mostly breached (this
    replica count never reached its knee)."""
    best, best_err = None, np.inf
    for thr in np.arange(1.0, np.ceil(np.nanmax(load)) + 1.0):
        above = load >= thr
        if above.sum() < MIN_BIN_ROWS:
            continue
        err = int((breach & ~above).sum() + (~breach & above).sum())
        if err < best_err:
            best, best_err = float(thr), err
    if best is None or breach[load >= best].mean() < 0.5:
        return None
    return best


def learn_safe_capacity(df: pd.DataFrame) -> SafeCapacity:
    """Knee per replica count from fault-free rows (see module docstring, rule 2)."""
    keep = (df["active_fault"] == "none") & (df["qa_flag"] == "ok") & (df["replicas_ready"] >= 1)
    if "use" in df:                      # also drop rows masked by make_labels (residual faults)
        keep &= df["use"].astype(bool)
    ok = df[keep]
    load = demand_rps(ok) / ok["replicas_ready"].to_numpy(float)
    breach = ((ok["p95"].fillna(0) >= SLO_P95_S) | (ok["error_rate"] >= SLO_ERR)).to_numpy()
    r_all = ok["replicas_ready"].to_numpy(float)
    knees: dict[int, float] = {}
    for r in np.unique(r_all):
        m = r_all == r
        k = _knee(load[m], breach[m]) if m.sum() >= 4 * MIN_BIN_ROWS else None
        if k is not None:
            knees[int(r)] = k
    fallback = min(knees.values()) if knees else 14.0
    return SafeCapacity(table=knees, fallback=fallback)


def _future_max(values: np.ndarray, ts: np.ndarray, lo_s: float, hi_s: float) -> np.ndarray:
    """max(values over [t+lo_s, t+hi_s]) for each t (NaN-safe)."""
    out = np.full(len(ts), np.nan)
    for i, t in enumerate(ts):
        w = values[(ts >= t + lo_s) & (ts <= t + hi_s)]
        w = w[np.isfinite(w)]
        if len(w):
            out[i] = w.max()
    return out


def _fault_state(active: str) -> str | None:
    if active in ("none", "", None) or (isinstance(active, float) and np.isnan(active)):
        return None
    cands = [S.FAULT_TO_STATE[a] for a in str(active).split("+") if a in S.FAULT_TO_STATE]
    for s in S.FAULT_PRIORITY:
        if s in cands:
            return s
    return None


def residual_hog_mask(run: pd.DataFrame, pods: pd.DataFrame | None,
                      faults: pd.DataFrame | None) -> np.ndarray:
    """True on rows after a cpu_hog's logged end where its target pod still burns
    >= ``RESIDUAL_HOG_CORES`` more than the median pod (the hog survived the reset).

    Designed on the round-2 hog runs, which include the held-out d-cpu-hog-fixed3-p2-r2 (it
    removes 38 of its rows from scoring); train_all's ``holdout_blind`` scores them kept."""
    out = np.zeros(len(run), dtype=bool)
    if pods is None or faults is None or pods.empty:
        return out
    rid = run["run_id"].iloc[0]
    hogs = faults[(faults["run_id"] == rid) & (faults["type"] == "cpu_hog")]
    if hogs.empty:
        return out
    ready = pods[pods["ready"] == 1]
    med = ready.groupby("ts")["cpu"].median()
    secs = run["seconds_since_start"].to_numpy(float)
    for h in hogs.itertuples():
        tgt = ready[ready["pod"] == h.target].set_index("ts")["cpu"]
        excess = (tgt - med.reindex(tgt.index)).reindex(run["ts"].to_numpy()).to_numpy(float)
        out |= (secs > h.end_rel_s) & (np.nan_to_num(excess) >= RESIDUAL_HOG_CORES)
    return out


def label_run(run: pd.DataFrame, cap: SafeCapacity, pods: pd.DataFrame | None = None,
              faults: pd.DataFrame | None = None) -> pd.DataFrame:
    """Labels for one run (sorted by ts). Returns columns: state, use, residual_hog (rows masked
    only because a cpu_hog outlived its logged end; kept apart for the blind held-out score)."""
    run = run.sort_values("ts")
    ts = run["ts"].to_numpy(float)
    ready = run["replicas_ready"].to_numpy(float)
    ready_c = np.maximum(ready, 1)
    gp = demand_rps(run)
    c_now = cap.at(ready_c)
    breach = ((run["p95"].fillna(0) >= SLO_P95_S) | (run["error_rate"] >= SLO_ERR)).to_numpy()
    sat = (gp / ready_c >= 0.85 * c_now) | (run["throttle_ratio"].to_numpy(float) >= 0.5)

    up_now = breach & sat
    up_soon = _future_max(gp, ts, 1, 60) / ready_c >= c_now
    peak = np.fmax(_future_max(gp, ts, -60, 180), gp)
    fewer = np.maximum(ready - 1, 1)
    fits = (ready >= 2) & (peak <= TARGET_UTIL * cap.at(fewer) * fewer) & ~breach
    sustained = np.array([fits[(ts >= t - 60) & (ts <= t)].all() for t in ts])

    state = np.where(up_now | up_soon, S.SCALE_UP,
                     np.where(sustained, S.SCALE_DOWN, S.HEALTHY)).astype(object)
    fault_state = run["active_fault"].map(_fault_state).to_numpy()
    has_fault = pd.notna(fault_state)
    state[has_fault] = fault_state[has_fault]
    residual = residual_hog_mask(run, pods, faults)
    use = (run["qa_flag"] == "ok").to_numpy() & ~residual
    for scenario, from_s, _ in KNOWN_LABEL_ISSUES:
        use &= ~((run["scenario"] == scenario) & (run["seconds_since_start"] >= from_s)).to_numpy()
    return pd.DataFrame({"state": state, "use": use, "residual_hog": residual}, index=run.index)


def make_labels(df: pd.DataFrame, cap: SafeCapacity | None = None,
                pods: pd.DataFrame | None = None, faults: pd.DataFrame | None = None) -> pd.DataFrame:
    """Labels for every run in ``df`` (index-aligned). Learns capacity when not given.

    ``pods`` (dataset_pods) and ``faults`` (faults_index) are optional; with both, rows where a
    cpu_hog outlived its logged end are masked (``residual_hog_mask``)."""
    cap = cap or learn_safe_capacity(df)
    by_run = dict(tuple(pods.groupby("run_id"))) if pods is not None else {}
    parts = [label_run(g, cap, by_run.get(r), faults) for r, g in df.groupby("run_id", sort=False)]
    return pd.concat(parts).loc[df.index]
