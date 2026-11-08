"""Offered-load traces for the replay simulator.

The lab's k6 driver is open loop (``ramping-arrival-rate``), so the *offered* arrival rate is known
exactly from the shape spec: ``loadlab/load/shapes.js`` turns (shape, base_rps, params) into
piecewise-linear points, and each run's ``k6_summary.json`` stores those points verbatim
(``data.loadlab.points``). :func:`shape_points` is a line-for-line port of ``shapes.js`` (including
the mulberry32 PRNG used by ``diurnal``), so synthetic traces can be generated for any spec, and
:func:`run_trace` reconstructs a recorded run's trace and aligns it to the dataset clock.

k6 rounds every stage target to an integer requests/minute; we keep that rounding.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data import REPO

RUNS_DIR = REPO / "loadlab" / "runs"
RPS_WINDOW_S = 20          # dataset rps = rate(http_requests_total[20s])


def _mulberry32(seed: int):
    a = seed & 0xFFFFFFFF

    def imul(x: int, y: int) -> int:
        return (x * y) & 0xFFFFFFFF

    def nxt() -> float:
        nonlocal a
        a = (a + 0x6D2B79F5) & 0xFFFFFFFF
        t = imul(a ^ (a >> 15), 1 | a)
        t = ((t + imul(t ^ (t >> 7), 61 | t)) & 0xFFFFFFFF) ^ t
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296

    return nxt


def shape_points(shape: str, base_rps: float, duration_s: int, params: dict | None = None
                 ) -> list[tuple[float, float]]:
    """Port of ``shapePoints()`` in loadlab/load/shapes.js: [(t_s, rps), ...]."""
    p = params or {}
    B, D = float(base_rps), int(duration_s)
    pts: list[tuple[float, float]] = []
    if shape == "steady":
        ramp = p.get("ramp_s", 10)
        pts += [(0, B * 0.2), (ramp, B), (D, B)]
    elif shape == "step_ramp":
        start, step = p.get("start_rps", B), p.get("step_rps", max(1, B * 0.25))
        step_s, max_r, ramp = p.get("step_s", 45), p.get("max_rps", math.inf), p.get("ramp_s", 3)
        t, r = 0.0, start
        pts.append((0, start))
        while t + step_s <= D + 1e-9:
            if t > 0:
                pts.append((t + ramp, r))
            pts.append((t + step_s, r))
            t += step_s
            r = min(r + step, max_r)
        if pts[-1][0] < D:
            pts.append((D, pts[-1][1]))
    elif shape == "diurnal":
        period, amp = p.get("period_s", 600), p.get("amplitude", 0.6)
        noise, dt, rand = p.get("noise", 0.08), p.get("point_s", 10), _mulberry32(p.get("seed", 42))
        for t in range(0, D + 1, dt):
            base = B * (1 - amp * math.cos(2 * math.pi * t / period))
            pts.append((t, max(0.5, base * (1 + noise * (2 * rand() - 1)))))
    elif shape == "spike":
        at, mult = p.get("spike_at_s", round(D * 0.35)), p.get("spike_mult", 4)
        dur, ramp = p.get("spike_s", 120), p.get("ramp_s", 10)
        pts += [(0, B), (at, B), (at + ramp, B * mult), (at + ramp + dur, B * mult),
                (at + 2 * ramp + dur, B), (D, B)]
    elif shape == "flash_crowd":
        at, mult, ramp = p.get("at_s", round(D * 0.3)), p.get("mult", 3), p.get("ramp_s", 5)
        pts += [(0, B), (at, B), (at + ramp, B * mult), (D, B * mult)]
    elif shape == "sawtooth":
        period = p.get("period_s", 120)
        lo, hi = p.get("low_mult", 0.5) * B, p.get("high_mult", 2.0) * B
        pts.append((0, lo))
        t = 0
        while t + period <= D:
            pts += [(t + period - 1, hi), (t + period, lo)]
            t += period
        if pts[-1][0] < D:
            pts.append((D, lo + (hi - lo) * ((D % period) / period)))
    else:
        raise ValueError(f"unknown shape {shape}")
    return [pt for pt in pts if pt[0] <= D]


def points_to_rate(points: list, duration_s: int, res_s: float = 1.0) -> np.ndarray:
    """Offered RPS sampled every ``res_s`` from t=0 to duration, as k6 executes it.

    k6 stages: stage i ramps linearly from the previous target to round(rps*60)/60 over
    round(dt) seconds. The start rate is round(points[0]*60)/60.
    """
    knots_t, knots_r = [0.0], [round(points[0][1] * 60) / 60]
    t = 0.0
    for (t0, _), (t1, r1) in zip(points[:-1], points[1:]):
        dt = round(t1 - t0)
        if dt <= 0:
            continue
        t += dt
        knots_t.append(t)
        knots_r.append(max(0, round(r1 * 60)) / 60)
    grid = np.arange(0, duration_s + 1e-9, res_s)
    return np.interp(grid, knots_t, knots_r, right=0.0)


@dataclass
class Trace:
    """Offered load on the simulator grid. ``rps[i]`` = mean offered RPS over [t[i]-step, t[i])."""
    name: str
    t: np.ndarray          # seconds since start (row time, matches dataset seconds_since_start)
    rps: np.ndarray
    step_s: float = 5.0
    meta: dict | None = None


def to_grid(rate_1s: np.ndarray, t_rows: np.ndarray, offset_s: float = 0.0, step_s: float = 5.0
            ) -> np.ndarray:
    """Mean offered rate over (t-step, t] for each row time, k6 clock shifted by ``offset_s``.

    ``offset_s`` > 0 means k6 started that many seconds after the dataset's t=0.
    """
    tt = np.arange(len(rate_1s), dtype=float) + offset_s
    out = np.empty(len(t_rows))
    for i, t in enumerate(t_rows):
        sub = np.arange(t - step_s + 0.5, t, 1.0)
        out[i] = np.interp(sub, tt, rate_1s, left=0.0, right=0.0).mean()
    return out


def windowed(rate_1s: np.ndarray, t_rows: np.ndarray, offset_s: float, window_s: float) -> np.ndarray:
    """Trailing ``window_s`` mean of the offered rate at each row (what a rate[20s] would see)."""
    tt = np.arange(len(rate_1s), dtype=float) + offset_s
    out = np.empty(len(t_rows))
    for i, t in enumerate(t_rows):
        sub = np.arange(t - window_s + 0.5, t, 1.0)
        out[i] = np.interp(sub, tt, rate_1s, left=0.0, right=0.0).mean()
    return out


def run_dir(run_id: str) -> Path:
    return RUNS_DIR / run_id


def load_k6_points(run_id: str) -> dict:
    with open(run_dir(run_id) / "k6_summary.json") as f:
        return json.load(f)["loadlab"]


def fit_offset(rate_1s: np.ndarray, rows: pd.DataFrame) -> tuple[float, float]:
    """Align the k6 clock to the dataset clock by matching the 20 s windowed offered rate to the
    observed ``rps`` on unsaturated rows. Returns (offset_s, MAPE on those rows)."""
    t = rows["seconds_since_start"].to_numpy(float)
    obs = rows["rps"].to_numpy(float)
    ok = ((rows["error_rate"] < 0.01) & (rows["inflight"] < 5 * rows["replicas_ready"].clip(lower=1))
          & (rows["p95"] < 0.3)).to_numpy() & (obs > 1)
    best = (0.0, math.inf)
    for off in np.arange(-10, 20.5, 0.5):
        w = windowed(rate_1s, t, off, RPS_WINDOW_S)
        err = np.mean(np.abs(w[ok] - obs[ok])) if ok.any() else math.inf
        if err < best[1]:
            best = (float(off), err)
    w = windowed(rate_1s, t, best[0], RPS_WINDOW_S)
    mape = float(np.mean(np.abs(w[ok] - obs[ok]) / obs[ok])) if ok.any() else float("nan")
    return best[0], mape


def run_trace(run_id: str, rows: pd.DataFrame, step_s: float = 5.0) -> Trace:
    """Offered-load trace of a recorded run on that run's dataset row times.

    ``rows`` are the run's dataset_deploy rows. Checks the k6 points against :func:`shape_points`
    (``meta['shape_port_ok']``) and the reconstructed rate against the observed rps on unsaturated
    rows (``meta['rps_mape']``).
    """
    spec = load_k6_points(run_id)
    pts = spec["points"]
    port = shape_points(spec["shape"], spec["base_rps"], spec["duration_s"], spec["params"])
    port_ok = len(port) == len(pts) and all(
        abs(a[0] - b[0]) < 1e-6 and abs(a[1] - b[1]) < 1e-6 for a, b in zip(port, pts))
    rate = points_to_rate(pts, spec["duration_s"])
    rows = rows.sort_values("seconds_since_start")
    off, mape = fit_offset(rate, rows)
    t = rows["seconds_since_start"].to_numpy(float)
    return Trace(name=str(rows["scenario"].iloc[0]), t=t, rps=to_grid(rate, t, off, step_s),
                 step_s=step_s,
                 meta={"run_id": run_id, "offset_s": off, "rps_mape": mape, "shape": spec["shape"],
                       "shape_port_ok": port_ok, "rate_1s": rate})


def synthetic_trace(name: str, shape: str, base_rps: float, duration_s: int,
                    params: dict | None = None, step_s: float = 5.0) -> Trace:
    """A trace for a shape spec that was never run (e.g. longer runs where scale-down matters)."""
    rate = points_to_rate(shape_points(shape, base_rps, duration_s, params), duration_s)
    t = np.arange(step_s, duration_s + 1e-9, step_s)
    return Trace(name=name, t=t, rps=to_grid(rate, t, 0.0, step_s), step_s=step_s,
                 meta={"shape": shape, "synthetic": True, "offset_s": 0.0, "rate_1s": rate})
