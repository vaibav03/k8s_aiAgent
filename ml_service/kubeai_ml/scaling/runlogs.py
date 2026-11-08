"""Read Kubernetes events recorded by the lab runner (``runs/<run_id>/events.json``).

Used to calibrate controller timing that the 5 s Prometheus rows cannot resolve:

* HPA sync phase: HPA events (FailedGetResourceMetric, SuccessfulRescale) land on the controller's
  15 s sync grid, so ``event_time mod 15`` gives the phase relative to the run clock.
* actual HPA rescale decisions (time, new size), the ground truth for simulator validation.
* pod startup delay: ReplicaSet SuccessfulCreate -> pod first Ready in pods.csv.
"""
from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .traces import run_dir

HPA_SYNC_S = 15.0


def _ts(s: str) -> float:
    return dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc).timestamp()


def run_t0(run_id: str) -> float:
    with open(run_dir(run_id) / "scenario.json") as f:
        return float(json.load(f)["_run"]["t0"])


def _events(run_id: str) -> list[dict]:
    p = run_dir(run_id) / "events.json"
    if not p.exists():
        return []
    with open(p) as f:
        return json.load(f)["items"]


@dataclass
class HpaEvents:
    rescales: list[tuple[float, int]]      # (seconds since t0, new size)
    sync_times: list[float]                # all HPA event times (seconds since t0)

    @property
    def phase_s(self) -> float | None:
        """Sync phase in [0, 15): circular mean of event times mod 15 (events are 1 s resolution)."""
        if not self.sync_times:
            return None
        ang = np.asarray(self.sync_times) % HPA_SYNC_S / HPA_SYNC_S * 2 * np.pi
        m = np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())
        return float((m / (2 * np.pi) * HPA_SYNC_S) % HPA_SYNC_S)


def hpa_events(run_id: str) -> HpaEvents:
    t0 = run_t0(run_id)
    resc, sync = [], []
    for e in _events(run_id):
        if e["involvedObject"]["kind"] != "HorizontalPodAutoscaler" or not e.get("firstTimestamp"):
            continue
        # k8s aggregates identical events (same reason + message): ``count`` > 1 keeps only the
        # first and last occurrence, so a run that scales 3 -> 4 twice has one "New size: 4"
        # event. Both timestamps are kept; the middle occurrences of count > 2 are lost (use
        # :func:`rescales_from_rows` for a complete list).
        times = {_ts(e["firstTimestamp"]) - t0}
        if e.get("lastTimestamp") and int(e.get("count") or 1) > 1:
            times.add(_ts(e["lastTimestamp"]) - t0)
        for t in sorted(times):
            sync.append(t)
            if e["reason"] == "SuccessfulRescale":
                resc.append((t, int(e["message"].split("New size: ")[1].split(";")[0])))
    return HpaEvents(rescales=sorted(resc), sync_times=sync)


def rescales_from_rows(rows: pd.DataFrame, ev: HpaEvents, step_s: float = 5.0,
                       max_event_lead_s: float = 15.0) -> list[tuple[float, int]]:
    """Complete list of actual rescales (seconds since t0, new size).

    Every change of ``replicas_desired`` in the 5 s rows is one rescale (complete, but 5 s
    resolution and up to ~10 s late: kube-state-metrics scrape). Its time is refined to the HPA
    event with the same new size within ``max_event_lead_s`` before the row, when one exists
    (event dedup can drop it), else the middle of the preceding row interval."""
    t = rows["seconds_since_start"].to_numpy(float)
    r = rows["replicas_desired"].to_numpy(float)
    out: list[tuple[float, int]] = []
    for i in range(1, len(r)):
        if r[i] == r[i - 1] or not np.isfinite(r[i]):
            continue
        n = int(r[i])
        cand = [te for te, k in ev.rescales if k == n and t[i] - max_event_lead_s <= te <= t[i] + 1.0]
        out.append((max(cand) if cand else t[i] - step_s / 2, n))
    return out


def pod_startup_delays(run_ids: list[str]) -> list[float]:
    """SuccessfulCreate event -> first pods.csv sample with ready == 1 (seconds; 5 s sampling,
    1 s event resolution). Only pods created after the run's t0 (scale-ups / replacements)."""
    out: list[float] = []
    for rid in run_ids:
        t0 = run_t0(rid)
        pods_p = run_dir(rid) / "pods.csv"
        if not pods_p.exists():
            continue
        pods = pd.read_csv(pods_p, usecols=["ts", "pod", "ready"])
        first_ready = pods[pods["ready"] == 1].groupby("pod")["ts"].min()
        for e in _events(rid):
            if e["reason"] != "SuccessfulCreate" or e["involvedObject"]["kind"] != "ReplicaSet":
                continue
            ts = _ts(e["firstTimestamp"])
            pod = e["message"].split(": ")[-1]
            if ts >= t0 and pod in first_ready.index:
                out.append(float(first_ready[pod] - ts))
    return out
