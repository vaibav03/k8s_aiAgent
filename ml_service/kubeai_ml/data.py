"""Load the collected CSVs, find crash events and build time-ordered examples.

Units:
- k8s_metrics.csv ("Memory Usage") comes from DevTrails server-prom.js, which
  reads metrics-server working set in Ki and divides by 1000, so
  bytes = value * 1024 * 1000.
- nirmal_dataset_1.csv ("Memory Usage") is MiB, bytes = value * 2**20.

Limit: the memory limit used while k8s_metrics.csv was collected is unknown
(the 768Mi in the DevTrails yaml is exceeded by 2,800 rows). Pods are
OOMKilled when they hit the limit, so the observed maximum (887.345 Ki/1000,
~866.6 MiB) is used as a proxy for the limit. nirmal_dataset_1.csv was
collected with a 900Mi limit (podCreation.yaml).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import HISTORY_S, STEP_S, compute_features

KI_OVER_1000 = 1024 * 1000
MIB = 2 ** 20
GAP_S = 120          # a raw gap longer than this starts a new segment (no forward-fill across it)
RESTART_GRACE_S = 120  # a restart-count increment this soon after a non-Running row is the restart itself
EVENT_MERGE_S = 60     # crash markers closer than this to the previous one are the same crash


@dataclass
class Series:
    name: str
    df: pd.DataFrame          # ts (unix s), bytes, status, restart_count, segment
    limit_bytes: float
    limit_note: str
    sha256: str


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _unix_seconds(timestamps):
    dt = pd.to_datetime(timestamps, utc=True)
    return (dt - pd.Timestamp(0, tz="UTC")).dt.total_seconds()


def _finish(df, name, limit_bytes, limit_note, path):
    df = df.sort_values("ts", kind="stable").reset_index(drop=True)
    df = df.drop_duplicates(subset=["ts"], keep="last").reset_index(drop=True)
    # A 0 after a non-zero restart count is a failed kubectl read, not a reset.
    rc = df["restart_count"]
    df["restart_count"] = rc.mask((rc == 0) & (rc.cummax() > 0)).ffill()
    df["segment"] = (df["ts"].diff() > GAP_S).cumsum()
    return Series(name, df, float(limit_bytes), limit_note, file_sha256(path))


def load_k8s_metrics(path):
    raw = pd.read_csv(path)
    df = pd.DataFrame({
        "ts": _unix_seconds(raw["Timestamp"]),
        "bytes": raw["Memory Usage"].astype(float) * KI_OVER_1000,
        "status": raw["Node Status"].astype(str),
        "restart_count": raw["Restart Count"].astype(float),
    })
    df = df[df["bytes"] > 0]
    limit = df["bytes"].max()
    note = ("proxy: observed max working set (887.345 Ki/1000 = %.1f MiB); "
            "true limit during collection unknown" % (limit / MIB))
    return _finish(df, "k8s_metrics.csv", limit, note, path)


def load_nirmal(path, limit_mib=900):
    raw = pd.read_csv(path)
    df = pd.DataFrame({
        "ts": _unix_seconds(raw["Timestamp"]),
        "bytes": raw["Memory Usage"].astype(float) * MIB,
        "status": raw["Node Status"].astype(str),
        "restart_count": raw["Restart Count"].astype(float),
    })
    df = df[df["bytes"] > 0]  # 0 means Prometheus returned null
    return _finish(df, "nirmal_dataset_1.csv", limit_mib * MIB, f"{limit_mib}Mi (pod spec)", path)


def crash_events(df):
    """Times the pod crashed: Running -> OOMKilled/CrashLoopBackOff, or a
    restart-count increment while Running that is not the restart that follows
    a crash we already saw."""
    running = (df["status"] == "Running").to_numpy()
    ts = df["ts"].to_numpy()
    rc = df["restart_count"].to_numpy()
    events = []
    last_not_running = -np.inf
    for i in range(1, len(df)):
        if not running[i - 1]:
            last_not_running = ts[i - 1]
        if running[i - 1] and not running[i]:
            candidate = ts[i]
        elif (running[i] and rc[i] > rc[i - 1] and ts[i] - ts[i - 1] <= GAP_S
              and ts[i] - last_not_running > RESTART_GRACE_S):
            candidate = ts[i]
        else:
            continue
        # Status flaps (OOMKilled -> Running -> CrashLoopBackOff) are one crash.
        if not events or candidate - events[-1] > EVENT_MERGE_S:
            events.append(candidate)
    return np.array(events)


def restart_times(df, events):
    """When the container (re)started: first Running row after a non-Running
    row, plus silent restarts (increment events with no status change)."""
    running = (df["status"] == "Running").to_numpy()
    ts = df["ts"].to_numpy()
    starts = [ts[i] for i in range(1, len(df)) if running[i] and not running[i - 1]]
    silent = [e for e in events if running[np.searchsorted(ts, e)]]
    return np.unique(np.array(starts + silent, dtype=float))


AGENT_POLL_S = 15     # the agent's POLL_MS / 1000: one Prometheus instant query per tick
AGENT_WINDOW = 40     # the agent's WINDOW (ring buffer length)


def history_window(ts, b, lo, hi, t, sampler):
    """The samples the agent would hold at time t, restricted to raw rows [lo, hi).

    sampler="tick": what the deployed agent sends. One instant query every
    AGENT_POLL_S seconds (ts = query time, value = the newest scrape at or
    before it), AGENT_WINDOW of them.
    sampler="raw": every raw collector row in the last HISTORY_S + STEP_S s.
    """
    if sampler == "raw":
        lo = max(lo, np.searchsorted(ts, t - HISTORY_S - STEP_S, side="left"))
        return ts[lo:hi], b[lo:hi]
    ticks = t - AGENT_POLL_S * np.arange(AGENT_WINDOW - 1, -1, -1, dtype=float)
    j = np.searchsorted(ts, ticks, side="right") - 1
    ok = j >= lo
    return ticks[ok], b[j[ok]]


def build_examples(series, stride=STEP_S, sampler="tick"):
    """One example every `stride` seconds while the pod is Running.

    Each example sees only the history the agent would have at that moment:
    samples from the same segment and only since the container's last
    (re)start (the agent clears its window when restartCount goes up), shaped
    by `sampler` (see history_window). Features come from
    features.compute_features. Future-looking columns (next event, future
    memory) are stored separately for labelling and forecast scoring.
    """
    df = series.df
    ts = df["ts"].to_numpy()
    b = df["bytes"].to_numpy()
    status = df["status"].to_numpy()
    rc = df["restart_count"].to_numpy()
    seg = df["segment"].to_numpy()
    events = crash_events(df)
    starts = restart_times(df, events)

    rows = []
    for s in np.unique(seg):
        idx = np.flatnonzero(seg == s)
        t0, t1 = ts[idx[0]], ts[idx[-1]]
        for t in np.arange(t0 + stride, t1 + 1e-9, stride):
            hi = np.searchsorted(ts, t, side="right")
            if status[hi - 1] != "Running":
                continue
            prev_starts = starts[starts <= t]
            lo = idx[0]
            if len(prev_starts):
                lo = max(lo, np.searchsorted(ts, prev_starts[-1], side="left"))
            h_ts, h_b = history_window(ts, b, lo, hi, t, sampler)
            if len(h_ts) < 2:
                continue
            since = t - prev_starts[-1] if len(prev_starts) else None
            feats = compute_features(h_ts, h_b, series.limit_bytes, int(rc[hi - 1]), since)
            nxt = events[events > t]
            row = dict(feats)
            row["t"] = t
            row["next_event_in_s"] = float(nxt[0] - t) if len(nxt) else np.inf
            row["last10_frac"] = b[max(hi - 10, lo):hi] / series.limit_bytes
            for h in (15, 30, 60, 120):
                j = np.searchsorted(ts, t + h, side="right") - 1
                valid = (t + h <= t1 and row["next_event_in_s"] > h
                         and status[j] == "Running" and seg[j] == s)
                row[f"future_bytes_{h}s"] = b[j] if valid else np.nan
            rows.append(row)
    return pd.DataFrame(rows), events


def label(examples, horizon_s):
    """y = 1 if the pod crashes within the next horizon_s seconds."""
    return (examples["next_event_in_s"] <= horizon_s).astype(int).to_numpy()
