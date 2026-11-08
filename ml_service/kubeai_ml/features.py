"""The one feature definition shared by training and serving.

Input is a pod's raw memory samples (unix seconds, working-set bytes), its
memory limit in bytes (or None) and optional restart info. Output is a flat
dict of features. Every step is deterministic:

1. Clean: drop non-finite / negative samples, sort by time, and drop stale
   duplicates (a value identical to the previous sample is a re-read of the
   same Prometheus/metrics-server scrape, which happens every 15-30 s while
   collectors poll every 1 s).
2. Resample onto a fixed 15 s grid anchored at the newest sample ("now"),
   reaching back HISTORY_S seconds. Each grid point takes the last sample at or
   before it (forward fill). Points before the first sample are NaN.
3. usage fraction = working_set / limit.
4. For each window W in 30/60/120/300 s (grid points with t >= now - W):
   last, mean, max, std and a Theil-Sen slope (median of pairwise slopes,
   fraction per second).
5. headroom = 1 - last fraction; time_to_limit_s = headroom / slope over the
   TREND_WINDOW_S window when that slope is positive (else TTL_CAP_S).
6. restart_count and seconds_since_restart are passed through (NaN if unknown).

With no limit, fraction features are NaN; byte-level trend is still computed.
"""
from __future__ import annotations

import math

import numpy as np

STEP_S = 15
WINDOWS_S = (30, 60, 120, 300)
HISTORY_S = 300
TREND_WINDOW_S = 120
TTL_CAP_S = 3600.0
FORECAST_HORIZONS_S = (15, 30, 60, 120)

FRACTION_FEATURES = ["frac_last"] + [
    f"frac_{stat}_{w}s" for w in WINDOWS_S for stat in ("mean", "max", "std", "slope")
] + ["headroom", "time_to_limit_s"]


def clean_samples(ts, values):
    """Sort, drop invalid values and consecutive stale duplicates."""
    ts = np.asarray(ts, dtype=float)
    values = np.asarray(values, dtype=float)
    ok = np.isfinite(ts) & np.isfinite(values) & (values >= 0)
    ts, values = ts[ok], values[ok]
    order = np.argsort(ts, kind="stable")
    ts, values = ts[order], values[order]
    if len(values) == 0:
        return ts, values
    keep = np.ones(len(values), dtype=bool)
    keep[1:] = values[1:] != values[:-1]
    # Keep the newest sample too, so "now" is the latest observation time.
    keep[-1] = True
    return ts[keep], values[keep]


def resample(ts, values, step=STEP_S, history=HISTORY_S):
    """Forward-fill samples onto a grid ending at the newest timestamp.

    Returns (grid_ts, grid_values); grid points before the first sample are NaN.
    """
    ts, values = clean_samples(ts, values)
    if len(ts) == 0:
        return np.array([]), np.array([])
    n = int(history // step) + 1
    grid = ts[-1] - step * np.arange(n - 1, -1, -1, dtype=float)
    idx = np.searchsorted(ts, grid, side="right") - 1
    out = np.where(idx >= 0, values[np.clip(idx, 0, None)], np.nan)
    return grid, out


def theil_sen_slope(t, v):
    """Median of pairwise slopes; 0.0 if fewer than two points."""
    ok = np.isfinite(v)
    t, v = t[ok], v[ok]
    if len(t) < 2:
        return 0.0
    i, j = np.triu_indices(len(t), k=1)
    dt = t[j] - t[i]
    return float(np.median((v[j] - v[i]) / dt))


def window_stats(grid_t, grid_v, window):
    """last/mean/max/std/slope over grid points within `window` seconds of now."""
    sel = grid_t >= grid_t[-1] - window
    t, v = grid_t[sel], grid_v[sel]
    finite = v[np.isfinite(v)]
    if len(finite) == 0:
        return dict(mean=np.nan, max=np.nan, std=np.nan, slope=0.0)
    return dict(
        mean=float(finite.mean()),
        max=float(finite.max()),
        std=float(finite.std()),
        slope=theil_sen_slope(t, v),
    )


def compute_features(ts, working_set_bytes, limit_bytes=None,
                     restart_count=None, seconds_since_restart=None):
    """Feature dict for the newest point of one pod's sample series."""
    grid_t, grid_b = resample(ts, working_set_bytes)
    if len(grid_t) == 0:
        raise ValueError("no valid samples")

    last_bytes = float(grid_b[-1])
    trend_bytes_per_s = window_stats(grid_t, grid_b, TREND_WINDOW_S)["slope"]
    feats = {
        "last_bytes": last_bytes,
        "trend_bytes_per_s": trend_bytes_per_s,
        "restart_count": float(restart_count) if restart_count is not None else np.nan,
        "seconds_since_restart": (float(seconds_since_restart)
                                  if seconds_since_restart is not None else np.nan),
    }

    has_limit = limit_bytes is not None and limit_bytes > 0
    if not has_limit:
        feats.update({name: np.nan for name in FRACTION_FEATURES})
        feats["time_to_limit_raw_s"] = None
        return feats

    frac = grid_b / float(limit_bytes)
    feats["frac_last"] = float(frac[-1])
    for w in WINDOWS_S:
        for stat, value in window_stats(grid_t, frac, w).items():
            feats[f"frac_{stat}_{w}s"] = value

    headroom = 1.0 - feats["frac_last"]
    slope = feats[f"frac_slope_{TREND_WINDOW_S}s"]
    if headroom <= 0:
        ttl = 0.0
    elif slope > 1e-9:
        ttl = headroom / slope
    else:
        ttl = math.inf
    feats["headroom"] = headroom
    feats["time_to_limit_raw_s"] = None if math.isinf(ttl) else float(ttl)
    feats["time_to_limit_s"] = float(min(ttl, TTL_CAP_S))
    return feats


def trend_forecast(last_bytes, trend_bytes_per_s, limit_bytes=None,
                   horizons=FORECAST_HORIZONS_S, damping=1.0):
    """Robust-trend extrapolation, clipped to [0, limit]."""
    out = []
    for h in horizons:
        value = last_bytes + damping * trend_bytes_per_s * h
        value = max(value, 0.0)
        if limit_bytes:
            value = min(value, float(limit_bytes))
        out.append({"in_s": int(h), "working_set_bytes": float(value)})
    return out


def to_vector(feats, feature_names):
    """Order a feature dict into the model's input row (NaN for missing)."""
    return np.array([[feats.get(name, np.nan) if feats.get(name) is not None else np.nan
                      for name in feature_names]], dtype=float)
