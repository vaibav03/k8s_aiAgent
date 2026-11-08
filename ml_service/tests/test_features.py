import math

import numpy as np
import pytest

from kubeai_ml import features as F

MIB = 2 ** 20
LIMIT = 900 * MIB


def series(values, start=1_700_000_000.0, step=1.0):
    ts = [start + i * step for i in range(len(values))]
    return ts, list(values)


def test_resample_forward_fills_onto_15s_grid_anchored_at_newest_sample():
    ts, ws = [0.0, 1.0, 2.0, 40.0], [10.0, 10.0, 10.0, 20.0]
    grid_t, grid_v = F.resample(ts, ws)
    assert len(grid_t) == F.HISTORY_S // F.STEP_S + 1
    assert grid_t[-1] == 40.0 and np.allclose(np.diff(grid_t), F.STEP_S)
    # 25 s and 10 s see the value from t=0..2; 40 s sees the new value.
    assert grid_v[-1] == 20.0 and grid_v[-2] == 10.0 and grid_v[-3] == 10.0
    assert np.isnan(grid_v[0])  # before the first sample


def test_clean_drops_stale_duplicates_and_sorts():
    ts, ws = F.clean_samples([3, 1, 2, 4], [5, 5, 5, 6])
    assert list(ts) == [1, 4] and list(ws) == [5, 6]


def test_theil_sen_slope_is_robust_to_one_spike():
    t = np.arange(0, 150, 15, dtype=float)
    v = 2.0 * t
    v[4] += 10_000
    assert F.theil_sen_slope(t, v) == pytest.approx(2.0)


def test_linear_leak_gives_slope_and_time_to_limit():
    rate = 1 * MIB  # bytes per second
    ts, ws = series([0.5 * LIMIT + rate * i for i in range(301)])
    f = F.compute_features(ts, ws, LIMIT)
    assert f["trend_bytes_per_s"] == pytest.approx(rate)
    assert f["frac_slope_120s"] == pytest.approx(rate / LIMIT)
    headroom = 1 - f["frac_last"]
    assert f["time_to_limit_s"] == pytest.approx(headroom * LIMIT / rate)
    assert f["headroom"] == pytest.approx(headroom)


def test_flat_series_has_no_trend_and_capped_ttl():
    ts, ws = series([0.2 * LIMIT] * 120)
    f = F.compute_features(ts, ws, LIMIT, restart_count=3, seconds_since_restart=50)
    assert f["frac_last"] == pytest.approx(0.2)
    assert f["frac_slope_300s"] == 0.0
    assert f["time_to_limit_s"] == F.TTL_CAP_S and f["time_to_limit_raw_s"] is None
    assert f["restart_count"] == 3 and f["seconds_since_restart"] == 50


def test_no_limit_leaves_fraction_features_nan_but_keeps_trend():
    ts, ws = series([100 * MIB + i * MIB for i in range(60)])
    f = F.compute_features(ts, ws, None)
    assert all(math.isnan(f[name]) for name in F.FRACTION_FEATURES)
    assert f["trend_bytes_per_s"] == pytest.approx(MIB)


def test_features_are_deterministic_and_order_independent():
    rng = np.random.default_rng(0)
    ts, ws = series(rng.uniform(0.1, 0.9, 200) * LIMIT)
    a = F.compute_features(ts, ws, LIMIT)
    b = F.compute_features(ts[::-1], ws[::-1], LIMIT)
    assert F.to_vector(a, F.FRACTION_FEATURES).tolist() == F.to_vector(b, F.FRACTION_FEATURES).tolist()


def test_trend_forecast_is_clipped_to_limit():
    out = F.trend_forecast(0.95 * LIMIT, 1 * MIB, LIMIT)
    assert [p["in_s"] for p in out] == list(F.FORECAST_HORIZONS_S)
    assert out[0]["working_set_bytes"] == pytest.approx(0.95 * LIMIT + 15 * MIB)
    assert out[-1]["working_set_bytes"] == LIMIT


def _series_with_restart():
    import pandas as pd
    from kubeai_ml.data import Series
    # 1 s rows, value changes every 15 s; 600 s at 85% -> crash (CrashLoopBackOff 60 s)
    # -> restart at low usage.
    rows = []
    for t in range(0, 900):
        if t < 600:
            v, st, rc = 0.85 * LIMIT, "Running", 0
        elif t < 660:
            v, st, rc = 0.1 * LIMIT, "CrashLoopBackOff", 0
        else:
            v, st, rc = (0.2 + 0.0001 * (t // 15 * 15 - 660)) * LIMIT, "Running", 1
        rows.append((float(t), v, st, rc))
    df = pd.DataFrame(rows, columns=["ts", "bytes", "status", "restart_count"])
    df["segment"] = 0
    return Series("synthetic", df, LIMIT, "test", "0")


def test_examples_after_restart_see_only_post_restart_history():
    from kubeai_ml.data import build_examples
    ex, events = build_examples(_series_with_restart(), stride=15)
    after = ex[ex["t"] > 660]
    assert len(after) and (after["frac_max_300s"] < 0.5).all()  # no pre-crash 85% samples


def test_tick_sampler_matches_agent_shape():
    from kubeai_ml.data import history_window, AGENT_POLL_S, AGENT_WINDOW
    ts = np.arange(0, 1000, 1.0)
    b = np.repeat(np.arange(100), 10)[:1000].astype(float)
    h_ts, h_b = history_window(ts, b, 0, 1000, 999.0, "tick")
    assert len(h_ts) == AGENT_WINDOW and np.allclose(np.diff(h_ts), AGENT_POLL_S)
    assert h_b[-1] == b[999] and h_b[0] == b[int(999 - AGENT_POLL_S * (AGENT_WINDOW - 1))]
