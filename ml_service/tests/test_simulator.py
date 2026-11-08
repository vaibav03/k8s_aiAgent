"""Happy-path checks for the replay simulator (no dataset fit needed: default PodModel knots)."""
import json

import pytest

from kubeai_ml.scaling.hpa import HPAPolicy
from kubeai_ml.scaling.podmodel import PodModel
from kubeai_ml.scaling.simulator import CallablePolicy, FixedPolicy, SimConfig, simulate, summarize_sim
from kubeai_ml.scaling.traces import RUNS_DIR, shape_points, synthetic_trace


def _model() -> PodModel:
    knots = {n: ([0.0, 8.0, 12.0, 14.0], [0.03, 0.03, 0.07 + 0.01 * n, 0.12 + 0.03 * n]) for n in (1, 2, 3, 4)}
    return PodModel(lat_knots=knots)


def test_shape_port_matches_k6_points():
    runs = sorted(p for p in RUNS_DIR.glob("b-hpa-*") if (p / "k6_summary.json").exists())
    if not runs:
        pytest.skip("no loadlab runs on disk")
    for p in runs:
        spec = json.loads((p / "k6_summary.json").read_text())["loadlab"]
        pts = shape_points(spec["shape"], spec["base_rps"], spec["duration_s"], spec["params"])
        assert len(pts) == len(spec["points"])
        assert all(abs(a[1] - b[1]) < 1e-9 for a, b in zip(pts, spec["points"]))


def test_fixed_sizes_bracket_the_slo():
    tr = synthetic_trace("steady30", "steady", 30.0, 300)
    m = _model()
    big = summarize_sim(simulate(tr, FixedPolicy(4), m, SimConfig(initial_replicas=4, max_replicas=4)))
    small = summarize_sim(simulate(tr, FixedPolicy(1), m, SimConfig(initial_replicas=1, max_replicas=1)))
    assert big["slo_violation_min"] == 0
    assert small["slo_violation_min"] > 2 and small["dropped_pct"] > 10


def test_hpa_scales_up_on_flash_crowd():
    tr = synthetic_trace("flash", "flash_crowd", 10.0, 400, {"at_s": 100, "mult": 4})
    res = simulate(tr, HPAPolicy(0.7), _model(), SimConfig(max_replicas=6))
    assert res.df["true_spec"].iloc[-1] >= 5
    assert all(b > a for _, a, b in res.actions)          # no scale-down within 300 s window


def test_policy_sees_only_observable_columns():
    seen = []

    def fn(history, state):
        seen.append(list(history.columns))
        return 2

    simulate(synthetic_trace("s", "steady", 10.0, 60), CallablePolicy(fn), _model())
    assert seen and not any(c.startswith("true_") for c in seen[-1])


def test_hpa_sd30_scales_back_in_after_spike():
    tr = synthetic_trace("spike", "spike", 8.0, 600, {"spike_at_s": 100, "spike_mult": 4, "spike_s": 120})
    res = simulate(tr, HPAPolicy(0.7, scale_down_window_s=30.0), _model(), SimConfig(max_replicas=6))
    peak = res.df["true_spec"].max()
    assert peak >= 3 and res.df["true_spec"].iloc[-1] < peak
    assert any(b < a for _, a, b in res.actions)
