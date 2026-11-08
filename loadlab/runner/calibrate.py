"""Analyse a capacity-probe run (1 pod, step_ramp) and write loadlab/calibration.json.

For each step we take the steady part of the hold (skip the first `settle_s` seconds) and compute,
directly from Prometheus over that window: achieved RPS, p50/p95/p99 (histogram over the whole
window, not an average of percentiles), error ratio, CPU cores, throttle ratio.

per_pod_rps_at_slo   = highest offered RPS step with p95 < 300ms, errors < 1% and achieved >= 95% offered
per_pod_max_rps      = highest achieved RPS of any step (throughput ceiling)

    python runner/calibrate.py runs/<capacity-probe-run-id>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import promexport  # noqa: E402

SLO_P95_S = 0.300
SLO_ERR = 0.01


def q1(query: str, t: float) -> float | None:
    r = promexport.prom_instant(query, t)
    if not r:
        return None
    v = float(r[0]["value"][1])
    return None if v != v else v  # NaN -> None


def main(run_dir: Path, settle_s: int = 15):
    sc = json.loads((run_dir / "scenario.json").read_text())
    ns, t0 = sc["_run"]["namespace"], sc["_run"]["t0"]
    p = sc["traffic"]["params"]
    step_s, start, inc = p["step_s"], p["start_rps"], p["step_rps"]
    n = int(sc["duration_s"] // step_s)
    a = f'namespace="{ns}"'
    c = f'namespace="{ns}",pod=~"loadlab-.*",scope="pod"'
    steps = []
    for i in range(n):
        w0, w1 = t0 + i * step_s + settle_s, t0 + (i + 1) * step_s
        W = f"{int(w1 - w0)}s"
        WC = f"{max(int(w1 - w0), 40)}s"  # cAdvisor refreshes every ~15s: need a longer window
        offered = start + i * inc
        row = {"step": i, "offered_rps": offered, "window": [round(w0), round(w1)]}
        row["achieved_rps"] = q1(f'sum(rate(http_requests_total{{{a}}}[{W}]))', w1)
        for qq in (0.5, 0.95, 0.99):
            row[f"p{int(qq * 100)}_s"] = q1(
                f'histogram_quantile({qq}, sum by (le)(increase(http_request_duration_seconds_bucket{{{a}}}[{W}])))', w1)
        row["error_ratio"] = q1(f'sum(rate(http_requests_total{{{a},code=~"5.."}}[{W}])) / '
                                f'sum(rate(http_requests_total{{{a}}}[{W}]))', w1) or 0.0
        row["cpu_cores"] = q1(f'sum(rate(container_cpu_usage_seconds_total{{{c}}}[{WC}]))', w1)
        row["throttle_ratio"] = q1(f'sum(rate(container_cpu_cfs_throttled_periods_total{{{c}}}[{WC}])) / '
                                   f'sum(rate(container_cpu_cfs_periods_total{{{c}}}[{WC}]))', w1)
        row["cpu_cores_proc"] = q1(f'sum(rate(process_cpu_seconds_total{{{a},job="annotated-pods"}}[{W}]))', w1)
        row["inflight_avg"] = q1(f'avg_over_time(sum(http_inflight_requests{{{a}}})[{W}:5s])', w1)
        ach = row["achieved_rps"] or 0
        row["meets_slo"] = bool(row["p95_s"] is not None and row["p95_s"] < SLO_P95_S
                                and row["error_ratio"] < SLO_ERR and ach >= 0.95 * offered)
        steps.append(row)
        print(json.dumps(row))
    ok = [s for s in steps if s["meets_slo"]]
    at_slo = max((s["offered_rps"] for s in ok), default=None)
    max_rps = max((s["achieved_rps"] or 0) for s in steps)
    cpu_per_req = [s["cpu_cores_proc"] / s["achieved_rps"] for s in steps
                   if s["cpu_cores_proc"] and s["achieved_rps"] and s["meets_slo"]]
    cal = {
        "source_run": run_dir.name,
        "slo": {"p95_s": SLO_P95_S, "error_ratio": SLO_ERR},
        "app": sc["app"],
        "pod_resources": {"cpu_request": "250m", "cpu_limit": "500m", "mem_request": "128Mi", "mem_limit": "256Mi"},
        "per_pod_rps_at_slo": at_slo,
        "per_pod_max_rps": round(max_rps, 2),
        "cpu_core_seconds_per_req": round(sum(cpu_per_req) / len(cpu_per_req), 5) if cpu_per_req else None,
        "hpa_70pct_cpu_equiv_rps_per_pod": None,
        "steps": steps,
        "notes": ("per_pod_rps_at_slo = highest offered step meeting p95<300ms, err<1%, achieved>=95% offered. "
                  "Use traffic.base_load_pods in scenarios to express load in pods-worth of SLO capacity."),
    }
    if cal["cpu_core_seconds_per_req"]:
        # RPS at which one pod sits at the HPA target (70% of the 250m request = 175m)
        cal["hpa_70pct_cpu_equiv_rps_per_pod"] = round(0.175 / cal["cpu_core_seconds_per_req"], 2)
    out = HERE.parent / "calibration.json"
    out.write_text(json.dumps(cal, indent=2))
    print(f"\nper_pod_rps_at_slo={at_slo}  per_pod_max_rps={cal['per_pod_max_rps']}  "
          f"cpu_s_per_req={cal['cpu_core_seconds_per_req']}  hpa70_rps={cal['hpa_70pct_cpu_equiv_rps_per_pod']}"
          f"\nwrote {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
