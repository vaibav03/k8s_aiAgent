"""Markdown tables from a bundle's metrics.json.

    PYTHONPATH=. python -m kubeai_ml.scaling.report [bundle_dir]     # default models/scaling/LATEST
"""
from __future__ import annotations

import json
import sys
from typing import Any

from .bundle import resolve


def md(rows: list[dict[str, Any]], cols: list[str]) -> str:
    def fmt(v: Any) -> str:
        if isinstance(v, float):
            return f"{v:.3f}".rstrip("0").rstrip(".") if abs(v) < 1000 else f"{v:.0f}"
        return "" if v is None else str(v)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(fmt(r.get(c)) for c in cols) + " |" for r in rows]
    return "\n".join(out)


def classifier(m: dict) -> str:
    cols = ["model", "macro_f1_strict", "macro_f1_settled", "accuracy_settled", "false_scale_up_on_fault_rows",
            "false_scale_up_on_fault_rows_excl_spike_overlap", "false_page_rate_on_load_rows", "missed_scale_up"]
    parts = []
    for split in ("loro", "holdout"):
        rows = [{"model": k, **v} for k, v in m["classifier"][split].items()]
        parts.append(f"### classifier, {split}\n" + md(rows, cols))
        ch = m["classifier"]["chosen"]
        pc = [{"class": c, **v} for c, v in m["classifier"][split][ch]["per_class_settled"].items()]
        parts.append(f"per class ({ch}, settled, {split})\n" + md(pc, ["class", "precision", "recall", "f1", "n"]))
        if split == "holdout" and "holdout_blind" in m["classifier"]:
            bl = [{"variant": k, **{kk: v[kk] for kk in ("macro_f1_strict", "macro_f1_settled", "accuracy_settled",
                                                          "rows_scored_settled")}}
                  for k, v in m["classifier"]["holdout_blind"].items()]
            parts.append("blind held-out variants\n" + md(bl, ["variant", "macro_f1_strict", "macro_f1_settled",
                                                               "accuracy_settled", "rows_scored_settled"]))
            pcs = [{"class": c, **v} for c, v in m["classifier"][split][ch].get("per_class_strict", {}).items()]
            parts.append(f"per class ({ch}, strict, {split})\n" + md(pcs, ["class", "precision", "recall", "f1", "n"]))
        det = m["classifier"][split][ch]["detection"]
        parts.append(f"detection ({ch}, {split})\n" + md(det, ["fault_id", "type", "delay_s", "any_fault_alert_s", "row_recall"]))
    return "\n\n".join(parts)


def forecast(m: dict) -> str:
    cols = ["h", "method", "pinball50", "pinball90", "rel_pinball_mean_pct", "cov90", "cov90_surge", "n_surge", "q90_over_y", "n"]
    parts = [f"### forecast, {s}\n" + md(m["forecast"][s], cols) for s in ("loro", "holdout")]
    fp = m["forecast"].get("flat_traffic_probe")
    if fp:
        parts.append("flat-traffic probe (p90 / current rate)\n" + md(fp, list(fp[0])))
    return "\n\n".join(parts)


def capacity(m: dict) -> str:
    c = m["capacity"]
    parts = []
    for s in ("loro", "holdout"):
        parts.append(f"### capacity models, {s}\n" + md(c[f"{s}_models"], ["model", "logloss", "brier", "auc",
                                                                           "bad_deploy_mean_p", "bad_deploy_breach_rate"]))
        parts.append(f"capacity as a rule, {s}\n" + md(c[f"{s}_rules"], ["rule", "recall", "precision",
                                                                          "false_alarm_rate", "bad_deploy_recall"]))
    parts.append("safe rps per pod\n" + md(c["safe_table"], list(c["safe_table"][0])))
    if "breach_rows_by_run" in c:
        parts.append(f"breach rows by run\n```\n{json.dumps(c['breach_rows_by_run'], indent=1)}\n```")
    return "\n\n".join(parts)


def simulation(m: dict) -> str:
    s = m.get("simulation")
    if not s:
        return ""
    cols = ["max_cap", "policy", "traces", "slo_violation_min", "violation_events", "pod_minutes", "pod_min_over_need",
            "scaling_actions", "max_replicas", "ttr_after_spike_max_s", "dropped_pct", "gate_false_fire_pct"]
    parts = [f"### simulator validation (HPA replay)\n```\n{json.dumps(s['hpa_replay_validation'], indent=1)}\n```",
             "### policy sweep (dev traces, cap 8)\n" + md(s["policy_sweep_dev"], list(s["policy_sweep_dev"][0]))]
    for split in ("dev", "synthetic", "holdout", "holdout_warm"):
        if f"table_{split}" not in s:
            continue
        parts.append(f"### policies, {split} traces\n" + md(s[f"table_{split}"], cols))
        cm = s.get(f"cost_matched_{split}")
        if cm:
            parts.append(f"cost-matched, {split}\n" + md(cm, list(cm[0])))
    pt = [r for r in s["per_trace"] if r.get("split") == "holdout"]
    parts.append("### per trace, holdout\n" + md(pt, ["trace", "max_cap", "policy", "slo_violation_min",
                                                       "violation_events", "pod_minutes", "pod_min_over_need",
                                                       "scaling_actions"]))
    return "\n\n".join(parts)


def faults(m: dict) -> str:
    f = m.get("fault_runs")
    if not f:
        return ""
    cols = ["run", "split", "fault", "lab_policy", "engine_top_state", "engine_state_recall", "engine_upscale_pct",
            "engine_blocked_pct", "actual_replicas_at_start", "actual_peak_replicas", "actual_scale_ups",
            "actual_extra_pod_min", "slo_viol_min", "viol_frac_before_first_upscale", "viol_frac_after_first_upscale",
            "engine_restart_actions", "engine_restart_on_target_pct", "engine_rollback_pct"]
    return "### fault runs: engine (open loop) vs actual HPA\n" + md(f["per_fault"], cols)


def main() -> None:
    path = resolve(sys.argv[1] if len(sys.argv) > 1 else None)
    m = json.loads((path / "metrics.json").read_text())
    print(f"# {path.name}\n")
    for part in (classifier(m), forecast(m), capacity(m), simulation(m), faults(m)):
        if part:
            print(part + "\n")


if __name__ == "__main__":
    main()
