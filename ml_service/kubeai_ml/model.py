"""Model bundle: load once, score one pod's samples into a prediction dict.

A bundle directory holds model.txt (LightGBM booster), calibrator.json
(isotonic map raw score -> probability, fit on the validation split) and
manifest.json (features, horizon, threshold, metrics, provenance).
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np

from . import features as F

MIB = 2 ** 20


def resolve_bundle_dir(model_dir):
    """`model_dir` may be a bundle directory or a LATEST pointer file."""
    path = Path(model_dir)
    if path.is_file():
        version = path.read_text().strip()
        path = path.parent / version
    return path


class OomModel:
    def __init__(self, booster, calibrator, manifest):
        self.booster = booster
        self.cal_x = np.asarray(calibrator["x"], dtype=float)
        self.cal_y = np.asarray(calibrator["y"], dtype=float)
        self.manifest = manifest
        self.feature_names = manifest["features"]
        self.threshold = float(manifest["threshold"])
        self.horizon_s = int(manifest["horizon_s"])
        self.version = manifest["model_version"]
        self.damping = float(manifest.get("forecast", {}).get("damping", 1.0))
        # Trend gate: is_anomaly also needs memory to be heading for the limit
        # (time_to_limit within this many seconds). The training data has
        # almost no pods that sit flat at high usage, so p_oom alone flags
        # every steady pod above ~2/3 of its limit. None = no gate.
        gate = manifest.get("alert_rule", {}).get("max_time_to_limit_s")
        self.max_ttl_s = None if gate is None else float(gate)

    @classmethod
    def load(cls, model_dir):
        path = resolve_bundle_dir(model_dir)
        booster = lgb.Booster(model_file=str(path / "model.txt"))
        calibrator = json.loads((path / "calibrator.json").read_text())
        manifest = json.loads((path / "manifest.json").read_text())
        return cls(booster, calibrator, manifest)

    def calibrate(self, raw):
        return np.clip(np.interp(raw, self.cal_x, self.cal_y), 0.0, 1.0)

    def p_oom_batch(self, X):
        """Calibrated probabilities for a feature matrix (training/eval use)."""
        return self.calibrate(self.booster.predict(X))

    def gate_ok(self, ttl_raw):
        """True when the trend gate lets an alert through (ttl_raw: s, None = not rising)."""
        if self.max_ttl_s is None:
            return True
        return ttl_raw is not None and ttl_raw <= self.max_ttl_s

    def alerts(self, p, ttl_raw):
        """Vectorised is_anomaly for training/eval (ttl_raw: array, NaN = not rising)."""
        p = np.asarray(p, dtype=float)
        if self.max_ttl_s is None:
            return p >= self.threshold
        ttl = np.asarray(ttl_raw, dtype=float)
        return (p >= self.threshold) & np.isfinite(ttl) & (ttl <= self.max_ttl_s)

    def p_oom(self, feats):
        return float(self.p_oom_batch(F.to_vector(feats, self.feature_names))[0])

    def predict(self, pod, namespace, samples, limit_bytes=None,
                restart_count=None, seconds_since_restart=None):
        ts = [s["ts"] for s in samples]
        ws = [s["working_set_bytes"] for s in samples]
        feats = F.compute_features(ts, ws, limit_bytes, restart_count, seconds_since_restart)
        trend = feats["trend_bytes_per_s"]
        has_limit = limit_bytes is not None and limit_bytes > 0

        if has_limit:
            p = self.p_oom(feats)
            usage = feats["frac_last"]
            ttl = feats["time_to_limit_raw_s"]
        else:
            p, usage, ttl = 0.0, None, None
        gate = self.gate_ok(ttl) if has_limit else False
        is_anomaly = bool(has_limit and p >= self.threshold and gate)

        return {
            "pod": pod,
            "namespace": namespace,
            "model_version": self.version,
            "p_oom": p,
            "is_anomaly": is_anomaly,
            "threshold": self.threshold,
            "horizon_s": self.horizon_s,
            "usage_fraction": usage,
            "trend_bytes_per_s": trend,
            "time_to_limit_s": ttl,
            "forecast": F.trend_forecast(feats["last_bytes"], trend,
                                         limit_bytes if has_limit else None,
                                         damping=self.damping),
            "reasons": self.reasons(feats, p, is_anomaly, has_limit, ts, restart_count, gate),
        }

    def reasons(self, feats, p, is_anomaly, has_limit, ts, restart_count, gate=True):
        out = []
        trend_mib = feats["trend_bytes_per_s"] / MIB
        if not has_limit:
            out.append("no memory limit set: the container cannot be OOMKilled by its own "
                       "limit, so p_oom is not scored (0); trend and forecast still reported")
        else:
            out.append(f"usage {feats['frac_last']:.0%} of limit "
                       f"({feats['last_bytes'] / MIB:.0f} MiB)")
        if trend_mib > 0.01:
            msg = f"rising {trend_mib:.2f} MiB/s"
            ttl = feats.get("time_to_limit_raw_s")
            if has_limit and ttl is not None:
                msg += f", limit in ~{ttl:.0f}s"
            out.append(msg)
        elif trend_mib < -0.01:
            out.append(f"falling {-trend_mib:.2f} MiB/s")
        else:
            out.append("memory flat over the last 2 min")
        if has_limit:
            verdict = ">=" if p >= self.threshold else "<"
            out.append(f"P(OOM within {self.horizon_s}s) = {p:.2f} {verdict} threshold "
                       f"{self.threshold:.2f}")
            if p >= self.threshold and not gate:
                out.append(f"not flagged: memory is not on course to reach the limit within "
                           f"{self.max_ttl_s:.0f}s (alerts need a rising trend; the model has "
                           "seen almost no pods that stay flat at high usage)")
        span = max(ts) - min(ts)
        if span < F.HISTORY_S:
            out.append(f"only {span:.0f}s of history (model uses up to {F.HISTORY_S}s)")
        if restart_count:
            out.append(f"container has restarted {restart_count} times")
        return out
