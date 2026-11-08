"""Layout of a scaling bundle directory (``models/scaling/<version>/``) and its loader.

::

    <version>/
      manifest.json          version, data sha256, split, chosen models, headline metrics + baselines
      metrics.json           every evaluation table (CV, held-out, baselines, simulator, fault runs)
      feature_spec.json      engine input schema (PromQL), state features, forecast features
      label_rules.json       labelling rules and fault-gate thresholds
      policy.json            replica policy parameters (risk, stabilisation, defaults)
      capacity.json          logistic P(SLO breach | utilisation, replicas)
      classifier/            meta.json + LightGBM boosters + calibration.json (P(true | predicted))
      forecast/              6 LightGBM quantile boosters + forecast_meta.json
      sim_calibration.json   pod model the policy evaluation ran on
    LATEST                   text file naming the shipped version
"""
from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from .capacity import CapacityModel
from .classifier import StateClassifier
from .data import MODELS_DIR
from .forecast import LgbmQuantileForecaster
from .policy import Stabilization

LATEST = MODELS_DIR / "LATEST"


def resolve(bundle_dir: str | Path | None = None) -> Path:
    """A bundle directory, a LATEST pointer file, or None (= models/scaling/LATEST)."""
    path = Path(bundle_dir) if bundle_dir is not None else LATEST
    if path.is_file():
        path = path.parent / path.read_text().strip()
    if not (path / "manifest.json").exists():
        raise FileNotFoundError(f"no scaling bundle at {path}")
    return path


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=_json_default))


def _json_default(o: Any) -> Any:
    try:
        import numpy as np
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return None if o != o else float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:  # pragma: no cover
        pass
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


@dataclass(frozen=True)
class PolicyParams:
    """Everything the replica policy needs besides the models (stored as policy.json)."""
    risk: float = 0.10
    stabilization: Stabilization = Stabilization()
    decision_period_s: float = 15.0
    default_cpu_per_req_s: float = 0.0265
    cpu_limit_cores: float = 0.5
    cpu_request_cores: float = 0.25
    history_s: float = 1800.0          # history the engine keeps for features (cpu/req baseline)

    def to_dict(self) -> dict[str, Any]:
        st = {f.name: getattr(self.stabilization, f.name) for f in fields(Stabilization)}
        return {"risk": self.risk, "stabilization": st, "decision_period_s": self.decision_period_s,
                "default_cpu_per_req_s": self.default_cpu_per_req_s,
                "cpu_limit_cores": self.cpu_limit_cores, "cpu_request_cores": self.cpu_request_cores,
                "history_s": self.history_s}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PolicyParams":
        st = dict(d["stabilization"])
        for k in ("up_horizons_s", "down_horizons_s"):
            st[k] = tuple(int(x) for x in st[k])
        return cls(risk=float(d["risk"]), stabilization=Stabilization(**st),
                   decision_period_s=float(d["decision_period_s"]),
                   default_cpu_per_req_s=float(d["default_cpu_per_req_s"]),
                   cpu_limit_cores=float(d["cpu_limit_cores"]),
                   cpu_request_cores=float(d["cpu_request_cores"]), history_s=float(d["history_s"]))


@dataclass
class Bundle:
    path: Path
    manifest: dict[str, Any]
    classifier: StateClassifier
    calibration: dict[str, dict[str, float]]      # predicted state -> {true state: P}
    forecaster: LgbmQuantileForecaster
    capacity: CapacityModel
    policy: PolicyParams

    @classmethod
    def load(cls, bundle_dir: str | Path | None = None) -> "Bundle":
        path = resolve(bundle_dir)
        manifest = json.loads((path / "manifest.json").read_text())
        cal_path = path / "classifier" / "calibration.json"
        calibration = json.loads(cal_path.read_text()) if cal_path.exists() else {}
        policy = PolicyParams.from_dict(json.loads((path / "policy.json").read_text()))
        capacity = CapacityModel.load(path / "capacity.json")
        capacity.risk = policy.risk
        return cls(path=path, manifest=manifest, classifier=StateClassifier.load(path / "classifier"),
                   calibration=calibration, forecaster=LgbmQuantileForecaster.load(path / "forecast"),
                   capacity=capacity, policy=policy)
