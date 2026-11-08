"""State classifiers compared in ``train_classifier``; ``hybrid`` is the shipped design.

* ``rules``     runbook only (``rules.predict``).
* ``lgbm``      one LightGBM multiclass over all 10 states.
* ``two_stage`` LightGBM fault-vs-load, then LightGBM subtype within each branch.
* ``hybrid``    runbook fault gates (physics signatures, generalise to fault types seen once)
                then LightGBM over the load states (healthy / scale_up / scale_down), which have
                hundreds of rows from many runs.

All models consume ``features.FEATURES`` plus capacity-normalised load features computed from a
learned ``labels.SafeCapacity`` (``capacity_features``), so the knee is learned, not hardcoded.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import rules
from . import states as S
from .features import FEATURES
from .labels import TARGET_UTIL, SafeCapacity

CAP_FEATURES = ["load_frac", "load_frac_1m", "load_frac_max_3m", "proj_load_frac_1m", "down_fit"]
MODEL_FEATURES = FEATURES + CAP_FEATURES
MODES = ("rules", "lgbm", "two_stage", "hybrid")

LGB_PARAMS = dict(n_estimators=120, learning_rate=0.08, num_leaves=15, min_child_samples=10,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                  class_weight="balanced", verbose=-1, random_state=0, n_jobs=1)


def capacity_features(f: pd.DataFrame, cap: SafeCapacity) -> pd.DataFrame:
    """Add load normalised by the learned knee c(r) and the 'fits in r-1 pods' ratio."""
    out = f.copy()
    r = np.maximum(out["replicas_ready"].fillna(1).to_numpy(float), 1)
    c = cap.at(r)
    out["load_frac"] = out["load_per_pod"] / c
    out["load_frac_1m"] = out["load_per_pod_1m"] / c
    out["load_frac_max_3m"] = out["load_per_pod_max_3m"] / c
    out["proj_load_frac_1m"] = (out["load_per_pod_1m"] + out["load_trend_1m"].clip(lower=0).fillna(0)) / c
    fewer = np.maximum(r - 1, 1)
    fit = out["load_per_pod_max_3m"].to_numpy(float) * r / (TARGET_UTIL * cap.at(fewer) * fewer)
    out["down_fit"] = np.where(r >= 2, fit, np.nan)
    return out


def _lgbm(x: pd.DataFrame, y: np.ndarray) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(**LGB_PARAMS).fit(x[MODEL_FEATURES], y)


@dataclass
class StateClassifier:
    mode: str = "hybrid"
    cap: SafeCapacity = field(default_factory=SafeCapacity)
    models: dict[str, lgb.LGBMClassifier] = field(default_factory=dict)

    def fit(self, f: pd.DataFrame, y: np.ndarray) -> "StateClassifier":
        x = capacity_features(f, self.cap)
        y = np.asarray(y, dtype=object)
        is_fault = np.isin(y, S.FAULT_STATES)
        if self.mode == "lgbm":
            self.models["all"] = _lgbm(x, y)
        elif self.mode == "two_stage":
            self.models["stage1"] = _lgbm(x, is_fault.astype(int))
            self.models["fault"] = _lgbm(x[is_fault], y[is_fault])
            self.models["load"] = _lgbm(x[~is_fault], y[~is_fault])
        elif self.mode == "hybrid":
            self.models["load"] = _lgbm(x[~is_fault], y[~is_fault])
        return self

    def predict(self, f: pd.DataFrame) -> np.ndarray:
        x = capacity_features(f, self.cap)
        if self.mode == "rules":
            return rules.predict(x)
        if self.mode == "lgbm":
            return self.models["all"].predict(x[MODEL_FEATURES]).astype(object)
        load = self.models["load"].predict(x[MODEL_FEATURES]).astype(object)
        if self.mode == "two_stage":
            fault = self.models["fault"].predict(x[MODEL_FEATURES]).astype(object)
            is_fault = self.models["stage1"].predict(x[MODEL_FEATURES]).astype(bool)
            return np.where(is_fault, fault, load)
        gate = rules.fault_gate(x)
        return np.where(gate != None, gate, load).astype(object)  # noqa: E711

    def decide(self, feats: dict[str, float]) -> dict[str, object]:
        """Serving: one feature dict (``features.features_at``) -> state + remediation."""
        state = str(self.predict(pd.DataFrame([feats]))[0])
        return {"state": state, "remediation": S.REMEDIATION[state],
                "scaling_helps": S.SCALING_HELPS[state]}

    # ------------------------------------------------------------------ persistence
    def save(self, out_dir: Path, extra: dict[str, object] | None = None) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, m in self.models.items():
            m.booster_.save_model(str(out_dir / f"{name}.txt"))
        classes = {n: [str(c) for c in m.classes_] for n, m in self.models.items()}
        meta = {"mode": self.mode, "capacity": self.cap.to_dict(), "features": MODEL_FEATURES,
                "classes": classes, **(extra or {})}
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=float))

    @classmethod
    def load(cls, out_dir: Path) -> "StateClassifier":
        meta = json.loads((out_dir / "meta.json").read_text())
        cap = SafeCapacity({int(k): v for k, v in meta["capacity"]["table"].items()},
                           meta["capacity"]["fallback"])
        obj = cls(mode=meta["mode"], cap=cap)
        for name, classes in meta["classes"].items():
            obj.models[name] = _BoosterClassifier(out_dir / f"{name}.txt", classes)  # type: ignore[assignment]
        return obj


class _BoosterClassifier:
    """Minimal predict() over a saved booster (avoids pickling sklearn wrappers)."""

    def __init__(self, path: Path, classes: list[str]):
        self.booster = lgb.Booster(model_file=str(path))
        self.classes_ = np.array(classes, dtype=object)

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        p = self.booster.predict(x)
        idx = np.argmax(p, axis=1) if p.ndim == 2 else (p >= 0.5).astype(int)
        out = self.classes_[idx]
        if set(self.classes_) <= {"0", "1"}:
            return out.astype(int)
        return out
