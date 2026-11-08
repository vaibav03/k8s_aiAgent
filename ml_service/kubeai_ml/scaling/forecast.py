"""Demand forecaster: request rate at +60 / +120 / +300 s, quantiles p50 and p90.

Everything works in log space, z = log1p(rps), so one model serves 5 RPS and 500 RPS
deployments alike and multiplicative surges look the same at any base load.

Input is the last HIST_STEPS+1 samples of the demand series on a 5 s grid (oldest first,
NaN where there is no history yet). ``resample_history`` builds that grid from raw
(timestamp, value) samples so training and serving share one code path.

Forecasters (all expose ``predict(Z) -> dict[(horizon_s, q)] -> rps array``):
    PersistenceForecaster   z(t+h) = z(t)
    TrendForecaster         OLS slope over the last 120 s, extrapolated
    HoltForecaster          statsmodels damped-trend Holt on the last 300 s
    LgbmQuantileForecaster  LightGBM quantile regression of z(t+h) - z(t) on lag / rolling /
                            derivative features (direct, one model per horizon x quantile)
The first three get their quantiles from empirical residual quantiles on training runs.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

STEP_S = 5
HIST_STEPS = 60                           # 300 s of history (+ the current sample)
HORIZONS_S = (60, 120, 300)
QUANTILES = (0.5, 0.9)
Key = tuple[int, float]

_LAGS = (1, 3, 6, 12, 24, 60)
_WINDOWS = (6, 12, 24, 60)
_SLOPES = (6, 12, 24)
FEATURE_NAMES: list[str] = (
    [f"d_{k}" for k in _LAGS]
    + [f"{s}_{w}" for w in _WINDOWS for s in ("mean_rel", "std", "max_rel", "min_rel")]
    + [f"slope_{w}" for w in _SLOPES]
    + ["pos_60", "curv"]
)
# No "how much history is there" feature, and leading gaps are back-filled with the first
# observed value: in the lab every run starts at t=0 and most surges come at a fixed time after
# the start, so any run-age signal lets the model forecast the lab schedule (a phantom spike on
# flat traffic) instead of the traffic. A deployment with a short history looks like a long flat one.
# No absolute level either (review 3 dropped ``z_last``): the lab's spike / flash runs share a
# few base rates, so the level alone told the model "a 3-4x surge is coming". Every feature is
# now relative to the current value (log ratios), so the forecast is level-invariant apart from
# the noise level (std of log rps is larger at low rates, which is physical).


def to_z(x: np.ndarray | float) -> np.ndarray:
    return np.log1p(np.maximum(np.asarray(x, dtype=float), 0.0))


def from_z(z: np.ndarray | float) -> np.ndarray:
    return np.maximum(np.expm1(np.asarray(z, dtype=float)), 0.0)


def resample_history(ts: np.ndarray, values: np.ndarray, now: float) -> np.ndarray:
    """Last-value-carried-forward onto the 5 s grid ending at ``now``; NaN before the first sample."""
    ts = np.asarray(ts, dtype=float)
    values = np.asarray(values, dtype=float)
    order = np.argsort(ts)
    ts, values = ts[order], values[order]
    grid = now - STEP_S * np.arange(HIST_STEPS, -1, -1)
    idx = np.searchsorted(ts, grid, side="right") - 1
    out = np.where(idx >= 0, values[np.clip(idx, 0, None)], np.nan)
    return out


def _slope(z: np.ndarray, w: int) -> np.ndarray:
    """OLS slope (per step) over the last w points of each row; NaN-aware."""
    seg = z[:, -w:]
    x = np.arange(w, dtype=float)
    ok = ~np.isnan(seg)
    n = ok.sum(1)
    xm = np.where(ok, x, 0).sum(1) / np.maximum(n, 1)
    ym = np.where(ok, seg, 0).sum(1) / np.maximum(n, 1)
    dx = np.where(ok, x - xm[:, None], 0)
    dy = np.where(ok, seg - ym[:, None], 0)
    den = (dx * dx).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        s = (dx * dy).sum(1) / den
    return np.where(n >= 3, s, np.nan)


def make_features(Z: np.ndarray) -> np.ndarray:
    """Z: (n, HIST_STEPS+1) log-demand histories -> (n, len(FEATURE_NAMES))."""
    Z = backfill(np.atleast_2d(np.asarray(Z, dtype=float)))
    last = Z[:, -1]
    cols = [last - Z[:, -1 - k] for k in _LAGS]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for w in _WINDOWS:
            seg = Z[:, -w - 1:]
            cols += [np.nanmean(seg, 1) - last, np.nanstd(seg, 1),
                     np.nanmax(seg, 1) - last, np.nanmin(seg, 1) - last]
        slopes = {w: _slope(Z, w + 1) for w in _SLOPES}
        cols += [slopes[w] for w in _SLOPES]
        lo, hi = np.nanmin(Z, 1), np.nanmax(Z, 1)
        cols.append((last - lo) / (hi - lo + 1e-3))
    cols.append(slopes[6] - slopes[24])
    return np.column_stack(cols)


def backfill(Z: np.ndarray) -> np.ndarray:
    """Replace leading NaNs of each history row with the row's first observed value."""
    Z = np.array(Z, dtype=float, copy=True)
    ok = ~np.isnan(Z)
    first = np.where(ok.any(1), ok.argmax(1), Z.shape[1])
    for i in np.flatnonzero(first > 0):
        if first[i] < Z.shape[1]:
            Z[i, :first[i]] = Z[i, first[i]]
    return Z


class Forecaster(Protocol):
    name: str

    def fit(self, Z: np.ndarray, Y: dict[int, np.ndarray]) -> "Forecaster": ...

    def predict(self, Z: np.ndarray) -> dict[Key, np.ndarray]: ...


@dataclass
class _ResidualQuantiles:
    """Turns a point forecast in z space into quantiles via training residual quantiles."""
    offsets: dict[Key, float] = field(default_factory=dict)

    def fit(self, center: dict[int, np.ndarray], Y: dict[int, np.ndarray]) -> None:
        for h in HORIZONS_S:
            r = Y[h] - center[h]
            r = r[~np.isnan(r)]
            for q in QUANTILES:
                self.offsets[(h, q)] = float(np.quantile(r, q))

    def apply(self, center: dict[int, np.ndarray]) -> dict[Key, np.ndarray]:
        return {(h, q): from_z(center[h] + self.offsets[(h, q)]) for h in HORIZONS_S for q in QUANTILES}


class _CenterForecaster:
    name = "base"

    def __init__(self) -> None:
        self.rq = _ResidualQuantiles()

    def center(self, Z: np.ndarray) -> dict[int, np.ndarray]:
        raise NotImplementedError

    def fit(self, Z: np.ndarray, Y: dict[int, np.ndarray]):
        self.rq.fit(self.center(Z), Y)
        return self

    def predict(self, Z: np.ndarray) -> dict[Key, np.ndarray]:
        return self.rq.apply(self.center(Z))


class PersistenceForecaster(_CenterForecaster):
    name = "persistence"

    def center(self, Z):
        return {h: Z[:, -1].copy() for h in HORIZONS_S}


class TrendForecaster(_CenterForecaster):
    """Linear trend over the last ``window`` steps (default 120 s) extrapolated h ahead."""
    name = "linear_trend"

    def __init__(self, window: int = 24) -> None:
        super().__init__()
        self.window = window

    def center(self, Z):
        s = np.nan_to_num(_slope(Z, self.window + 1))
        return {h: Z[:, -1] + s * (h / STEP_S) for h in HORIZONS_S}


class HoltForecaster(_CenterForecaster):
    """Damped-trend Holt (statsmodels), refit on every origin's last 300 s."""
    name = "holt_damped"

    def center(self, Z):
        from statsmodels.tsa.holtwinters import Holt
        steps = max(HORIZONS_S) // STEP_S
        out = {h: np.empty(len(Z)) for h in HORIZONS_S}
        for i, row in enumerate(Z):
            y = row[~np.isnan(row)]
            if len(y) < 10 or np.ptp(y) < 1e-9:
                for h in HORIZONS_S:
                    out[h][i] = y[-1] if len(y) else 0.0
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fc = Holt(y, damped_trend=True, initialization_method="estimated").fit().forecast(steps)
            for h in HORIZONS_S:      # a damped trend fitted on a short window can still run away
                out[h][i] = float(np.clip(fc[h // STEP_S - 1], y.min() - 1.0, y.max() + 1.0))
        return out


@dataclass
class LgbmParams:
    n_estimators: int = 300
    learning_rate: float = 0.03
    num_leaves: int = 15
    min_child_samples: int = 30
    subsample: float = 0.8
    subsample_freq: int = 1
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    n_jobs: int = 1               # tiny data: threading is 10x slower


class LgbmQuantileForecaster:
    """Direct multi-horizon LightGBM quantile regression on z(t+h) - z(t).

    ``q90_margin`` (z units, per horizon) is an additive conformal correction learned on
    inner leave-run-out residuals so the p90 keeps ~90 % coverage on unseen runs.
    """
    name = "lgbm_quantile"

    def __init__(self, params: LgbmParams | None = None) -> None:
        self.params = params or LgbmParams()
        self.boosters: dict[Key, object] = {}
        self.q90_margin: dict[int, float] = {h: 0.0 for h in HORIZONS_S}

    def fit(self, Z: np.ndarray, Y: dict[int, np.ndarray]):
        import lightgbm as lgb
        X = make_features(Z)
        for h in HORIZONS_S:
            target = Y[h] - Z[:, -1]
            ok = ~np.isnan(target)
            for q in QUANTILES:
                m = lgb.LGBMRegressor(objective="quantile", alpha=q, verbose=-1,
                                      **self.params.__dict__)
                m.fit(X[ok], target[ok])
                self.boosters[(h, q)] = m.booster_
        return self

    def predict_z(self, Z: np.ndarray) -> dict[Key, np.ndarray]:
        X = make_features(Z)
        out = {}
        for h in HORIZONS_S:
            q50 = Z[:, -1] + self.boosters[(h, 0.5)].predict(X)
            q90 = Z[:, -1] + self.boosters[(h, 0.9)].predict(X) + self.q90_margin[h]
            out[(h, 0.5)], out[(h, 0.9)] = q50, np.maximum(q90, q50)   # no quantile crossing
        return out

    def predict(self, Z: np.ndarray) -> dict[Key, np.ndarray]:
        return {k: from_z(v) for k, v in self.predict_z(Z).items()}

    # ---- persistence -----------------------------------------------------------------
    def save(self, path: Path, info: dict | None = None) -> None:
        """Boosters + forecast_meta.json; ``info`` (training provenance) is stored under "info"."""
        path.mkdir(parents=True, exist_ok=True)
        for (h, q), b in self.boosters.items():
            b.save_model(str(path / f"lgbm_h{h}_q{int(q * 100)}.txt"))
        meta = {"horizons_s": list(HORIZONS_S), "quantiles": list(QUANTILES), "step_s": STEP_S,
                "hist_steps": HIST_STEPS, "features": FEATURE_NAMES, "target": "log1p(rps)",
                "q90_margin": {str(h): m for h, m in self.q90_margin.items()}, "info": info or {}}
        (path / "forecast_meta.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, path: Path) -> "LgbmQuantileForecaster":
        import lightgbm as lgb
        meta = json.loads((path / "forecast_meta.json").read_text())
        if meta["features"] != FEATURE_NAMES:
            raise ValueError("forecast model was trained with a different feature set")
        f = cls()
        f.q90_margin = {int(h): float(m) for h, m in meta["q90_margin"].items()}
        for h in HORIZONS_S:
            for q in QUANTILES:
                f.boosters[(h, q)] = lgb.Booster(model_file=str(path / f"lgbm_h{h}_q{int(q * 100)}.txt"))
        return f


def forecast_from_samples(model: Forecaster, ts: np.ndarray, rps: np.ndarray,
                          now: float) -> dict[Key, float]:
    """Serving helper: raw (ts, rps) samples of one deployment -> {(h, q): rps}."""
    Z = to_z(resample_history(ts, rps, now))[None, :]
    return {k: float(v[0]) for k, v in model.predict(Z).items()}
