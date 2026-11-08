"""Learned capacity: P(SLO breach | load per pod, CPU cost per request, replicas).

The physical driver is CPU utilisation of the pod's limit,

    u = offered_rps_per_pod * cpu_s_per_request / cpu_limit_cores,

so a bad deploy that triples the CPU cost per request triples u at the same traffic and
the safe rate per pod drops by 3x without any retraining. Replica count enters because
the measured latency knee moves earlier with more replicas (CFS throttling starts sooner:
p95 crosses 300 ms at 20 / 18 / 16 rps/pod with 1 / 2 / 3 replicas).

Model: logistic regression on [u, u^2, log N, u*log N] (N clamped to the trained range,
because the replica effect must not be extrapolated). ``safe_rps_per_pod(cpu, N, risk)`` is
the largest per-pod rate with P(breach) <= risk, solved by bisection on u.

``robust_cpu_per_req`` estimates the per-request CPU cost from deployment telemetry the
same way in training and serving: trailing median of cpu_cores / goodput over rows with
traffic and < 1 % errors (so 5xx-heavy overload rows do not distort it).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data import CPU_LIMIT_CORES, SLO_ERR

FEATURES = ("u", "u2", "log_n", "u_log_n")
CPU_PER_REQ_WINDOW = 12          # 60 s of 5 s samples
DEFAULT_RISK = 0.10


def robust_cpu_per_req(cpu_cores: pd.Series, goodput_rps: pd.Series, error_rate: pd.Series,
                       replicas: pd.Series, window: int = CPU_PER_REQ_WINDOW) -> pd.Series:
    """Trailing median CPU seconds per successful request (NaN until the first valid row)."""
    valid = (goodput_rps > 0.5 * replicas.clip(lower=1)) & (error_rate < SLO_ERR)
    raw = (cpu_cores / goodput_rps).where(valid)
    return raw.rolling(window, min_periods=1).median().ffill()


def utilisation(rps_per_pod: np.ndarray, cpu_per_req_s: np.ndarray,
                cpu_limit: float = CPU_LIMIT_CORES) -> np.ndarray:
    return np.asarray(rps_per_pod, float) * np.asarray(cpu_per_req_s, float) / cpu_limit


@dataclass
class CapacityModel:
    coef: list[float]
    intercept: float
    n_min: float = 1.0
    n_max: float = 4.0                 # replica range seen in training; clamp outside it
    cpu_limit: float = CPU_LIMIT_CORES
    risk: float = DEFAULT_RISK

    def _x(self, u: np.ndarray, n: np.ndarray) -> np.ndarray:
        u = np.asarray(u, float)
        ln = np.log(np.clip(np.asarray(n, float), self.n_min, self.n_max))
        u, ln = np.broadcast_arrays(u, ln)
        return np.column_stack([u.ravel(), (u * u).ravel(), ln.ravel(), (u * ln).ravel()])

    def p_breach_u(self, u: np.ndarray, n: np.ndarray) -> np.ndarray:
        z = self._x(u, n) @ np.asarray(self.coef) + self.intercept
        return 1.0 / (1.0 + np.exp(-z))

    def p_breach(self, rps_per_pod, cpu_per_req_s, replicas) -> np.ndarray:
        return self.p_breach_u(utilisation(rps_per_pod, cpu_per_req_s, self.cpu_limit), replicas)

    def safe_utilisation(self, replicas: float, risk: float | None = None) -> float:
        """Largest u in [0, 2] with P(breach) <= risk (P is increasing in u over this range)."""
        risk = self.risk if risk is None else risk
        lo, hi = 0.0, 2.0
        if self.p_breach_u(np.array([lo]), np.array([replicas]))[0] > risk:
            return 0.0
        for _ in range(40):
            mid = (lo + hi) / 2
            if self.p_breach_u(np.array([mid]), np.array([replicas]))[0] <= risk:
                lo = mid
            else:
                hi = mid
        return lo

    def safe_rps_per_pod(self, cpu_per_req_s: float, replicas: float, risk: float | None = None) -> float:
        return self.safe_utilisation(replicas, risk) * self.cpu_limit / max(cpu_per_req_s, 1e-4)

    def replicas_needed(self, demand_rps: float, cpu_per_req_s: float, max_replicas: int = 50,
                        risk: float | None = None) -> int:
        """Smallest N with demand / N <= safe_rps_per_pod(cpu, N)."""
        for n in range(1, max_replicas + 1):
            if demand_rps / n <= self.safe_rps_per_pod(cpu_per_req_s, n, risk):
                return n
        return max_replicas

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"features": list(FEATURES), **asdict(self)}, indent=2))

    @classmethod
    def load(cls, path: Path) -> "CapacityModel":
        d = json.loads(path.read_text())
        if tuple(d.pop("features")) != FEATURES:
            raise ValueError("capacity model was trained with a different feature set")
        return cls(**d)


def design_matrix(u: np.ndarray, n: np.ndarray, n_max: float = 4.0) -> np.ndarray:
    return CapacityModel(coef=[0.0] * len(FEATURES), intercept=0.0, n_max=n_max)._x(u, n)
