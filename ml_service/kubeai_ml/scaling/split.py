"""The fixed development / held-out split of the loadlab runs, and dataset provenance.

Held-out runs are whole runs, never used to fit, select or tune anything (models, thresholds,
policy parameters, simulator calibration). Each is a round-2 run whose class has a twin in
round 1, so the held-out score measures generalisation to a new intensity / seed / replica
setup of a known situation, not to an unseen class:

==================================  =========================================  ==================
held-out run (round 2)              what is new                                round-1 twin
==================================  =========================================  ==================
b-hpa-diurnal-sd30-s59-r2           diurnal seed 59, HPA with 30 s scale-in    b-hpa-diurnal-s7
b-hpa-spike-sd30-b06-r2             4x spike on 10.8 rps, 30 s scale-in        b-hpa-spike-b05
c-under-fixed1-flash-r2             flash crowd onto 1 fixed pod               c-under-fixed1-diurnal
c-over-fixed4-diurnal-r2            4 fixed pods, low diurnal load             c-over-fixed4-steady
d-leak-medium-r2                    560 KB/s leak under diurnal load           d-leak-slow
d-dep-latency-75p600-r2             75 % of requests +600 ms, HPA              d-dep-latency
d-error10-hpa-diurnal-r2            10 % 5xx under diurnal load, HPA           d-error-burst
d-bad-deploy-x2-hpa-r2              x2 CPU per request, HPA                    d-bad-deploy
d-cpu-hog-fixed3-p2-r2              480m hog on pod 2 of 3 fixed               d-cpu-hog
d-crashloop-fixed3-r2               crashloop rollout on 3 fixed pods          d-crashloop
d-pod-kill-hpa-r2                   3 pod kills under HPA                      d-pod-kill
==================================  =========================================  ==================

Every class keeps at least one run in development (crashloop: exactly one).

Caveats (review 3) -- the held-out runs are not fully blind:

* Two held-out traffic schedules nearly copy a dev run: ``c-under-fixed1-flash-r2`` (flash at
  150 s, base 10.8 rps, x2.5) vs dev ``b-hpa-flash-b06`` (flash at 150 s, base 10.8, x3), and
  ``b-hpa-spike-sd30-b06-r2`` (x4 spike at 150 s) vs dev ``b-hpa-spike-b05`` (x4 at 150 s). The
  forecaster and simulator numbers on them measure a known onset and shape.
* Rules designed while held-out runs were visible (the round-2 forward test): the error_fault
  ``error_rate`` floor (d-error10-hpa-diurnal-r2), the residual cpu_hog mask
  (d-cpu-hog-fixed3-p2-r2), the policy's "measured load only during a fault" gate
  (d-crashloop-fixed3-r2), and the adoption of the outer conformal p90 correction (round-2
  coverage). ``train_all`` reports blind classifier variants; the card lists the rest.
* There is no round 3 yet: the fix is to freeze the rules and hold out runs collected after.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from .data import LAB_DATA

HOLDOUT_SCENARIOS: tuple[str, ...] = (
    "b-hpa-diurnal-sd30-s59-r2",
    "b-hpa-spike-sd30-b06-r2",
    "c-under-fixed1-flash-r2",
    "c-over-fixed4-diurnal-r2",
    "d-leak-medium-r2",
    "d-dep-latency-75p600-r2",
    "d-error10-hpa-diurnal-r2",
    "d-bad-deploy-x2-hpa-r2",
    "d-cpu-hog-fixed3-p2-r2",
    "d-crashloop-fixed3-r2",
    "d-pod-kill-hpa-r2",
)

# Traffic with a time shape (what a forecaster and an autoscaler are judged on).
SHAPES_WITH_DYNAMICS: tuple[str, ...] = ("diurnal", "spike", "flash_crowd", "sawtooth")

DATA_FILES: tuple[str, ...] = ("dataset_deploy.csv.gz", "dataset_pods.csv.gz", "faults_index.csv",
                               "runs_index.csv")


def is_holdout(scenario: pd.Series | str) -> pd.Series | bool:
    if isinstance(scenario, str):
        return scenario in HOLDOUT_SCENARIOS
    return scenario.isin(HOLDOUT_SCENARIOS)


def is_shaped(shape: pd.Series | str) -> pd.Series | bool:
    if isinstance(shape, str):
        return shape in SHAPES_WITH_DYNAMICS
    return shape.isin(SHAPES_WITH_DYNAMICS)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def data_provenance(lab_data: Path = LAB_DATA) -> dict[str, str]:
    """sha256 of every dataset file the bundle was trained and evaluated on."""
    return {name: sha256(lab_data / name) for name in DATA_FILES if (lab_data / name).exists()}
