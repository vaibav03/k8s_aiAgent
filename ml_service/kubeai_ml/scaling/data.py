"""loadlab dataset access and constants shared by the scaling modules."""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

PKG_DIR = Path(__file__).resolve().parent
ML_ROOT = PKG_DIR.parent.parent                      # ml_service/
REPO = ML_ROOT.parent                                # k8s_aiAgent/
LAB_DATA = REPO / "loadlab" / "data"
MODELS_DIR = ML_ROOT / "models" / "scaling"
PLOTS_DIR = PKG_DIR / "plots"

STEP_S = 5                     # dataset sample period
SLO_P95_S = 0.300
SLO_ERR = 0.01
CPU_LIMIT_CORES = 0.5          # per pod (loadlab Deployment)
CPU_REQUEST_CORES = 0.25
HPA_TARGET_UTIL = 0.70         # of request

# Runs whose traffic has a shape (used to evaluate the forecaster).
SHAPED_GROUPS = ("B", "C", "E")
# Collection rounds: models are fit on TRAIN_ROUNDS; later rounds are a true out-of-sample test
# (new seeds / shapes / HPA settings collected after the round-1 design was frozen).
TRAIN_ROUNDS = (1,)
HOLDOUT_ROUNDS = (2,)


def load_deploy(rounds: Iterable[int] | None = None) -> pd.DataFrame:
    """Deployment-level rows, sorted, with run duration and derived columns.

    ``rounds`` keeps only those collection rounds (None = every round in the dataset).
    """
    df = pd.read_csv(LAB_DATA / "dataset_deploy.csv.gz")
    if "round" not in df.columns:
        df["round"] = 1
    if rounds is not None:
        df = df[df["round"].isin(list(rounds))]
    runs = pd.read_csv(LAB_DATA / "runs_index.csv")[["run_id", "duration_s"]]
    df = df.merge(runs, on="run_id", how="left")
    df = df.sort_values(["run_id", "ts"]).reset_index(drop=True)
    df["active_fault"] = df["active_fault"].fillna("none")
    df["breach"] = ((df["p95"] >= SLO_P95_S) | (df["error_rate"] >= SLO_ERR)).astype(int)
    pods = df["replicas_ready"].clip(lower=1)
    df["rps_per_pod"] = df["rps"] / pods                    # offered (all responses)
    df["goodput_per_pod"] = df["goodput_rps"] / pods
    with np.errstate(divide="ignore", invalid="ignore"):
        df["cpu_per_req_s"] = np.where(df["goodput_rps"] > 0.5,
                                       df["cpu_cores_used"] / df["goodput_rps"], np.nan)
    return df
