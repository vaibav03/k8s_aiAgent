"""Deployment states the decision engine reports, and the remediation for each.

The classifier answers "WHAT is the state". ``REMEDIATION`` turns a state into the action an
on-call engineer would take; ``SCALING_HELPS`` is the "do NOT scale" gate the replica policy
must respect (scaling a deployment whose problem is not load wastes money and hides the fault).
"""
from __future__ import annotations

from typing import Final

HEALTHY: Final = "healthy"
SCALE_UP: Final = "scale_up"
SCALE_DOWN: Final = "scale_down"
MEMORY_LEAK: Final = "memory_leak"
DEPENDENCY_LATENCY: Final = "dependency_latency"
ERROR_FAULT: Final = "error_fault"
NOISY_POD: Final = "noisy_pod"
BAD_DEPLOY: Final = "bad_deploy"
CRASHLOOP: Final = "crashloop"
POD_RESTART: Final = "pod_restart"

LOAD_STATES: Final = (HEALTHY, SCALE_UP, SCALE_DOWN)
FAULT_STATES: Final = (MEMORY_LEAK, DEPENDENCY_LATENCY, ERROR_FAULT, NOISY_POD, BAD_DEPLOY,
                       CRASHLOOP, POD_RESTART)
STATES: Final = LOAD_STATES + FAULT_STATES

# loadlab ground-truth fault type -> state.
FAULT_TO_STATE: Final = {
    "memory_leak": MEMORY_LEAK,
    "dep_latency": DEPENDENCY_LATENCY,
    "error_rate": ERROR_FAULT,
    "cpu_hog": NOISY_POD,
    "bad_deploy": BAD_DEPLOY,
    "crashloop": CRASHLOOP,
    "pod_kill": POD_RESTART,
}

# Which state wins when two faults overlap (root cause first).
FAULT_PRIORITY: Final = (CRASHLOOP, BAD_DEPLOY, MEMORY_LEAK, NOISY_POD, DEPENDENCY_LATENCY,
                         ERROR_FAULT, POD_RESTART)

REMEDIATION: Final = {
    HEALTHY: "none",
    SCALE_UP: "scale_out to needed replicas (load-driven)",
    SCALE_DOWN: "scale_in one step after cool-down (over-provisioned)",
    MEMORY_LEAK: "restart the leaking pod before OOM (rolling); do NOT scale; open bug",
    DEPENDENCY_LATENCY: "do NOT scale; page owner of the slow dependency; consider circuit-breaker",
    ERROR_FAULT: "do NOT scale; page human; roll back if a rollout is recent",
    NOISY_POD: "delete/restart the hot pod (dispersion outlier); do NOT scale",
    BAD_DEPLOY: "roll back the last rollout; scale only as a stop-gap if SLO is breached",
    CRASHLOOP: "pause/roll back the rollout; do NOT scale (old ReplicaSet still serves)",
    POD_RESTART: "watch; replacement pod should be ready in ~30 s; do NOT scale",
}

SCALING_HELPS: Final = {
    HEALTHY: False, SCALE_UP: True, SCALE_DOWN: True, MEMORY_LEAK: False,
    DEPENDENCY_LATENCY: False, ERROR_FAULT: False, NOISY_POD: False,
    BAD_DEPLOY: False, CRASHLOOP: False, POD_RESTART: False,
}
