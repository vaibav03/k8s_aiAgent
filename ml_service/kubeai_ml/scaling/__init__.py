"""Load forecasting, learned capacity and replica policy for the SRE decision engine.

Modules:
    data      loadlab dataset loading and shared constants
    forecast  demand forecaster (features + persistence / trend / Holt / LightGBM quantile)
    capacity  P(SLO breach | load, cpu per request, replicas) and safe_rps_per_pod
    policy    desired_replicas: fast-up / slow-safe-down with a non-load-fault gate
    states    the 10 deployment states, remediation and the "scaling helps?" gate
    features  state features from deploy rows (+ optional pod snapshot), train == serve
    labels    state labels: ground-truth faults + learned-knee scale_up / scale_down
    rules     SRE runbook: physics fault gates (+ load rules); the gates front the shipped hybrid
    classifier rules / LightGBM / two-stage / hybrid wrappers, save/load, decide()

Shipped pipeline (rounds 1 + 2, held-out runs, bundle ``models/scaling/<version>`` + LATEST):
``python -m kubeai_ml.scaling.train_all``; serving API ``engine.DecisionEngine``; tables
``python -m kubeai_ml.scaling.report``; model card ``ml_service/SCALING_MODEL_CARD.md``.
Other modules: split (held-out runs, data sha256), bundle (layout/loader), policy_eval
(simulator comparison with the engine as a policy), fault_eval (engine vs actual HPA on faults).

State classifier exploration: ``python -m kubeai_ml.scaling.train_classifier``; findings in
EXPLORE_CLASSIFIER.md.

Training / evaluation entry points: ``python -m kubeai_ml.scaling.train_forecast`` and
``python -m kubeai_ml.scaling.train_capacity``. Findings: EXPLORE_FORECAST_CAPACITY.md.
"""
