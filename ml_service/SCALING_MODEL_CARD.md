# Scaling decision engine: model card

**Bundle:** `models/scaling/scaling-20261108194802` (`models/scaling/LATEST`). **Date:** 2026-11-08 (review-3 revision).
**Reproduce:** `cd ml_service && PYTHONPATH=. python -m kubeai_ml.scaling.train_all` (about 14 min on 6 cores).
To print every table below from the bundle, run `python -m kubeai_ml.scaling.report`.
**Serve:** `kubeai_ml.scaling.engine.DecisionEngine().decide(deployment_obs, pod_obs, current_replicas, min, max, now_ts=...)`.
**Tests:** the whole `ml_service` suite passes (69 tests). `tests/test_scaling.py` runs the shipped bundle on recorded held-out windows; `tests/test_scaling_safety.py` covers the safety interlocks on synthetic telemetry.

Every 15 s, for each deployment, the engine answers three questions:

1. **What state is it in?** One of healthy, scale_up, scale_down, memory_leak, dependency_latency, error_fault, noisy_pod, bad_deploy, crashloop or pod_restart.
2. **How many replicas are needed?** Now, and at +60, +120 and +300 s.
3. **What should be done?** scale, restart_pod, rollback, page, or none. It says "do NOT scale" explicitly when scaling would not fix the problem.

The OOM-ahead model (`models/LATEST`) is unchanged. It is an optional per-pod memory-risk input.

**Bottom line of this revision.** The earlier claim that the predictive policy "avoids SLO violations" was an artefact. It came from a wide forecast that parked the fleet near the replica cap. Once the forecast must be backed by measurements, and costs are compared at equal pod-minutes, **the predictive replica policy is not better than a stock HPA**. On held-out traces it has more violation minutes at about the same cost (section 3). The engine's demonstrated value is the **state classifier and the fault gate**:

* it does not add pods during non-load faults (0 % of decisions on every held-out fault run);
* it names the remediation (restart the right pod, page, roll back only with rollout evidence);
* it is much cheaper than the HPA on flat or over-provisioned traffic.

## 1. Data and evaluation protocol

* **Data.** The loadlab dataset, rounds 1 + 2: 44 runs, 5,368 deployment rows at 5 s, 16,608 pod rows and 27 injected faults. The manifest stores the sha256 of the four data files.
* **Held-out set.** 11 whole runs, listed in `split.py`. Each is a round-2 run whose class has a round-1 twin. The shipped models are fitted on the 33 development runs only.
* **The held-out set is not fully blind.** Some choices were made while held-out runs were visible:

  | decision | held-out run it was tuned on | how this card handles it |
  |---|---|---|
  | error_fault gate `error_rate >= 0.005` (`rules.ERROR_RATE_FLOOR`) | d-error10-hpa-diurnal-r2 | blind score reported (§2.1) |
  | residual cpu_hog mask (`labels.residual_hog_mask`), which removes 38 held-out rows | d-cpu-hog-fixed3-p2-r2 | blind score reported (§2.1) |
  | policy rule "during a fault only *measured* load adds pods" | d-crashloop-fixed3-r2, d-leak-medium-r2 | pre-fix numbers reported (§4) |
  | adoption of the outer conformal p90 correction | round-2 coverage (0.79 / 0.80 / 0.87 without it) | held-out p90 coverage is a post-decision number |

  Two held-out traffic schedules also nearly copy dev runs:
  * `c-under-fixed1-flash-r2` vs dev `b-hpa-flash-b06`: both flash at 150 s from a 10.8 RPS base;
  * `b-hpa-spike-sd30-b06-r2` vs dev `b-hpa-spike-b05`: both x4 spikes at 150 s.

  The forecaster and simulator numbers on these two runs therefore measure a known onset and shape. There is no round 3 yet. The proper fix is to freeze the rules now and hold out runs collected afterwards.
* **Cross-validation.** Leave-one-run-out on the development runs.
  * Classifier: 33 folds. Each fold re-learns the capacity knee *and re-derives the load labels* from that knee, so neither the training labels nor the test labels depend on the test run. This changes 19 labels compared with the global knee.
  * Forecaster: 14 folds, one per traffic-shaped development run.
  * Capacity model: 33 folds.
* **Masking.** Rows with `qa_flag != ok` are excluded, as are rows where a cpu_hog outlived its logged end (68 dev rows and 38 held-out rows).
* **Scoring.** "Strict" scores every unmasked row. "Settled" drops rows within 30 s of a label change, because labels trail the signals by up to about 25 s. A label segment no longer than 30 s is kept whole. Without that exception the settled score could never include pod_restart, whose windows are exactly 30 s; the previous card's settled F1 silently left that class out.

## 2. Components

### 2.1 State classifier: **hybrid** (runbook fault gates, then a LightGBM load head)

The fault gates are rules checked in priority order: crashloop, memory_leak, pod_restart, bad_deploy, noisy_pod, error_fault, dependency_latency. The thresholds are in `label_rules.json`. If no gate fires, a LightGBM head picks healthy, scale_up or scale_down from 52 features plus 5 features normalised by the learned capacity knee (17 / 17 / 13 RPS per pod for 1 / 2 / 3 replicas). Between rules alone and the hybrid, the pipeline picks by leave-run-out settled macro-F1; it picked the hybrid.

| model | LORO F1 strict | LORO F1 settled | **held-out F1 strict** | **held-out F1 settled** | held-out acc. settled | held-out false scale_up on fault rows | held-out false pages on load rows | held-out missed scale_up |
|---|---|---|---|---|---|---|---|---|
| **hybrid (shipped)** | 0.821 | 0.860 | **0.797** | **0.844** | 0.896 | **0.000** | 0.014 | 0.125 |
| rules | 0.814 | 0.858 | 0.787 | 0.837 | 0.881 | 0.000 | 0.014 | 0.170 |
| LightGBM, 10 classes | 0.678 | 0.703 | 0.771 | 0.806 | 0.894 | 0.000 | 0.019 | 0.091 |
| two-stage LightGBM | 0.638 | 0.651 | 0.681 | 0.711 | 0.855 | 0.000 | 0.009 | 0.102 |
| baseline: HPA 70 % rule | 0.120 | 0.121 | 0.135 | 0.139 | 0.387 | 0.243 | 0 | 0.034 |
| baseline: majority class | 0.058 | 0.059 | 0.033 | 0.033 | 0.199 | 0 | 0 | 1.000 |

All macro-F1 values above cover 10 classes, including pod_restart. The previous card's 0.941 held-out figure covered 9 classes.

**Blind held-out score** (hybrid, settled). This undoes the two decisions made with held-out runs visible:

| variant | F1 strict | F1 settled | error_fault P / R | noisy_pod P / R |
|---|---|---|---|---|
| shipped rules, shipped mask | 0.797 | 0.844 | 1.00 / 1.00 | 1.00 / 0.92 |
| without the `error_rate` floor | 0.790 | 0.835 | 0.84 / 1.00 | 1.00 / 0.92 |
| hog rows kept at their nominal label | 0.768 | 0.811 | 1.00 / 1.00 | 0.52 / 0.92 |
| **both undone (blind figure)** | **0.761** | **0.803** | 0.84 / 1.00 | 0.52 / 0.92 |

On the 38 restored rows the model predicts noisy_pod. That is physically right, because the hog really is still running, so the mask itself is defensible. It was simply designed on held-out data.

Per-class held-out results (hybrid):

| class | P | R | F1 settled | n | F1 strict |
|---|---|---|---|---|---|
| healthy | 0.758 | 0.913 | 0.828 | 230 | 0.774 |
| scale_up | 0.895 | 0.875 | 0.885 | 88 | 0.828 |
| scale_down | 0.938 | 0.879 | 0.908 | 531 | 0.868 |
| memory_leak | 0.888 | 0.990 | 0.936 | 96 | 0.909 |
| dependency_latency | 0.955 | 1.000 | 0.977 | 42 | 0.913 |
| error_fault | 1.000 | 1.000 | 1.000 | 36 | 0.963 |
| noisy_pod | 1.000 | 0.919 | 0.958 | 37 | 0.883 |
| bad_deploy | 1.000 | 0.895 | 0.944 | 38 | 0.872 |
| crashloop | 1.000 | 1.000 | 1.000 | 42 | 0.957 |
| pod_restart | 0.000 | 0.000 | 0.000 | 18 | 0.000 |

Under leave-run-out the per-class settled F1 is:

* dependency_latency 0.81;
* healthy, scale_up and scale_down 0.86–0.90;
* the other fault classes 0.92–1.00;
* pod_restart **0.40** (recall 0.25).

Detection delay is the time from fault start to the first correct state, on held-out runs:

| fault | delay |
|---|---|
| crashloop | 12 s |
| error_rate | 16 s |
| dependency_latency | 33 s |
| memory leak | 40 s |
| cpu_hog | 47 s |
| bad_deploy | 52 s |

**pod_kill was not detected at all on the held-out run**: all 3 kills, row recall 0, and the 18 rows were predicted scale_down. On the dev runs, under leave-run-out, it is caught 3–5 s after the kill on 1 of the 6 rows in each window. A missed pod_kill is harmless, because the Deployment replaces the pod within about 10 s.

### 2.2 Load forecaster: **LightGBM quantile** (p50 and p90 at +60 / +120 / +300 s)

* **Model.** Direct models on `log1p(rps)`. The features are lags, rolling statistics, slopes and curvature over 300 s, **all relative to the current value**.
* **Change in this revision: no absolute level.** The previous model's `z_last` input let it recognise the lab's spike and flash base rates. On perfectly flat traffic its p90 at +120 s was 4.5x at 4 RPS, 3.6x at 10.8, 4.1x at 14.4 and 1.8x at 20–27 RPS.
* **p90 margin.** Two conformal stages, inner and outer: +0.08 / +0.33 / +0.37 log units.
* **The target is `rps` (offered demand), not goodput.**

| horizon | method | LORO pinball % | LORO p90 cov. | held-out pinball % | held-out p90 cov. | held-out cov. on surges |
|---|---|---|---|---|---|---|
| +60 s | **LightGBM** | **10.0** | 0.83 | **9.3** | 0.86 | 0.44 |
| | persistence | 13.7 | 0.78 | 11.9 | 0.86 | 0.40 |
| +120 s | **LightGBM** | **16.3** | 0.81 | **17.6** | 0.92 | 0.76 |
| | persistence | 23.2 | 0.79 | 21.5 | 0.84 | 0.53 |
| +300 s | **LightGBM** | **17.8** | 0.80 | 18.3 | 1.00 | 1.00 (n=12) |
| | persistence | 23.3 | 0.84 | **16.1** | 0.94 | 0.42 |

Damped Holt and linear trend are worse than persistence at every horizon (`report.py`).

**Flat-traffic probe** (`train_forecast.flat_probe`): the p90 as a multiple of the current rate on traffic with no dynamics. The multiple is now level-invariant, but the prior is still wide:

| traffic | p90 / rate at +60 s | at +120 s | at +300 s |
|---|---|---|---|
| perfectly flat, 4–60 RPS | 1.76–1.93x | 3.2–3.7x | 4.3–5.0x |
| Poisson seen through rate[20s], 4–60 RPS | 1.2–1.6x | 1.8–2.6x | 1.9–3.6x |
| (p50 on Poisson) | 1.00–1.02x | 1.03–1.07x | 1.01–1.24x |

This is the lab's surge-heavy prior. The p90 is calibrated for runs where a surge usually comes, so **the policy must not follow it on its own** (section 2.4).

### 2.3 Capacity: **logistic P(SLO breach | u, N)**

* **Inputs.** u = rps per pod × CPU seconds per successful request / 0.5-core limit. CPU per request is a trailing 60 s median of telemetry.
* **Model.** logit = −10.07 + 0.883 u + 10.51 u² + 3.06 ln N − 0.308 u·ln N, with N clamped to [1, 4].
* **Training data.** 3,058 dev rows from 33 runs.

| model | LORO log-loss | LORO AUC | held-out log-loss | held-out AUC |
|---|---|---|---|---|
| **utilisation logistic (shipped)** | 0.123 | 0.955 | **0.052** | **0.967** |
| load-only logistic | 0.208 | 0.765 | 0.083 | 0.950 |
| monotone LightGBM (u, N) | 0.118 | 0.958 | 0.075 | 0.963 |
| utilisation + throttle nowcast | 0.079 | 0.983 | 0.074 | 0.942 |

**Read the held-out column with care.**

* All 79 held-out breach rows come from 2 runs: 64 from `c-under-fixed1-flash-r2`, a single 1-replica run, and 15 from `b-hpa-spike-sd30-b06-r2`. The held-out AUC of 0.967 is therefore mostly an N = 1 result.
* Held-out bad-deploy rows have a 0 % breach rate. The claim that CPU cost per request moves capacity is validated **only on dev**: the dev bad-deploy breach rate is 0.91. Even there, the logistic under-predicts it (mean P 0.29).
* Dev breaches are spread over 13 runs.

Used as a rule on the held-out rows ("load above capacity means a breach"):

| rule | recall | precision | false-alarm rate |
|---|---|---|---|
| fixed 18 RPS/pod | 0.73 | 1.00 | 0.0 % |
| HPA 70 % of the CPU request | 1.00 | 0.26 | 25.6 % |
| learned, risk 0.1 (**shipped risk**) | 0.94 | 0.79 | 2.3 % |
| learned, risk 0.2 | 0.92 | 0.79 | 2.2 % |
| learned, risk 0.3 | 0.90 | 0.83 | 1.7 % |

Safe RPS per pod at risk 0.1 is 15.6 / 13.4 / 11.9 / 10.7 for N = 1 / 2 / 3 / ≥4 at 26 ms per request, and 5.3 / 4.6 / 4.0 / 3.6 at 78 ms.

### 2.4 Replica policy (`policy.py`, parameters in `policy.json`)

* **Measured need.** The smallest N whose safe rate covers *measured* demand × `up_headroom` (1.3, chosen by the sweep). It is applied at once, capped at doubling or +4 pods per step.
* **Forecast-driven need** may add pods only when a measurement backs it:
  * either P(SLO breach now) is above the risk budget,
  * or demand is measurably rising: the mean over the last 30 s is ≥ 1.2x the mean 60–120 s earlier.

  The forecast need is capped at max(need_now + 1, ⌈1.5 × need_now⌉) and must persist for 2 consecutive decisions. The classifier's scale_up state alone is **not** evidence. Its LightGBM head calls flat 10.8 RPS on one pod "scale_up", because that is the base rate of the lab's flash-crowd run, which is a schedule prior.
* **Scale down** one pod at a time, at most once per 60 s. The target is the maximum need over a **180 s** window; 60 s is no longer allowed, because it made the policy flap.
* **Gate.**
  * memory_leak, dependency_latency, error_fault, noisy_pod, crashloop, pod_restart: only measured load adds pods, sized at the healthy baseline CPU cost; scale-down is frozen.
  * bad_deploy: scales up only as a **stop-gap** while the SLO is breached or P(breach) is over budget. It is sized from measured demand at the new CPU cost, never from the forecast. The scale action carries `stop_gap: true`, and normal scale-in removes those pods once the state clears after the rollback.
  * Insufficient telemetry or an outage: scale-down is frozen and the forecast is ignored.
* **Tuning.** The sweep covered 24 settings (risk 0.1 / 0.2 / 0.3 × up horizons +60 or +60/+120 × headroom 1.0 / 1.3 × window 180 / 300 s). It ran on the 14 dev traffic traces plus 7 synthetic traces, including flat and noisy-flat traffic, with a cap of 8.
  * Rule: the fewest pod-minutes whose violation minutes do not exceed the HPA-30 s baseline. **No setting met that rule**: the best got 9.1 violation minutes against the HPA's 6.7.
  * The fallback, fewest violation minutes, chose **risk 0.1, +60 s horizon, headroom 1.3, 180 s window**.
  * Headroom 1.0 gives 12.7–15.5 violation minutes at 1,001–1,335 pod-minutes. Headroom 1.3 gives 9.1–9.4 minutes at 1,330–1,706 pod-minutes.

## 3. Policy evaluation (replay simulator)

**Simulator calibration** (dev runs only):

* CPU per request is 26.1 ms.
* Per-pod capacity is 19.3 / 17.7 / 17.4 RPS for 1 / 2 / 3 pods. **These are fitted only for 1–3 pods.** Values for 4–8 pods are linear extrapolation (16.5 … 13.5), so every cap-8 number is extrapolated.
* Pod startup is 10 s.
* HPA grouping now follows `replica_calculator.go`. A not-Ready pod with no sample is *missing*: on scale-down it counts as max(100 %, target) of the request. A Ready pod whose sample predates readiness plus the window is *unready*: it is dropped on scale-down. Both count as 0 % on scale-up.

**Simulator validation**: the stock HPA replayed on the recorded B runs.

| | held-out (2 runs) | dev (10 runs) |
|---|---|---|
| replica MAE | 0.046 pods | 0.050 pods |
| rows with the exact replica count | 92.8 % | 94.2 % |
| signed scale-up timing error | −7 s | −8 s |
| violation minutes, actual vs simulated | 1.25 vs 0.75 | 2.5 vs 2.0 |

* **Timing.** The simulated HPA usually scales up one 15 s sync early.
* **Metrics-server lag.** The fitted lag of 0 s is a **boundary fit**: it is the lowest value in the grid, and is not physical. Round-1 runs alone fit best at 10 s and round-2 runs at 0 s.
* **Violations.** Across all B runs, actual violation minutes are **1.36x** the simulated ones. The simulator models recovery after a scale-up too optimistically.
* **Bias direction.** These biases favour reactive policies, which are the ones that suffer transients. Treat every policy's violation minutes as a lower bound, with an uncertainty of about ×1.4.
* **No PREDICTIVE run has ever been executed on the cluster.**

**Policy setup.** Every policy except fixed-N starts at 1 replica. PREDICTIVE is the serving `DecisionEngine`. New columns in the tables:

* **violation events**: independent breach episodes;
* **pod-min over need**: pods above the perfect-foresight oracle need of the pod model.

**Held-out traffic runs**: 6 runs, 45 trace-minutes.

| cap | policy | SLO-viol. min | viol. events | pod-min | pod-min over need | actions |
|---|---|---|---|---|---|---|
| 4 | PREDICTIVE | 2.17 | 2 | 179.3 | 93.9 | 29 |
| 4 | HPA cpu70, 30 s scale-down | 0.67 | 1 | 170.8 | 84.5 | 32 |
| 4 | HPA cpu70, 300 s (stock) | 0.67 | 1 | 200.6 | 114.3 | 13 |
| 4 | HPA cpu50, 300 s | 0.50 | 1 | 229.8 | 143.4 | 13 |
| 4 | HPA cpu35, 300 s | 0.00 | 0 | 248.5 | 161.9 | 9 |
| 4 | fixed-4 | 0.00 | 0 | 257.0 | 170.4 | 0 |
| 4 | fixed-3 / fixed-2 | 2.75 / 3.17 | 1 / 1 | 192.8 / 128.5 | 108.7 / 47.0 | 0 |
| 8 (extrapolated) | PREDICTIVE | 2.17 | 2 | 223.6 | 138.3 | 37 |
| 8 (extrapolated) | HPA cpu70, 300 s | 0.67 | 1 | 227.8 | 141.5 | 15 |
| 8 (extrapolated) | HPA cpu70, 30 s | 0.67 | 1 | 185.1 | 98.9 | 35 |

**Per trace** (cap 4; violation minutes / pod-minutes). All the violations come from the two surge runs whose schedules repeat dev runs. There are **n = 1 to 2 independent violation events per policy**, which is too few for a confidence statement.

| trace | PREDICTIVE | HPA 30 s | HPA stock |
|---|---|---|---|
| b-hpa-spike-sd30-b06-r2 | 1.25 / 29.8 | 0.67 / 26.1 | 0.67 / 34.9 |
| c-under-fixed1-flash-r2 | 0.92 / 24.6 | 0 / 27.6 | 0 / 27.6 |
| b-hpa-diurnal-sd30-s59-r2 | 0 / 45.8 | 0 / 40.0 | 0 / 49.3 |
| c-over-fixed4-diurnal-r2 | 0 / 16.4 | 0 / 16.2 | 0 / 18.8 |
| d-error10-hpa-diurnal-r2 | 0 / 31.2 | 0 / 29.2 | 0 / 35.7 |
| d-leak-medium-r2 | 0 / 31.5 | 0 / 31.8 | 0 / 34.3 |

**Cost-matched comparison.** PREDICTIVE is set next to the non-predictive policy with the closest pod-minutes:

| traces | cap | PREDICTIVE viol. min / pod-min | baseline | baseline viol. min / pod-min |
|---|---|---|---|---|
| held-out | 4 | 2.17 / 179 | HPA 70 % 30 s | 0.67 / 171 |
| held-out | 8 | 2.17 / 224 | HPA 70 % 300 s | 0.67 / 228 |
| held-out, 600 s warm start | 4 | 2.17 / 269 | fixed-2 | 3.17 / 248 |
| held-out, 600 s warm start | 8 | 2.17 / 318 | HPA 70 % 30 s | 0.67 / 307 |
| dev (tuned on) | 4 | 4.92 / 524 | HPA 70 % 30 s | 3.50 / 502 |
| dev (tuned on) | 8 | 4.50 / 774 | HPA 70 % 300 s | 3.17 / 796 |
| synthetic | 4 | 4.92 / 479 | HPA 70 % 30 s | 3.58 / 491 |
| synthetic | 8 | 4.58 / 732 | HPA 70 % 300 s | 3.50 / 785 |

**Synthetic flat and noisy traces** (cap 8). These are the cases the reviewers used to show parking at the cap:

| trace | PREDICTIVE: pods mean / max, actions | HPA 30 s: pods mean / max, actions |
|---|---|---|
| flat 10.8 RPS, 15 min | 1.9 / 2, 1 | 2.0 / 2, 1 |
| flat 40 RPS, 15 min | 7.0 / 8, 5 | 6.8 / 7, 2 |
| noisy flat 30 ± 35 %, 30 min | 6.2 / 8, 15 | 5.5 / 7, 28 |

The previous bundle went 1→3→4 within 45 s on flat 10.8 RPS and stayed at the cap. The new one runs at 2 pods, like the HPA.

**Reading.**

* **PREDICTIVE does not beat the HPA at equal cost.** On every cost-matched comparison except the warm-started cap-4 one (where the closest baseline is the under-sized fixed-2), the HPA has fewer violation minutes at the same pod-minutes.
* The previous "0.00 violation minutes" came from the forecast-driven fleet sitting at or near the cap.
* With the +120 s horizon removed or the forecast gated, the dev sweep never matches the HPA. The forecaster cannot predict a step before it happens, and the stock HPA's 70 % target (6.7 RPS per pod) is itself generous headroom.
* What PREDICTIVE does well: no flapping on noisy traffic (15 vs 28 actions), no parking at the cap, about HPA-level cost, and the fault gate (section 4).
* A CPU-target HPA cannot be tuned to the learned capacity. At a 147 % target, violation minutes are 5–7x higher.

## 4. Faults: the gate against what the HPA actually did

This is an open-loop replay of the engine every 15 s on every recorded fault run. Its recommendation is recorded, not applied. The HPA columns are what actually happened in the lab.

| run (split) | engine top state (recall) | adds pods | blocks scaling | restarts issued (on the faulted pod) | rollback | HPA in reality |
|---|---|---|---|---|---|---|
| d-dep-latency-75p600-r2 (**held-out**) | dependency_latency (0.88) | **0 %** | 88 % | – | 0 % | 3→4; SLO share 0.90 → 1.00, the extra pod did not help |
| d-dep-latency / -25p300-r2 (dev) | dependency_latency (0.80 / 0.71) | 0 % | 80 / 71 % | – | 0 % | 3 / fixed 3 |
| e-spike-dep-latency (dev) | scale_up (0.33) | 11 % (the 4x spike is real load) | 22 % | – | 0 % | 3→4 |
| d-error10-hpa-diurnal-r2 (**held-out**) | error_fault (0.93) | **0 %** | 93 % | – | 0 % | 3 |
| d-error-burst / d-error30-fixed3-r2 (dev) | error_fault (1.00 / 0.92) | 0 % | 100 / 92 % | – | 0 % | 3 / fixed 3 |
| d-leak-medium-r2 (**held-out**) | memory_leak (0.91) | **0 %** | 91 % | 2 (100 %) | 0 % | 2→3, diurnal load |
| e-diurnal-leak (dev) | memory_leak (0.92) | 2.6 % | 92 % | 2 (100 %) | 0 % | 3→4 |
| d-leak-oom / -slow / -fast-oom-r2 (dev) | memory_leak (0.86–0.89) | 0 % | 86–91 % | 1–2 (100 %) | 0 % | 3 / 3 / fixed 3 |
| d-cpu-hog / -p0-r2 (dev) | noisy_pod (0.79) | 0 % | 79–86 % | 1 (100 %) | 0 % | 3→4 / fixed 3 |
| d-cpu-hog-fixed3-p2-r2 (**held-out**) | noisy_pod (0.80) | 0 % | 80 % | 1 (100 %) | 0 % | fixed 3 |
| d-crashloop-fixed3-r2 (**held-out**) | crashloop (0.94) | **0 %** | 94 % | – | 94 % (a rollout is present) | fixed 3 |
| d-bad-deploy (dev) | bad_deploy (0.73) | 0 % | 80 % | – | 73 % | 3→4; SLO share 0.56 → 1.00 |
| d-bad-deploy-x2-hpa-r2 (**held-out**) | bad_deploy (0.79) | **0 %** (SLO fine, no stop-gap) | 79 % | – | 79 % | 3→4, no breach |
| d-bad-deploy-x3-fixed3-r2 (dev) | bad_deploy (0.73) | 73 % (stop-gap; SLO breached 91 % of the time) | 7 % | – | 73 % | fixed 3 |
| d-pod-kill ×3 runs | healthy / scale_down | 0 % | 0–50 % | – | 0 % | (not detected, harmless) |

**Post-hoc caveat.** The policy rule "during a fault only measured load adds pods" was added because of the held-out crashloop run. These are the numbers with that rule disabled, measured by the reviewers on the previous bundle:

| run | scale-up share without the rule | with the rule, previous bundle | this bundle |
|---|---|---|---|
| d-crashloop-fixed3-r2 | **43.8 %** | 6.2 % | 0 % |
| d-leak-medium-r2 | **8.8 %** | 2.9 % | 0 % |
| dev d-crashloop (for comparison) | 0.0 % | | |

The design is sound, but its held-out validation needs a fresh crashloop run. In this bundle the forecast evidence gate also blocks those scale-ups, independently of the rule.

**Changes since the previous bundle.**

* Restarts: each faulted pod now gets 1–2 restart actions per fault, all on the faulted pod. Before, the engine issued a restart on 30–38 of 45–49 decisions, mostly on the wrong pod (20 of 25 on d-leak-fast-oom-r2).
* Bad deploy: no scale-up on the held-out run (previously 7.1 %, and 4→6 recommendations with p95 at 150 ms).

## 5. Serving API and safety interlocks

* **Inputs.** The schema is documented field by field, with PromQL, in `engine.py` and `feature_spec.json`.
  * Required: `ts, rps, p95, error_rate, cpu_cores_used, replicas_ready`.
  * Optional: as before, plus **`revision`**: the `deployment.kubernetes.io/revision` annotation. This is rollout evidence when `pod_obs` is absent. Do not use `metadata.generation`, which also changes on every scale.
  * `pod_obs`: as before.
  * `now_ts`: optional wall-clock time of the decision.
* **Output.** As before, plus `telemetry_ok`.
* **Interlocks** (review 3; tests in `tests/test_scaling_safety.py`):
  * **Missing data is unknown, not zero.** If the newest valid `rps` sample is older than 30 s, or the last 5 min contain a gap over 60 s, then scale-down is frozen, the forecast is ignored, and the reasons say "insufficient telemetry". Demand comes from the last valid sample. In the reviewers' 4-minute dropout at 50 RPS on 7 pods, the old engine scaled 7→3. It now holds at 7.
  * **Bounds.** `min_replicas > max_replicas` raises an error. A current count above `max_replicas` is clamped to max at once, as the Kubernetes HPA does, because max is a hard operator bound. This is intentional.
  * **Outage.** With 0 ready pods, or ≥ 50 % 5xx on the last two samples, the engine pages at critical severity first, whatever the classifier says, and freezes scale-down.
  * **Restarts.**
    * At most one per deployment per 5 min.
    * None while ready < desired.
    * With a single ready pod, a surge `rollout_restart` (`max_surge 1, max_unavailable 0`) instead of `delete_pod`.
    * When no pod can be named, a page instead of a delete.
    * The engine is open loop, so an emitted restart is assumed to be applied.
  * **Leaking pod.** The OOM-ahead model's pick is used only when it flags the pod (anomaly, or p_oom ≥ 0.5). Otherwise the engine restarts the pod whose working set grows fastest over 2 min (≥ 10 KB/s), and falls back to the highest memory / limit.
  * **Rollback only with rollout evidence**: a new ReplicaSet hash, or a `revision` change, within 15 min.
    * bad_deploy without evidence is reported as a **cost-per-request regression** and paged.
    * crashloop without evidence pages at high severity and does not roll back.
    * crashloop with OOMKills is routed to the memory remediation (raise the limit or fix the leak).
* **Latency.** About 15–30 ms per decision.

## 6. Limitations

1. **The predictive policy has no demonstrated advantage over the HPA** (section 3). Ship the engine for its state, gate and remediation output. Use its replica count as advice, next to the HPA, until closed-loop cluster runs say otherwise.
2. **Not blind.** The held-out set influenced 4 decisions (section 1). The blind classifier figure is 0.803 settled / 0.761 strict.
3. **One lab, one app, one node.** The capacity model is clamped to 1–4 replicas, and the simulator is extrapolated beyond 3 pods.
4. **Forecaster prior.** On flat traffic the p90 at +60 s is 1.2–1.9x the current rate and at +120 s 1.8–3.7x. This is now level-invariant but still wide. The policy only uses the forecast when a measurement backs it.
5. **The classifier's load head has a schedule prior.** It calls flat 10.8 RPS on 1 pod scale_up. That state is therefore not used as scaling evidence.
6. **pod_restart is effectively undetected**: held-out F1 0, LORO F1 0.40.
7. **Simulator biases.** Violations are under-counted by about 1.36x. HPA scale-ups come about 7 s early. The metrics-server lag sits on the edge of the grid.
8. **Fault coverage is thin.** There are 2–5 runs per class and no benign rollouts. A x1.5 CPU regression sits below the 1.8x gate.
9. **Lab bug.** The cpu_hog child survives the chaos reset; those rows are masked.

## 7. What to collect next

1. **A frozen round 3**: new crashloop, error, hog and pod-kill runs, plus surges at unseen times, shapes and base rates. Score all of it once, after this bundle.
2. **Closed-loop cluster runs** where the engine drives replicas, to check the simulator's ranking and violation bias.
3. Benign rollouts; 5–8 replica step ramps; long flat and diurnal runs without surges, to retrain the forecaster's prior.

## 8. Files

| file | role |
|---|---|
| `kubeai_ml/scaling/train_all.py` | the pipeline: per-fold relabelled LORO, blind held-out variants, forecaster with a flat probe, capacity with per-run breach counts, simulator with warm-start, cost-matched and oracle-need metrics, fault replay |
| `kubeai_ml/scaling/engine.py` | `DecisionEngine`, input schema, interlocks |
| `kubeai_ml/scaling/policy.py` | replica policy (evidence-gated forecast, stop-gap bad deploy, telemetry freeze) |
| `kubeai_ml/scaling/hpa.py` | simulated HPA (`replica_calculator.go` grouping) |
| `kubeai_ml/scaling/split.py` | held-out runs and their caveats |
| `models/scaling/<version>/` | manifest, metrics, specs, policy, models, `fault_replay_decisions.csv.gz` |
| `tests/test_scaling*.py`, `tests/test_simulator.py` | held-out windows, interlocks, policy, simulator |

The exploration notes (`kubeai_ml/scaling/EXPLORE_*.md`) are superseded by this card.
