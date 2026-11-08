# Model card: `oom-gbm-h120-20261108121800`

## What it predicts

`p_oom` is the calibrated probability that a pod that is **Running now** will crash
(become OOMKilled or CrashLoopBackOff, or its restart count goes up) **within the next
120 s**. The model only sees memory: working-set bytes over the last 5 minutes and the
memory limit.

`is_anomaly` is **`p_oom >= 0.50` AND `time_to_limit_s <= 300`**: the alert also needs
memory to be on course to reach the limit within 5 minutes (the "trend gate", see
below). A pod that sits flat, however high, is not flagged unless it is at or above
its limit.

This is a different label from the old models. They used `Node Status != Running`,
which marks pods that have **already** crashed: 89% of those rows are CrashLoopBackOff
at about 79 MB. The new label only looks at rows that are still Running, so a correct
prediction is a real early warning.

## Data

| | |
|---|---|
| Training file | `data/k8s_metrics.csv`, sha256 `6d130e4a…eff3b` |
| Content | 1 pod (`server-image-*`), 1 workload, 2025-08-30, 24,116 rows (24,014 after removing null/0 memory and duplicate timestamps) |
| Sampling | about 1 s, but 94% of rows repeat the last value (the scrape interval is 15-30 s). Resampled to a 15 s grid |
| Crashes | 40 crash events after merging status flaps less than 60 s apart |
| Memory unit | `Memory Usage` = metrics-server working set in Ki / 1000 (DevTrails `server-prom.js`), so bytes = value × 1,024,000 |
| **Limit (assumption)** | The real limit during collection is unknown: the 768Mi in the DevTrails YAML is exceeded by 2,800 rows. OOMKills happen at the limit, so the **observed maximum, 887.345 (≈ 866.5 MiB), is used as a proxy for the limit**. Every usage fraction in training depends on this assumption |
| Cross-dataset file | `data/nirmal_dataset_1.csv`: a different pod and workload (Chaos Mesh stress), MiB, real limit 900Mi, 3 OOM events, 2,643 rows |

## Method

- **Input shape = what the agent sends.** Every training, validation and test example
  is built from the history the deployed agent would hold at that moment: one sample
  every 15 s (a Prometheus instant query returns the newest scrape, timestamped at the
  query time), 40 samples, and **only samples since the container's last (re)start**,
  because the agent clears its window when `restartCount` goes up. The serving path is
  therefore identical: replaying those exact sample lists through `/v1/predict`
  reproduces the offline `p_oom` exactly (max |difference| 0 over all 1,272 rows).
- **Features** (`kubeai_ml/features.py`, the same code runs in training and serving):
  usage fraction on a 15 s forward-filled grid. For windows of 30/60/120/300 s:
  last, mean, max, std and Theil-Sen slope. Also headroom, and time-to-limit from
  the 120 s slope (capped at 3600 s).
- **Features left out on purpose:** `restart_count` keeps rising through the file
  (40 to 80), so it works as a clock. A shuffled split plus a time feature is what
  inflated the old notebook results. `seconds_since_restart` is available behind
  `--with-restart-feature` but is not shipped; the agent may not always send it.
- **Split:** time-ordered and never shuffled. The cuts fall right after crash events:
  24 events for train (until 10:58:22 UTC), 8 for validation (until 11:53:55), 8 for
  test. Training examples are taken every 5 s (so the 15 s sampling phase varies).
  Validation and test examples are taken every 15 s, the same rate as the agent.
- **Model:** LightGBM, 200 rounds, 15 leaves. `scale_pos_weight = neg/pos` is the only
  correction for class imbalance. Monotone constraints: risk can only go up with
  usage, max and slope, and only go down with headroom and time-to-limit.
- **Calibration, threshold and trend gate:** isotonic calibration fit on validation.
  The threshold and the trend gate (`max_time_to_limit_s` from {300, 600, 900, 1800})
  are chosen together for the best F1 on validation (ties go to the tighter gate).
  **The test split was not used for any choice.**
- **Horizon:** 30, 60 and 120 s were each trained. The horizon with the best validation
  F1 is shipped (120 s: 0.881, against 0.833 at 60 s and 0.788 at 30 s). A longer
  horizon also gives the agent more lead time before the crash.

## Why the trend gate

The training data has almost no pod that stays flat at high usage: in this file high
usage always ended in an OOM. So `p_oom` for a **flat** pod (40 samples, 15 s apart)
climbs with usage alone: 0.27 at 60-66% of the limit, about 0.33-0.35 at 66.5-68.5%,
and **0.72-0.93 from 69% of the limit up** (the boundary for `p_oom >= 0.5` is about
0.69 flat). Without the gate, every pod that runs steadily above about two-thirds of
its limit would alert on every tick, and the agent's hysteresis and cooldown would not
help because the condition never clears. With the gate, flat pods at 50-99% of the
limit all return `is_anomaly: false` (a `reasons` line says why); a flat pod at or over
its limit (`time_to_limit_s = 0`) is still flagged.

What the gate costs and buys (threshold re-tuned on validation for the ungated row):

| | Val F1 | Test F1 | Test false alarms/h | Cross-dataset F1 | Cross false alarms/h |
|---|---|---|---|---|---|
| **Gated (shipped)** | 0.881 | 0.828 | 1.5 | **0.638** | **9.1** |
| No gate | 0.900 | 0.828 | 1.5 | 0.484 | 27.3 |

`p_oom` itself is not gated, so a consumer that wants the raw score still has it.

## Results on the test split (8 crashes, 272 Running rows, 58 positive)

The rule baselines use thresholds tuned on validation. All inputs are agent-shaped.

| H = 120 s | PR-AUC | ROC-AUC | F1 | Precision | Recall | Crashes caught | Mean lead | False alarms/h |
|---|---|---|---|---|---|---|---|---|
| **LightGBM + trend gate (shipped)** | 0.828 | 0.945 | 0.828 | **0.828** | 0.828 | 8/8 | 122 s | **1.5** |
| Usage-fraction rule (usage ≥ 0.687) | 0.899 | **0.974** | **0.833** | 0.806 | 0.862 | 8/8 | 126 s | 4.5 |
| Time-to-limit rule (ttl ≤ 118 s) | **0.909** | 0.973 | 0.800 | 0.701 | **0.931** | 8/8 | 180 s | 3.0 |
| Old agent (LSTM → LightGBM, 0.5 cut) | 0.748 | 0.877 | 0.726 | 0.682 | 0.776 | 8/8 | 186 s | 11.9 |

"Crashes caught" counts a crash when an alert fires in the 300 s before it. Lead time is
measured from the first such alert. False alarms per hour count alerts on Running rows
more than 300 s from any crash.

All three horizons (test split; thresholds and gate always chosen on validation):

| Horizon | Model F1 / PR-AUC | Usage rule F1 / PR-AUC | TTL rule F1 / PR-AUC | Old agent F1 / PR-AUC |
|---|---|---|---|---|
| 30 s | 0.774 / 0.909 | 0.741 / 0.923 | 0.828 / 0.942 | 0.390 / 0.444 |
| 60 s | 0.769 / 0.904 | 0.857 / 0.936 | 0.853 / 0.977 | 0.604 / 0.726 |
| 120 s | 0.828 / 0.828 | 0.833 / 0.899 | 0.800 / 0.909 | 0.726 / 0.748 |

### Input shape

The numbers above use agent-shaped input (15 s instant-query ticks). The same shipped
model fed the raw ~1 s collector rows instead (cut at restarts too) gives val F1 0.881 /
PR-AUC 0.962 and test F1 0.817 / PR-AUC 0.828, so the two shapes agree closely. The
earlier bundle was trained and evaluated on raw rows only; its reported test F1 of
0.891 described an input the agent never sends.

### Honest reading

- **The model does not beat simple rules in-dataset.** At 120 s its F1 is level with
  the usage rule and its PR-AUC is below both rules; at 60 s both rules beat it. Its
  one clear in-dataset advantage is fewer false alarms (1.5/h against 3.0-4.5/h).
  With only 8 test crashes, a difference of a few F1 points is noise.
- In this data, every crash comes after memory climbs steadily to the limit, and
  "high usage" alone already explains most of the label. The ML adds little beyond
  usage plus trend.
- The old agent pair is much worse than everything else. It flags all the time (12
  false alarms per hour), which is why its recall and lead time look high.

### Cross-dataset (`nirmal_dataset_1.csv`, limit 900Mi, 3 crashes, 184 Running rows, 24 positive)

The same model, threshold and gate were applied, with no retraining.

| H = 120 s | PR-AUC | ROC-AUC | F1 | Precision | Recall | Crashes caught | False alarms/h |
|---|---|---|---|---|---|---|---|
| LightGBM + trend gate | 0.413 | 0.781 | **0.638** | **0.652** | 0.625 | 3/3 | **9.1** |
| LightGBM, no gate | 0.413 | 0.781 | 0.484 | 0.395 | 0.625 | 3/3 | 27.3 |
| Usage-fraction rule | **0.750** | **0.893** | 0.436 | 0.315 | 0.708 | 3/3 | 43.6 |
| Time-to-limit rule | 0.689 | 0.832 | 0.597 | 0.515 | 0.708 | 3/3 | 14.5 |
| Old agent | 0.623 | 0.859 | 0.432 | 0.320 | 0.667 | 3/3 | 47.3 |

**The model's probability ranking does not transfer well**: on a different workload
(stress-chaos memory spikes) its PR-AUC is the worst of all. The gated alert has the
best F1 and the fewest false alarms there, but that comes mostly from the trend gate,
i.e. from the time-to-limit signal. With 3 events this is weak evidence either way. At
60 s the model ranks better on this file (PR-AUC 0.896 against 0.929 for the usage
rule and 0.929 for the TTL rule).

## Forecast (`forecast` in the API)

The forecast extrapolates a robust trend: `last + 0.75 × TheilSen slope(120 s) × h`,
clipped to `[0, limit]`. The damping (0.75) was picked on validation from
{0, 0.25, 0.5, 0.75, 1}, where 0 means plain persistence. No ML forecaster is
shipped: the old LSTM is worse than persistence.

MAE in MiB on the test split. Targets are taken only while the pod is still Running.

| Horizon | Persistence | Trend (shipped) | Old LSTM |
|---|---|---|---|
| +15 s | 19.3 | **14.8** | 40.6 |
| +30 s | 40.1 | **21.8** | 58.1 |
| +60 s | 82.8 | **43.2** | 96.4 |
| +120 s | 170.0 | **80.9** | 176.3 |

On the cross-dataset the trend is **worse** than persistence (+15 s: 39.4 vs 32.1;
+120 s: 231.6 vs 201.1). Stress-chaos memory jumps up and down in steps and does not
ramp. The old LSTM there is roughly equal to persistence (43.3 / 196.5).

## Why the service is still useful

Each response also carries the model-free signals: `usage_fraction`,
`trend_bytes_per_s`, `time_to_limit_s` and a plain-language `reasons` list. On the test
split a `time_to_limit_s <= 118` rule is about as good as the model. Consumers can alert
on either signal. The agent applies k-of-n hysteresis and a cooldown on top of
`is_anomaly`.

## Known limitations

- One pod, one workload and one 10.5 h run. The 40 crashes are highly correlated, and
  the test split has 8 crashes, so the confidence intervals are wide.
- The limit used in training is a proxy (the observed max). Every usage fraction
  depends on it.
- **Steady high usage.** `p_oom` for a pod that sits flat at about 69% or more of its
  limit is 0.72-0.93, because in this data high usage always ended in an OOM. The
  trend gate keeps `is_anomaly` false for those pods, but the flip side is that a pod
  sitting flat just below its limit that then OOMs from a sudden jump (no ramp) is
  not flagged before the jump. Do not alert on raw `p_oom` without a trend condition.
- With fewer than about 2 minutes of history (for example right after a restart, when
  the agent clears its window), the slope features rest on few points. The response
  says so in `reasons`.
- No limit (`limit_bytes: null`): `p_oom` is 0 and not scored, because a container
  with no limit cannot be OOMKilled by its own limit. Node-pressure eviction is not
  modelled. Trend and forecast are still returned.
- Scores are per container: the kernel OOM-kills each container against its own
  limit, so the agent sends one item per container (see the agent README).
- Retrain with `python -m kubeai_ml.train` when data from more pods, limits and
  workloads is available. The same script reports every baseline next to the model.
