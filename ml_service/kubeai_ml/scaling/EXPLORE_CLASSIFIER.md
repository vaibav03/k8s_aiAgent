# State / decision classifier: exploration (round 1 + first round-2 runs)

**Question.** Every 15 s, for each deployment, which state is it in? The states are healthy, scale_up, scale_down, memory_leak, dependency_latency, error_fault, noisy_pod, bad_deploy, crashloop and pod_restart. And does scaling help?

**Answer.** Ship the **hybrid**: the runbook's physics fault gates (`rules.fault_gate`) go first. When no gate fires, a LightGBM head picks healthy, scale_up or scale_down from the shared features (`features.py`) and a capacity knee learned from data (`labels.learn_safe_capacity`). `train_classifier` picks between rules and hybrid by settled macro-F1 (rules wins a tie). On the current data it picks hybrid. The fault path is the same rules either way.

Headline numbers. The snapshot is 4,422 rows from 36 runs: all 24 round-1 runs plus the first 12 round-2 runs. Round 2 was still running when this was written.

| evaluation | model | macro-F1 settled | accuracy | false scale_up on fault rows (excl. spike overlap) | false pages on load rows | missed scale_up |
|---|---|---|---|---|---|---|
| leave-one-run-out, 36 folds | **hybrid** | **0.913** | 0.901 | 0.034 (0.005) | 0.004 | 0.152 |
| | rules | 0.906 | 0.883 | 0.033 (0.007) | 0.004 | 0.132 |
| | LightGBM multiclass | 0.761 | 0.861 | 0.035 (0.000) | 0.022 | 0.177 |
| | two-stage LightGBM | 0.758 | 0.862 | 0.037 (0.000) | 0.009 | 0.142 |
| **forward test**: fit on round 1, score round 2 (12 runs) | **hybrid** | **0.932** | 0.906 | 0.012 | 0.004 | 0.093 |
| | rules | 0.924 | 0.900 | 0.012 | 0.004 | 0.151 |
| | LightGBM multiclass | 0.941 | 0.901 | 0.000 | 0.009 | 0.058 |
| | two-stage LightGBM | 0.819 | 0.875 | 0.000 | 0.002 | 0.070 |

* **Forward test.** The round-1 rules were written before round 2 existed. They generalise to new fault intensities and fixed-replica setups: error_rate at 10 % and 30 %, cpu_hog at 350m and 480m on 3 fixed pods, bad_deploy at ×2 under the HPA and ×3 on 3 fixed pods. Their F1 on those classes is 0.94–1.00.
* **Why not LightGBM, even though it wins the forward test.** Its fault side scores well only on types it already saw in round 1. It also takes a shortcut: it calls bad_deploy on *any* rollout before the cost per request has moved (see "LightGBM rollout shortcut"). Under leave-run-out it still scores 0 on crashloop, a type seen in one run only.

Reproduce:

```bash
cd ml_service
PYTHONPATH=. python -m kubeai_ml.scaling.train_classifier
```

This takes about 3 minutes. It writes `classifier_cv.json` (keys `results` for leave-run-out and `round_holdout` for the forward test) and `models/scaling/state_classifier/`. The code reads whatever rows are in `loadlab/data/*.csv.gz`, so appending more round-2 runs needs no code change. The forward test always holds out the newest `round`.

## Files

| file | role |
|---|---|
| `states.py` | The 10 states, their remediation text, and `SCALING_HELPS` (the "do NOT scale" gate for the replica policy). |
| `features.py` | `build_features(deploy_rows, pod_rows=None)` for training and `features_at(...)` for serving (newest row). Windows are time-based, so the same code runs at a 5 s or 15 s cadence. 52 features. |
| `labels.py` | `learn_safe_capacity` (the knee per replica count), `make_labels` (fault ground truth plus capacity-derived load states), `residual_hog_mask`, and the `use` mask. |
| `rules.py` | The runbook: fault gates in priority order, then the load state. |
| `classifier.py` | `StateClassifier(mode=rules\|lgbm\|two_stage\|hybrid)`, plus `capacity_features`, `decide()`, `save` and `load`. |
| `train_classifier.py` | Leave-run-out CV, a forward test on the newest round, metrics, the rules-vs-hybrid auto-pick, and save. |
| `tests/test_scaling_classifier.py` | Happy-path synthetic tests (3, passing). |

## Labels (`labels.py`)

1. **Fault rows** take the fault's state. When faults overlap, the priority is crashloop > bad_deploy > memory_leak > noisy_pod > dependency_latency > error_fault > pod_restart. In `e-spike-dep-latency`, rows are labelled dependency_latency even while the spike saturates the pods, because the root cause decides the action.
2. **Capacity knee c(r).** For each replica count, a decision stump on per-pod *demand* separates SLO-ok rows from breached rows (fault-free rows with `qa_flag == ok` and `use`).
   * A replica count whose rows above the stump are not mostly breached never reached its knee. It inherits the most conservative knee.
   * Learned: **c(1)=17, c(2)=17, c(3)=13 RPS/pod**. c(4+) falls back to 13. The round-2 runs did not change it.
   * The knee falls as replicas are added, as the QA report found (20/18/16 offered).
3. **Demand, not goodput.** Under deep overload, goodput *collapses* to 2–5 RPS/pod, because 504s at the 10 s deadline are slow errors, not served work. `features.demand_rps` therefore uses `rps` while p95 breaches the SLO and `goodput_rps` otherwise, which removes fast injected 5xx from the load.
4. **scale_up** on either of two conditions:
   * the SLO is breached now and the pods are saturated (demand/pod ≥ 0.85·c(r), or throttle ≥ 0.5);
   * the peak demand over the next 60 s, divided by the current ready count, reaches c(r). This is the predictive part.
5. **scale_down** when all of these hold: the SLO is OK, ready ≥ 2, and the peak demand over [t−60 s, t+180 s] fits in `ready−1` pods at 75 % of c(ready−1). This must hold for every row of the trailing 60 s.
6. Otherwise the row is **healthy**.
7. `use=False` excludes these rows from training and scoring (196 rows in this snapshot):
   * rows with `qa_flag != ok`: post-fault tails and node glitches;
   * rows caught by **`residual_hog_mask`**: after a cpu_hog's logged end, the target pod still burns at least 0.15 core more than the median pod. This is detected from the pod table, with no run names (88 rows across the 3 hog runs; see the data findings). It replaced the round-1 hard-coded `KNOWN_LABEL_ISSUES` entry, which is now an empty manual override.

Label counts on usable rows: healthy 1635, scale_down 1293, scale_up 478, memory_leak 282, bad_deploy 132, noisy_pod 130, error_fault 122, dependency_latency 98, crashloop 44, pod_restart 12.

## Features (`features.py`)

All features are per deployment. They come from `dataset_deploy` columns plus an optional `dataset_pods` snapshot.

| group | features |
|---|---|
| load | demand per ready pod (now, 1 m mean, 3 m max, 1 m trend), goodput per pod, 1 m demand growth, in-flight per pod |
| capacity (`classifier.capacity_features`) | load / c(r) (now, 1 m, 3 m max), 1 m projected load / c(r), `down_fit` = 3 m peak load / (0.75·c(r−1)·(r−1)) |
| efficiency | CPU per goodput request (now and 1 m median); `cpr_ratio` = 1 m cpu/req ÷ the pre-change baseline (see below) |
| SLO | p95 and p99 / 0.3 s (now, 1 m mean, 1 m max, 30 s min), error rate (now, 1 m, 3 m), `fast_error_share` (errors while p95 < SLO) |
| compute | throttle, CPU vs request, CPU vs limit, `lat_cpu_decoupling` = p95_slo·(1−throttle) |
| memory | growth in KB/s (now, 1 m and 3 m medians), max pod fraction of the limit and its 3 m delta |
| lifecycle | restarts over 1 m and 3 m, OOMKills over 3 m, crashloop pods, not-ready pods, ready deficit vs desired, ready drop vs the 1 m max, HPA desired − ready |
| pods | max−median CPU, throttle and mem fraction across ready pods; max/median **CPU per request** across pods; number of ReplicaSet hashes; seconds since a new hash appeared (a rollout) |

The cpu/req baseline is the median over [t−10 m, t−2 m]. When pod names show a new ReplicaSet hash in the last 15 m, the baseline is the 10 m before that rollout instead.

Separating look-alikes:

* **cpu_hog vs bad_deploy.** Both raise CPU per request (median `cpr_ratio` is 1.8 for the hog and 1.9–2.9 for bad_deploy). Pod dispersion tells them apart: `pod_cpr_disp` is about 3.7 for the hog, because one pod is hot, and about 1.07 for bad_deploy, because every new pod is hot. `secs_since_rollout` helps too.
* **dependency_latency vs saturation.** Throttle is 0.04 at a CPU of 0.32 of the limit, versus throttle above 0.5.
* **error_fault vs overload.** Injected errors run at 10–30 % with p95 ≈ 30 ms. Overload 504s come with p95 ≈ 10 s.

## Leave-one-run-out detail (36 folds, groups = run_id)

For every held-out run, capacity is re-learned on the other runs only. Models train on `use` rows.

"Settled" metrics drop rows within 30 s after a label change, because labels trail signals by up to about 25 s. This drops all 12 pod_restart rows (30 s windows), so pod_restart is judged by detection delay only.

### Per class, settled (hybrid; the fault rows are identical to rules)

| class | precision | recall | F1 | n |
|---|---|---|---|---|
| healthy | 0.871 | 0.926 | 0.897 | 1540 |
| scale_up | 0.817 | 0.848 | 0.832 | 401 |
| scale_down | 0.939 | 0.879 | 0.908 | 1195 |
| memory_leak | 0.938 | 0.973 | 0.955 | 264 |
| dependency_latency | 0.902 | 0.640 | 0.748 | 86 |
| error_fault | 1.000 | 1.000 | 1.000 | 104 |
| noisy_pod | 0.980 | 0.893 | 0.935 | 112 |
| bad_deploy | 1.000 | 0.895 | 0.944 | 114 |
| crashloop | 1.000 | 1.000 | 1.000 | 38 |

The full confusion matrices are in `classifier_cv.json` → `results.<model>.confusion_settled`. The main confusions:

* scale_down ↔ healthy (136 / 53 rows). This is the 75 % target-utilisation boundary, and the replica policy applies its own hysteresis.
* scale_up ↔ healthy (57 / 52 rows). These are the 1 m projection on noisy ramps (sawtooth and flash runs).
* dependency_latency → scale_up (20 rows). This is the spike overlap in `e-spike-dep-latency`.

### The costly mistake: false scale_up on fault rows

The rate is 3.4 % overall and **0.5 %** without the e-spike overlap. Every remaining case is one of these:

* dependency_latency during a real 4× spike. Scaling does help with the spike part.
* the first rows of bad_deploy (3 %). There the HPA lags, the pods are throttled, and the cost jump has not yet been measured.

Pure LightGBM gets 0 % outside the overlap, but only because it calls every rollout bad_deploy.

### Detection delay after fault start (rules = hybrid, leave-run-out)

| fault | first correct state | row recall in window |
|---|---|---|
| memory_leak 768 KB/s (`d-leak-oom`) | 36 s | 0.92 |
| memory_leak 96 KB/s (`d-leak-slow`) | 58 s | 0.88 |
| memory_leak 640 KB/s + diurnal (`e-diurnal-leak`) | 39 s | 0.94 |
| dependency_latency (`d-dep-latency`) | 42 s | 0.80 |
| dependency_latency + spike (`e-spike-dep-latency`) | 42 s | 0.37 |
| error_rate 20 % / 10 % (HPA diurnal, r2) / 30 % (fixed 3, r2) | 12 s / 16 s / 11 s | 0.92 / 0.90 / 0.91 |
| cpu_hog 450m / 350m (r2) / 480m (r2) | 58 s / 51 s / 47 s; the first fault alert of any type comes at 33–41 s | 0.78 / 0.80 / 0.82 |
| bad_deploy ×3 / ×2 HPA (r2) / ×3 fixed 3 (r2) | 51 s / 52 s / 53 s | 0.74 / 0.68 / 0.68 |
| crashloop | 9 s | 0.86 |
| pod_kill × 2 | 3 s each | 0.17 (ready drops only briefly) |

Bad_deploy is labelled from the moment the rollout starts. The rolled-out pods only change `cpr_ratio` once they carry traffic, about 40–45 s later (1.0 → 1.8 at +45 s, about 1.9–2.9 by +55 s). So the gate fires 5–10 s after the physical evidence exists.

## Forward test: round 1 → round 2

This is the most honest number here, because the gates were written after seeing round 1 only.

* **Rules (frozen round-1 thresholds): 0.910 settled macro-F1 out of the box.** It rose to **0.924** after one fix to the error_fault gate (below). That fix was motivated by `d-error-burst` in round 1, but it also confirmed on round-2 rows, so treat 0.910 as the strictly blind number.
* **Hybrid: 0.916 → 0.932.** The LightGBM load head trained on round 1 cut missed scale_up from 15 % to 9 % on round-2 traffic: new HPA runs with a 30 s scale-down window, under- and over-sized fixed runs.
* **Per class (rules):**
  * error_fault 1.00 / 1.00;
  * noisy_pod 0.97 / 0.91;
  * bad_deploy 1.00 / 0.90.
  * The ×2 bad deploy reaches `cpr_ratio` 1.94, just over the 1.8 gate. A ×1.5 regression would be missed (see the next steps).

### LightGBM rollout shortcut (why its 0.941 does not ship)

In both round-2 bad_deploy runs, LightGBM trained on round 1 says bad_deploy at **+7 s**, while `cpr_ratio` is still 1.00. Its only evidence is `n_rs_hashes = 2`: a rollout is in progress. The only rollouts in the lab are faults (bad_deploy, crashloop, and their roll-backs), so the model learned "rollout ⇒ bad_deploy". On a real cluster that means a rollback page on every deploy.

The gate waits for the measured cost jump instead. That is slower (about 50 s) but causal. The same shortcut explains LightGBM's apparent 0 % false scale_up on bad_deploy.

Round 3 needs **benign rollouts** (the same image with a new `APP_VERSION`) before any learned fault head can be trusted.

## Fixes in this pass

1. **`residual_hog_mask`** (in `labels.py`) replaces the hard-coded `("d-cpu-hog", 420 s)` entry.
   * Without it, the appended round-2 hog runs dropped noisy_pod precision to 0.63 (0.98 with the mask), and false pages on load rows were 3.0 % (1.1 % with the mask, 0.4 % after fix 2).
   * The rows are physically still a noisy pod, so they are masked, not relabelled.
   * `learn_safe_capacity` now also respects `use`.
2. **The error_fault gate also requires `error_rate ≥ 0.5 %` now.**
   * The 1 m mean lingers about 60 s after the 5xx stop, which gave 22 false error_fault pages on healthy rows across the 3 error runs.
   * error_fault precision went from 0.825 to 1.000 and false pages from 1.1 % to 0.4 %, with no recall or delay loss.
3. **The forward test (`round_holdout`) was added** to `train_classifier`. It runs automatically when the dataset has more than one `round`.

## Design decisions

1. **Faults first, then load.** The gates check the root causes that scaling cannot fix, in this order: crashloop, memory_leak, pod_restart, bad_deploy, noisy_pod, error_fault, dependency_latency. `scale_up` is only reachable when no gate fires. `states.SCALING_HELPS` gives the replica policy a hard "do NOT scale" input.
2. **Gates and the physics behind them.** None of the thresholds is fitted to the folds.
   * **crashloop:** CrashLoopBackOff pods, or a not-ready pod with at least 2 restarts in 3 m.
   * **leak:** the 1 m median growth is at least 50 KB/s (at that rate a 256 Mi pod OOMs in under 90 min), or there was an OOMKill in the last 3 m. The growth path is ignored during a request backlog (≥ 10 in-flight per pod, or throttle ≥ 0.8) and on a noisy pod.
   * **bad_deploy:** cpu/req is at least 1.8× the baseline, uniform across pods (max/median < 1.6), with a rollout in the last 15 m. Without pod data the rollout condition is replaced by "not overloaded".
   * **noisy_pod:** max/median cpu/req across pods is at least 2, and max−median CPU is at least 0.15 cores, with no recent rollout.
   * **error_fault:** fast 5xx ≥ 1 % over 1 m, and ≥ 0.5 % now.
   * **dependency_latency:** p95 is above the SLO for 30 s straight, while throttle < 0.3 and CPU < 60 % of the limit.
3. **Capacity is learned, not hardcoded.** It is re-learned inside every CV fold. `capacity.py` (P(breach | load, cpu/req, replicas)) can replace `SafeCapacity` inside `capacity_features` without changing the classifier interface.
4. **Hybrid over rules.** The load head beats the rules on healthy and scale_down precision, and on the forward test it also misses fewer scale_ups: 9 % against 15 %. It does miss more scale_ups than the rules under leave-run-out (15.2 % against 13.2 %). Its fault behaviour is identical, because the gates come first. The auto-pick re-decides on every retrain.
5. **Serving.**
   * The call chain is `features_at(last_N_deploy_rows, pod_rows)` → `StateClassifier.load(models/scaling/state_classifier).decide(feats)` → `{state, remediation, scaling_helps}`.
   * At least 15 min of history gives the cpu/req baseline.
   * Without pod rows, only the dispersion and rollout features are disabled. noisy_pod then cannot be detected, and bad_deploy falls back to "not overloaded".
   * The OOM-ahead model (`models/LATEST`) stays the per-pod memory-risk input, and the leak gate is its deployment-level counterpart. The engine should page if either fires.

## Data findings (for the lab owner)

* **The cpu_hog child process survives the chaos reset in every hog run (rounds 1 and 2).** The target pod keeps 0.45–0.5 core and throttle 0.4–0.98 until the run ends, although `faults.jsonl` logs a successful reset. The most likely cause is in `app/main.py` `_apply_hog`: the forked child inherits uvicorn's SIGTERM handler. Kill it with SIGKILL, or reset the signal handlers in the child. `residual_hog_mask` handles this until the fix lands.
* **Goodput collapses under overload.** Anything that uses `goodput_rps` as demand under-provisions exactly at saturation. Use `features.demand_rps`.
* **Memory growth is load-sensitive.** Deep request queues show 50–80 KB/s of growth with no leak, so gate it on backlog.
* **No benign rollouts.** Every rollout in the data is a fault, which lets learned models shortcut bad_deploy (see above).
* **Coverage still thin:** crashloop and pod_kill have 1 run each in this snapshot. The `d-crashloop-fixed3-r2` and `d-dep-latency-*-r2` runs were still running.

## Limits and next steps

* The thresholds were set by an engineer who had seen the round-1 fault runs. The forward test on round 2 is the honest check, and it held at 0.91–0.93. Re-run `train_classifier` when round 2 finishes: the crashloop and dep_latency round-2 runs will test the remaining gates blind.
* Leak, dependency, hog and bad_deploy detection takes 40–60 s, because of the 1 m windows and the time the rollout needs to carry traffic. LightGBM detects leaks and dependency latency faster (11 s and 7 s). An "early-warning" LightGBM fault head behind the gates is worth trying once round 3 adds benign rollouts and at least 3 runs per fault type.
* bad_deploy margin: ×2 regressions sit at 1.94 against the 1.8 gate. Lowering the gate to about 1.4 *with* the rollout condition kept fault-free false hits at 5 rows here, but it cannot be validated without benign rollouts, so it was not changed.
* The scale_up and scale_down labels depend on c(r) and the 75 % target. Most healthy ↔ scale_down disagreements sit on that boundary.
