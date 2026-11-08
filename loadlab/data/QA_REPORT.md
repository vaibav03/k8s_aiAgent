# loadlab dataset: independent data QA

**Date:** 2026-11-08 (after collection finished at 09:22).
**Scope:** `data/dataset_deploy.csv.gz` (2,914 rows), `data/dataset_pods.csv.gz` (9,175 rows), and the 24 raw run folders in `../runs/`.
**Reproduce:** `$PY runner/build_dataset.py && $PY runner/qa_dataset.py`. The second command writes this report's numbers to `qa_results.json` and its plots to `plots/`.

## Verdict

**The dataset is usable as collected. No run needs to be re-collected for training.**

* 18 of 24 runs pass outright. 6 have warnings, all explained below, and none fail.
* Every fault shows the expected physical signature inside its ground-truth window, and the measured rates match what was injected:
  * leak slope 753 vs 768 KB/s, 95 vs 96 KB/s and 627 vs 640 KB/s;
  * error rate 0.189 vs 0.2;
  * bad-deploy CPU per request ×2.98 vs ×3.0.
* Integrity is clean in all 24 runs:
  * no gaps;
  * no duplicate timestamps;
  * timestamps monotonic in both files;
  * counters never decrease;
  * no out-of-range values;
  * every labelled window fully labelled, with no label rows outside a window.
* The QA pass fixed four small data issues in `runner/build_dataset.py` and rebuilt the files. Every original column of `dataset_deploy` is byte-for-byte unchanged (verified).

There are two substantive findings for whoever trains on this data:

1. **The SLO capacity per pod is not a constant 18 RPS.** With 1, 2 and 3 replicas, p95 crosses 300 ms at 20, 18 and 16 RPS per pod. Throughput tops out at about 17.8 RPS per pod in all three runs, so the throughput knee is the same and clear; only the latency knee moves. The card claimed 18–19 RPS per pod for all three, and it is now corrected. See `a-cap-fixed2/3` below.
2. **There is no HPA scale-in anywhere in group B.** All 8 runs scale from 1 to 4 pods and never back down. The only "scale down" example is `c-over-fixed4-steady`. This is a coverage gap, not a data fault; see *Recommended additional runs*.

## Per-run results

The columns are I (integrity: gaps, NaN, monotonicity, counters, bounds), L (labels match `faults.jsonl`), S (physical signature) and the overall result.
"Pre" is the clean stretch from 120 s to 10 s before the fault; "during" runs from 30 s after the fault starts to the fault end.

| run | overall | I | L | S | evidence |
|---|---|---|---|---|---|
| `a-cap-fixed1` | PASS | PASS | PASS | PASS | Last SLO-ok step 18 RPS/pod (p95 38 ms), first breach at 20 RPS/pod (p95 1.9 s), served rate flat at 17.9 RPS/pod past it, final p95 10 s, err 0.90 |
| `a-cap-fixed2` | **WARN** | PASS | PASS | WARN | Last SLO-ok step 16 RPS/pod (p95 227 ms), breach at 18 RPS/pod (0.67 s), plateau 17.8 RPS/pod. Throughput knee clear; latency knee soft and earlier (p95 already 212 ms at 14 RPS/pod) |
| `a-cap-fixed3` | **WARN** | PASS | PASS | WARN | Last SLO-ok step 14 RPS/pod (p95 244 ms), breach at 16 RPS/pod (0.44 s), plateau 17.7 RPS/pod. Same softening as fixed2 |
| `b-hpa-diurnal-s7` | PASS | PASS | PASS | PASS | Replicas 1→4, no scale-in, 0 SLO rows, corr(rps, hpa_desired) 0.49 |
| `b-hpa-diurnal-s23` | PASS | PASS | PASS | PASS | 1→4, no scale-in, 0 SLO rows, corr 0.45 |
| `b-hpa-spike-b08` | PASS | PASS | PASS | PASS | 1→4, 11 SLO rows, longest breach 40 s while the HPA lagged, p95 30 ms at the end |
| `b-hpa-spike-b05` | PASS | PASS | PASS | PASS | 1→4, 3 SLO rows, longest breach 15 s |
| `b-hpa-flash-b06` | PASS | PASS | PASS | PASS | 1→4, 0 SLO rows, corr 0.88 |
| `b-hpa-flash-b09` | PASS | PASS | PASS | PASS | 1→4, 9 SLO rows, longest breach 25 s, corr 0.82 |
| `b-hpa-saw-b10` | PASS | PASS | PASS | PASS | 1→4, 4 SLO rows, longest breach 20 s |
| `b-hpa-saw-b14` | PASS | PASS | PASS | PASS | 1→4, 0 SLO rows |
| `c-under-fixed1-diurnal` | PASS | PASS | PASS | PASS | Top-20 % load rows: p95 9.75 s, CPU/request 2.00 (at the limit), throttle 1.00; 39 % of rows break the SLO |
| `c-under-fixed2-spike` | PASS | PASS | PASS | PASS | Top-20 % load rows: p95 10 s, CPU/request 2.00, throttle 1.00; 29 % of rows break the SLO |
| `c-over-fixed4-steady` | PASS | PASS | PASS | PASS | CPU/request median 0.24 (HPA target 0.70), p95 30 ms, 2.2 RPS/pod against a capacity of 18 |
| `d-leak-oom` | PASS | PASS | PASS | PASS | Target pod slope 753 KB/s (768 injected), memory 0.14→0.96 of the limit, OOMKill expected and seen (`oomkills_total` at 481 s; restart at 478.8 s), p95 ratio 0.96 (flat) |
| `d-leak-slow` | **WARN** | PASS | PASS | PASS | Slope 95 KB/s (96 injected), memory 0.14→0.29, no OOM, which is expected because 96 KB/s × 420 s is less than the headroom. The warning is the baseline: 3 rows at p95 0.66 s at t = 114–124 s, from the node glitch (now `qa_flag=node_glitch`) |
| `d-dep-latency` | PASS | PASS | PASS | PASS | p95 30→490 ms (×16.3), CPU/request 0.63→0.63 (×1.00), in-flight 0→4, replicas stay at 3, errors 0 |
| `d-error-burst` | PASS | PASS | PASS | PASS | `error_rate` 0.000→0.189 (0.2 injected), p95 ×1.05, CPU ×1.02 |
| `d-cpu-hog` | PASS | PASS | PASS | PASS | Target pod throttle 0.03→**0.90** vs 0.03 on the other pods; target CPU 0.49 cores vs 0.12 on the others; deployment CPU/request 0.63→0.86; HPA 3→4; p95 134 ms |
| `d-pod-kill` | PASS | PASS | PASS | PASS | Both kills: `replicas_ready` 3→2, back to 3 within 60 s; p95 max 34 / 72 ms |
| `d-bad-deploy` | **WARN** | WARN | PASS | PASS | CPU ms/request 26.4→78.7 (**×2.98**, ×3.0 injected), throttle 0.04→0.55, p95 0.52 s, replicas 3→4 (capped), p95 back to 30 ms after the rollback. The warning comes from NaNs: `mem_growth` 20.5 % (pods younger than 90 s after two rollouts) and pod `cpu` 7.4 %. Both are expected. |
| `d-crashloop` | **WARN** | WARN | PASS | PASS | `crashloop_pods` max 1, restarts 0→5, new pod `pods_not_ready`=1 for the whole window (0 in every `none` row). `replicas_ready` **never drops** (stays at 3), because maxSurge lets the old ReplicaSet keep serving. The warning comes from NaNs: pod `p95` is NaN in 11.4 % of rows, because the crashing pod serves no traffic, which is expected. |
| `e-spike-dep-latency` | PASS | PASS | PASS | PASS | Dependency-only phase (210–240 s): p95 30→489 ms, CPU ×1.02, replicas 3. Spike + dependency phase: CPU/request 1.56, p95 0.68 s, replicas 4 |
| `e-diurnal-leak` | PASS | PASS | PASS | PASS | Slope 627 KB/s (640 injected), memory 0.14→0.98, OOM seen at 484 s, one restart; p95 ratio 1.41 is the diurnal traffic, not the leak |

### Capacity-knee detail (median of each 40 s step, first 15 s dropped)

| offered RPS/pod | fixed1 p95 | fixed2 p95 | fixed3 p95 | fixed1 throttle | fixed2 throttle | fixed3 throttle |
|---|---|---|---|---|---|---|
| 10 | 30 ms | 30 ms | 93 ms | 0.00 | 0.05 | 0.14 |
| 12 | 30 ms | 66 ms | 117 ms | 0.00 | 0.19 | 0.25 |
| 14 | 30 ms | 212 ms | 244 ms | 0.04 | 0.38 | 0.39 |
| 16 | 34 ms | 227 ms | **439 ms** | 0.09 | 0.63 | 0.61 |
| 18 | 38 ms | **671 ms** | 1.67 s | 0.33 | 0.88 | 0.76 |
| 20 | **1.90 s** | 4.53 s | 4.57 s | 0.64 | 0.99 | 0.91 |

* CPU per pod at equal RPS per pod is the same in all three runs, about 0.31 cores at 12 RPS/pod against a 0.5-core limit.
* CFS throttling starts much earlier when there are more replicas.
* Node CPU from cAdvisor pod-scope cgroups stayed under about 3.3 of 12 host cores in all three runs (Prometheus, re-queried), so this is not node saturation in the accounting sense. The likely causes are burstier per-pod arrivals and CPU-quota scheduling noise with more busy cgroups. The cause was not proven, and nothing in the data is wrong.

**Implication:** the agent must not treat "18 RPS per pod" as capacity. Throttle ratio and p95 are the real headroom signals, and the dataset supports learning that.

## Signal-separation checks (scale vs fix)

The dataset does carry the information needed to tell "needs replicas" apart from "is broken" (see `plots/fault_signatures.png`).

* **dep_latency vs saturation.** Dependency latency gives p95 around 0.49 s at CPU/request 0.4–0.65. Saturation gives high p95 only at CPU/request at or above 1.4 with throttle near 1. The two clusters do not overlap except in `e-spike-dep-latency`'s combined phase, where both causes are real.
* **bad_deploy vs load.** CPU per request moves from about 26 ms to about 79 ms. `none` rows in groups C–E have a median of 26.4 ms, but their 95th percentile is 44 ms because of post-fault tails and throttled overload, so per-request cost needs the ±25 s tolerance too. Under deep overload, fast 5xx pull the apparent CPU per request down to about 17 ms.
* **cpu_hog is a confounder for bad_deploy** at the deployment level: CPU per request rises to about 48 ms. Only pod-level skew separates them. Under `cpu_hog`, one pod runs at throttle 0.90 while the others sit at 0.03. Under `bad_deploy`, every new pod is throttled. **A model needs at least one pod-dispersion feature,** such as max minus median pod throttle or max pod CPU over median pod CPU.
* **crashloop** is invisible in p95, errors and `replicas_ready`. Only `crashloop_pods`, `restarts_total` and the new `pods_not_ready` show it.
* **error_rate** shows up only in `error_rate`. p95 and CPU stay within 5 %.
* **memory_leak** shows up only in the memory columns until the OOMKill. p95 and CPU stay flat, and the pod restart is the only traffic-side trace.

## NaN ratios

`plots/nan_heatmap.png` shows these per run and per column.

* **Deploy:** only `mem_growth_bytes_per_s` has NaNs, in 8.1 % of rows overall.
  * Per run it ranges from 4.4 % to 15.5 %, and reaches 20.5 % in `d-bad-deploy`.
  * By design, it is NaN until a pod is 90 s old.
  * Every other signal column is 0 % NaN in all 24 runs.
* **Pods:** after the fix, each signal column has 0.6–1.8 % NaN.
  * These rows are pods in their first one or two scrapes (no cAdvisor sample yet) or terminating pods.
  * The worst cases are `d-crashloop` pod `p95` at 11.4 % (the crashing pod serves no traffic) and `d-bad-deploy` pod `cpu` at 7.4 % (pods replaced twice).
* **`fault_target_pod` / `fault_ids`:** these read as NaN in pandas wherever no fault is active, because they were written as empty strings. Use `fillna("")`. This was left unchanged on purpose.

## Unlabelled anomalies

Scan: steady-traffic runs, rows labelled `none`, p95 above 300 ms.

| where | rows | explanation |
|---|---|---|
| `d-dep-latency` t = 422–437 s | 4 | 20 s rate window trailing the fault end (`post_fault_tail`) |
| `d-bad-deploy` t = 421–441 s | 5 | Rollback is itself a rollout; trails about 25 s (`post_fault_tail`) |
| `d-leak-oom` t = 636–646 s and `d-leak-slow` t = 114–124 s | 3 + 3 | **Node-wide glitch** at unix 1794151445–55 (08:24:05–15 MST), the same instant in both lanes; CPU/request and throttle rose too (0.78 and 0.10 in `d-leak-oom`). Cause unknown. Now tagged `qa_flag=node_glitch` |

Nothing else is unexplained.

## Fixes applied (in `runner/build_dataset.py`, then rebuilt)

1. **Pod `mem_ws` / `mem_frac_limit` == 0 → NaN.** 129 pod rows in 14 runs had a working set of exactly 0. cAdvisor reports 0 before a container's first stats refresh, and a live Python process is never 0 bytes, so these zeros would have looked like a huge memory drop to any growth or min feature. Deploy-level aggregates were not affected, because they use max and sum.
2. **New `qa_flag` column (both files):** `ok`, `node_glitch` (6 deploy rows) or `post_fault_tail` (66 deploy rows labelled `none` within 30 s of a fault end). Labels are unchanged; this column only lets scoring mask or down-weight those rows instead of hard-coding the ±25 s tolerance.
3. **New `pods_not_ready` (deploy):** the count of pods present but not ready at each step. It gives the crashloop / stuck-rollout signal that `replicas_ready` cannot, because of maxSurge.
4. **New `goodput_rps` (deploy):** `rps * (1 - error_rate)`. `rps` counts fast 5xx, so it climbs back to 22.7 RPS/pod at the last overload step (error rate 0.90) and fakes extra throughput.
5. `summary.json` now also reports pod NaN shares and row counts by `qa_flag`.
6. **`DATASET_CARD.md` corrections:**
   * `faults_index.csv` has **11** rows, not 13;
   * `seconds_since_start` is never negative (the lead-in is trimmed);
   * the capacity-knee statement is corrected;
   * the QA columns are documented.

Pre-fix copies of the build script and data are in the session scratchpad (not in the repo). `git diff` shows the script change.

## Re-collection list

**Required: none.**

**Optional, low priority:** `a-cap-fixed2` and `a-cap-fixed3`, each run **alone on the node**. This means the other lane is idle, so use `matrix.py --lanes 1 --only a-cap-fixed2 a-cap-fixed3`. The point is to separate "more replicas soften the knee" from cross-lane noise. Both were collected while an HPA diurnal run used 0.3–1.3 cores in the other lane. The current data is still valid as "capacity under background load".

## Recommended additional runs (coverage gaps, not defects)

* **HPA scale-in.** Repeat 2–3 B scenarios with `behavior.scaleDown.stabilizationWindowSeconds: 30`, or run for at least 15 min, so that "scale down" decisions exist beyond the single `c-over` run.
* **A second instance of each single fault** (`error_rate`, `cpu_hog`, `bad_deploy`, `crashloop`). Each has exactly 1 injection (44 rows), so a train/test split by run is impossible for those classes.
* **cpu_hog on a fixed-replica deployment.** The HPA hid part of the p95 effect by adding a pod.

## Plots (`plots/`)

* `run_<scenario>.png` (24 files): rps, replicas ready/desired, p95 (log scale, SLO line), CPU/request + throttle + error rate, and per-pod memory fraction (leak target in orange, restarts dashed), with the fault label window shaded.
* `capacity_knee.png`: p95 vs offered RPS/pod and served vs offered for the three A runs.
* `fault_signatures.png`: p95 vs CPU/request, and error rate vs CPU ms per request, for fault rows against `none` rows in groups C–E.
* `nan_heatmap.png`: NaN share per run and column in `dataset_deploy`.
