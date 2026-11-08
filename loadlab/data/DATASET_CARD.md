# loadlab dataset card

Labelled Kubernetes autoscaling and remediation telemetry from the loadlab (see `../README.md`).

> **Round 2 added (2026-11-08 09:36–11:36 MST):** 20 more runs, so there are now **44 runs, 5,368 deploy rows and 16,608 pod rows**, and a `round` column. The round-1 description below is unchanged. See [Round 2](#round-2) at the end.

**Collection:** 2026-11-08, 07:04–09:22 MST, on minikube profile `kubeai-lab` (6 CPU / 6 GiB, one node).
All 24 runs in `../scenarios/matrix.yaml` passed on the first attempt.

**Purpose:** train and evaluate an agent that chooses among three actions:

* scale up or scale down;
* leave the deployment alone;
* remediate, because scaling will not help (memory leak, slow dependency, 5xx errors, CPU hog, bad deploy, crashloop, pod loss).

**SLO:** p95 < 300 ms and error rate < 1 %.

## Files

| file | rows | content |
|---|---|---|
| `dataset_deploy.csv.gz` | **2,914** | one row per run per 5 s, deployment-level signals plus ground truth |
| `dataset_pods.csv.gz` | **9,175** | one row per pod per 5 s, plus ground truth |
| `runs_index.csv` | 24 | one row per run: scenario, scaling spec, row counts, max p95, SLO-violation rows, k6 summary |
| `faults_index.csv` | 11 | one row per injected fault: type, target, start/end relative to t0, duration, params, re-applications |
| `summary.json` | – | machine-readable counts: rows by fault, by group, by `qa_flag`, NaN shares |
| `QA_REPORT.md`, `qa_results.json`, `plots/` | – | independent data QA (`runner/qa_dataset.py`): per-run pass/warn/fail with evidence, plots |

Raw per-run folders are kept in `../runs/<run_id>/`. Each holds `deploy.csv`, `pods.csv`,
`faults.jsonl` (the ground truth), `scenario.json`, `k6_summary.json`, `k6.log`, `events.json`,
`pods_final.json`, `runner.log` and `queries.json` (the exact PromQL used).

## Scenario matrix (24 runs, 237 min of load)

Notes for the tables below:

* `base_rps` is in RPS. The calibrated capacity of one pod is **18 RPS at the SLO** and 19.6 RPS at maximum throughput.
* HPA runs use `autoscaling/v2` with a CPU target of 70 % and the default behaviour. `max` is 6 in the spec but was clipped to **4** by the 2-lane cap.
* "Rows" is the number of rows in `dataset_deploy`.
* "Viol" is the number of deploy rows breaking the SLO.
* "k6 fail" is the share of requests that failed on the client side.

### A. Capacity: fixed replicas, step ramp past saturation

Each run lasts 400 s, in 40 s steps.

| scenario | replicas | traffic | rows | max p95 | viol | k6 fail |
|---|---|---|---|---|---|---|
| a-cap-fixed1 | 1 | 6 → 24 RPS in 2-RPS steps | 85 | 10 s | 26 | 0.186 |
| a-cap-fixed2 | 2 | 12 → 48 RPS in 4-RPS steps | 85 | 10 s | 36 | 0.175 |
| a-cap-fixed3 | 3 | 18 → 72 RPS in 6-RPS steps | 84 | 10 s | 46 | 0.176 |

The **throughput** ceiling is the same in all three runs: about 17.8 RPS served per pod (0.5-core
limit). The **SLO** knee is not: p95 crosses 300 ms at 20 RPS/pod with 1 replica, at 18 with 2 and at
16 with 3, because CFS throttling starts earlier as replicas are added (throttle 0.19 / 0.25 at
12 RPS/pod with 2 / 3 replicas vs 0.0 with 1). Do not hard-code 18 RPS/pod as SLO capacity; see
`QA_REPORT.md`. Past the ceiling, in-flight requests climb to
200+ and p95 rises to the 10 s deadline. Overload then shows up as 504s in `error_rate`, which reaches 0.9.

### B. Reactive baseline: hpa_cpu, starting at 1 pod (held-out traffic for autoscaler replay)

| scenario | duration | traffic | base_rps | rows | max p95 | viol |
|---|---|---|---|---|---|---|
| b-hpa-diurnal-s7 | 780 s | diurnal, period 520 s, amplitude 0.6, seed 7 (10.8–43 RPS) | 27 | 159 | 0.14 s | 0 |
| b-hpa-diurnal-s23 | 780 s | diurnal, period 400 s, amplitude 0.5, noise 0.12, seed 23 | 18 | 158 | 0.09 s | 0 |
| b-hpa-spike-b08 | 540 s | 4× spike at 180 s for 120 s (14.4 → 57.6 RPS) | 14.4 | 111 | 1.37 s | 11 |
| b-hpa-spike-b05 | 540 s | 4× spike at 150 s for 120 s (9 → 36 RPS) | 9 | 110 | 0.42 s | 3 |
| b-hpa-flash-b06 | 540 s | sustained 3× from 150 s (10.8 → 32.4 RPS) | 10.8 | 111 | 0.21 s | 0 |
| b-hpa-flash-b09 | 540 s | sustained 3× from 180 s (16.2 → 48.6 RPS) | 16.2 | 111 | 2.65 s | 9 |
| b-hpa-saw-b10 | 540 s | sawtooth, period 120 s, 0.5×–2× (9–36 RPS) | 18 | 110 | 2.53 s | 4 |
| b-hpa-saw-b14 | 540 s | sawtooth, period 180 s, 0.5×–2× (12.6–50 RPS) | 25.2 | 110 | 0.23 s | 0 |

The SLO breaks briefly while the HPA lags behind fast ramps; it takes about 15–60 s to scale out.
These rows are "needs more replicas" examples. Because the HPA never scales down within 5 min, the
B runs also hold over-provisioned stretches after each peak.

### C. Wrong fixed size

| scenario | duration | replicas | traffic | rows | max p95 | viol | k6 fail |
|---|---|---|---|---|---|---|---|
| c-under-fixed1-diurnal | 720 s | 1 | diurnal 5.8–23 RPS (peaks above the 18 RPS capacity) | 149 | 10 s | 58 | 0.100 |
| c-under-fixed2-spike | 540 s | 2 | 4× spike, 14.4 → 57.6 RPS (capacity 36) | 111 | 10 s | 32 | 0.392 |
| c-over-fixed4-steady | 480 s | 4 | steady 9 RPS (2.25 RPS per pod, about 12 % of capacity) | 98 | 0.03 s | 0 | 0 |

Under-provisioned runs are clear "scale up" examples. The over-provisioned run is a clear "scale down" example.

### D. Single faults under hpa_cpu

Traffic is steady 18 RPS, starting at 3 pods (6 RPS per pod, about 63 % CPU of request). A clean
stretch comes before and after each fault.

| scenario | duration | fault (scheduled window) | rows | max p95 | viol | signature observed |
|---|---|---|---|---|---|---|
| d-leak-oom | 720 s | memory_leak 768 KB/s, one pod, 180–600 s, freed at end | 147 | 0.60 s* | 3* | memory 0.14 → 0.88 of limit at ~0.78 MB/s, OOMKill at ~478 s, leak re-applied, memory grows again. RPS, CPU and p95 stay flat. |
| d-leak-slow | 720 s | memory_leak 96 KB/s, one pod, 180–600 s, memory kept | 147 | 0.66 s* | 3* | memory 0.14 → 0.29, no OOM. Growth stops at 600 s. |
| d-dep-latency | 600 s | dep_latency 400 ms on 50 %, all pods, 200–420 s | 122 | 0.49 s | 47 | p95 0.49 s while CPU stays flat at about 0.47 cores (63 % of request) and the HPA stays at 3. In-flight requests 0 → 3–5. |
| d-error-burst | 600 s | error_rate 0.2, all pods, 200–420 s | 123 | 0.04 s | 47 | `error_rate` about 0.2. Latency and CPU unchanged. |
| d-cpu-hog | 600 s | cpu_hog 450m, one pod, 200–420 s | 122 | 0.21 s | 0 | CPU 0.47 → 0.85 and throttle 0.04 → 0.3–0.4. The HPA scales 3 → 4, which does not help (noisy neighbour). It stays at 4 afterwards. |
| d-pod-kill | 600 s | pod_kill at 200 s and 380 s | 123 | 0.07 s | 0 | ready replicas 3 → 2 → 3 within about 30 s. |
| d-bad-deploy | 600 s | bad_deploy, CPU_MS_PER_REQ 25 → 75, 200–420 s, then rollback | 122 | 1.10 s | 45 | CPU 0.48 → 1.4 cores, throttle 0.5–0.77. The HPA scales 3 → 4 (capped). p95 stays at 0.4–1.0 s because 4 pods are not enough. Recovers after the rollback. |
| d-crashloop | 600 s | crashloop rollout (CRASH_ON_START=1), 200–420 s, then rollback | 122 | 0.04 s | 0 | `crashloop_pods` 1 and restarts 0 → 5. The old ReplicaSet keeps serving, so the SLO holds. |

\* Not caused by the fault: see the cross-lane interference note below.

### E. Combinations

| scenario | duration | traffic / fault | rows | max p95 | viol | signature |
|---|---|---|---|---|---|---|
| e-spike-dep-latency | 660 s | 4× spike at 240 s for 120 s (14.4 → 57.6 RPS), plus dep_latency 400 ms / 50 % on all pods, 180–450 s | 135 | 1.98 s | 56 | p95 stays about 0.49 s at low CPU, and the spike adds saturation on top (CPU 1.5–1.95× request, HPA 3 → 4). The two causes overlap. |
| e-diurnal-leak | 780 s | diurnal 10.8–32 RPS, plus memory_leak 640 KB/s on one pod, 120–690 s | 159 | 0.07 s | 0 | OOMKill at about 480 s while traffic and the HPA swing. The leak is re-applied after the restart. |

## Ground-truth fault time

The table counts actual applied windows from `faults.jsonl`.

| fault type | injections | fault minutes | deploy rows labelled | pod rows where the pod is a target |
|---|---|---|---|---|
| memory_leak | 3 | **23.50** | 282 | 282 |
| dep_latency | 2 | **8.17** | 98 | 330 |
| error_rate | 1 | **3.67** | 44 | 132 |
| cpu_hog | 1 | **3.70** | 44 | 44 |
| bad_deploy | 1 | **3.67** | 44 | 163 |
| crashloop | 1 | **3.67** | 44 | 44 |
| pod_kill | 2 | 0 (instant). Labelled for 30 s each, 1.00 min in total | 12 | 6 |
| **none** | – | – | 2,346 | 8,174 |

There are 46.4 fault minutes in total (47.4 with the pod_kill windows), and 568 of the 2,914 deploy
rows (19.5 %) carry a fault label. In group E, 54 rows have dep_latency active alongside a spike.

Deploy rows per group: A 254, B 980, C 358, D 1,028, E 294.

426 deploy rows break the SLO.

## Schema

### Added columns (both files)

| column | type | meaning |
|---|---|---|
| `run_id` | str | `<scenario>-<MMDD-HHMMSS>-<rand>`; the folder name in `../runs/` |
| `scenario` | str | id in `scenarios/matrix.yaml` |
| `group` | str | A capacity, B reactive HPA, C wrong fixed size, D single fault, E combination |
| `policy` | str | `fixed` or `hpa_cpu` |
| `shape` | str | k6 traffic shape |
| `base_rps` | float, RPS | base offered load of the shape |
| `seconds_since_start` | float, s | `ts - t0`, where t0 is when the k6 pod was Running. Always > 0: `promexport` trims the lead-in until traffic is flowing. |
| `active_fault` | str | fault types whose ground-truth window contains `ts`, joined with `+` in start order. `none` if there are none. |
| `fault_target_pod` | str | one entry per active fault, joined with `\|`. An entry is `all`, `rollout`, or the target pod names joined with `;`. |
| `fault_ids` | str | `<run_id>-f<idx>` per active fault, joined with `\|`. Joins to `faults.jsonl` and `faults_index.csv`. |

Pods file only:

| column | type | meaning |
|---|---|---|
| `pod_is_fault_target` | 0/1 | 1 if the pod is hit by an active fault. For target `all`, every pod. For a rollout fault, pods of the new ReplicaSet (their pod-template hash differs from the baseline). For a one-pod fault, the pods listed in `faults.jsonl` `pods`. |
| `pod_active_fault` | str | the active fault types that hit this pod, or `none` |

**How fault windows are defined:**

* Chaos faults (`memory_leak`, `dep_latency`, `error_rate`, `cpu_hog`) use `[start_ts, end_ts]`, the
  actual apply and reset times recorded by the runner. These faults are re-applied within 3–5 s after
  a container restart or the arrival of a new pod (`reapplied` in `faults.jsonl`).
* `bad_deploy` and `crashloop` use `[start_ts, end_ts]`, from the rollout of the bad env to the
  start of the rollback.
* `pod_kill` uses `[delete_ts, delete_ts + 30 s]`.

### QA columns (added by the data QA pass)

| column | file | meaning |
|---|---|---|
| `qa_flag` | both | `ok`; `node_glitch` (the unlabelled 08:24:03–08:24:17 node-wide hiccup, 3 rows each in `d-leak-oom` and `d-leak-slow`); `post_fault_tail` (labelled `none`, ≤ 30 s after a fault label ends, signals still trailing; 66 deploy rows). Mask or down-weight non-`ok` rows when scoring. |
| `goodput_rps` | deploy | `rps * (1 - error_rate)`. `rps` counts every response including fast 5xx, so it rises again under deep overload (22.7 RPS/pod at err 0.9); use goodput for throughput. |
| `pods_not_ready` | deploy | pods present but not ready at the step. `replicas_ready` never drops in `d-crashloop` (maxSurge: the old ReplicaSet keeps serving), but this column is 1 during the whole crashloop window and 0 in every `none` row of that run. |

Pod file: `mem_ws` / `mem_frac_limit` exactly 0 (cAdvisor before the container's first stats refresh,
129 rows) are set to NaN.

### `dataset_deploy.csv.gz`: signal columns

All columns are floats unless noted. Rates use a 20 s window for app metrics and a 45 s window for cAdvisor.

| column | unit | definition |
|---|---|---|
| `ts` | unix s, int | sample time; 5 s step |
| `rps` | req/s | `sum(rate(http_requests_total[20s]))` over `/work` |
| `inflight` | requests | `sum(http_inflight_requests)`, a point sample |
| `p50`, `p95`, `p99` | s | `histogram_quantile` over the deployment-wide bucket rate (20 s). Capped at 10 s, the server deadline. NaN when there is no traffic. |
| `error_rate` | ratio 0–1 | 5xx / all. Includes 504s from the 10 s server deadline; 0 when there is no traffic. |
| `cpu_cores_used` | cores | sum over live pods of `max(cAdvisor pod cgroup, app process CPU)` |
| `cpu_util_vs_request` | ratio | `cpu_cores_used` / sum of CPU requests of Running pods (0.25 each). Can reach 2.0 at the limit. This is what the HPA targets (0.7). |
| `throttle_ratio` | ratio 0–1 | CFS throttled periods / periods (45 s) |
| `mem_ws_bytes_total` | bytes | sum of pod working sets |
| `mem_ws_max_pod_frac_of_limit` | ratio 0–1 | max over pods of working set / 256Mi limit |
| `mem_growth_bytes_per_s` | bytes/s | max over pods of `deriv(working_set[60s])`, counting only pods at least 90 s old. NaN when no pod is that old (8.1 % of rows). |
| `replicas_desired` | pods | `kube_deployment_spec_replicas` |
| `replicas_ready` | pods | `kube_deployment_status_replicas_ready` |
| `restarts_total` | count | cumulative container restarts over every pod seen in the run; never decreases |
| `oomkills_total` | count | cumulative restarts whose last terminated reason is OOMKilled. Lags the kill by 5–10 s. |
| `crashloop_pods` | pods | pods waiting in CrashLoopBackOff |
| `hpa_desired` | pods | `kube_horizontalpodautoscaler_status_desired_replicas`; the fixed count under the fixed policy |
| `cpu_cores_proc` | cores | sum of app process CPU only |
| `cpu_fallback_pods` | pods | live pods that have no cAdvisor CPU yet |

### `dataset_pods.csv.gz`: signal columns

| column | unit | definition |
|---|---|---|
| `ts` | unix s | sample time |
| `pod` | str | pod name. The second-to-last dash-separated part is the pod-template hash; `5965489b55` is the good v1 template. |
| `cpu` | cores | `max(cAdvisor, process)` |
| `throttle` | ratio | CFS throttle ratio |
| `mem_ws` | bytes | working set |
| `mem_frac_limit` | ratio | working set / 256Mi |
| `restarts` | count | container restarts |
| `rps` | req/s | requests per second for the pod |
| `p95` | s | the pod's own p95 |
| `errors` | 5xx/s | rate of 5xx responses |
| `ready` | 0/1 | 1 if the pod is ready |
| `cpu_proc` | cores | app process CPU only |
| `oomkills` | count | OOMKills derived for this pod |

## Caveats

1. **Labels end when the fault ends; signals do not.** The rate and quantile windows (20 s for app
   metrics, 45 s for cAdvisor) make signals trail the fault window by up to about 20 s at the start and the end.
   * `d-dep-latency` and `d-bad-deploy` keep p95 > 0.3 s for 15–30 s after `end_ts`, on rows labelled `none`.
   * The bad-deploy tail lasts longer (about 30 s) because the rollback is itself a rollout.
   * `d-error-burst` has `error_rate` decaying over the 20 s after the end.

   If you train classifiers on these labels, allow for a ±25 s tolerance or use event-level metrics.
2. **Cross-lane interference, unlabelled.** At 08:24:10 MST a node-wide hiccup lasted about 15 s.
   It hit both lanes at the same instant. The cause was not identified; it came about 3 min after the
   deep-overload `c-under-fixed2-spike` finished and its namespace was deleted.
   * p95 rose to about 0.6 s for 3 rows in both `d-leak-oom` (t ≈ 636–646 s) and `d-leak-slow` (t ≈ 114–124 s).
   * Those rows are labelled `none` (and `d-leak-oom`'s fall just after its leak ended).
   * They are genuine noise. Treat them as hard negatives, or mask them if they get in the way.
3. **Lane cap.**
   * Every HPA run had `max` clipped from 6 to 4 (`_uncapped_max` in `scenario.json`).
   * In `d-bad-deploy`, the HPA wanted more than 4 pods (CPU at about 140 % of request), so the bad
     deploy was not fully absorbed. The run is therefore an example where scaling helps only partly
     and rollback fixes it.
   * The over-provisioned run uses 4 fixed pods, not 5.
4. **Python heap after `free_on_end`.**
   * The process does not return all freed memory to the OS. After the leak is freed,
     `mem_frac_limit` settles at about 0.2, not the 0.14 baseline. Growth does drop to about 0.
5. **cAdvisor / CPU.** cAdvisor data here is per pod only, sampled every 10–15 s. Throttle has no
   fallback and lags. See the README's known limitations 1–4 for details.
6. **Connection errors.** Client-side connection resets on an OOMKill or pod delete appear only in
   `k6_summary.json` (`runs_index.csv` `k6_failed_rate`), not in `error_rate`. In the capacity and
   under-provisioned runs, `k6_failed_rate` (10–39 %) is mostly 504s plus client timeouts under deep overload.
7. **No scale-in in HPA runs.** The default HPA scale-down stabilisation (5 min) plus runs of 9–13 min
   mean the HPA almost never scales in. Post-peak stretches are therefore over-provisioned. Group C
   gives an explicit fixed over-provisioned example.

## How to regenerate

```bash
cd k8s_aiAgent/loadlab
PY=/Users/vaibav/kubeai-work/.venv/bin/python
monitoring/deploy.sh && nohup monitoring/port-forward.sh >/tmp/prom-pf.log 2>&1 &   # once per cluster
nohup $PY runner/matrix.py > runs/_matrix/nohup.out 2>&1 &   # ~2h15m, 2 lanes; state in runs/_matrix/state.json
$PY runner/matrix.py --list                                  # progress
$PY runner/build_dataset.py                                  # -> data/*.csv.gz, runs_index.csv, faults_index.csv, summary.json
```

`matrix.py` skips scenarios already marked `ok` in `runs/_matrix/state.json`. Delete that file, or
pass `--only <ids>`, to collect again. Prometheus keeps 24 h, so a run can be re-exported with
`$PY runner/promexport.py runs/<run_id>`.

---

## Round 2

**Collection:** 2026-11-08, 09:36–11:36 MST, on the same cluster, image and calibration.

* The source is `../scenarios/matrix_round2.yaml`, run with
  `runner/matrix.py --matrix scenarios/matrix_round2.yaml` (2 lanes). The solo capacity run followed with `--lanes 1`.
* All 20 runs passed matrix validation on the first attempt.
* State is in `../runs/_matrix/state_matrix_round2.json`.

**Why:** round 1 had a single injection of error_rate, cpu_hog, bad_deploy and crashloop, which made leave-run-out evaluation impossible. It also had no HPA scale-in, and an open question about the capacity knee. Every round-2 run uses parameters, seeds, target pods or replica counts that differ from round 1.

### Combined totals

| | round 1 | round 2 | total |
|---|---|---|---|
| runs | 24 | 20 | **44** |
| `dataset_deploy` rows | 2,914 | 2,454 | **5,368** |
| `dataset_pods` rows | 9,175 | 7,433 | **16,608** |
| `faults_index` rows | 11 | 16 | 27 |
| SLO-violating deploy rows | 426 | 352 | 778 |
| `qa_flag` ok / post_fault_tail / node_glitch | 2,842 / 66 / 6 | 2,358 / 96 / 0 | 5,200 / 162 / 6 |

### New column

| column | file | meaning |
|---|---|---|
| `round` | both (and `runs_index.csv`, `faults_index.csv`) | collection round, 1 or 2. It is the **last** column, so every earlier column keeps its position. Round-1 rows are identical to the previous build. |

### Round 2 scenarios

All runs use the 2-lane cap of 4 pods unless marked solo.

* "Rows" is the number of deploy rows.
* "Viol" is the number of SLO-violating rows.

| scenario | policy | traffic | fault | rows | max p95 | viol |
|---|---|---|---|---|---|---|
| a-cap-fixed2-solo-r2 | fixed 2, **alone on the node** | step ramp 16→44 RPS, 2-RPS steps of 40 s (8→22 RPS/pod) | – | 124 | 10 s | 39 |
| b-hpa-diurnal-sd30-s41-r2 | HPA, **scaleDown window 30 s** | diurnal 1.2 pods, period 360 s, amplitude 0.65, seed 41 | – | 159 | 0.11 s | 0 |
| b-hpa-diurnal-sd30-s59-r2 | HPA, window 30 s | diurnal 0.9 pods, period 300 s, amplitude 0.55, seed 59 | – | 159 | 0.09 s | 0 |
| b-hpa-saw-sd30-b12-r2 | HPA, window 30 s | sawtooth 8.6→38.9 RPS, period 240 s | – | 146 | 0.58 s | 3 |
| b-hpa-spike-sd30-b06-r2 | HPA, window 30 s | 4× spike at 150 s for 150 s (10.8→43 RPS) | – | 122 | 6.9 s | 15 |
| c-under-fixed1-flash-r2 | fixed 1 | flash crowd 10.8→27 RPS at 150 s | – | 100 | 10 s | 69 |
| c-over-fixed4-diurnal-r2 | fixed 4 | low diurnal 3.6–10.8 RPS | – | 122 | 0.03 s | 0 |
| d-error10-hpa-diurnal-r2 | HPA, start 3 | diurnal 0.9 pods | error_rate 0.10, all pods, 240–450 s | 122 | 0.05 s | 45 |
| d-error30-fixed3-r2 | fixed 3 | steady 15 RPS | error_rate 0.30, all pods, 180–360 s | 111 | 0.03 s | 39 |
| d-cpu-hog-fixed3-p0-r2 | fixed 3 | steady 13 RPS | cpu_hog 350m, pod 0, 180–390 s | 110 | 0.12 s | 0 |
| d-cpu-hog-fixed3-p2-r2 | fixed 3 | steady 16 RPS | cpu_hog 480m, pod 2, 150–360 s | 111 | 0.25 s | 0 |
| d-bad-deploy-x2-hpa-r2 | HPA, start 3 | steady 16 RPS | bad_deploy 25→50 ms/request, 200–420 s | 123 | 0.20 s | 0 |
| d-bad-deploy-x3-fixed3-r2 | fixed 3 | steady 15 RPS | bad_deploy 25→75 ms/request (v3-bad), 180–400 s | 123 | 1.03 s | 44 |
| d-crashloop-fixed3-r2 | fixed 3 | steady 14 RPS | crashloop rollout, 150–390 s | 110 | 0.03 s | 0 |
| d-dep-latency-25p300-r2 | fixed 3 | steady 15 RPS | dep_latency +300 ms on 25 %, 180–390 s | 111 | 0.34 s | 44 |
| d-dep-latency-75p600-r2 | HPA, start 3 | steady 18 RPS | dep_latency +600 ms on 75 %, 150–390 s | 111 | 1.69 s | 50 |
| d-leak-fast-oom-r2 | fixed 3 | steady 15 RPS | memory_leak 2,048 KB/s, pod 1, 120–450 s, freed at end | 111 | 1.74 s | 4 |
| d-leak-medium-r2 | HPA, start 2 | diurnal 0.8 pods | memory_leak 560 KB/s, pod 0, 120–630 s, retained | 146 | 0.04 s | 0 |
| d-pod-kill-fixed2-r2 | fixed 2 | steady 16 RPS | pod_kill at 150 s (pod 0) and 330 s (pod 1) | 110 | 0.03 s | 0 |
| d-pod-kill-hpa-r2 | HPA, start 3 | steady 16.2 RPS | pod_kill at 140 / 290 / 440 s (pods 2 / 0 / 1) | 123 | 0.03 s | 0 |

### Ground truth per fault class (both rounds)

| fault type | injections r1 + r2 | runs r1 / r2 | deploy rows r1 + r2 |
|---|---|---|---|
| memory_leak | 3 + 2 | 3 / 2 | 282 + 168 = 450 |
| dep_latency | 2 + 2 | 2 / 2 | 98 + 90 = 188 |
| bad_deploy | 1 + 2 | 1 / 2 | 44 + 88 = 132 |
| cpu_hog | 1 + 2 | 1 / 2 | 44 + 86 = 130 |
| error_rate | 1 + 2 | 1 / 2 | 44 + 78 = 122 |
| crashloop | 1 + 1 | 1 / 1 | 44 + 48 = 92 |
| pod_kill | 2 + 5 | 1 / 2 | 12 + 30 = 42 |
| none | – | 24 / 20 | 2,346 + 1,866 = 4,212 |

Load-state counts for `none` rows (a heuristic proxy for scale_up, scale_down and healthy) are in `QA_REPORT.md` → Round 2.

### Round 2 caveats

1. **The capacity knee depends on replicas, confirmed solo.** With the node to itself, 2 replicas keep the SLO at 17 RPS/pod (p95 284 ms) and break it at 18 RPS/pod. One replica breaks at 20. Learn capacity per replica count.
2. **Scale-in exists.** The 4 HPA runs with `scale_down_window_s: 30` scale from 4 to 2 replicas (11 step-downs). Round-1 HPA runs use the default 300 s window and never scale in. `scale_down_window_s` is recorded in each run's `scenario.json` scaling spec.
3. **pod_kill under HPA (`d-pod-kill-hpa-r2`) is invisible at deployment level.** The replacement pod was ready within one 5 s step, so `replicas_ready` never dips. It is visible only as pod churn in `dataset_pods`. The fixed-2 run shows the dip (2 → 1).
4. **The fast leak under-delivers** (measured 1.49 MB/s against 2 MB/s injected), and it raises p95 to about 1.7 s for about 20 s just before each OOMKill (memory pressure). The medium leak matches its injected rate (554 vs 560 KB/s), and p95 stays flat.
5. **The HPA scales out during dep_latency** (`d-dep-latency-75p600-r2`, 3 → 4 at 203 s) without any p95 benefit. This is a labelled example of scaling that does not help.
6. **NaN in `mem_growth_bytes_per_s`** is 22–24 % in the bad_deploy runs (two rollouts, so pods are younger than 90 s). Over all rows it is 9.2 %.
7. The lane cap still applies: HPA `max` 6 was clipped to 4.

### Query version note (2026-11-08, after the live A/B review)

The committed dataset was exported with PromQL **v1** (each run's `queries.json`). `runner/promexport.py` now
exports **v2**, which `lib/promDeploy.js` (the live agent) also uses:

* `mem_growth_bytes_per_s` counts only pods that are still live scrape targets (`and on(pod) group by (pod)(up{...})`).
  v1 kept a deleted pod's last 60 s of `deriv` in the max for up to ~60 s after the deletion. In the live A/B that
  kept a just-restarted leaking pod's 1 MB/s slope alive for ~65 s and latched the agent in `memory_leak`.
* Pod regex `loadlab-[a-z0-9]+-[a-z0-9]{5}` instead of `loadlab-.*`; kube-state-metrics pod selectors (restarts,
  OOM reason, CrashLoopBackOff, requests, limits, phase) restricted to it; `hpa_desired` restricted to the
  `loadlab` HPA. In these one-Deployment namespaces this selects exactly the same series (checked live on the A/B runs).

Impact on this dataset, measured by re-querying all 44 runs (`runner/check_query_v2.py`, Prometheus still had them):
v2 changes **70 of 4,861** `mem_growth_bytes_per_s` points (1.4 %, all at scale-in, pod-kill or rollout pod
deletions), **none by more than 100 KB/s**, because no leaking pod was ever deleted in these runs. The dataset was
therefore not re-exported; a re-export with the current exporter would differ only in those points.
