# Live A/B on kubeai-lab: KubeAI agent (policy v2) vs the stock HPA

**Date:** 2026-11-08, 13:57 to 15:22 local time. **Cluster:** minikube profile `kubeai-lab` (6 CPU / 6 GB).
**Images:** rebuilt in minikube right before the run. The ML service served bundle `scaling-20261108194802` with
`policy_mode=hpa_gated` (policy v2, confirmed via `/v1/model`), and the agent ran the current `lib/scaler.js`.
**Reproduce:** `cd loadlab && $PY ab/ab.py run` (all slots, then `analyze`), or `$PY ab/ab.py analyze` to rebuild the
tables and plots from `ab/runs/`.

> **Revision note (post-review, same data).** A review of this report found counting and wording errors; they are
> corrected below and the tables were regenerated from the unchanged raw run data (`ab.py analyze`). Changes:
> scale changes are now counted from the value just before a window (the old count dropped a change 3–4 s after the
> window opened); the HPA arms' leak recovery is timed from the OOMKill itself, not from the next 5 s pod sample;
> client-side (k6) latency is now reported; the spike misread happened in 2 runs, not 1; the HPA-300s caveat was
> wrong and is replaced. Several agent defects the review found are fixed in code after this run (see
> [Fixed after this run](#fixed-after-this-run)); the A/B itself was **not** re-run with the fixed agent.

## Verdict (short)

**On pure load the agent ties the HPA it copies; it does not beat it.** On the faults it behaves differently from
the HPA, but with **no measured SLO benefit** (n = 1 run per arm):

* It **fixed the memory leak before it OOMKilled**. It restarted the leaking pod 44.7 s after the leak started, when
  the last scraped working set was 23 % of the memory limit (cAdvisor refreshes every ~15 s; the leaking
  pod's own next sample, at +45 s, read 30 %, as did the twin pod in the HPA-30s arm from +42 s, so the true value at
  the restart was about 0.3). Both HPA arms left the pod
  to OOMKill at +221–222 s, at 97–99 % of the limit.
* During the slow-dependency fault it **added no pods after it detected the fault** (first `dependency_latency`
  decision at +45 s). It did add one pod 4 s after the fault started (2→3, load-driven, before detection). The HPA-30s
  arm went 2→3 at +3 s and 3→4 at +113 s; the HPA-300s arm went 2→3 at +10 s and added nothing after that. So
  "no fault-time scale-up after detection" separates the agent only from the HPA-30s arm's one late pod.

The SLO impact of both faults was the **same in all arms**. The dependency fault cost 3.0 violation-minutes everywhere,
and the leak caused no violations anywhere. That is because:

* scaling can't fix a slow dependency;
* the leak's only user-visible damage in the HPA arms was the OOMKill itself, which cost 2–4 failed requests out of
  13,368.

**Client-side latency disagrees with the server-side SLO metric.** k6's own p95 (`http_req_duration`) is 1.07–2.76 s
in 8 of 10 runs, far above the 300 ms SLO, while the Prometheus-grid metric counts only 1–1.5 violation-minutes on
load. The server-side histogram does not see queueing that clients see (mostly during the spike). On client p95 the
agent beat HPA-30s in all 3 paired runs (1068 vs 2056, 1281 vs 2234, 2538 vs 2761 ms; n = 1 each, not significant),
and the two HPA-300s runs of slot 5 were by far the best of the study (280 and 327 ms).

**Where the agent loses:**

1. **Its fault gate delayed a needed load scale-up by 60 s.** For 75 s after the leak remediation the agent kept
   classifying `memory_leak`, and scale-up is frozen in that state. It scaled 2→3 at t≈245 s; the HPA-30s arm did it
   at t≈185 s. It cost no SLO here only because the 2 pods still had headroom. **Cause, found in review:** a
   measurement artefact, not real growth. The `mem_growth_bytes_per_s` query kept the deleted pod's last 60 s
   (`deriv` over a range selector) for ~65 s after the delete, at ~1.03 MB/s. Fixed after the run (query v2, live
   scrape targets only). An offline replay of this run's decisions against the stored Prometheus data with the v2
   query (`replay_v2.mjs`, output `replay_v2_c-leak.jsonl`) reproduces every logged state with the old query and,
   with the new one, leaves `memory_leak` 2 decisions after the restart instead of 4 and asks for 2→3 30 s earlier.
   That is a counterfactual on recorded inputs, not a live re-run.
2. **During the dependency fault, the HPA-300s arm used fewer pod-minutes inside the fault window (8.9) than the
   agent (9.0).** The agent saved pods only compared with the HPA-30s arm (10.2). "The agent saves pods during
   faults" is therefore not a robust claim from this data.
3. **The differences on load are noise.**
   * Agent vs HPA-30s: violation-minutes 1.00 vs 1.25, 3.75 vs 4.00 and 1.50 vs 1.25.
   * Pod-minutes are within about ±6 %.
   * There is one run per arm per trial. Each run has 48 SLO grid points, and one grid point is 0.25 min.

4. **It misread the spike overload in 2 of 3 runs.** In a-load (t=600 s) and b-dep (t=600 s), 15 s after it had
   scaled 2→4 for the spike, it labelled the overload `dependency_latency` (p = 0.91) and sent a false page. Both
   times it cost nothing because the arm was already at max replicas, but that state blocks scale-up, so below the
   cap the same misread would freeze scaling during a real overload. This is a systematic failure mode of the
   classifier (high p95 with CPU not yet saturated on new pods looks like a slow dependency), not a one-off.

The second-seed repeat of trial (a) **failed twice** because the API server stalled. Only its HPA-300s arm exists, so
it has no paired comparison. It is not used in any verdict.

## Design

| | |
|---|---|
| Trace (all arms) | k6 `diurnal_spike` (new shape in `load/shapes.js`), open loop, 720 s, base 14.4 RPS (0.8 pods of SLO capacity), period 600 s, amplitude 0.5, noise ±8 %, seed 42. A ×6 flash spike runs from t=560 s (ramp 10 s, hold 90 s): 7 → 22 RPS diurnal, spike 45–52 RPS (about 2.9 pods of SLO capacity). The trace is deterministic for a given seed. |
| App | loadlab:v1, 25 ms CPU per request, 250m/500m CPU, 128/256 Mi memory. Identical in all arms. |
| Bounds | min 1, max 4 (the lane cap). Every arm starts at 1 replica. |
| `agent` arm | No HPA. lab.py `fixed` policy at 1 replica, plus the in-cluster `kubeai-agent` pointed at the run namespace. Settings: `ACT_MODE=act`, `ALLOW_EXEC=true`, `DEPLOYMENTS=loadlab`, `SCALE_MIN/MAX=1/4` (plus the `kubeai.io/{min,max}-replicas` annotations), a decision every 15 s, shipped cooldowns (up 15 s, down 60 s, remediation 300 s). `OOM_LOOP=false`, so only the deployment loop under test acts. A RoleBinding is created in the run namespace only. |
| `hpa30` arm | autoscaling/v2 HPA, CPU 70 % of request, `behavior.scaleDown.stabilizationWindowSeconds: 30`. This is the twin of policy v2, which is the same formula with a 30 s window. |
| `hpa300` arm | Same HPA with the default behaviour (300 s scale-down window). |
| Trials | (a) pure load; (b) the same load plus `dep_latency` on all pods, 400 ms on 50 % of requests, t=180–360 s; (c) the same load plus `memory_leak` 1 MB/s on one pod from t=120 s. |
| Pairing | Slots 1–3 ran the `agent` and `hpa30` arms of one trial **at the same time** in two namespaces, so both saw the same node conditions. Their load starts differed by 26–37 s (36.5, 26.5, 27.6 s) because the HPA arm waits for HPA metrics. The `hpa300` arms ran in slots 4–5, paired with each other. |
| SLO | p95 > 300 ms or 5xx > 1 %, read on a 15 s grid (the nearest 5 s Prometheus row) over [t0, t0+720 s]. |
| Cost | Pod-minutes = ∫ `kube_deployment_spec_replicas` dt over the same window. Ready-pod-minutes are also reported. |

**Leak semantics: the one change to the lab.** `lab.py` gained a fault flag `reapply_on_restart: false`. In trial (c)
it means that *any* restart clears the leak in **both** arms: the agent's pod delete and the HPA arm's OOMKill
restart. Without the flag, the lab re-applies a leak after an OOM container restart but not after a pod delete. That
would have favoured the agent unfairly. This models a state-triggered leak. For a code leak that recurs in every new
process, a restart only buys time, and that case was **not** tested (see Limitations).

## Results

### Totals per run

| trial | arm | viol-min | viol events | viol-min in spike | pod-min | ready pod-min | scale ups / downs | max replicas | restarts / OOMKills | k6 failed reqs (of 13,368) | k6 p90 / p95 / max (ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| a-load | agent | **1.00** | 3 | **0.50** | 35.8 | 35.4 | 4 / 2 | 4 | 0 / 0 | 4 | 115 / **1068** / 3534 |
| a-load | hpa30 | 1.25 | 3 | 0.75 | **33.7** | 33.3 | 4 / 3 | 4 | 0 / 0 | **0** | 198 / 2056 / 4528 |
| a-load | hpa300 | 1.50 | 1 | 1.50 | 34.0 | 33.8 | 3 / 0 | 4 | 0 / 0 | **0** | 556 / 2371 / 5248 |
| b-dep | agent | **3.75** | 2 | **0.50** | **33.5** | 33.2 | 3 / 1 | 4 | 0 / 0 | **0** | 435 / **1281** / 3563 |
| b-dep | hpa30 | 4.00 | 2 | 0.75 | 34.3 | 34.0 | 4 / 3 | 4 | 0 / 0 | 3 | 451 / 2234 / 5461 |
| b-dep | hpa300 | 4.75 | 3 | 1.50 | 34.2 | 34.2 | 3 / 0 | 4 | 0 / 0 | **0** | 661 / 2443 / 6018 |
| c-leak | agent | 1.50 | 3 | 1.25 | **32.2** | 31.8 | 3 / 1 | 4 | **0 / 0** | **0** | 449 / 2538 / 5103 |
| c-leak | hpa30 | **1.25** | 2 | **1.00** | 33.8 | 33.2 | 4 / 3 | 4 | 1 / 1 | 2 | 354 / 2761 / 5544 |
| c-leak | hpa300 | **1.25** | 3 | **1.00** | 38.0 | 37.7 | 3 / 0 | 4 | 1 / 1 | 4 | 107 / **280** / 3483 |
| a2-load-s7 | hpa300 (unpaired) | 1.25 | 3 | 1.00 | 37.8 | 37.6 | 3 / 0 | 4 | 0 / 0 | 0 | 111 / 327 / 2663 |

Notes:
* Bold = best value of the three arms in that trial (per column, for viol-min, spike viol-min, pod-min, failed
  requests, k6 p95, OOMKills). One run per arm: none of these differences is statistically meaningful.
* Scale up/down counts are changes of the Deployment's `spec.replicas` in the load window, counted from the value just
  before the window.
* The agent's last scale-down (4→2 or 4→3) happens at t≈720 s, at the window edge, so it is not counted. In a-load it
  is the cause of the agent's 4 failed requests: k6 logged 4 `connection refused` at t=720 s, the second of the
  agent's `scale:4->2` (connections to the 2 removed pods).
* The k6 columns are client-side `http_req_duration` over the whole run, from `k6_summary.json`.

### Behaviour during the faults

| trial | arm | fault viol-min | replicas at fault start → max in fault | scale changes in fault | pod-min in fault window | what the arm did |
|---|---|---|---|---|---|---|
| b-dep (180 s) | agent | 3.0 | 2 → 3 | +1 (2→3 at +4 s, load-driven, before detection) | 9.0 | Classified `dependency_latency` on 10 decisions in the window (first at +45 s, p=0.91), 1 just after it (t=375 s) and 1 false one at the spike (t=600 s), 12 in all. Every in-window one said "do_not_scale: scaling will not fix it" and paged. **No scale-up after detection.** |
| b-dep | hpa30 | 3.0 | 2 → **4** | +2 (2→3 at +3 s, 3→4 at +113 s) | 10.2 | CPU formula. The second pod came on the diurnal peak inside the fault; it did not change p95, which stayed at about 500 ms in all arms. |
| b-dep | hpa300 | 3.0 | 2 → 3 | +1 (2→3 at +10 s) | 8.9 | Nothing after +10 s, so it had fewer pod-minutes in the window than the agent. |
| c-leak (first 300 s) | agent | 0.0 | 2 → 3 | +1 (t≈245 s) | 13.0 | Detected `memory_leak` **44.7 s** after the leak started and **deleted the leaking pod in the same decision**; last scraped working set before the decision 23 % of the limit (about 30 % at the restart itself, see the verdict). It then paged for 4 more cycles (the measurement artefact above). Scaling stayed frozen until the state returned to healthy. |
| c-leak | hpa30 | 0.0 | 2 → 4 | +2 / −1 | 14.5 | No reaction to memory. **OOMKill at +222.4 s** (peak scraped 98.9 % of limit), 1 restart, 2 failed requests. |
| c-leak | hpa300 | 0.0 | 2 → 4 | +2 | 14.8 | **OOMKill at +221.1 s** (peak scraped 97.4 % of limit), 1 restart, 4 failed requests. |

**MTTR for the leak** (leak start → leaked memory released):

| arm | time | how |
|---|---|---|
| agent | **44.7 s** | `kubectl delete pod` (the decision's timestamp) |
| hpa30 | **222.4 s** | OOMKill, timed by the first k6 `connection refused` (the pod no longer existed at the end, so its container status was not captured) |
| hpa300 | **221.1 s** | OOMKill, container `lastState.terminated.finishedAt` |

The first version of this report gave 237.4 s and 229.1 s for the HPA arms: those were the first 5 s pod samples with
the memory halved, which lag the kill by 8–15 s. The agent cut the time to recovery by about 5.0× and avoided the OOMKill. With this leak, though, the HPA arms'
"recovery" was the crash itself, and its user-visible cost was small: 2–4 connection errors and no SLO grid
violation.

### Agent decision log (the 3 paired agent runs)

* **Volume and errors:** 50 decisions per run, 0 `decision_error` records.
* **Latency:** median ML `/v1/decide` latency 50–56 ms; median full cycle (Prometheus fetch, ML call, kubectl) 168–179 ms.
* **States:**
  * a-load: healthy 36, scale_down 11, scale_up 2, dependency_latency 1.
    * The `dependency_latency` decision is a **misclassification** at t≈600 s, inside the spike overload, 15 s after the agent had already scaled 2→4.
    * It produced a false page. It did not block a scale, because the arm was already at the max of 4, but in general a misread like this would freeze scale-up for that cycle.
  * b-dep: healthy 31, dependency_latency 12, scale_down 6, scale_up 1.
    * The 12 `dependency_latency` decisions are 10 inside the fault window (t=225–360 s), 1 just after it (t=375 s,
      the 20 s rate window still covers the fault) and 1 **misclassification** at t=600 s in the spike, the same
      false page as in a-load.
  * c-leak: 5 memory_leak decisions.
* **Executed actions:**

| trial | executed actions |
|---|---|
| a-load | `scale:1->2`, `scale:2->3`, `scale:3->4`, `scale:4->3`, `scale:3->2`, `scale:2->4`, `scale:4->2` |
| b-dep | `scale:1->2`, `scale:2->3`, `scale:3->2`, `scale:2->4`, `scale:4->2` |
| c-leak | `scale:1->2`, `restart_pod:<leaking pod>`, `scale:2->3`, `scale:3->2`, `scale:2->4`, `scale:4->3` |

* No rollback was executed, and no action was refused by a guard rail.
* The full logs are in `runs/<trial>/agent/decisions.jsonl`, with the agent's stdout in `agent.log`.

### Plots

Each plot in `plots/` has three panels: replicas (solid = desired, dotted = ready), p95 on a log scale with the SLO
line, and served RPS. Fault windows are shaded and the agent's executed actions are marked with arrows.

* `plots/a-load.png`
* `plots/b-dep.png`
* `plots/c-leak.png`
* `plots/a2-load-s7.png` (HPA-300s arm only)

## How to read the load numbers (honest caveats)

* **The flash spike dominates.** Agent and HPA-30s both entered the spike at 2 replicas, because both had scaled in
  during the trough. Both went straight to 4, and both violated for 0.5–1.25 min. The agent's scale-up landed 0–5 s
  earlier, which is within the noise of a 15 s decision loop and a 15 s HPA sync.
* **Short p95 blips outside the fault and spike windows** (≈15 s each, p95 near 500 ms):
  * a-load: agent 2 (t=135, 405 s), HPA-30s 2 (t=105, 375 s), HPA-300s none;
  * b-dep: none in any arm, apart from the fault's own tail at t=375 s (15 s after the fault ended) in all three;
  * c-leak: 1 in each arm, at t≈495–510 s, just before the spike ramp;
  * a2-load-s7 (HPA-300s): 1, at t=510 s.

  They are not consistently tied to the arm's own scale events. Each is worth 0.25 violation-minutes, which is the
  same size as every agent-vs-HPA difference above.
* **The HPA-300s arms are unpaired with the agent and ran in other time slots** (slots 4–5, against each other).
  * Entry into the spike: a-load and b-dep (slot 4) at 3 pods; c-leak and a2-load-s7 (slot 5) at 4 pods (they scaled
    3→4 at t=369 s and t=365 s).
  * Spike violation-minutes: 1.50 and 1.50 in slot 4, 1.00 and 1.00 in slot 5. In c-leak that ties HPA-30s (1.00)
    and beats the agent (1.25).
  * Slot 5 had the best client latency of the whole study (k6 p95 280 and 327 ms, against 1068–2761 ms in every
    other run).
  * The first version of this report called the HPA-300s spike numbers low-confidence because "the node ran out of
    CPU with 2 lanes × 4 pods". That was an untested hypothesis: slots 1–3 ran agent and HPA-30s at 4 pods each
    through overlapping spikes too, and slot 5, with 2 × 4 pods, was the best slot. It is withdrawn. No mechanism for
    the slot-to-slot differences was verified.
* **The live result matches the offline evaluation.** The simulator also found HPA_GATED tied with HPA 70 % / 30 s on
  load (held-out 0.67 vs 0.67 violation-minutes at about 1 % pod-minute difference), and that its value lies in the
  fault gate and remediation. This A/B reproduces that live. It found no load advantage, and none should be claimed.
  The simulator's "wins on faults" is behavioural (fewer fault-time scale-ups); live, that behaviour showed up (no
  scale-up after detection, leak restart before OOM) with **zero measured SLO benefit**.

## Limitations

1. **Small sample.** One paired run per trial, three trials. The second seed failed (see below). Nothing here is
   statistically separable except the leak behaviour, which is categorical: restart before OOM vs OOMKill.
2. **The arms are not perfectly time-aligned.** The HPA arm's load started 26–37 s after the agent arm's in each slot,
   because it waits for HPA metrics. So the two spikes overlapped but did not coincide.
3. **Leak model.** The leak is state-triggered: any restart clears it. A code leak would come back in the new pod.
   The agent's 300 s remediation cooldown would then allow one restart per 5 min, and at 1 MB/s the pod reaches the
   limit in about 3.5 min, so OOMKills would return. That case is untested.
4. **The dependency fault cannot show "HPA wastes pods" strongly.** The app's slow dependency is an async sleep, so it
   uses no CPU and the HPA's CPU signal barely moves. The one extra HPA pod came from the diurnal peak inside the fault
   window, not from the fault.
5. **Cluster stability.**
   * The second-seed slot ran twice and failed both times. At 15:15 and again at 15:21 the API server stopped
     answering (`TLS handshake timeout`). At the same time the kube-apiserver, kube-state-metrics and metrics-server
     probes failed.
   * The first failure came at the spike, the second at the diurnal peak with about 4 app pods on the node. Node or
     host saturation is the likely cause, but this is not verified.
   * The failed runs are kept in `runs/_failed/` and are not scored.
   * Slots 1–5 completed with no runner errors and no agent `decision_error`. Their timing could still have been
     affected by load on the node, which nobody measured.
6. **`OOM_LOOP=false`.** The older per-pod OOM loop was disabled so that only the deployment loop acted. (The review
   found that loop unsafe next to the deployment loop at the time; fixed after the run, see below.)
7. **Server-side vs client-side latency.** The SLO metric is the server's own histogram on a 15 s grid. Client p95
   is 4–9× higher in most runs and ranks the arms differently (see the totals table). A client-side SLO would have
   been the better primary metric.

## Fixed after this run

The review that corrected this report also found agent defects. They are fixed in code and covered by tests
(`test/scaler.smoke.mjs`, `test/agent.smoke.mjs`, `ml_service/tests/test_scaling_safety.py`); none of them was
exercised by these runs except the first, and **the A/B was not re-run** with the fixed agent.

| defect | effect in this A/B | fix |
|---|---|---|
| `mem_growth_bytes_per_s` kept a deleted pod's slope ~60 s | caused the 75 s `memory_leak` latch and the 60 s-late scale-up in c-leak | query v2: live scrape targets only (`lib/promDeploy.js`, `runner/promexport.py`); offline replay above |
| `rollout undo` without `--to-revision`; the agent's own undo/restart counted as rollout evidence | none (no rollback ran) | undo to the newest revision the agent saw healthy, at most one rollback per bad revision, own rollouts excluded (`agent_rollout_ts`) |
| per-pod OOM loop ignored `ACT_MODE`, could `scale` via the LLM, no shared cooldown | none (`OOM_LOOP=false` here) | gated on `ACT_MODE`; `scale` removed from its allowlist; shared per-Deployment cooldown; off by default when the deployment loop runs |
| pod delete could fall back to Prometheus pod names (sibling Deployments match `web-.*`); `kubectl delete` ignores PDBs | none (one Deployment per namespace; no PDB) | owner-verified pods only; Eviction API with PDB 429 → surge rollout restart |
| kube-state-metrics and `hpa_desired` queries were namespace-wide; app metrics namespace-wide | none (one Deployment per namespace; a live re-query showed the scoped v2 queries return identical series for these runs, except `mem_growth`) | pod-regex scoping, HPA-name scoping, per-Deployment `APP_METRIC_MATCHER` template, refusal otherwise |

## Files

* `ab.py`: the runner. It holds the plan, a per-arm subclass of `lab.Run` that points the agent at the namespace
  and collects the decision log, and the analysis and plots.
* `scenarios/*.yaml`: the exact scenario per trial and arm (generated).
* `runs/<trial>/<arm>/`: the standard loadlab outputs (`deploy.csv`, `pods.csv`, `faults.jsonl`, `k6_summary.json`,
  `events.json`, …) plus, for the agent arm, `decisions.jsonl` and `agent.log`.
* `results.csv` / `results.json`: one row per run with every metric above.
* `plots/*.png`.
* `run_slots1-6.out`, `run_slot6_retry.out`, `ab.log`: the console logs.

**Changes outside `ab/`:**
* `load/shapes.js`: a new `diurnal_spike` shape (additive).
* `runner/lab.py`: the optional per-fault `reapply_on_restart` flag (default `true`, so existing scenarios behave the
  same).

* `replay_v2.mjs`, `replay_v2_c-leak.jsonl`: the post-review offline replay of the c-leak agent decisions (v1 vs v2
  query).

**Teardown:** all `lab-ab-*` namespaces were deleted. The `kubeai` and `monitoring` namespaces were kept. The agent
ConfigMap was restored to `ACT_MODE=observe`, `ALLOW_EXEC=false`, `NAMESPACE=default` (and, after the review,
`OOM_LOOP` empty = off while the deployment loop runs), and the agent was restarted.
