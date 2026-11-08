# KubeAI agent: predicting pod OOM kills, deciding per Deployment

The agent runs two loops.

1. **Per pod (OOM-ahead).** It watches the memory of Kubernetes pods. For each pod it asks a trained model how likely an OOM kill is within the next horizon. It alerts only on repeated anomalies (k-of-n). It can also ask Gemini for a remediation (raise a memory limit or a rollout restart; never a replica change), which it checks against an allowlist and executes according to the same `ACT_MODE` as the deployment loop (`observe`: no kubectl). It is off by default when the deployment loop runs; if both run they share one per-Deployment remediation cooldown.
2. **Per Deployment (SRE decision).** Every `DECIDE_INTERVAL_S` it pulls a telemetry window for each target Deployment from Prometheus (CPU/throttle, memory, RPS/inflight, p95/p99/error rate, replicas, per-pod rows; the exact PromQL that produced the training data), asks the ML service's `POST /v1/decide` for the state (healthy, scale_up, scale_down, memory_leak, dependency_latency, error_fault, noisy_pod, bad_deploy, crashloop, pod_restart), the desired replica count and the remediation, logs the decision as one JSON line, and acts according to `ACT_MODE` (`observe` by default). See [The deployment loop](#the-deployment-loop-per-deployment-decisions).

```
                 +--------------------+   sum by (pod, container)(container_memory_working_set_bytes{...})
                 |    Prometheus      |<-----------------------------------------+
                 +--------------------+                                          |
                                                                                 |
 +----------------+   pods?labelSelector=  +-------------------------------------+------+
 | Kubernetes API |<-----------------------|  agent (Node, index.js + lib/)             |
 | (proxy or SA)  |   limits, restarts,    |  - one sequential loop, POLL_MS            |
 +----------------+   CrashLoopBackOff     |  - per-container ring buffer {ts, bytes}   |
        ^                                  |  - k-of-n hysteresis + cooldown            |
        |  kubectl (allowlisted argv,      |  - Express :3001 /healthz /status /pods/x  |
        |  --dry-run=server by default)    +-----+-------------------------+-----------+
        +----------------------------------------+                         |
                                                   POST /v1/predict/batch  |   strict JSON
                                         +-------------------------+       |   recommendation
                                         | ML service (FastAPI)    |<------+   +-----------+
                                         | ml_service/  :8000      |           |  Gemini   |
                                         | p_oom, time_to_limit,   |           | (optional)|
                                         | forecast, reasons       |           +-----------+
                                         +-------------------------+
```

## Components

| Path | What it is |
|---|---|
| `index.js` | Entry point. Loads config, starts the Express API and the monitoring loop. |
| `lib/config.js` | Reads every setting from the environment (see `.env.example`). |
| `lib/kube.js` | Pod discovery by label selector. Reads each container's own memory limit (real quantity parser; never summed across containers), restart count and seconds since its last termination, CrashLoopBackOff on any container, and the owning Deployment. Sends a bearer token from the service-account file when one is present. |
| `lib/prom.js` | One PromQL query per tick for all pods, per container. Missing data stays missing. |
| `lib/ml.js` | Client for `POST /v1/predict/batch`: chunks of at most `ML_BATCH_MAX` (500, the server's cap); a chunk rejected with 422 is retried item by item via `POST /v1/predict`, so one bad item costs only its own prediction. A limit that is not > 0 is sent as `null`. |
| `lib/alert.js` | k-of-n hysteresis and cooldown on the service's boolean `is_anomaly`. |
| `lib/remediation.js` | Gemini via `@langchain/google-genai` (client created once), the strict JSON schema, the allowlist validators (LLM and engine actions), the kubectl argv builder (incl. the Eviction API call and `rollout undo --to-revision`), `execFile` (no shell), and the per-Deployment remediation ledger both loops share. |
| `lib/quantity.js` | Kubernetes quantity parser (`Ki Mi Gi Ti`, `K M G`, plain bytes, e-notation). |
| `lib/promDeploy.js` | Deployment telemetry window: the same PromQL as `loadlab/runner/promexport.py` (query v2; `queryVersion: "v1"` reproduces the training export byte for byte), `query_range`, 5 s step, `HISTORY_S`, and its post-processing (cumulative restarts/OOMKills, fills, trims), in the `/v1/decide` schema. |
| `lib/scaler.js` | The per-Deployment loop: window → `POST /v1/decide` → decision record (stdout + `DECISION_LOG`) → `ACT_MODE` (observe / dryrun / act) with the HPA guard, replica bounds, cooldowns and the allowlist. |
| `ml_service/` | Training pipeline and FastAPI model server. See `ml_service/README.md`. |
| `k8s/` | Namespace, ML Deployment and Service, agent Deployment, Service, ConfigMap, ServiceAccount, ClusterRole and a RoleBinding per watched namespace. The LLM key Secret is created out of band (see below), never applied from this directory. |
| `legacy/` | The old stdin LSTM+LightGBM hybrid, and why it was replaced. |
| `test/agent.smoke.mjs` | Runs the real agent against a fake Prometheus, a fake kube API and a fake ML service (OOM loop). |
| `test/scaler.smoke.mjs` | The deployment loop against a fake Prometheus (`query_range`), a fake `/v1/decide`, a fake kube API and a fake kubectl; plus `index.js` as a child process. |

## The ML service contract

The agent depends only on this contract. The full schema is in `ml_service/`.

- `POST /v1/predict/batch` with body `{"items":[{"pod","namespace","samples":[{"ts","working_set_bytes"}...],"limit_bytes","restart_count","seconds_since_restart"}]}` returns `{"results":[{"pod","namespace","model_version","p_oom","is_anomaly","threshold","horizon_s","usage_fraction","trend_bytes_per_s","time_to_limit_s","forecast","reasons"}]}`.
- `POST /v1/decide` with body `{"deployment","namespace","series":[{ts, rps, p95, error_rate, cpu_cores_used, replicas_ready, ...}],"pods":[{ts, pod, cpu, mem_ws, ...}],"current_replicas","min_replicas","max_replicas","now_ts","revision"}` returns `{"state","state_probs","confidence","desired_replicas","actions":[{"type","target","params","reason"}],"reasons","telemetry_ok","forecast","slo_risk",...}` (full schema: `ml_service/README.md`).
- `GET /healthz`, `GET /readyz` (503 until **both** the OOM and the scaling bundle have loaded), `GET /v1/model` (both bundles: the OOM manifest at the top level, `bundles.{oom,scaling}`), and `GET /metrics` (Prometheus format, incl. decide latency and state counts).

The agent sends **one item per container**, with that container's own samples and its own `limit_bytes`, because the kernel OOM-kills each container against its own limit. (Comparing a pod-level working set with the sum of the container limits hides an app container at its limit behind a sidecar with a large limit, and counts an unlimited sidecar's memory as usage.) The pod's prediction is its most at-risk container (`latest_prediction.container`).

Model caveats, documented in more detail in `ml_service/`: the training data is one pod and one 10.5 h run, and the true memory limit for that run is unknown. Because OOM kills happen at the limit, the observed maximum (887.345 in the collector's `Ki/1000` units) is used as a stand-in for the limit. Cross-dataset results on `nirmal_dataset_1.csv` (900Mi limit) are reported alongside the in-dataset ones.

## The OOM loop (one tick)

1. List the pods in `NAMESPACE` that match `POD_SELECTOR`. Forget pods that are gone. When a container's restart count goes up, clear that container's window, because memory resets after a restart and the old samples would fake a trend. (The model is trained on histories cut the same way.)
2. Run `sum by (pod, container) (container_memory_working_set_bytes{namespace="ns",pod=~"a|b",container!="",container!="POD"})` and append to each container's ring buffer (`WINDOW` samples). A container with no result gets no sample.
3. Send every container that has at least 2 samples to the ML service, one item each, in batches of at most `ML_BATCH_MAX`. If the service is down, the agent logs the error and retries on the next tick. A 422 batch is retried item by item.
4. Take each pod's most at-risk container as the pod's prediction and record its `is_anomaly` in the pod's last-N history. An alert fires when at least `ALERT_K` of the last `ALERT_N` predictions are anomalous and the pod is not in cooldown (`COOLDOWN_MS`).
5. When an alert fires:
   - If the pod is already in CrashLoopBackOff, the agent logs `CRITICAL` (notify the administrator) and makes no LLM call.
   - If another remediation of the same Deployment (by this loop for another pod, or by the deployment loop) happened within `REMEDIATION_COOLDOWN_S`, the alert is recorded as suppressed and no LLM call is made.
   - If `GOOGLE_API_KEY` is not set, the recommendation is recorded as `none` with the reason.
   - Otherwise Gemini returns `{"action","target","params","reason"}`. The validator accepts only:
     - `patch_memory_limit` (a container in the pod; the new limit must be above the current one, at least 1.2x the container's latest working set, at least 64Mi, and at most `MAX_MEMORY_LIMIT`; rejected when the container has neither a current limit nor an observed working set)
     - `rollout_restart` (dry run: `kubectl rollout restart` has no `--dry-run` flag, so the agent runs the equivalent `kubectl patch deployment/x -p '{"spec":{"template":{"metadata":{"annotations":{"kubectl.kubernetes.io/restartedAt":"<now>"}}}}}' --dry-run=server`; with `ALLOW_EXEC=true` it runs the real `rollout restart`)
     - `none`

     `scale` was removed from this allowlist: an LLM-chosen replica count bypassed the deployment loop's HPA guard, replica bounds, `--current-replicas` precondition and cooldowns (and could scale down during an incident). The target must be the pod's own Deployment. The agent then builds the argv itself, for example `kubectl -n default set resources deployment/x -c app --limits=memory=1024Mi --dry-run=server`. The text the model returns is never executed. Execution follows `ACT_MODE` exactly like the deployment loop: `observe` never calls kubectl (the step is logged as planned), `dryrun` runs `--dry-run=server`, and only `act` with `ALLOW_EXEC=true` changes anything.

The next tick is scheduled only after the current one finishes, so ticks never overlap.

**Steady pods.** The model's training data has almost no pods that sit flat at high usage, so its `p_oom` for a pod that runs steadily at about 69% or more of its limit is high (0.72-0.93). The service therefore sets `is_anomaly` only when `p_oom >= threshold` **and** memory is on course to reach the limit within 300 s (`time_to_limit_s <= 300`), so steady pods, even at 90% of their limit, do not alert. The flip side: a pod that sits flat just below its limit and is then OOM-killed by a sudden jump gets no early warning. Do not build alerts on raw `p_oom` without a trend condition. See `ml_service/MODEL_CARD.md`.

## The deployment loop (per-Deployment decisions)

Enabled when `DEPLOYMENTS` (comma list) or `DEPLOY_SELECTOR` (label selector on Deployments) is set, in `NAMESPACE`. Every `DECIDE_INTERVAL_S` (15 s), sequentially for each target:

1. Read the Deployment (`spec.replicas` = current, `deployment.kubernetes.io/revision`, optional `kubeai.io/min-replicas` / `kubeai.io/max-replicas` annotations, `spec.paused`) and the namespace's HPAs.
2. Run the 24–25 PromQL queries the engine needs (`hpa_desired` only when an HPA targets the Deployment) as `query_range` over the last `HISTORY_S` (900 s) at a 5 s step. They are `loadlab/runner/promexport.py`'s `deploy_queries` / `pod_queries` (query v2) with the namespace, the Deployment name, its pod regex (`<deployment>-[a-z0-9]+-[a-z0-9]{5}`: ReplicaSet hash + pod suffix, so `web` never matches `web-api`'s pods), the HPA's name, the app container (`APP_CONTAINER`), the scrape job (`APP_JOB`) and the per-Deployment `APP_METRIC_MATCHER` as parameters. v2 differs from the v1 strings that exported the training data (still reproducible byte for byte with `queryVersion: "v1"`, checked against a run's `queries.json`) in three ways: kube-state-metrics pod selectors are restricted to the pod regex, `hpa_desired` to the Deployment's HPA, and `mem_growth_bytes_per_s` to live scrape targets (v1 kept a deleted pod's slope ~60 s, which latched `memory_leak` after the agent's own restart in the live A/B). Re-querying the A/B runs, v2 returns identical series to v1 except `mem_growth`; on the 44 dataset runs it changes 70 of 4,861 `mem_growth` points, none by more than 100 KB/s. (With v1, replaying the window builder for the held-out run `d-leak-medium-r2` reproduced its exported `deploy.csv` exactly: 114/114 rows, max abs difference 0 on all 15 engine columns.) Pod rows whose name does not match the regex are dropped again in the window builder. When `APP_METRIC_MATCHER` has no `{deployment}` template and the namespace has more than one Deployment, the loop refuses to decide (`decision_error`, `stage: "config"`), because the app metrics would be namespace-wide. The post-processing is promexport's / build_dataset's (cumulative restarts and OOMKills, `hpa_desired ← replicas_desired` without an HPA, `goodput_rps`, `pods_not_ready`, cAdvisor's 0-byte first sample → missing, leading rows without traffic trimmed), with one live-serving difference: `rps`/`inflight` are zero-filled only while the app's `inflight` gauge exists, so a dead scrape reaches the engine as missing (its stale-telemetry interlock freezes scale-down) instead of as zero traffic. The five cross-check columns the engine never reads are not queried.
3. `POST /v1/decide` with the window, the per-pod rows, `current_replicas`, the bounds, `now_ts`, `revision` (revisions the agent itself created are reported as the revision before them) and, once the agent has done one, `agent_rollout_ts` (its own rollout undo / restart times, so the engine does not count the new ReplicaSet as rollout evidence).
4. Turn each returned action into a step, then act according to `ACT_MODE`:

| Engine action | Agent step (argv built by `lib/remediation.js buildEngineAction`) | Guards |
|---|---|---|
| `scale` | `kubectl -n NS scale deployment/D --current-replicas=CUR --replicas=N` | clamped to [min, max]; **refused if an HPA targets the Deployment, or the HPA list cannot be read, unless `HPA_OVERRIDE=true`**; `SCALE_UP_COOLDOWN_S` / `SCALE_DOWN_COOLDOWN_S` since the last executed scale; `--current-replicas` makes the API server reject the call if someone else changed the count meanwhile |
| `restart_pod` (`delete_pod`) | **eviction**: `kubectl create --raw /api/v1/namespaces/NS/pods/P/eviction -f -` (body: a `policy/v1` Eviction; dry run: `?dryRun=All`) | P must be in the owner-verified live pod list of this Deployment (kube API by the Deployment's selector incl. `matchExpressions`, owner ReplicaSet); **refused when that list cannot be read or the selector is empty** (never falls back to Prometheus pod names); a PodDisruptionBudget refusal (429) is logged as skipped and replaced by a surge `rollout restart`; one per `REMEDIATION_COOLDOWN_S`, shared with the OOM loop |
| `restart_pod` (`rollout_restart`) | `kubectl rollout restart deployment/D` (dry run: the equivalent `patch` of `restartedAt`) | cooldown |
| `rollback` | `kubectl -n NS rollout undo deployment/D --to-revision=R` | only when the engine emits it (it does so only with rollout evidence). R = the newest revision older than the current code that this agent saw healthy for ≥ 4 decisions; **refused and paged** when there is none, when the current revision was created by the agent's own rollback, or when the agent already rolled back once from that revision (no undo ping-pong). An agent rollout restart of revision N is treated as N's code. Cooldown. |
| `page` | logged as a `{"level":"PAGE",...}` line, never executed | |
| `none` | nothing (`params.scaling: "blocked"` is the "do NOT scale" gate) | |

Nothing runs while the Deployment is paused.

| `ACT_MODE` | Behaviour |
|---|---|
| `observe` (default) | log the decision and the steps that would run (`outcome: "planned"`); kubectl is never called |
| `dryrun` | every allowed step runs as `kubectl ... --dry-run=server` (RBAC + admission are checked, nothing changes) |
| `act` | real execution, **only with `ALLOW_EXEC=true`**; without it the agent logs a warning and runs as `dryrun` |

Every decision is one JSON line on stdout and, if `DECISION_LOG` is set, appended to that file (the live A/B parses it):

```json
{"kind":"decision","timestamp":"2026-11-08T20:23:30.604Z","ts":1794169410.604,"namespace":"kubeai-demo",
 "deployment":"loadlab","mode":"observe","state":"healthy","confidence":0.9635,"current":2,"ready":2,"desired":2,
 "action":"none","actions":[{"type":"none","target":"loadlab","params":{},"reason":"within capacity; ...","outcome":"none"}],
 "reasons":["state healthy (p=0.96): load / learned knee 0.42","policy: hold: ...","..."],
 "telemetry_ok":true,"slo_risk":0.0031,"hpa":null,"min_replicas":1,"max_replicas":10,
 "model_version":"scaling-20261108194802","rows":14,"pod_rows":28,
 "latency_ms":{"prometheus":584,"ml":247,"total":899}}
```

`action` is the first non-`none` step as a short string (`scale:2->4`, `restart_pod:<pod>`, `rollback`, `page`, `none`); `actions[]` carries every step with its `outcome` (`planned`, `dry_run_ok`, `executed`, `failed` + `error`, `skipped` + `why` for cooldowns / no change, `refused` + `why` for the HPA guard / allowlist / paused, `paged`, `none`) and, when kubectl ran, its `argv`. A cycle that cannot decide writes `{"kind":"decision_error","stage":"kube|prometheus|window|ml|discovery|config|internal","error":...}` and retries on the next cycle.

| Env | Default | |
|---|---|---|
| `DEPLOYMENTS` | empty | comma list of Deployments in `NAMESPACE` |
| `DEPLOY_SELECTOR` | empty | label selector on Deployments (used when `DEPLOYMENTS` is empty) |
| `SCALER` | on if either is set | force the loop on/off |
| `ACT_MODE` | `observe` | `observe` / `dryrun` / `act` (act also needs `ALLOW_EXEC=true`) |
| `DECIDE_INTERVAL_S` | 15 | decision period |
| `HISTORY_S` | 900 | telemetry window (≥ 180; the engine keeps up to 1800 s) |
| `SCALE_MIN`, `SCALE_MAX` | 1, `MAX_REPLICAS` (10) | default bounds; per-Deployment annotations `kubeai.io/min-replicas`, `kubeai.io/max-replicas` win |
| `SCALE_UP_COOLDOWN_S`, `SCALE_DOWN_COOLDOWN_S` | 15, 60 | the agent's own cooldowns, on top of the engine's stabilisation |
| `REMEDIATION_COOLDOWN_S` | 300 | per Deployment and remediation kind; restart-type remediations also share it with the OOM loop |
| `HPA_OVERRIDE` | `false` | `true` lets the agent scale a Deployment an HPA also targets (they will fight) |
| `DECISION_LOG` | empty | file to append decision lines to (`/tmp/decisions.jsonl` in `k8s/agent.yaml`) |
| `APP_CONTAINER`, `APP_JOB` | `app`, `annotated-pods` | the app container name and its Prometheus job, as in the lab |
| `APP_METRIC_MATCHER` | empty | per-Deployment label matcher template for the app-level metrics, e.g. `app="{deployment}"`. Empty = namespace-wide (the lab: one served Deployment per namespace); the loop then refuses to decide in a namespace with several Deployments |
| `POD_REGEX` | `{deployment}-[a-z0-9]+-[a-z0-9]{5}` | pod regex template for the cAdvisor and kube-state-metrics pod selectors |
| `PROM_CONCURRENCY` | 6 | parallel `query_range` calls per window |
| `DECIDE_TIMEOUT_MS` | 20000 | `/v1/decide` timeout |
| `OOM_LOOP` | on only when the deployment loop is off | `true` / `false` force the per-pod OOM loop on / off |

## Decision engine & live A/B

The deployment loop's brain is `ml_service`'s scaling engine (bundle `scaling-20261108194802`, policy v2 = HPA_GATED: a CPU-70 % HPA formula plus the state classifier's fault gate and remediation). Full detail: `ml_service/SCALING_MODEL_CARD.md` and `loadlab/ab/AB_REPORT.md`. Headline numbers, stated as narrowly as the data allows:

* **Classifier** (10 states): held-out settled macro-F1 0.844 (blind 0.803) vs 0.139 for an HPA-style rule.
* **Simulator, load (held-out, cap 4):** HPA_GATED 0.67 SLO-violation minutes at 172.0 pod-minutes vs its HPA 70 % / 30 s twin 0.67 at 170.8: a tie. It is *not* on the HPA Pareto frontier (0.58 at that cost; HPA 100 % reaches 0.67 at 137 pod-minutes), and all held-out violations are one event (n = 1). No policy tried, including forecast-driven ones, beat the HPA on load.
* **Simulator, faults (open-loop replay):** 2.3 % of held-out fault decisions add pods vs 16.5 % for an HPA; 0.50 vs 3.75 wasted pod-minutes. This is fewer fault-time scale-ups, not a measured SLO gain.
* **Live closed-loop A/B on kubeai-lab** (agent in `act` vs HPA 30 s / 300 s; 3 trials; **one run per arm**):
  * load: violation-minutes agent vs HPA-30s 1.00 vs 1.25 (pure load), 3.75 vs 4.00 (dependency fault), 1.50 vs 1.25 (leak); pod-minutes within about ±6 %: a tie;
  * dependency fault: 3.0 violation-minutes in every arm; the agent added no pod after detecting the fault at +45 s (one load-driven pod at +4 s before that), HPA-30s added one at +113 s;
  * leak: the agent restarted the leaking pod at +44.7 s, the HPA arms were OOMKilled at +221–222 s (≈ 5× faster recovery); no SLO violation in any arm, 2–4 failed requests in the HPA arms;
  * client-side k6 p95 was 1.1–2.8 s in 8 of 10 runs, far above the server-side metric; it favoured the agent over HPA-30s in all 3 paired runs (n = 1 each);
  * regressions: a ~75 s `memory_leak` latch after the restart delayed a needed scale-up by ~60 s (a PromQL artefact, fixed in query v2; an offline replay shortens it to 2 decisions), and false `dependency_latency` pages at the spike overload in 2 of 3 runs (not fixed).
* **Fixed after the A/B (not re-run live):** undo ping-pong (`--to-revision`, one rollback per bad revision, own rollouts excluded from rollout evidence), the OOM loop ignoring `ACT_MODE` and scaling via the LLM, pod deletes that could hit a sibling Deployment or ignore PDBs (now owner-verified evictions), namespace-wide kube-state-metrics / HPA / app-metric selectors, and the `mem_growth` artefact.

## Running locally

Prerequisites: Node 18.17+, Python 3.12, and a cluster with Prometheus (kube-prometheus-stack or istio's) and metrics-server.

```bash
# 1. Cluster access
kubectl proxy --port=8001 &
kubectl -n monitoring port-forward svc/prometheus-operated 9090 &   # adjust to your Prometheus svc

# 2. ML service (train + serve) - exact commands in ml_service/README.md
cd ml_service && ...            # serves on :8000; check: curl localhost:8000/readyz

# 3. Agent
cd ..
npm install
cp .env.example .env            # set POD_SELECTOR, optionally GOOGLE_API_KEY
npm start
curl localhost:3001/status | jq
curl localhost:3001/pods/<pod-name> | jq

# deployment loop against the lab Prometheus (loadlab/monitoring/port-forward.sh), observe only:
NAMESPACE=<lab-ns> DEPLOYMENTS=loadlab ACT_MODE=observe DECISION_LOG=/tmp/decisions.jsonl npm start
curl localhost:3001/decisions?n=5 | jq
```

`ALLOW_EXEC` defaults to `false`. In that mode every remediation runs as a server-side dry run: the API server validates it against RBAC and admission, and nothing changes. (`rollout_restart` is dry-run as the equivalent `kubectl patch` of the `restartedAt` annotation, because `kubectl rollout restart` has no `--dry-run` flag.) Set `ALLOW_EXEC=true` only when you want the agent to change Deployments.

## Running in the cluster

```bash
# build straight into the minikube node's Docker (no registry, no `image load`); both verified on kubeai-lab
minikube -p kubeai-lab image build -t kubeai-ml:latest ml_service
minikube -p kubeai-lab image build -t kubeai-agent:latest .
# (elsewhere: docker build ... && minikube image load ..., or push to your registry)

kubectl apply -f k8s/            # safe to re-run: k8s/ contains no Secret
# the agent may act only in namespaces you bind (k8s/agent-rbac.yaml binds "default"):
kubectl -n <target-ns> create rolebinding kubeai-agent --clusterrole=kubeai-agent --serviceaccount=kubeai:kubeai-agent
kubectl -n kubeai patch configmap kubeai-agent-config --type merge \
  -p '{"data":{"NAMESPACE":"<target-ns>","DEPLOYMENTS":"<deployment>","ACT_MODE":"observe"}}'
kubectl -n kubeai rollout restart deployment/kubeai-agent
kubectl -n kubeai exec deploy/kubeai-agent -- tail -n 3 /tmp/decisions.jsonl   # or: kubectl logs, GET /decisions
# optional LLM key, created once and never committed (the agent's secretRef is optional):
kubectl -n kubeai create secret generic kubeai-agent-secrets --from-literal=GOOGLE_API_KEY=...
kubectl -n kubeai rollout restart deployment/kubeai-agent   # env is read at start
kubectl -n kubeai port-forward svc/kubeai-agent 3001 & curl localhost:3001/status
```

Inside the cluster the agent talks to `https://kubernetes.default.svc`. It uses the projected service-account token (re-read on every call because the token rotates) and the cluster CA (`NODE_EXTRA_CA_CERTS`). The permissions are a ClusterRole that only takes effect in namespaces where a RoleBinding grants it (namespace-scoped):
- `pods`: get, list, watch
- `pods/eviction`: create (restart one owner-verified live pod of the decided Deployment through the Eviction API, which honours PDBs; there is no pod `delete` verb)
- `deployments`: get, list, watch, patch
- `replicasets`: get, list (`kubectl rollout undo` reads them)
- `deployments/scale`: get, patch, update
- `horizontalpodautoscalers`: get, list (the HPA guard)

No secrets, no create, no delete of Deployments (`kubectl auth can-i` checked on kubeai-lab). Upgrading from the earlier namespaced Role: `roleRef` is immutable, so `kubectl -n default delete rolebinding kubeai-agent` before `kubectl apply -f k8s/`. Run the agent as a single replica, because alert, cooldown and scaler state is held in memory, and the ML service as one replica with one worker for the same reason.

## Agent API

- `GET /healthz` returns `{"status":"ok"}`.
- `GET /status` returns the config (without secrets), the tick count, the last error per dependency, and, for each pod: its containers (limit, restarts, samples, latest working set), its latest prediction (the most at-risk container), alert state (recent history, active, fired_at, cooldown_until), last recommendation, and the kubectl results.
- `GET /pods/:name` returns one pod's view, or 404.
- `GET /decisions?n=50` returns the last `n` (≤ 200) deployment decision records (the same objects as the log lines), or 404 when the deployment loop is off. `GET /status` also has `scaler`: mode, cycles, last error and the last decision per Deployment.

## Tests

```bash
npm run check   # node --check on every source file
npm test        # agent.smoke.mjs (OOM loop) + scaler.smoke.mjs (deployment loop)
```

The smoke test checks the following:
- the label selector is used, and one PromQL query runs per tick
- pods with no data are not sent to the model
- limits are converted to bytes
- the agent recovers after the ML service returns 503
- an alert fires only after 2 of 3 anomalies, and cooldown stops it from re-firing
- a healthy pod stays quiet, and a crash-looping pod logs CRITICAL
- without an API key, kubectl is never executed
- with a fake LLM, the validated recommendation runs as `--dry-run=server`; the LLM cannot `scale`
- `ACT_MODE=observe` with `ALLOW_EXEC=true` never calls kubectl from the OOM loop; two alerting pods of one Deployment cause one LLM call and one rollout restart; a remediation by the deployment loop (shared ledger) suppresses it
- every allowlisted action's dry-run argv passes a fake kubectl that, like the real one, rejects `--dry-run` on `rollout restart`
- new memory limits below 64Mi, below 1.2x the working set, or with no limit and no working set are rejected
- items are per container with each container's own limit, and a sidecar with a large limit does not hide an app container at its own limit
- 1,020 items go out in chunks of 500, and one invalid item (422) costs only its own prediction

`test/scaler.smoke.mjs` checks the deployment loop:
- PromQL v1 is byte-identical to the committed `queries.json` of a lab run (30 queries) and v2 to the current `promexport.py` (`test/fixtures/promql_v2_loadlab.json`); parameters (deployment, escaped pod regex, HPA name, `APP_METRIC_MATCHER`); two Deployments sharing a prefix (`web`, `web-api`): no foreign pod rows
- the window builder: promexport's OOMKill counting, fills, `pods_not_ready`, trims; a dead app scrape stays `null`
- the allowlist: scale bounds and `--current-replicas`, eviction of the Deployment's own owner-verified pods only (unknown ownership → refused), `rollout undo --to-revision` only to an older known-good revision, rollout restart argv; `matchExpressions` selectors
- observe never runs kubectl and writes the decision line to stdout and `DECISION_LOG`; dryrun runs `--dry-run=server` and clamps to the `kubeai.io/max-replicas` annotation
- the HPA guard (an HPA targets the Deployment, or HPAs are unreadable → refused; `HPA_OVERRIDE=true` → scales)
- act with `ALLOW_EXEC=true` scales for real and honours the up/down cooldowns; act without it degrades to dryrun
- remediation: evict own pod, refuse a foreign pod, rollback to the known-good revision + rollout restart dry runs, page logged, remediation cooldown
- no undo ping-pong: after one executed rollback the engine keeps asking for 25 cycles (> 6 min) and the agent refuses and pages every time; its own undo is reported as the previous revision with `agent_rollout_ts`; a rollback after its own rollout restart targets the revision before the restarted code
- eviction refused when the pod list errors or the selector is empty; a PDB 429 → skipped + surge rollout restart
- a namespace with two Deployments and no `{deployment}` matcher → `decision_error` `config`; with the template the matcher is expanded per Deployment
- ML down / Prometheus down → `decision_error`, recovery next cycle; `DEPLOY_SELECTOR` discovery
- `index.js` as a child process: the loop runs every `DECIDE_INTERVAL_S`, `/decisions`, `/status.scaler`, `DECISION_LOG`

## Security notice: leaked API key

A real Google API key was committed to this repository's public history (`.env` in commits `c700f68` "Integrated llm call" and `48952ce` "Working", both on `origin/main`). Deleting the file does not remove it from history. **That key must be revoked and rotated in Google Cloud**; treat it as compromised whatever happens to the history. Rewriting history (`git filter-repo --path .env --invert-paths` plus a force-push) is optional and needs every clone to re-fetch. `.env` and `.env.*` are git-ignored; keep the key only in your local `.env` or the cluster Secret.
