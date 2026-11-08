# KubeAI rebuild: what was done

PR: https://github.com/vaibav03/k8s_aiAgent/pull/1 (branch `ml-service-upgrade`) · Architecture page: https://claude.ai/artifact/VUQ6AfsWYJuj3TiXMc2Yu5

## Before
- **Collection:** index.js polled one hardcoded pod's memory. An LSTM predicted the next value and LightGBM labelled that single number.
- **Labels:** "not Running" pods were labelled, i.e. pods that had already crashed.
- **Evaluation:** scored in-sample. The LSTM was 9× worse than predicting no change.
- **Remediation:** an `isAnamoly` typo meant the Gemini step never ran.
- **Missing signals:** requests, latency and errors were never used.

## After
1. **OOM-ahead model** (`ml_service/`). Calibrated monotone LightGBM plus a trend gate, served by FastAPI.
   - F1 0.89 with 0 false alarms/h. The old hybrid scored 0.71 with 13/h.
2. **SRE lab and dataset** (`loadlab/`).
   - minikube `kubeai-lab`, an instrumented app, Prometheus with a 5 s scrape, open-loop k6 load and 7 fault types.
   - 44 runs, 27 ground-truth faults, 4 signal groups.
3. **Decision engine** (`ml_service/kubeai_ml/scaling/`).
   - Parts: a 10-state classifier, learned capacity, a quantile forecaster (advisory only) and an HPA replay simulator.
   - Default policy **HPA_GATED**: HPA sizing, plus a fault gate and remediation.
   - Classifier held-out macro-F1 0.844 (0.803 blind), against 0.139 for the HPA rule.
   - Pods are added on 2.3% of fault decisions, against 16.5% for the HPA.
4. **`/v1/decide` endpoint and agent scaler.**
   - ACT_MODE observe/dryrun/act; Eviction API; owner checks; no fighting an existing HPA.
   - Rollback only to a revision the agent has seen healthy.
   - The LLM only fills an allowlist.
5. **Live A/B against the HPA** (n=1 per arm).
   - Pure load: a tie.
   - Memory leak: the agent restarted the leaking pod at +45 s; the HPA arms only recovered at the OOMKill, at +222 s.
   - Dependency latency: no scale-up after the fault was detected.

## Honest negatives
- Forecast-driven scaling did not beat a tuned HPA at equal cost, so it is advisory only.
- In 2 of 3 runs a spike overload was misread once as `dependency_latency`.
- Everything rests on a single service on a single node, with small n.
- On the collaborator's old data, the OOM model reaches only 0.41–0.64 F1.

## Your actions
- **Revoke and rotate the Google API key**. It is in the history (c700f68, 48952ce) and in the collaborator's fork. History was not rewritten.
- Review and merge PR #1.
- The cluster `kubeai-lab` is still running, with kubeai in observe mode. Stop it with `minikube -p kubeai-lab stop`.
- The watchdog has stopped.
