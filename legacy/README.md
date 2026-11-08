# legacy/

Kept for reference only. Nothing here is loaded at runtime any more.

| File | What it was | Why it was replaced |
|---|---|---|
| `hybrid_predict.py` | stdin/stdout child process: 10-sample window, then LSTM forecast, then LightGBM flag | One hard-coded pod, no error handling or restart, and Keras progress bars corrupted the `pred,flag` stdout protocol. Replaced by the HTTP service in `../ml_service/` (`POST /v1/predict/batch`, many pods per call). |
| `hybrid_lstm_model.keras`, `hybrid_lstm_model.h5` | LSTM memory forecaster (committed 2025-04-09, RMSprop) | No training code exists in any repo. It does not match the notebook models (`k8s_Notebooks`, 2025-09-24, Adam, different md5), so the deployed pair could not be reproduced. On held-out data its MAE is about 9x worse than persistence (just repeating the last value). It shrinks toward the mean near the limit, so it could never warn about a leak. |
| `lightgbm_anomaly.pkl` | LightGBM classifier on one feature (the LSTM's predicted memory) | It came from a different run than the LSTM. The notebook metrics were in-sample (fit and scored on the same rows: 0.963, against 0.931 held out). Its label (`Node Status != Running`) marks pods that have **already** crashed (89% are CrashLoopBackOff rows at about 79 MB), so it detects crashes after the fact instead of predicting them. Held out, the agent pair caught 46 of 1306 anomalies on `k8s_metrics.csv` and got F1 0.052 on `dataset_1`. |
| `exp.js` | scratch script | Dead code. It requested `/api/v1/namespaces/default/pod/<name>` (singular `pod`, a 404). |

Train/serve skew in the old setup: the models were trained on metrics-server working set in `Ki/1000` units divided by the file maximum (887.345). The agent instead fed them cAdvisor `container_memory_usage_bytes`, which includes page cache, divided by `2^20` and by `parseInt(limit.split("M")[0])`. That parse is wrong for `Gi`, for plain bytes, and for pods with no limit.

Agent bugs fixed by the rewrite (see `../index.js` and `../lib/`):
- The alert branch tested `isAnamoly`, a typo of `isAnomaly` that is always undefined, so it always printed "Healthy".
- Missing Prometheus data was turned into `0`.
- The agent read only `result[0]` of a multi-series result, and `containerStatuses[0]` (the istio sidecar).
- `express` was imported but `listen()` was never called.
- It used the retired `gemini-1.5-pro` model, with a prompt that asked for a raw shell command (the `exec` was commented out).

The full analysis is in `understand.json` in the parent workspace.

Note: an earlier version of this agent committed a real `GOOGLE_API_KEY` in `.env` (commits `c700f68` and `48952ce`, still in the public history). That key is compromised and must be revoked; see "Security notice" in the top-level README.
