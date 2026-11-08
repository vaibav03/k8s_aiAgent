# KubeAI ML service: OOM-risk prediction (/v1/predict) + deployment-level SRE decisions
# (/v1/decide) over HTTP (FastAPI + LightGBM + pandas, no TensorFlow).
#
# Build straight into the minikube node's Docker (no registry, no `image load`):
#   minikube -p kubeai-lab image build -t kubeai-ml:latest ml_service
FROM python:3.12-slim

# LightGBM needs the OpenMP runtime.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY kubeai_ml ./kubeai_ml
# models/LATEST (OOM-ahead) and models/scaling/LATEST (decision engine) are both baked in.
COPY models ./models

ENV MODEL_DIR=/app/models/LATEST SCALING_MODEL_DIR=/app/models/scaling/LATEST PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp OMP_NUM_THREADS=1
USER 10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:8000/healthz')" || exit 1
# One worker: the decision engine keeps per-deployment stabilisation memory in process.
CMD ["uvicorn", "kubeai_ml.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
