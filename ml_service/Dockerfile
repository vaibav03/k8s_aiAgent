# KubeAI ML service: OOM-risk prediction over HTTP (FastAPI + LightGBM, no TensorFlow).
FROM python:3.12-slim

# LightGBM needs the OpenMP runtime.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY kubeai_ml ./kubeai_ml
COPY models ./models

ENV MODEL_DIR=/app/models/LATEST PYTHONUNBUFFERED=1
USER 10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:8000/healthz')" || exit 1
CMD ["uvicorn", "kubeai_ml.server:app", "--host", "0.0.0.0", "--port", "8000"]
