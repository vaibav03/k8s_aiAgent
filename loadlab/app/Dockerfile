FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt && useradd -u 10001 -M -s /usr/sbin/nologin app
COPY main.py .
USER 10001
EXPOSE 8080 8081
# exec form: python is PID 1, so an OOM kill / crash terminates the container directly
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1", "--loop", "uvloop", "--http", "httptools", "--no-access-log", "--timeout-keep-alive", "30"]
