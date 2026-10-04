# NovaTech Copilot API. Built by docker-compose.yml at the repo root.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/hf \
    FASTEMBED_CACHE_PATH=/cache/fastembed \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /app

# CPU-only torch: the default Linux wheel bundles CUDA (several GB).
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Non-root user; volume mount points must exist here so volumes inherit its ownership.
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/data/state /app/data/logs /cache \
    && chown -R app:app /app/data /cache
USER app

EXPOSE 8000
CMD ["sh", "-c", "python -m scripts.bootstrap && exec uvicorn app.api.main:app --host 0.0.0.0 --port 8000"]
