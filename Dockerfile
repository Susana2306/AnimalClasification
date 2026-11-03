FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    MPLCONFIGDIR=/tmp/matplotlib \
    PORT=8080

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api.py animalclassification.keras animalclassificationpreentrenado.keras ./
COPY templates ./templates

RUN useradd --create-home --uid 1000 appuser
USER appuser

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/', timeout=4)"

# Worker sync: pyplot no es thread-safe. Sin --preload: TensorFlow no soporta fork tras cargar modelos.
CMD ["sh", "-c", "exec gunicorn api:app_to_flask --bind 0.0.0.0:${PORT} --timeout 120 --worker-tmp-dir /dev/shm --access-logfile -"]
