FROM python:3.12-slim

# Keep model files in a mounted cache and store uploads on the data volume.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/model-cache \
    DOCUMENT_STORAGE_DIR=/data/uploads

WORKDIR /srv/app
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY frontend ./frontend
COPY alembic ./alembic
COPY alembic.ini .

RUN useradd --create-home --uid 10001 rag \
    && mkdir -p /data/uploads /opt/model-cache \
    && chown -R rag:rag /data /opt/model-cache /srv/app
USER rag

# Apply schema changes before serving API requests.
EXPOSE 8000
CMD ["sh", "-c", "python -m alembic upgrade head && exec python -m uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
