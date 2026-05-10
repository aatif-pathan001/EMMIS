FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    API_PORT=8000 \
    UI_PORT=8501

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libglib2.0-0 \
        libgomp1 \
        tini \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install .

COPY app.py ./

RUN useradd --create-home --uid 1000 emmis \
    && chown -R emmis:emmis /app

USER emmis

EXPOSE 8000 8501

ENTRYPOINT ["/usr/bin/tini", "--"]

# Runs the FastAPI service and the Streamlit UI side by side.
# MongoDB Atlas is external — pass MONGODB_URI / DATABASE_NAME / COLLECTION_NAME
# (and the encryption keys) via `docker run --env-file .env` or `-e`.
# To run only one service, override CMD, e.g.:
#   docker run ... emmis uvicorn emmis.api.routes:app --host 0.0.0.0 --port 8000
#   docker run ... emmis streamlit run app.py --server.port 8501 --server.address 0.0.0.0
CMD ["bash", "-c", "uvicorn emmis.api.routes:app --host 0.0.0.0 --port ${API_PORT} & streamlit run app.py --server.port ${UI_PORT} --server.address 0.0.0.0 --server.headless true & wait -n; exit $?"]
