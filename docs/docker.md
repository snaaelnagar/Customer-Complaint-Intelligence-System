# Docker Deployment Guide

## Purpose

This document describes how to build, run, validate, and troubleshoot the Machine Learning / NLP Customer Complaint Intelligence System using Docker.

The Docker image contains:

* Streamlit dashboard (`app/`)
* Analytics pipeline source code (`src/`)
* Configuration files (`config.yaml`, `.streamlit/config.toml`)
* Pre-computed dashboard artifacts (`data/processed/`)
* Trained NLP model artifacts (`models/nlp/`)

The raw CFPB dataset (`data/raw/`) is intentionally excluded from the image.

This deployment strategy prioritizes:

* One-command execution
* Fast demo startup
* Reproducibility
* Portfolio-friendly deployment

See `architecture.md` for the deployment architecture rationale.

---

# Image Characteristics

| Property                     | Value            |
| ---------------------------- | ---------------- |
| Base Image                   | Python 3.11 Slim |
| Runtime                      | Streamlit        |
| Deployment Type              | Single Container |
| Approximate Size             | 3–4 GB           |
| Includes Dashboard Artifacts | Yes              |
| Includes NLP Models          | Yes              |
| Includes Raw Dataset         | No               |
| Runs as Non-Root User        | Yes              |

The image is intentionally larger than a minimal application container because it bundles pre-computed dashboard outputs and trained NLP models.

This allows the dashboard to start immediately after `docker run` without requiring preprocessing, analytics generation, forecasting, or NLP model training inside the container.

---

# Build Image

Build the Docker image locally.

```bash
docker build -t customer-complaint-intelligence:v1 .
```

Verify image creation:

```bash
docker images
```

Expected repository:

```text
customer-complaint-intelligence
```

---

# Run Container

Run the locally built image.

```bash
docker run -p 8501:8501 customer-complaint-intelligence:v1
```

Open:

```text
http://localhost:8501
```

---

# Docker Hub Image

Repository:

```text
https://hub.docker.com/repository/docker/<your-dockerhub-username>/customer-complaint-intelligence
```

Pull the latest published image:

```bash
docker pull <your-dockerhub-username>/customer-complaint-intelligence:latest
```

Run:

```bash
docker run -p 8501:8501 <your-dockerhub-username>/customer-complaint-intelligence:latest
```

Open:

```text
http://localhost:8501
```

---

# Verify Container Health

List running containers:

```bash
docker ps
```

View logs:

```bash
docker logs -f <container_id>
```

A healthy container should:

* Stay in `Up` status
* Show Streamlit startup logs
* Respond on port 8501
* Load the dashboard successfully

Example:

```text
You can now view your Streamlit app in your browser.
```

---

# Common Port Conflict

If port 8501 is already allocated:

```bash
docker run -p 8502:8501 customer-complaint-intelligence:v1
```

Open:

```text
http://localhost:8502
```

---

# Production-Style Multi-Stage Dockerfile

```dockerfile
FROM python:3.11-slim AS base

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

FROM base AS dependencies

COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt

FROM dependencies AS runtime

COPY app/ app/
COPY src/ src/
COPY config.yaml .
COPY data/processed/ data/processed/
COPY models/nlp/ models/nlp/
COPY .streamlit/ .streamlit/

RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501')" || exit 1

CMD ["streamlit","run","app/streamlit_app.py","--server.address=0.0.0.0","--server.port=8501"]
```

### Why Multi-Stage?

Benefits:

* Smaller image
* Faster rebuilds
* Dependency caching
* Cleaner runtime layer
* Better separation of concerns

### Why Non-Root User?

The container runs as:

```text
appuser
```

instead of root.

This is a standard container-hardening practice because it limits permissions if the container is ever compromised.

---

# Docker Compose

Create:

```text
docker-compose.yaml
```

```yaml
version: "3.9"

services:
  complaint-intelligence:
    image: <your-dockerhub-username>/customer-complaint-intelligence:latest
    container_name: customer-complaint-intelligence

    ports:
      - "8501:8501"

    restart: unless-stopped

    environment:
      - PYTHONUNBUFFERED=1

    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8501')"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 30s
```

Start:

```bash
docker compose up -d
```

View logs:

```bash
docker logs -f customer-complaint-intelligence
```

Stop:

```bash
docker compose down
```

---

# Files Excluded from the Image

The following files are intentionally excluded via `.dockerignore`:

```text
data/raw/
assets/
reports/
tests/
notebooks/
logs/
.env
```

These files are not required for runtime execution.

---

# Files Included in the Image

```text
app/
src/
config.yaml
.streamlit/
requirements.txt
data/processed/
models/nlp/
```

These files are required for dashboard execution.

---

# Why Pre-Computed Artifacts Are Bundled

The image intentionally includes:

```text
data/processed/
models/nlp/
```

instead of generating them during startup.

Advantages:

* Faster startup
* No preprocessing required
* No forecasting required
* No NLP retraining required
* Reproducible demo environment

Trade-off:

```text
Larger image size
```

This trade-off was intentionally chosen because the project is designed for demonstration and portfolio deployment rather than distributed production retraining.

---

# Health Check Notes

The container uses a Streamlit health check:

```text
http://localhost:8501
```

During initial startup, health checks may briefly fail while:

* Streamlit starts
* Dashboard artifacts load
* NLP models load

This is expected behavior during the startup period.

---

# Troubleshooting

## Port Already Allocated

Error:

```text
Bind for 0.0.0.0:8501 failed: port is already allocated
```

Fix:

```bash
docker run -p 8502:8501 customer-complaint-intelligence:v1
```

---

## NLP Model Loading Error

Possible cause:

```text
numpy
scikit-learn
joblib
```

versions differ from the training environment.

Fix:

* Pin dependency versions
* Rebuild image
* Re-export model artifacts

---

## Missing Dashboard Files

Error:

```text
Summary file not found
```

Cause:

Pipeline outputs were not included during image build.

Fix:

```bash
python -m src.preprocessing
python -m src.pipeline
docker build -t customer-complaint-intelligence:v1 .
```

---

## Streamlit Theme Not Applied

Cause:

```text
.streamlit/config.toml
```

missing from the image.

Verify:

```text
.streamlit/config.toml
```

exists before rebuilding.

---

## Container Exits Immediately

Check logs:

```bash
docker ps -a
docker logs <container_id>
```

Common causes:

* Missing NLP artifacts
* Missing dashboard artifacts
* Dependency version mismatch
* Incorrect file paths

---

# Deployment Validation Checklist

Before publishing:

```text
Docker image builds successfully
Container starts successfully
Dashboard loads successfully
Executive Dashboard loads
Risk Analysis loads
Forecasting loads
NLP Prediction loads
Recommendation Engine loads
Health check passes
```

If all items above pass, the Docker deployment is considered successful.
