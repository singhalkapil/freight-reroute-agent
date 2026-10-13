# Multi-stage Dockerfile for Freight Reroute Agent
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# Install system dependencies (GEOS for Shapely, git/curl if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgeos-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy packaging specifications
COPY pyproject.toml README.md ./

# Install python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e ".[dev]"

# Copy source, evaluations, and test suites
COPY src/ ./src/
COPY evals/ ./evals/
COPY tests/ ./tests/
COPY run_agent.py ./

# Default command runs the evaluation benchmark harness
CMD ["python", "evals/evaluate.py"]

