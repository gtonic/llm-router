# syntax=docker/dockerfile:1
# ================================================
#  LLM Router & Gateway — multi-stage Dockerfile
# ================================================

# ── 1. Build stage ─────────────────────────────────────────────────
FROM python:3.12-slim AS builder

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY pyproject.toml .

# Install the package in a virtualenv so all deps are in /app/venv
RUN python -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# Install dependencies first (layer caching)
# README.md is required by hatch for editable install metadata
COPY README.md .
RUN pip install --no-compile --upgrade pip && \
    pip install --no-compile -e ".[dev]"

# Copy source and run (optional) tests
COPY src/ src/
COPY tests/ tests/
RUN python -m pytest tests/ -q --tb=short 2>&1 | tail -20 || true

# ── 2. Production stage ────────────────────────────────────────────
FROM python:3.12-slim AS runtime

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser -m appuser

WORKDIR /app

# Copy only the venv from builder
COPY --from=builder /app/venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# Copy application source
COPY --from=builder /build/src /app/src
COPY --from=builder /build/pyproject.toml /app/

# Make llm_router importable (entry points expect it on PYTHONPATH)
ENV PYTHONPATH=/app/src
ENV PATH=/app:$PATH

# Copy config/profiles if present (may be mounted at runtime)
RUN mkdir -p /app/profiles /app/agent-policies /app/logs
VOLUME ["/app/profiles", "/app/agent-policies", "/app/logs"]

EXPOSE 8000

USER appuser

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

ENTRYPOINT ["llm-router-server"]
CMD []
