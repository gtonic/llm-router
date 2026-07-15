# syntax=docker/dockerfile:1
# ================================================
#  LLM Router & Gateway — Apple Container Image
#  Build with: container build -t llm-router:latest .
# ================================================

FROM python:3.12-slim AS builder

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY pyproject.toml .

# Install the package in a virtualenv
RUN python -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# Install dependencies first (layer caching)
RUN pip install --no-compile --upgrade pip && \
    pip install --no-compile -e ".[dev]"

# Copy source
COPY src/ src/
COPY tests/ tests/

# ── Production stage ───────────────────────────────────────────────
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

# Create directories for mounted config/profiles
RUN mkdir -p /app/profiles /app/agent-policies /app/logs

EXPOSE 8000

USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

ENTRYPOINT ["llm-router-server"]
CMD []
