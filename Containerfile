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

# Readiness, not just liveness: fail if the process is up but every configured
# backend is unreachable (a single provider outage is tolerated — that's what
# routing fallback is for).
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import json,sys,urllib.request; d=json.load(urllib.request.urlopen('http://localhost:8000/v1/system/health', timeout=8)); b=d.get('backends',{}); sys.exit(0 if (not b or any(v.get('healthy') for v in b.values())) else 1)" || exit 1

ENTRYPOINT ["llm-router-server"]
CMD []
