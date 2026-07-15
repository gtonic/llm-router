# 🚀 LLM Router & Gateway

Policy-based LLM request router with guardrails, rate limiting, and OpenTelemetry tracing.

## Features

- **Multi-provider routing**: Local (Llama.cpp/Ollama) + Remote (OpenAI/Anthropic/Azure)
- **Policy-based routing**: YAML-driven rules with priority ordering
- **Complexity-based routing**: Auto-classify requests by complexity
- **Hybrid routing**: Split complex requests across multiple models
- **Round-robin**: Load balancing across model instances
- **Guardrails pipeline**: PII detection/redaction, abuse filtering, content safety
- **Rate limiting**: Per-client RPM/TPM limits with token-bucket algorithm
- **OpenTelemetry tracing**: Full request tracing with cost/latency metrics
- **OpenAI-compatible API**: Drop-in replacement for OpenAI endpoints

## Architecture

```
Client → Rate Limiter → Input Guardrails → Policy Router → Model Call → Output Guardrails → Client
                 │                   │                    │                  │
                 ▼                   ▼                    ▼                  ▼
              Rate Check        PII/Abuse           Local/Remote       PII Leak/Content
                                        │            (Llama.cpp/      Safety Check
                                        │             OpenAI/etc)
                                        ▼
                                   Routing Decision
                                   (Policy/Complexity/
                                    Hybrid/RoundRobin)
```

## Quick Start

```bash
# Install
pip install -e .

# Configure
cp .env.example .env
# Edit .env with your settings

# Start server
llm-router-server

# Test
curl http://localhost:8000/health
# {"status":"ok"}
```

## Project Structure

```
llm-router/
├── src/llm_router/
│   ├── config.py        # Settings & dataclasses
│   ├── models.py        # OpenAI-compatible request/response models
│   ├── router.py        # Main router engine
│   ├── pool/            # Model backends
│   ├── routing/         # Routing strategies
│   ├── guardrails/      # Input/output filtering
│   ├── tracing/         # OpenTelemetry setup
│   └── server/          # FastAPI routes
├── profiles/            # Model profiles (YAML)
├── agent-policies/      # Routing policies (YAML)
└── tests/
```

## Configuration

### Model Profiles

Create YAML files in `profiles/` to define model backends:

```yaml
# profiles/local.yaml
- id: llama-local
  name: Llama CPP Local
  type: local
  base_url: http://localhost:8888/v1
  model_name: Qwen3.6-35B-A3B-local
  temperature: 0.3
  max_tokens: 4096
```

### Routing Policies

Define rules in `agent-policies/`:

```yaml
# agent-policies/default.yaml
rules:
  - id: no_remote_pii
    name: No external calls for PII
    conditions:
      contains_pii: true
    target_model: llama-local
    priority: 100
  - id: default_remote
    name: Default Remote
    conditions: {}
    target_model: gpt4o
    priority: 0
```

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/v1/chat/completions` | POST | Main chat endpoint |
| `/v1/models` | GET | List available models |
| `/health` | GET | Health check |
| `/guardrails/check` | POST | Manual guardrail check |
| `/admin/reload` | POST | Reload policies |

## Development

```bash
# Install dev dependencies
pip install -e '.[dev]'

# Run linter
ruff check src/ tests/

# Format code
ruff format src/ tests/

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src/llm_router
```

## Docker & Container

### Docker

```bash
# Build image
docker build -t llm-router:latest .

# Run single container
docker run -d --name llm-router \
  -p 8000:8000 \
  -e OTEL_ENABLED=false \
  -e APP_DEBUG=true \
  llm-router:latest

# With Docker Compose (App + Jaeger)
docker compose up -d

# View logs
docker compose logs -f app

# Stop
docker compose down
```

### Apple Container (macOS native)

```bash
# Build
container build -t llm-router:latest -f Containerfile .

# Run
container run -d --name llm-router -p 8000:8000 llm-router:latest

# Compose (App + Jaeger)
container compose up -d
```

### Health Check

```bash
curl http://localhost:8000/health
# {"status":"ok","version":"0.1.0"}
```

### API Docs

```
http://localhost:8000/docs        # Swagger UI
http://localhost:8000/redoc       # ReDoc
http://localhost:16686            # Jaeger Tracing UI (docker compose)
```

## License

MIT
