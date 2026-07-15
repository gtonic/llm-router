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
- **Runtime administration**: Model backend CRUD, enable/disable toggles, and live client rebuilds
- **Runtime rollback**: Atomic guardrail configuration persistence with bounded rollback history
- **Self-describing discovery**: Agent Skill index, hosted `SKILL.md`, system manifest, and capabilities
- **OpenTelemetry tracing**: Full request tracing with cost/latency metrics
- **Prometheus metrics**: Request, routing, guardrail, and runtime admin action metrics
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
| `/admin/models` | GET, POST | List or create model backends |
| `/admin/models/{model_id}` | GET, PUT, DELETE | Inspect, update, or remove a backend |
| `/admin/models/{model_id}/toggle` | PATCH | Enable or disable a backend |
| `/admin/guardrails` | GET, PUT | Inspect or update runtime guardrails |
| `/admin/guardrails/{guardrail_name}/toggle` | PATCH | Toggle a guardrail |
| `/admin/guardrails/pii/patterns` | POST, DELETE | Manage custom PII patterns |
| `/admin/rollback` | POST | Restore the previous runtime configuration |
| `/.well-known/agent-skills/index.json` | GET | Agent Skill discovery index |
| `/.well-known/agent-skills/llm-router-gateway/SKILL.md` | GET | Hosted Agent Skill artifact |
| `/system/manifest` | GET | Full self-describing system manifest |
| `/system/health` | GET | Backend and guardrail health |
| `/system/metrics` | GET | JSON metrics snapshot |
| `/system/capabilities` | GET | Allowed lifecycle and admin actions |

Admin endpoints are intended for trusted operator access. Authentication and
authorization should be placed in front of them before exposing the router
beyond a trusted local or internal network.

## Agent Discovery

Agents can discover and understand a running router without prior integration
configuration:

```bash
curl http://localhost:8000/.well-known/agent-skills/index.json
curl http://localhost:8000/.well-known/agent-skills/llm-router-gateway/SKILL.md
curl http://localhost:8000/system/manifest
curl http://localhost:8000/system/capabilities
```

The discovery index returns the hosted skill URL and a SHA-256 digest. The
system manifest describes models, routing strategies, guardrails, health,
metrics, lifecycle actions, and the API catalog. Capabilities expose the
operations available to an automation or DevOps agent.

## Runtime Configuration

Mutable guardrail and rate-limit settings are persisted atomically in
`config/runtime.yaml` by default. The path can be changed with
`ROUTER_RUNTIME_CONFIG`. The runtime configuration includes explicit enable
flags for PII, abuse filtering, content safety, and rate limiting.

Model profiles remain in `profiles/`. Disabled profiles stay visible through
the admin API but are excluded from routing. Changes to backend connection
settings rebuild the live backend client.

```bash
curl http://localhost:8000/admin/models
curl http://localhost:8000/admin/guardrails
curl -X POST http://localhost:8000/admin/rollback
```

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
  -e ROUTER_OTELP_ENABLED=false \
  -e APP_DEBUG=true \
  llm-router:latest

# With Docker Compose (App + Jaeger)
docker compose up -d

# With Monitoring Stack (Jaeger + Prometheus + Grafana)
docker compose -f docker-compose.yml -f docker-compose.monitoring.yml up -d

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
http://localhost:16686            # Jaeger Tracing UI
http://localhost:3001             # Grafana Dashboard (admin/admin)
http://localhost:9090             # Prometheus Metrics
```

## License

MIT
