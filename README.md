<div align="center">

# 🧭 LLM Router & Gateway

### One OpenAI-compatible endpoint in front of every model you run — routed by policy, cost, latency and complexity, with guardrails, resilience and full observability built in.

[![CI](https://github.com/gtonic/llm-router/actions/workflows/ci.yml/badge.svg)](https://github.com/gtonic/llm-router/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![OpenAI Compatible](https://img.shields.io/badge/API-OpenAI--compatible-412991.svg)](#-drop-in-openai-compatibility)
[![Observability](https://img.shields.io/badge/observability-Prometheus%20%7C%20Grafana%20%7C%20Jaeger-e6522c.svg)](#-observability-that-actually-tells-you-whats-wrong)

</div>

---

**llm-router** sits between your applications and your models — local (llama.cpp, Ollama, vLLM) and cloud (OpenAI or any OpenAI-compatible API) — and makes one smart decision per request: *which* model should serve it, *whether* it's safe, and *what* it costs. Your app keeps talking plain OpenAI; the router does the hard part.

```
        Your apps  ──►  ┌───────────── llm-router ─────────────┐  ──►  llama.cpp / Ollama / vLLM
   (OpenAI SDK, curl,   │  auth · guardrails · routing ·        │       OpenAI / Azure / any
    LangChain, agents)  │  circuit breaking · metrics · tracing │       OpenAI-compatible API
                        └──────────────────────────────────────┘
```

## ✨ Why teams use it

| | |
|---|---|
| 💸 **Spend less, automatically** | Keep cheap/local models for trivial work and escalate to premium models only when complexity demands it. Per-model **token & cost accounting** — for streaming *and* non-streaming — makes the local-vs-API tradeoff visible in dollars. |
| 🔒 **Keep sensitive data in-house** | Built-in **PII detection & redaction**, abuse filtering and content-safety guardrails. A single policy rule keeps PII-bearing prompts on local models and out of third-party APIs. |
| 🛟 **Stay up when a provider doesn't** | Per-backend **circuit breaker**, health-aware selection, single-hop fallback and retries. A flapping backend is skipped, not hammered — and streams never get corrupted by a mid-stream failover. |
| 🔌 **Change nothing in your app** | 100% **OpenAI Chat Completions compatible**, streaming included. Point any OpenAI SDK at the router and you're done. |
| 📊 **See everything** | First-class **Prometheus metrics** (incl. TTFT & tokens/sec), a ready-made **Grafana dashboard**, **OpenTelemetry/Jaeger** tracing and **alert rules** ship in the box. |
| ⚙️ **Operate it live** | Add, update, toggle or roll back models and guardrails at runtime via an authenticated admin API — no restart, no redeploy. |

## 🚀 Quick start (60 seconds)

```bash
# 1. Clone & configure
git clone https://github.com/gtonic/llm-router.git && cd llm-router
cp .env.example .env                      # edit as needed

# 2. Bring up the full stack (router + Jaeger + Prometheus + Grafana)
docker compose -f docker-compose.yml -f docker-compose.monitoring.yml up -d

# 3. It's alive
curl http://localhost:8000/health
# {"status":"ok","version":"0.1.0"}
```

| Service | URL | Notes |
|---|---|---|
| 🧭 Router API | http://localhost:8000 | OpenAI-compatible; docs at `/docs` |
| 📊 Grafana | http://localhost:3001 | Dashboard auto-provisioned (`admin` / `admin`) |
| 🔎 Prometheus | http://localhost:9090 | Metrics + alert rules |
| 🕸️ Jaeger | http://localhost:16686 | Distributed traces |

Prefer bare metal?

```bash
pip install -e .
cp .env.example .env
llm-router-server            # starts uvicorn on :8000
```

## 🔌 Drop-in OpenAI compatibility

Your existing OpenAI code works unchanged — just swap the base URL (and, if auth is enabled, use one of your router keys):

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="sk-your-router-key")

resp = client.chat.completions.create(
    model="auto",                                   # let the router decide, or name a backend
    messages=[{"role": "user", "content": "Summarize this contract clause…"}],
    stream=True,                                    # streaming fully supported (+ token/cost accounting)
)
for chunk in resp:
    print(chunk.choices[0].delta.content or "", end="")
```

Set `model` to `"auto"` (or `"router-auto"`) to route by your configured strategy, or to a specific backend id to pin it.

---

## 🧩 Example configurations

Configuration is just YAML + env vars. Here are several ready-to-adapt setups.

### 1. Minimal — a single local model (Ollama / llama.cpp)

```yaml
# profiles/models.yaml
- id: local
  name: Local Llama
  type: local
  base_url: http://host.docker.internal:11434/v1   # Ollama default; llama.cpp: :8080/v1
  model_name: llama3.1:8b
  temperature: 0.3
  max_tokens: 4096
```

```dotenv
# .env
ROUTER_DEFAULT_STRATEGY=complexity
ROUTER_DEFAULT_MODEL=local
ROUTER_FALLBACK_MODEL=local
```

### 2. Local-first with an OpenAI safety net

```yaml
# profiles/models.yaml
- id: local
  name: Local Llama
  type: local
  base_url: http://host.docker.internal:11434/v1
  model_name: llama3.1:8b
  cost_per_1m_input: 0.0        # local = free
  cost_per_1m_output: 0.0

- id: gpt-4o-mini
  name: OpenAI GPT-4o mini
  type: remote
  base_url: https://api.openai.com/v1
  api_key: ${OPENAI_API_KEY}    # resolved from the environment, never written back to disk
  model_name: gpt-4o-mini
  cost_per_1m_input: 0.15
  cost_per_1m_output: 0.60
```

```dotenv
# .env
ROUTER_DEFAULT_STRATEGY=complexity   # simple → local, complex → gpt-4o-mini
ROUTER_DEFAULT_MODEL=local
ROUTER_FALLBACK_MODEL=gpt-4o-mini    # used if the primary backend fails
OPENAI_API_KEY=sk-...
```

### 3. Cost-optimized 3-tier fleet

```yaml
# profiles/models.yaml
- id: local           # Tier 1 — free, fast, trivial + PII-sensitive work
  name: Local Llama
  type: local
  base_url: http://host.docker.internal:11434/v1
  model_name: llama3.1:8b
  cost_per_1m_input: 0.0
  cost_per_1m_output: 0.0
  tags: [local, fast, low-cost]

- id: gpt-4o-mini     # Tier 2 — cheap cloud for the bulk of requests
  name: OpenAI GPT-4o mini
  type: remote
  base_url: https://api.openai.com/v1
  api_key: ${OPENAI_API_KEY}
  model_name: gpt-4o-mini
  cost_per_1m_input: 0.15
  cost_per_1m_output: 0.60
  tags: [remote, cheap]

- id: gpt-4o          # Tier 3 — premium, only for complex/critical work
  name: OpenAI GPT-4o
  type: remote
  base_url: https://api.openai.com/v1
  api_key: ${OPENAI_API_KEY}
  model_name: gpt-4o
  cost_per_1m_input: 2.50
  cost_per_1m_output: 10.00
  tags: [remote, premium]
```

### 4. Routing policies — PII stays home, tasks route by cost

Policies are evaluated by descending `priority`; the first matching rule wins. Supported conditions: `contains_pii`, `task_type` (`code` / `analysis` / `creative` / `general`), `complexity` (`low` / `medium` / `high` / `critical`), plus `all` / `any` combinators.

```yaml
# agent-policies/default.yaml
rules:
  - id: pii-stays-local          # 🔒 never send PII to a third-party API
    name: No external calls for PII
    conditions: { contains_pii: true }
    target_model: local
    priority: 100

  - id: code-to-mini             # 💻 code tasks → cheap cloud
    name: Code → GPT-4o mini
    conditions: { task_type: code }
    target_model: gpt-4o-mini
    priority: 60

  - id: heavy-analysis-premium   # 🧠 hard analysis → premium
    name: Critical analysis → GPT-4o
    conditions:
      all:
        - { task_type: analysis }
        - { complexity: critical }
    target_model: gpt-4o
    priority: 50

  - id: trivial-local            # 💤 simple stuff stays free
    name: Low complexity → local
    conditions: { complexity: low }
    target_model: local
    priority: 20

  - id: default                  # 🪃 everything else
    name: Default
    conditions: {}
    target_model: gpt-4o-mini
    priority: 0
```

Activate policy routing with `ROUTER_DEFAULT_STRATEGY=policy`.

### 5. Production `.env` — auth, resilience, guardrails, CORS

```dotenv
# ── Routing ────────────────────────────────────────────────
ROUTER_DEFAULT_STRATEGY=policy          # policy | complexity | hybrid | round_robin | latency | cost
ROUTER_DEFAULT_MODEL=local
ROUTER_FALLBACK_MODEL=gpt-4o-mini
ROUTER_STRICT_CONFIG=true               # fail fast on misconfiguration

# ── Data-plane auth ────────────────────────────────────────
# Secure by default: with no keys the inference API is fail-closed (401).
# Set keys to require them, or ROUTER_ALLOW_ANONYMOUS=true for a trusted network.
ROUTER_API_KEYS=sk-team-alpha,sk-team-beta
ROUTER_ADMIN_TOKEN=replace-with-a-long-random-token

# ── Resilience — circuit breaker ───────────────────────────
ROUTER_CIRCUIT_BREAKER_ENABLED=true
ROUTER_CIRCUIT_BREAKER_THRESHOLD=5      # consecutive failures before a backend is skipped
ROUTER_CIRCUIT_BREAKER_COOLDOWN=30      # seconds before a half-open probe

# ── Rate limits (also runtime-tunable & persistable via /admin) ─
ROUTER_RATE_LIMIT_RPM=120
ROUTER_RATE_LIMIT_TPM=200000

# ── CORS (explicit allow-list enables credentialed requests) ─
ROUTER_CORS_ORIGINS=["https://app.example.com"]

# ── Observability ──────────────────────────────────────────
ROUTER_OTLP_ENABLED=true
ROUTER_OTELP_ENDPOINT=http://jaeger:4317/v1/traces

OPENAI_API_KEY=sk-...
```

> 🔐 **Security note:** the inference API is **fail-closed by default** — with no `ROUTER_API_KEYS` set, gated endpoints return `401`. For a trusted/local network, set `ROUTER_ALLOW_ANONYMOUS=true` to opt into open access (a loud warning is logged).

---

## 🧠 Routing strategies

| Strategy | Picks the backend by… | Great for |
|---|---|---|
| `policy` | your YAML rules (PII, task type, complexity) | explicit, auditable control |
| `complexity` | a heuristic complexity score of the prompt | auto local↔cloud escalation |
| `hybrid` | local-first, escalating to remote on high/critical complexity | cheapest-that-works |
| `cost` | cheapest enabled backend (`cost_per_1m_*`) | hard budget optimization |
| `latency` | lowest-latency healthy backend (health-check latency) | interactive / low-latency UX |
| `round_robin` | even rotation across enabled backends | load spreading across replicas |

Every decision is guarded by the **circuit breaker** and a single-hop **fallback**, so an unhealthy target is skipped before it's called — not after it fails.

## 🛡️ Guardrails & security

- **PII detection & redaction** — emails, phone numbers, card numbers, API keys, plus custom regex patterns; redaction is applied to what's sent upstream *and* what's returned.
- **Abuse & content-safety filtering** — configurable categories; output is scanned as it streams (bounded, O(n)).
- **Rate limiting** — per-principal RPM/TPM (token-bucket), keyed to the authenticated API key, not spoofable body fields.
- **Request caps** — message-count, `max_tokens` (default ≤128k) and body-size limits reject oversized / DoS requests up front, all env-tunable (`ROUTER_MAX_OUTPUT_TOKENS`, …).
- **Two auth planes** — data-plane API keys for `/v1/*`, a separate admin token for the mutation API (both constant-time compared).

## 📈 Observability that actually tells you what's wrong

The monitoring stack is pre-wired — Prometheus scrapes the router, Grafana auto-loads the dashboard, Jaeger collects traces, and alert rules are loaded.

- **LLM-aware latency** — request-duration histograms plus **Time-to-First-Token** and **first-frame-vs-first-token**, so you can tell *"the model is thinking"* from *"the model streams slowly."*
- **Throughput & spend** — tokens/sec per model, cost per model, blended **$/1k tokens**, for streaming and non-streaming alike.
- **Routing & health** — request distribution per target model, errors by type (429 / guardrail-block / timeout / 5xx), concurrency, per-backend uptime and status history.
- **Alerts out of the box** — router down, all/any backend down, error-rate, p95 latency per model, rate-limit saturation, empty-response spikes, daily cost budget.

<div align="center"><em>Grafana → “LLM Router Dashboard” (auto-provisioned). Prometheus rules in <code>configs/alert_rules.yml</code>.</em></div>

## 🔧 Runtime administration & agent discovery

Operate the router live (admin token required):

```bash
# Inspect / mutate backends and guardrails without a restart
curl -H "X-Admin-Token: $TOKEN" http://localhost:8000/admin/models
curl -H "X-Admin-Token: $TOKEN" http://localhost:8000/admin/guardrails
curl -H "X-Admin-Token: $TOKEN" -X POST http://localhost:8000/admin/rollback   # revert last change
```

The router is **self-describing** — automation and DevOps agents can discover it with zero prior integration:

```bash
curl http://localhost:8000/.well-known/agent-skills/index.json    # skill discovery index
curl http://localhost:8000/system/manifest                        # models, strategies, guardrails, API catalog
curl http://localhost:8000/system/capabilities                    # allowed lifecycle / admin actions
```

## 📚 API reference

| Endpoint | Method | Description | Auth |
|---|---|---|---|
| `/v1/chat/completions` | POST | OpenAI-compatible chat (stream & non-stream) | 🔑 data-plane |
| `/v1/models` | GET | List available models | 🔑 data-plane |
| `/v1/guardrails/check` | POST | Manually run guardrails on text | 🔑 data-plane |
| `/health` | GET | Liveness probe | open |
| `/metrics` | GET | Prometheus metrics (text) | open |
| `/v1/system/health` | GET | Backend & guardrail health (readiness) | open |
| `/system/manifest` · `/system/metrics` | GET | Self-description / JSON metrics snapshot | 🔑 data-plane |
| `/system/capabilities` | GET | Allowed lifecycle / admin actions | open |
| `/admin/models` · `/admin/models/{id}` | GET/POST/PUT/DELETE | Backend CRUD | 🛠️ admin token |
| `/admin/models/{id}/toggle` | PATCH | Enable / disable a backend | 🛠️ admin token |
| `/admin/guardrails` · `/admin/guardrails/{name}/toggle` | GET/PUT/PATCH | Inspect / update / toggle guardrails | 🛠️ admin token |
| `/admin/guardrails/pii/patterns` | POST/DELETE | Manage custom PII patterns | 🛠️ admin token |
| `/admin/rollback` | POST | Restore previous runtime config | 🛠️ admin token |
| `/.well-known/agent-skills/…` | GET | Agent Skill discovery + hosted `SKILL.md` | open |

> `🔑 data-plane` endpoints require a valid API key. Fail-closed by default (`401` with no keys); set `ROUTER_ALLOW_ANONYMOUS=true` to allow unauthenticated access on a trusted network.

## ⚙️ Configuration reference

**Model backend fields** (`profiles/*.yaml`):

| Field | Default | Description |
|---|---|---|
| `id` | – | Unique backend id (used as the routing target) |
| `type` | – | `local` or `remote` |
| `base_url` | – | OpenAI-compatible base URL (`…/v1`) |
| `model_name` | `id` | Upstream model name to request |
| `api_key` | `""` | Use `${ENV_VAR}` — resolved from env, never persisted as plaintext |
| `temperature` / `max_tokens` | `0.3` / `4096` | Generation defaults |
| `timeout` / `retry_count` | `60` / `3` | Per-request timeout & retries |
| `cost_per_1m_input` / `cost_per_1m_output` | `0.0` | Prices that drive cost routing, metrics & budgets |
| `enabled` | `true` | Disabled backends stay visible to admin but off the routing path |
| `tags` | `[]` | Free-form labels |

**Key environment variables** (prefix `ROUTER_`, loaded from `.env`) — see [`.env.example`](.env.example) for the full list:
`DEFAULT_STRATEGY`, `DEFAULT_MODEL`, `FALLBACK_MODEL`, `API_KEYS`, `ADMIN_TOKEN`, `CIRCUIT_BREAKER_{ENABLED,THRESHOLD,COOLDOWN}`, `RATE_LIMIT_{RPM,TPM}`, `CORS_ORIGINS`, `OTLP_ENABLED`, `OTELP_ENDPOINT`, `STRICT_CONFIG`, `PREWARM`.

## 🐳 Deployment

```bash
# App + Jaeger
docker compose up -d

# App + full monitoring (Jaeger + Prometheus + Grafana)
docker compose -f docker-compose.yml -f docker-compose.monitoring.yml up -d

# Logs / stop
docker compose logs -f app
docker compose down
```

macOS-native builds are supported via Apple's `container` tool and the provided `Containerfile` (kept in sync with the `Dockerfile`).

## 🧪 Development

```bash
pip install -e '.[dev]'

ruff check src/ tests/         # lint
ruff format src/ tests/        # format
pytest tests/ -q               # 340+ tests
pytest tests/ --cov=src/llm_router --cov-report=term-missing   # with coverage
```

CI runs lint, the full test suite with a coverage gate, and a container build (which itself runs the tests as a gate).

## 🗂️ Project layout

```
src/llm_router/
├── router.py        # orchestration: guardrails → routing → model call → audit
├── routing/         # policy · complexity · hybrid · round_robin · (cost/latency in engine)
├── pool/            # backends (local/remote) + circuit breaker
├── guardrails/      # PII · abuse · content-safety · rate limiter
├── server/          # FastAPI app, routes, auth, metrics
└── tracing/         # OpenTelemetry setup
profiles/            # model backends (YAML)
agent-policies/      # routing policies (YAML)
configs/             # Prometheus, alert rules, Grafana dashboard + provisioning
```

## 📄 License

MIT — see [LICENSE](LICENSE).

<div align="center"><sub>Built for teams who want the economics of local models and the muscle of frontier models — without choosing.</sub></div>
