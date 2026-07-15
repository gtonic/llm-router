"""Skill manifest — machine-readable Agent Skill for llm-router discovery.

Implements the Cloudflare Agent Skills Discovery RFC (v0.2.0 draft):
  GET /.well-known/agent-skills/index.json   — Skill directory index
  GET /.well-known/agent-skills/<name>/SKILL.md  — Skill artifact

The SKILL.md is generic and agent-agnostic (Claude Code, Codex CLI, OpenClaw,
Cursor, Gemini CLI, Hermes Agent, etc.).
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

logger = logging.getLogger("llm-router")

# ──────────────────────────────────────────────────────────────────────
# SKILL.md content (the actual skill artifact)
# ──────────────────────────────────────────────────────────────────────

SKILL_MD_CONTENT = """---
name: llm-router-gateway
description: >-
  Configure and use the LLM Router & Gateway as a policy-based LLM request
  router with guardrails, rate limiting, and OpenTelemetry tracing. Use when
  setting up an LLM routing layer, configuring model backends, or integrating
  the router into an agent workflow.
version: 0.1.0
author: llm-router team
tags:
  - llm
  - router
  - gateway
  - guardrails
  - rate-limiting
  - opentelemetry
  - langchain
  - fastapi
agents:
  - claude-code
  - codex
  - openclaw
  - cursor
  - gemini-cli
  - hermes-agent
  - any-openai-compatible-agent
---

# LLM Router & Gateway — Agent Skill

## Overview

The LLM Router is a policy-based LLM gateway that routes requests to the
appropriate model backend based on complexity, policy, cost, or latency.
It provides built-in guardrails (PII detection, abuse filtering, content
safety, rate limiting) and OpenTelemetry tracing.

**Base URL:** `http://localhost:8000` (adjust for your deployment)

## Quick Start

### 1. Verify Connectivity

```bash
curl http://localhost:8000/health
# Expected: {"status":"ok","version":"0.1.0"}
```

### 2. Discover Available Models

```bash
curl http://localhost:8000/v1/models
# Returns list of available model backends
```

### 3. Send a Chat Completion

```bash
curl -X POST http://localhost:8000/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "router-auto",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

### 4. Streaming Response

```bash
curl -X POST http://localhost:8000/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "router-auto",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true
  }'
```

## Configuration

### Model Backends

Configure backends in `models.yaml` (path: `~/.llm-router/models.yaml` or
custom path via `--models-dir`):

```yaml
models:
  - id: llama-local
    name: "Llama 3.1 70B (local)"
    type: local
    base_url: "http://localhost:8001/v1"
    api_key: ""
    enabled: true
    tags: [local, fast, low-cost]

  - id: gpt-5.6-luna
    name: "GPT-5.6 Luna (remote)"
    type: remote
    base_url: "https://api.openai.com/v1"
    api_key: "${OPENAI_API_KEY}"
    enabled: true
    tags: [remote, high-quality, expensive]
```

### Routing Strategy

Set in `config.yaml`:

```yaml
default_strategy: complexity  # complexity | policy | hybrid | round_robin | latency | cost
```

- **complexity** — Auto-routes based on prompt complexity (default)
- **policy** — Routes based on policy rules
- **hybrid** — Combines complexity + policy
- **round_robin** — Distributes evenly across backends
- **latency** — Routes to fastest backend
- **cost** — Routes to cheapest backend

### Guardrails

All guardrails are enabled by default:

| Guardrail | Description | Endpoint |
|-----------|-------------|----------|
| PII Detection | Detects and optionally redacts PII | `/guardrails/check` |
| Abuse Filter | Blocks abusive/prompt-injection content | `/guardrails/check` |
| Content Safety | Filters unsafe content | `/guardrails/check` |
| Rate Limiting | RPM/TPM limits per user | Automatic |

## Model Selection

Use `router-auto` to let the router decide the best backend:

```json
{"model": "router-auto", "messages": [...]}
```

Or target a specific backend:

```json
{"model": "llama-local", "messages": [...]}
```

## System Endpoints

| Endpoint | Method | Description | Auth |
|----------|--------|-------------|------|
| `/health` | GET | Liveness check | None |
| `/v1/models` | GET | List available models | None |
| `/v1/chat/completions` | POST | OpenAI-compatible chat | API key |
| `/guardrails/check` | POST | Manual guardrail test | API key |
| `/admin/reload` | POST | Reload config from disk | API key |
| `/system/manifest` | GET | Full system self-description | None |
| `/system/health` | GET | Detailed health status | None |
| `/system/metrics` | GET | Prometheus metrics snapshot | None |
| `/system/capabilities` | GET | Allowed lifecycle actions | None |
| `/metrics` | GET | Raw Prometheus text format | None |

## Integration Patterns

### For Agent Frameworks

The router is OpenAI-compatible. Any framework that supports OpenAI
chat completions works out of the box:

```python
# LangChain
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    openai_api_key="any-key",  # Value doesn't matter for local
    openai_api_base="http://localhost:8000/v1",
    model="router-auto",
    temperature=0.7,
)
```

```python
# Direct HTTP
import httpx

async with httpx.AsyncClient() as client:
    resp = await client.post(
        "http://localhost:8000/v1/chat/completions",
        json={
            "model": "router-auto",
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )
    print(resp.json()["choices"][0]["message"]["content"])
```

### For Autonomous Agents

Agents can self-configure by:

1. Fetching `/.well-known/agent-skills/index.json` to discover capabilities
2. Reading `/system/manifest` for full system description
3. Using `/system/capabilities` to discover allowed actions
4. Monitoring `/system/health` for backend status

## Troubleshooting

| Symptom | Check |
|---------|-------|
| 503 "Router not initialized" | Server not started or config invalid |
| 404 "model not found" | Model not in models.yaml or disabled |
| Rate limited | Check RPM/TPM limits in config |
| PII redacted | Check PII patterns in `/guardrails/pii/patterns` |

## Installation

### From PyPI

```bash
pip install llm-router
llm-router-server  # Start the server
```

### From Source

```bash
git clone https://github.com/gtonic/llm-router.git
cd llm-router
pip install -e ".[dev]"
llm-router-server
```

### Configuration

Create `~/.llm-router/config.yaml`:

```yaml
host: 0.0.0.0
port: 8000
default_strategy: complexity
default_model: router-auto
rate_limit_rpm: 60
rate_limit_tpm: 100000
models_dir: ~/.llm-router/models.yaml
policies_dir: ~/.llm-router/policies.yaml
pii_redact: true
otlp_enabled: false
otlp_endpoint: http://localhost:4317
```
"""


def _compute_sha256(content: str) -> str:
    """Compute SHA-256 digest of content."""
    return f"sha256:{hashlib.sha256(content.encode('utf-8')).hexdigest()}"


def get_skill_index() -> dict[str, Any]:
    """Return the agent-skills discovery index (Cloudflare RFC v0.2.0)."""
    return {
        "$schema": "https://schemas.agentskills.io/discovery/0.2.0/schema.json",
        "skills": [
            {
                "name": "llm-router-gateway",
                "type": "skill-md",
                "description": (
                    "Policy-based LLM request router with guardrails, rate limiting, and OpenTelemetry tracing."
                ),
                "url": "/.well-known/agent-skills/llm-router-gateway/SKILL.md",
                "digest": _compute_sha256(SKILL_MD_CONTENT),
            }
        ],
    }


def get_skill_md() -> str:
    """Return the raw SKILL.md content."""
    return SKILL_MD_CONTENT
