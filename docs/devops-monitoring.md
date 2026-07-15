# DevOps Monitoring — System Manifest

## Overview

The LLM Router exposes a self-describing system manifest following the **Dragonbook Agent pattern**.
This enables automated DevOps agents (e.g. Pi-based monitoring agents) to discover, understand,
and operate on the system without manual configuration.

## Endpoints

| Endpoint | Method | Description | Auth |
|----------|--------|-------------|------|
| `/system/manifest` | GET | Full system self-description | None |
| `/system/health` | GET | Detailed health (backends + routing + guardrails) | None |
| `/system/metrics` | GET | Current metrics snapshot | None |
| `/system/capabilities` | GET | Allowed lifecycle actions | None |

## Manifest Structure

The `/system/manifest` endpoint returns a JSON object with the following sections:

### `system`
Core identity: name, display name, version, description.

### `metadata`
Runtime info: host, port, start time, uptime, model backends.

### `routing`
Available routing strategies and default.

### `guardrails`
List of active guardrails (PII detection, abuse filter, etc.).

### `health`
Current health status of all model backends with latency and error info.

### `api_endpoints`
Complete catalog of API endpoints with methods, descriptions, and auth requirements.

### `log_patterns`
Structured log patterns for anomaly detection:
- `model_unavailable` — model not found in pool
- `routing_fallback` — routing fallback triggered
- `upstream_error` — upstream backend error
- `memory_warning` — memory pressure detected
- `rate_limit` — rate limit exceeded
- `pii_detected` — PII detected and redacted
- `abuse_blocked` — abuse filter triggered
- `config_reload` — configuration reloaded

### `lifecycle`
Allowed actions with auth requirements and restart policies.

### `metrics_definitions`
Prometheus metric definitions with types, labels, and descriptions.

### `monitoring_thresholds`
Warning and critical thresholds for latency, model switches, and error rates.

### `bug_reporting`
Bug reporting configuration (target, labels, severity levels).

## Pi Agent Integration

### Discovery

A Pi DevOps agent can discover the system by:

1. **HTTP GET** `/system/manifest` — fetches the complete self-description
2. **Parse** the JSON to understand capabilities, health, and allowed actions
3. **Monitor** `/system/health` at configured intervals (default: 30s)
4. **Alert** on log pattern matches from the `log_patterns` section

### Example Pi Skill

```yaml
name: llm-router-monitor
description: Monitor the LLM Router via its self-describing manifest
version: "1.0.0"

tools:
  - name: check_router_health
    description: Check LLM Router health via /system/health
    method: GET
    url: "{{ROUTER_URL}}/system/health"
    interval: 30s

  - name: get_router_manifest
    description: Fetch system manifest for capability discovery
    method: GET
    url: "{{ROUTER_URL}}/system/manifest"
    on_startup: true

  - name: get_router_metrics
    description: Get current metrics snapshot
    method: GET
    url: "{{ROUTER_URL}}/system/metrics"
    interval: 60s

  - name: get_router_capabilities
    description: Get allowed lifecycle actions
    method: GET
    url: "{{ROUTER_URL}}/system/capabilities"
    on_startup: true

alerts:
  - name: backend_unhealthy
    condition: "health.backends[*].healthy == false"
    severity: warning
    action: alert

  - name: high_error_rate
    condition: "metrics.error_rate_percent > 10.0"
    severity: critical
    action: alert_and_restart

  - name: high_latency
    condition: "metrics.router_latency_ms > 1000"
    severity: warning
    action: alert
```

### Custom Tools (Pi Extension)

For more advanced monitoring, a Pi extension can implement:

```typescript
// devops-monitor/index.ts
import { Extension, Tool } from "pi";

@Extension({ name: "llm-router-monitor" })
export class LLmRouterMonitorExtension implements Extension {
  @Tool({ name: "check_router_health" })
  async checkHealth(): Promise<{ status: string; backends: Record<string, any> }> {
    const response = await fetch(`${process.env.ROUTER_URL}/system/health`);
    return response.json();
  }

  @Tool({ name: "get_router_manifest" })
  async getManifest(): Promise<any> {
    const response = await fetch(`${process.env.ROUTER_URL}/system/manifest`);
    return response.json();
  }

  @Tool({ name: "get_router_metrics" })
  async getMetrics(): Promise<any> {
    const response = await fetch(`${process.env.ROUTER_URL}/system/metrics`);
    return response.json();
  }

  @Tool({ name: "reload_router_config" })
  async reloadConfig(): Promise<{ status: string }> {
    const response = await fetch(`${process.env.ROUTER_URL}/admin/reload`, {
      method: "POST",
      headers: { "Authorization": `Bearer ${process.env.ROUTER_API_KEY}` },
    });
    return response.json();
  }
}
```

## Monitoring Thresholds

| Metric | Warning | Critical | Description |
|--------|---------|----------|-------------|
| Router latency | 200ms | 1000ms | Routing decision latency |
| Model switches | 10 | 50 | Switches per window |
| Error rate | 2% | 10% | Error rate percentage |

## Log Pattern Actions

| Pattern | Severity | Action |
|---------|----------|--------|
| `model_unavailable` | warning | alert |
| `routing_fallback` | warning | log_and_alert |
| `upstream_error` | critical | alert_and_restart |
| `memory_warning` | critical | alert_and_restart |
| `rate_limit` | warning | alert |
| `pii_detected` | info | log_only |
| `abuse_blocked` | warning | alert |
| `config_reload` | info | log_only |

## Security

- Health, metrics, and manifest endpoints are **public** (no auth required)
- Config reload requires API key authentication
- All monitoring data is read-only by default
- Critical actions (restart, shutdown) require explicit approval

## Prometheus Integration

The router also exposes raw Prometheus metrics at `/metrics`:

```
llm_router_requests_total{model="llama-3.1-8b",strategy="complexity",status="success"} 1234
llm_router_errors_total{model="gpt-4",error_type="timeout"} 5
llm_router_request_duration_seconds_bucket{model="llama-3.1-8b",le="0.5"} 800
llm_router_cost_total{model="gpt-4"} 2.45
llm_router_tokens_total{type="prompt",model="llama-3.1-8b"} 45000
llm_router_pii_detected_total{sensitivity="high"} 12
llm_router_abuse_blocked_total 3
llm_router_rate_limited_total 8
llm_router_route_decisions_total{strategy="complexity"} 1200
llm_router_response_status_total{status="200"} 1200
llm_router_active_requests 2
```
