"""System manifest — self-describing definition for DevOps monitoring.

Every system exposes its capabilities, health, metrics, log patterns, and
allowed actions via a structured manifest. This enables automated monitoring
agents (e.g. Pi-based DevOps agents) to discover, understand, and operate on
the system without manual configuration.

Endpoints:
  GET /system/manifest    — Full system definition (metadata + capabilities)
  GET /system/health      — Detailed health (backends + routing + guardrails)
  GET /system/metrics     — Current Prometheus metrics snapshot
  GET /system/capabilities — Allowed actions (lifecycle)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from llm_router.config import GatewaySettings, ModelBackendConfig

logger = logging.getLogger("llm-router")


@dataclass
class BackendHealth:
    """Health status of a single model backend."""

    id: str
    name: str
    type: str  # "local" | "remote" | "edge"
    healthy: bool
    latency_ms: float
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "healthy": self.healthy,
            "latency_ms": round(self.latency_ms, 1),
            "error": self.error,
        }


@dataclass
class SystemManifest:
    """Complete self-describing definition of the LLM Router system.

    Structured sections:
      - metadata       → harness, version, runtime info
      - health         → liveness + backend health
      - capabilities   → allowedNextActions / lifecycle
      - metrics        → monitoring data
      - log_patterns   → event signatures for anomaly detection
    """

    # ── Core metadata ──────────────────────────────────────────
    name: str = "llm-router"
    display_name: str = "LLM Router & Gateway"
    version: str = "0.1.0"
    description: str = "Policy-based LLM request router with guardrails, rate limiting, and OpenTelemetry tracing."

    # ── Runtime info ───────────────────────────────────────────
    runtime: str = "fastapi"
    port: int = 8000
    host: str = "0.0.0.0"
    start_time: str = ""  # ISO 8601, set at startup
    uptime_seconds: float = 0.0

    # ── Health (set dynamically) ───────────────────────────────
    health: dict[str, Any] = field(
        default_factory=lambda: {
            "status": "unknown",
            "backends": [],
        }
    )

    # ── Model backends ─────────────────────────────────────────
    models: list[dict[str, Any]] = field(default_factory=list)

    # ── Routing strategies ─────────────────────────────────────
    routing_strategies: list[str] = field(
        default_factory=lambda: [
            "policy",
            "complexity",
            "hybrid",
            "round_robin",
            "latency",
            "cost",
        ]
    )
    default_strategy: str = "complexity"

    # ── Guardrails ─────────────────────────────────────────────
    guardrails: list[str] = field(
        default_factory=lambda: [
            "pii_detection",
            "pii_redaction",
            "abuse_filter",
            "content_safety",
            "rate_limiting",
        ]
    )

    # ── API endpoints ──────────────────────────────────────────
    api_endpoints: list[dict[str, Any]] = field(
        default_factory=lambda: [
            {
                "path": "/v1/chat/completions",
                "method": "POST",
                "description": "OpenAI-compatible chat completion",
                "auth": "api_key",
            },
            {
                "path": "/v1/models",
                "method": "GET",
                "description": "List available models",
                "auth": None,
            },
            {
                "path": "/guardrails/check",
                "method": "POST",
                "description": "Manual guardrail check",
                "auth": "api_key",
            },
            {
                "path": "/admin/reload",
                "method": "POST",
                "description": "Reload policies from disk",
                "auth": "api_key",
            },
            {
                "path": "/system/manifest",
                "method": "GET",
                "description": "System self-description (DevOps monitoring)",
                "auth": None,
            },
            {
                "path": "/system/health",
                "method": "GET",
                "description": "Detailed health check",
                "auth": None,
            },
            {
                "path": "/system/metrics",
                "method": "GET",
                "description": "Current metrics snapshot",
                "auth": None,
            },
            {
                "path": "/system/capabilities",
                "method": "GET",
                "description": "Allowed lifecycle actions",
                "auth": None,
            },
        ]
    )

    # ── Log patterns for anomaly detection ─────────────────────
    log_patterns: list[dict[str, Any]] = field(
        default_factory=lambda: [
            {
                "name": "model_unavailable",
                "regex": "model.*unavailable|model.*not.*found|404.*model",
                "severity": "warning",
                "action": "alert",
                "description": "Requested model not found in pool",
            },
            {
                "name": "routing_fallback",
                "regex": "fallback|failover|switch.*to",
                "severity": "warning",
                "action": "log_and_alert",
                "description": "Routing fallback triggered",
            },
            {
                "name": "upstream_error",
                "regex": "upstream.*error|connection.*refused|502|503|504",
                "severity": "critical",
                "action": "alert_and_restart",
                "description": "Upstream backend error",
            },
            {
                "name": "memory_warning",
                "regex": "memory.*high|OOM|out.*of.*memory|heap.*error",
                "severity": "critical",
                "action": "alert_and_restart",
                "description": "Memory pressure detected",
            },
            {
                "name": "rate_limit",
                "regex": "rate.*limit|429|too.*many.*requests",
                "severity": "warning",
                "action": "alert",
                "description": "Rate limit exceeded",
            },
            {
                "name": "pii_detected",
                "regex": "PII detected|pii.*redact",
                "severity": "info",
                "action": "log_only",
                "description": "PII detected and redacted",
            },
            {
                "name": "abuse_blocked",
                "regex": "Abuse detected|abuse.*blocked",
                "severity": "warning",
                "action": "alert",
                "description": "Abuse filter triggered",
            },
            {
                "name": "config_reload",
                "regex": "config.*reloaded|reload.*policies",
                "severity": "info",
                "action": "log_only",
                "description": "Configuration reloaded",
            },
        ]
    )

    # ── Lifecycle / allowed actions ────────────────────────────
    lifecycle: dict[str, Any] = field(
        default_factory=lambda: {
            "restart_on_critical": True,
            "max_restarts": 3,
            "restart_window": "5m",
            "allowed_actions": [
                {
                    "action": "health_check",
                    "description": "Check system and backend health",
                    "method": "GET",
                    "endpoint": "/system/health",
                    "auth_required": False,
                },
                {
                    "action": "reload_config",
                    "description": "Reload policies and models from disk",
                    "method": "POST",
                    "endpoint": "/admin/reload",
                    "auth_required": True,
                },
                {
                    "action": "get_metrics",
                    "description": "Fetch current metrics",
                    "method": "GET",
                    "endpoint": "/system/metrics",
                    "auth_required": False,
                },
                {
                    "action": "get_manifest",
                    "description": "Get system self-description",
                    "method": "GET",
                    "endpoint": "/system/manifest",
                    "auth_required": False,
                },
            ],
        }
    )

    # ── Prometheus metrics definitions ─────────────────────────
    metrics_definitions: list[dict[str, Any]] = field(
        default_factory=lambda: [
            {
                "name": "llm_router_requests_total",
                "type": "counter",
                "description": "Total number of requests",
                "labels": ["model", "strategy", "status"],
            },
            {
                "name": "llm_router_errors_total",
                "type": "counter",
                "description": "Total number of errors",
                "labels": ["model", "error_type"],
            },
            {
                "name": "llm_router_request_duration_seconds",
                "type": "histogram",
                "description": "Request latency in seconds",
                "labels": ["model", "strategy"],
                "buckets": [0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
            },
            {
                "name": "llm_router_cost_total",
                "type": "counter",
                "description": "Total cost in USD",
                "labels": ["model"],
            },
            {
                "name": "llm_router_tokens_total",
                "type": "counter",
                "description": "Total tokens processed",
                "labels": ["type", "model"],
            },
            {
                "name": "llm_router_pii_detected_total",
                "type": "counter",
                "description": "Total PII detections",
                "labels": ["sensitivity"],
            },
            {
                "name": "llm_router_abuse_blocked_total",
                "type": "counter",
                "description": "Total abuse requests blocked",
                "labels": [],
            },
            {
                "name": "llm_router_rate_limited_total",
                "type": "counter",
                "description": "Total rate-limited requests",
                "labels": [],
            },
            {
                "name": "llm_router_route_decisions_total",
                "type": "counter",
                "description": "Total routing decisions",
                "labels": ["strategy"],
            },
            {
                "name": "llm_router_response_status_total",
                "type": "counter",
                "description": "Response status codes",
                "labels": ["status"],
            },
            {
                "name": "llm_router_active_requests",
                "type": "gauge",
                "description": "Number of active requests",
                "labels": [],
            },
        ]
    )

    # ── Bug reporting config ───────────────────────────────────
    bug_reporting: dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": True,
            "target": {
                "type": "github",
                "labels": ["bug", "llm-router", "devops-auto"],
            },
            "severity_levels": ["info", "warning", "critical"],
        }
    )

    # ── Monitoring thresholds ──────────────────────────────────
    monitoring_thresholds: dict[str, Any] = field(
        default_factory=lambda: {
            "router_latency_ms": {
                "warning": 200,
                "critical": 1000,
                "description": "Routing decision latency",
            },
            "model_switch_count": {
                "warning": 10,
                "critical": 50,
                "description": "Model switches per window",
            },
            "error_rate_percent": {
                "warning": 2.0,
                "critical": 10.0,
                "description": "Error rate percentage",
            },
        }
    )

    def to_dict(self) -> dict[str, Any]:
        """Serialize manifest to dict for JSON API response."""
        return {
            "system": {
                "name": self.name,
                "display_name": self.display_name,
                "version": self.version,
                "description": self.description,
            },
            "metadata": {
                "runtime": self.runtime,
                "port": self.port,
                "host": self.host,
                "start_time": self.start_time,
                "uptime_seconds": round(self.uptime_seconds, 1),
                "models": self.models,
            },
            "routing": {
                "strategies": self.routing_strategies,
                "default_strategy": self.default_strategy,
            },
            "guardrails": self.guardrails,
            "api_endpoints": self.api_endpoints,
            "health": self.health,  # Set dynamically
            "log_patterns": self.log_patterns,
            "lifecycle": self.lifecycle,
            "metrics_definitions": self.metrics_definitions,
            "monitoring_thresholds": self.monitoring_thresholds,
            "bug_reporting": self.bug_reporting,
        }


def build_manifest(
    settings: GatewaySettings,
    start_time: str,
    uptime_seconds: float,
    models: list[ModelBackendConfig],
    health_status: dict[str, BackendHealth] | None = None,
) -> SystemManifest:
    """Build a complete system manifest from runtime state."""
    manifest = SystemManifest(
        version=settings.__class__.__module__.split(".")[-1] or "0.1.0",
        start_time=start_time,
        uptime_seconds=uptime_seconds,
        port=settings.port,
        host=settings.host,
        default_strategy=settings.default_strategy.value,
    )

    # Model backends
    manifest.models = [
        {
            "id": m.id,
            "name": m.name,
            "type": m.type,
            "base_url": m.base_url,
            "enabled": m.enabled,
            "is_local": m.is_local,
            "is_remote": m.is_remote,
            "tags": m.tags,
        }
        for m in models
    ]

    # Health status (set dynamically)
    if health_status:
        manifest.health = {
            "status": "healthy" if all(h.healthy for h in health_status.values()) else "degraded",
            "backends": {mid: h.to_dict() for mid, h in health_status.items()},
        }
    else:
        manifest.health = {
            "status": "unknown",
            "backends": [],
        }

    return manifest
