"""Central configuration for the LLM Router & Gateway.

Uses pydantic-settings for .env loading and YAML files for model/policy configuration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum

from pydantic_settings import BaseSettings


# ───────────────────────────────────────────
# Enums
# ───────────────────────────────────────────

class RoutingStrategy(str, Enum):
    """Available routing strategies."""
    POLICY = "policy"
    COMPLEXITY = "complexity"
    HYBRID = "hybrid"
    ROUND_ROBIN = "round_robin"
    LATENCY = "latency"
    COST = "cost"


class PrivacyLevel(str, Enum):
    """Data privacy classification levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    RESTRICTED = "restricted"
    CLASSIFIED = "classified"


# ───────────────────────────────────────────
# Dataclasses
# ───────────────────────────────────────────

@dataclass
class ModelBackendConfig:
    """Configuration for a single model backend (local or remote)."""
    id: str
    name: str
    type: str  # "local" | "remote" | "edge"
    base_url: str
    api_key: str = ""
    model_name: str = ""
    enabled: bool = True
    temperature: float = 0.3
    max_tokens: int = 4096
    timeout: int = 60
    retry_count: int = 3
    cost_per_1m_input: float = 0.0
    cost_per_1m_output: float = 0.0

    def __post_init__(self):
        if not self.model_name:
            self.model_name = self.id


@dataclass
class PolicyRule:
    """A single routing policy rule loaded from YAML."""
    id: str
    name: str
    description: str = ""
    conditions: dict = field(default_factory=dict)
    target_model: str = ""
    priority: int = 0
    enabled: bool = True


# ───────────────────────────────────────────
# Settings
# ───────────────────────────────────────────

class GatewaySettings(BaseSettings):
    """Top-level gateway configuration.

    Loaded from .env with ``ROUTER_`` prefix and optional YAML files.
    """
    # ── Core ──────────────────────────────────
    default_strategy: RoutingStrategy = RoutingStrategy.POLICY
    default_model: str = "llama-local"

    # ── Model Backends ────────────────────────
    models_dir: str = "profiles"
    models: list[ModelBackendConfig] = field(default_factory=list)

    # ── Routing ───────────────────────────────
    policies_dir: str = "agent-policies"
    default_policy: str = "default"

    # ── Guardrails ────────────────────────────
    guardrails_input: bool = True
    guardrails_output: bool = True
    pii_redact: bool = True
    pii_max_tokens: int = 4096
    abuse_block_threshold: float = 0.8
    rate_limit_rpm: int = 60
    rate_limit_ppm: int = 60000

    # ── Observability ─────────────────────────
    otlp_enabled: bool = True
    otlp_endpoint: str = "http://localhost:4318/v1/traces"
    otlp_protocol: str = "http/protobuf"

    # ── Logging ───────────────────────────────
    log_dir: str = "logs"
    log_level: str = "INFO"

    model_config = {
        "env_prefix": "ROUTER_",
        "env_file": ".env",
        "extra": "ignore",
    }
