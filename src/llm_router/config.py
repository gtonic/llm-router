"""Central configuration for the LLM Router & Gateway.

Uses pydantic-settings for .env loading and YAML files for model/policy configuration.
All dataclasses support round-trip YAML serialisation via ``pydantic-yaml`` or
standard ``yaml.safe_load`` helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic_settings import BaseSettings

# ───────────────────────────────────────────
# Enums
# ───────────────────────────────────────────


class RoutingStrategy(StrEnum):
    """Available routing strategies."""

    POLICY = "policy"
    COMPLEXITY = "complexity"
    HYBRID = "hybrid"
    ROUND_ROBIN = "round_robin"
    LATENCY = "latency"
    COST = "cost"


class PrivacyLevel(StrEnum):
    """Data privacy classification levels."""

    PUBLIC = "public"
    INTERNAL = "internal"
    RESTRICTED = "restricted"
    CLASSIFIED = "classified"


class BackendType(StrEnum):
    """Supported backend kinds."""

    LOCAL = "local"
    REMOTE = "remote"
    EDGE = "edge"


# ───────────────────────────────────────────
# Dataclasses
# ───────────────────────────────────────────


@dataclass
class ModelBackendConfig:
    """Configuration for a single model backend (local or remote).

    Supports construction from a plain ``dict`` (e.g. YAML-loaded) via
    :meth:`from_dict`.
    """

    id: str
    name: str
    type: Literal["local", "remote", "edge"]
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
    tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.model_name:
            self.model_name = self.id

    @property
    def is_local(self) -> bool:
        """Return ``True`` when this backend points to a local LLM."""
        return self.type == "local"

    @property
    def is_remote(self) -> bool:
        """Return ``True`` when this backend is a cloud API."""
        return self.type == "remote"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelBackendConfig:
        """Create a :class:`ModelBackendConfig` from a parsed YAML / dict."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict (YAML-friendly)."""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "base_url": self.base_url,
            "api_key": self.api_key if self.api_key else None,
            "model_name": self.model_name,
            "enabled": self.enabled,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "retry_count": self.retry_count,
            "cost_per_1m_input": self.cost_per_1m_input,
            "cost_per_1m_output": self.cost_per_1m_output,
            "tags": self.tags,
        }


@dataclass
class PolicyRule:
    """A single routing policy rule loaded from YAML.

    Rules are evaluated in descending ``priority`` order; the first
    matching rule wins.
    """

    id: str
    name: str
    description: str = ""
    conditions: dict[str, Any] = field(default_factory=dict)
    target_model: str = ""
    priority: int = 0
    enabled: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PolicyRule:
        """Create a :class:`PolicyRule` from a parsed YAML / dict."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict (YAML-friendly)."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "conditions": self.conditions,
            "target_model": self.target_model,
            "priority": self.priority,
            "enabled": self.enabled,
        }


@dataclass
class GuardrailConfig:
    """Granular guardrail configuration."""

    pii_redact: bool = True
    pii_max_tokens: int = 4096
    pii_patterns: list[str] = field(
        default_factory=lambda: [
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # email
            r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",  # US phone
            r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b",  # credit card
            r"sk-[a-zA-Z0-9]{20,}",  # API key
        ]
    )
    abuse_block_threshold: float = 0.8
    safety_enabled: bool = True
    safety_categories: list[str] = field(
        default_factory=lambda: [
            "hate",
            "harassment",
            "self-harm",
            "sexual",
            "violence",
            "political",
        ]
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "pii_redact": self.pii_redact,
            "pii_max_tokens": self.pii_max_tokens,
            "pii_patterns": self.pii_patterns,
            "abuse_block_threshold": self.abuse_block_threshold,
            "safety_enabled": self.safety_enabled,
            "safety_categories": self.safety_categories,
        }


@dataclass
class RateLimitConfig:
    """Rate-limiting parameters."""

    enabled: bool = True
    rpm: int = 60
    tpm: int = 60_000
    burst_multiplier: float = 1.2

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "rpm": self.rpm,
            "tpm": self.tpm,
            "burst_multiplier": self.burst_multiplier,
        }


# ───────────────────────────────────────────
# Settings
# ───────────────────────────────────────────


class GatewaySettings(BaseSettings):
    """Top-level gateway configuration.

    Loaded from ``.env`` with ``ROUTER_`` prefix and optional YAML files
    for models and policies.  Extra env keys are silently ignored
    (``extra="ignore"``).

    Example .env::

        ROUTER_DEFAULT_STRATEGY=policy
        ROUTER_DEFAULT_MODEL=llama-3.1-8b
        ROUTER_OTLP_ENABLED=true
        ROUTER_OTLP_ENDPOINT=http://localhost:4318/v1/traces
        ROUTER_RATE_LIMIT_RPM=120
        ROUTE_RATE_LIMIT_TPM=120000
    """

    # ── Core ──────────────────────────────────
    default_strategy: RoutingStrategy = RoutingStrategy.POLICY
    default_model: str = "llama-3.1-8b"

    # ── Model Backends ────────────────────────
    models_dir: str = "profiles"
    models: list[ModelBackendConfig] = field(default_factory=list)

    # ── Routing ───────────────────────────────
    policies_dir: str = "agent-policies"
    default_policy: str = "default"

    # ── Guardrails ────────────────────────────
    guardrails_input: bool = True
    guardrails_output: bool = True
    guardrails: GuardrailConfig = field(default_factory=GuardrailConfig)
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)

    # ── Observability ─────────────────────────
    otlp_enabled: bool = True
    otlp_endpoint: str = "http://localhost:4318/v1/traces"
    otlp_protocol: str = "http/protobuf"

    # ── Logging ───────────────────────────────
    log_dir: str = "logs"
    log_level: str = "INFO"

    # ── Server ────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = field(default_factory=lambda: ["*"])

    model_config = {
        "env_prefix": "ROUTER_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    # ───────────────────────────────────────────
    # YAML / File helpers
    # ───────────────────────────────────────────

    def load_models_from_yaml(self, path: str | Path | None = None) -> list[ModelBackendConfig]:
        """Load model backends from a YAML file.

        The file is expected to contain a list of mapping objects, e.g.:

        .. code-block:: yaml

            - id: llama-8b
              name: "Llama 3.1 8B Instruct"
              type: local
              base_url: http://localhost:8080/v1
              model_name: llama-3.1-8b
              temperature: 0.2
              max_tokens: 4096
            - id: gpt-4o
              name: "OpenAI GPT-4o"
              type: remote
              base_url: https://api.openai.com/v1
              api_key: "${OPENAI_API_KEY}"
              model_name: gpt-4o
        """
        path = Path(path) if path else Path(self.models_dir)
        models: list[ModelBackendConfig] = []
        if not path.exists():
            return models
        for f in sorted(path.glob("*.yaml")) + sorted(path.glob("*.yml")):
            with open(f, encoding="utf-8") as fh:
                for raw in yaml.safe_load(fh) or []:
                    models.append(ModelBackendConfig.from_dict(raw))
        return models

    def save_models_to_yaml(
        self,
        models: list[ModelBackendConfig],
        path: str | Path | None = None,
    ) -> Path:
        """Write the full model list back to ``profiles/models.yaml``."""
        path = Path(path) if path else Path(self.models_dir, "models.yaml")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            yaml.dump([m.to_dict() for m in models], fh, default_flow_style=False, sort_keys=False)
        return path

    def load_policies_from_yaml(self, policy_name: str | None = None) -> list[PolicyRule]:
        """Load policy rules from ``<policies_dir>/<policy_name>.yaml``.

        Expected YAML shape::

            rules:
              - id: pii-detected
                name: "Block PII"
                conditions:
                  has_pii: true
                target_model: "gpt-4o"
                priority: 10
        """
        policy_name = policy_name or self.default_policy
        patterns = [
            Path(self.policies_dir) / f"{policy_name}.yaml",
            Path(self.policies_dir) / f"{policy_name}.yml",
        ]
        for p in patterns:
            if not p.exists():
                continue
            with open(p, encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            return [PolicyRule.from_dict(r) for r in data.get("rules", [])]
        return []

    def save_policies_to_yaml(
        self,
        rules: list[PolicyRule],
        policy_name: str | None = None,
    ) -> Path:
        """Write policy rules to YAML."""
        policy_name = policy_name or self.default_policy
        path = Path(self.policies_dir) / f"{policy_name}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            yaml.dump({"rules": [r.to_dict() for r in rules]}, fh, sort_keys=False)
        return path

    # ───────────────────────────────────────────
    # Helpers / resolved values
    # ───────────────────────────────────────────

    @property
    def rate_limit_rpm(self) -> int:
        """Backwards-compatible access to ``rate_limit.rpm``."""
        return self.rate_limit.rpm

    @rate_limit_rpm.setter
    def rate_limit_rpm(self, value: int) -> None:
        self.rate_limit.rpm = value

    @property
    def rate_limit_tpm(self) -> int:
        """Backwards-compatible access to ``rate_limit.tpm``."""
        return self.rate_limit.tpm

    @rate_limit_tpm.setter
    def rate_limit_tpm(self, value: int) -> None:
        self.rate_limit.tpm = value

    @property
    def abuse_block_threshold(self) -> float:
        """Backwards-compatible access to ``guardrails.abuse_block_threshold``."""
        return self.guardrails.abuse_block_threshold

    @abuse_block_threshold.setter
    def abuse_block_threshold(self, value: float) -> None:
        self.guardrails.abuse_block_threshold = value

    @property
    def pii_redact(self) -> bool:
        """Backwards-compatible access to ``guardrails.pii_redact``."""
        return self.guardrails.pii_redact

    @pii_redact.setter
    def pii_redact(self, value: bool) -> None:
        self.guardrails.pii_redact = value

    @property
    def pii_max_tokens(self) -> int:
        """Backwards-compatible access to ``guardrails.pii_max_tokens``."""
        return self.guardrails.pii_max_tokens

    @pii_max_tokens.setter
    def pii_max_tokens(self, value: int) -> None:
        self.guardrails.pii_max_tokens = value

    def model_by_id(self, model_id: str) -> ModelBackendConfig | None:
        """Look up a backend configuration by ``id``."""
        for m in self.models:
            if m.id == model_id:
                return m
        return None

    def enabled_models(self) -> list[ModelBackendConfig]:
        """Return only the enabled backends."""
        return [m for m in self.models if m.enabled]

    def local_models(self) -> list[ModelBackendConfig]:
        """Return only local (LlamaCPP / Ollama) backends."""
        return [m for m in self.models if m.is_local]

    def remote_models(self) -> list[ModelBackendConfig]:
        """Return only remote (OpenAI-compatible API) backends."""
        return [m for m in self.models if m.is_remote]

    def enable_all(self) -> None:
        """Enable every backend."""
        for m in self.models:
            m.enabled = True

    def disable_all(self) -> None:
        """Disable every backend."""
        for m in self.models:
            m.enabled = False
