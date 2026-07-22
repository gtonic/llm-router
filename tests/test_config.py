"""Tests for the central configuration module."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import yaml

from llm_router.config import (
    BackendType,
    GatewaySettings,
    GuardrailConfig,
    ModelBackendConfig,
    PolicyRule,
    PrivacyLevel,
    RateLimitConfig,
    RoutingStrategy,
)

# ── Enums ───────────────────────────────────────────


class TestRoutingStrategy:
    def test_all_values_present(self):
        assert RoutingStrategy.POLICY == "policy"
        assert RoutingStrategy.COMPLEXITY == "complexity"
        assert RoutingStrategy.HYBRID == "hybrid"
        assert RoutingStrategy.ROUND_ROBIN == "round_robin"
        assert RoutingStrategy.LATENCY == "latency"
        assert RoutingStrategy.COST == "cost"


class TestPrivacyLevel:
    def test_all_values_present(self):
        assert PrivacyLevel.PUBLIC == "public"
        assert PrivacyLevel.INTERNAL == "internal"
        assert PrivacyLevel.RESTRICTED == "restricted"
        assert PrivacyLevel.CLASSIFIED == "classified"


class TestBackendType:
    def test_all_values_present(self):
        assert BackendType.LOCAL == "local"
        assert BackendType.REMOTE == "remote"
        assert BackendType.EDGE == "edge"


# ── ModelBackendConfig ─────────────────────────────


class TestModelBackendConfig:
    def test_minimal(self):
        cfg = ModelBackendConfig(id="m1", name="M1", type="local", base_url="http://x")
        assert cfg.id == "m1"
        assert cfg.model_name == "m1"  # __post_init__ fallback
        assert cfg.enabled is True
        assert cfg.is_local is True
        assert cfg.is_remote is False

    def test_full(self):
        cfg = ModelBackendConfig(
            id="gpt",
            name="GPT-4",
            type="remote",
            base_url="https://api.openai.com/v1",
            api_key="sk-test",
            model_name="gpt-4",
            temperature=0.7,
            max_tokens=8192,
            cost_per_1m_input=0.03,
            cost_per_1m_output=0.06,
            tags=["production", "gpt"],
        )
        assert cfg.is_remote
        assert not cfg.is_local

    def test_from_dict(self):
        data = {"id": "llama", "name": "Llama", "type": "local", "base_url": "http://localhost:8080"}
        cfg = ModelBackendConfig.from_dict(data)
        assert cfg.id == "llama"

    def test_to_dict_roundtrip(self):
        cfg = ModelBackendConfig(id="m", name="M", type="remote", base_url="http://x")
        d = cfg.to_dict()
        assert d["id"] == "m"
        assert d["type"] == "remote"

    def test_serialise_to_yaml(self):
        cfg = ModelBackendConfig(id="t", name="T", type="local", base_url="http://x")
        yaml_str = yaml.dump(cfg.to_dict())
        loaded = yaml.safe_load(yaml_str)
        restored = ModelBackendConfig.from_dict(loaded)
        assert restored.id == "t"


# ── PolicyRule ─────────────────────────────────────


class TestPolicyRule:
    def test_defaults(self):
        rule = PolicyRule(id="r1", name="Rule 1", target_model="gpt-4")
        assert rule.enabled is True
        assert rule.priority == 0
        assert rule.conditions == {}

    def test_from_dict(self):
        data = {
            "id": "pii",
            "name": "Block PII",
            "conditions": {"has_pii": True},
            "target_model": "gpt-4o",
            "priority": 10,
        }
        rule = PolicyRule.from_dict(data)
        assert rule.target_model == "gpt-4o"
        assert rule.priority == 10

    def test_to_dict_roundtrip(self):
        rule = PolicyRule(id="r", name="R", conditions={"key": "val"}, target_model="m", priority=5)
        d = rule.to_dict()
        assert d["conditions"]["key"] == "val"


# ── GuardrailConfig ────────────────────────────────


class TestGuardrailConfig:
    def test_defaults(self):
        gc = GuardrailConfig()
        assert gc.pii_redact is True
        assert gc.pii_max_tokens == 4096
        assert gc.abuse_block_threshold == 0.8
        assert gc.safety_enabled is True

    def test_to_dict(self):
        d = GuardrailConfig().to_dict()
        assert "pii_patterns" in d
        assert "safety_categories" in d


# ── RateLimitConfig ────────────────────────────────


class TestRateLimitConfig:
    def test_defaults(self):
        rl = RateLimitConfig()
        assert rl.rpm == 60
        assert rl.tpm == 60_000
        assert rl.enabled is True

    def test_to_dict(self):
        d = RateLimitConfig(rpm=120, tpm=120_000).to_dict()
        assert d["rpm"] == 120


# ── GatewaySettings ────────────────────────────────


class TestGatewaySettings:
    def test_defaults(self):
        gs = GatewaySettings(_env_file=None)
        assert gs.default_strategy == RoutingStrategy.COMPLEXITY
        assert gs.default_model == "llama-local"
        assert gs.fallback_model == "gpt-5.4-nano"
        assert gs.otlp_endpoint == "http://localhost:4318/v1/traces"
        assert gs.host == "0.0.0.0"
        assert gs.port == 8000

    def test_env_prefix(self):
        with patch.dict(os.environ, {"ROUTER_DEFAULT_MODEL": "custom-model", "ROUTER_PORT": "9999"}):
            gs = GatewaySettings()
            assert gs.default_model == "custom-model"
            assert gs.port == 9999

    def test_rate_limit_compat(self):
        gs = GatewaySettings(_env_file=None)  # ignore any local .env override
        assert gs.rate_limit_rpm == 60
        gs.rate_limit_rpm = 120
        assert gs.rate_limit_rpm == 120

    def test_rate_limit_env_overrides_nested_config(self):
        with patch.dict(os.environ, {"ROUTER_RATE_LIMIT_RPM": "150", "ROUTER_RATE_LIMIT_TPM": "250000"}):
            gs = GatewaySettings(_env_file=None)
            assert gs.rate_limit.rpm == 150
            assert gs.rate_limit.tpm == 250000
            assert gs.rate_limit_rpm == 150  # backwards-compat property reflects it too

    def test_env_rate_limit_wins_over_persisted_runtime_config(self, tmp_path):
        rc = tmp_path / "runtime.yaml"
        rc.write_text("rate_limit:\n  tpm: 60000\n  rpm: 60\n")
        with patch.dict(os.environ, {"ROUTER_RATE_LIMIT_TPM": "2000000"}):
            gs = GatewaySettings(_env_file=None)
            gs.load_runtime_config(rc)
            assert gs.rate_limit.tpm == 2000000  # explicit env wins over stale runtime state
            assert gs.rate_limit.rpm == 60  # no env for rpm → persisted value applies

    def test_runtime_config_applies_when_no_env_override(self, tmp_path):
        rc = tmp_path / "runtime.yaml"
        rc.write_text("rate_limit:\n  tpm: 99999\n")
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ROUTER_RATE_LIMIT_TPM", None)
            gs = GatewaySettings(_env_file=None)
            gs.load_runtime_config(rc)
            assert gs.rate_limit.tpm == 99999  # no env → runtime config applies

    def test_guardrail_compat(self):
        gs = GatewaySettings()
        assert gs.pii_redact is True
        gs.pii_redact = False
        assert gs.pii_redact is False
        assert gs.guardrails.pii_redact is False

    def test_abuse_threshold_compat(self):
        gs = GatewaySettings()
        assert gs.abuse_block_threshold == 0.8
        gs.abuse_block_threshold = 0.9
        assert gs.abuse_block_threshold == 0.9

    def test_model_by_id_found(self):
        models = [
            ModelBackendConfig(id="a", name="A", type="local", base_url="http://x"),
            ModelBackendConfig(id="b", name="B", type="remote", base_url="http://y"),
        ]
        gs = GatewaySettings(models=models)
        assert gs.model_by_id("b").name == "B"

    def test_model_by_id_missing(self):
        gs = GatewaySettings()
        assert gs.model_by_id("missing") is None

    def test_enabled_models(self):
        models = [
            ModelBackendConfig(id="on", name="On", type="local", base_url="http://x"),
            ModelBackendConfig(id="off", name="Off", type="local", base_url="http://y", enabled=False),
        ]
        gs = GatewaySettings(models=models)
        assert len(gs.enabled_models()) == 1

    def test_local_models(self):
        models = [
            ModelBackendConfig(id="l1", name="L1", type="local", base_url="http://x"),
            ModelBackendConfig(id="r1", name="R1", type="remote", base_url="http://y"),
        ]
        gs = GatewaySettings(models=models)
        assert len(gs.local_models()) == 1
        assert gs.local_models()[0].id == "l1"

    def test_remote_models(self):
        models = [
            ModelBackendConfig(id="l1", name="L1", type="local", base_url="http://x"),
            ModelBackendConfig(id="r1", name="R1", type="remote", base_url="http://y"),
        ]
        gs = GatewaySettings(models=models)
        assert len(gs.remote_models()) == 1
        assert gs.remote_models()[0].id == "r1"

    def test_enable_disable_all(self):
        models = [
            ModelBackendConfig(id="a", name="A", type="local", base_url="http://x", enabled=False),
            ModelBackendConfig(id="b", name="B", type="local", base_url="http://y", enabled=False),
        ]
        gs = GatewaySettings(models=models)
        gs.enable_all()
        assert all(m.enabled for m in gs.models)
        gs.disable_all()
        assert all(not m.enabled for m in gs.models)

    def test_load_models_from_yaml(self, tmp_path: Path):
        f = tmp_path / "models.yaml"
        f.write_text(
            yaml.dump([{"id": "x", "name": "X", "type": "local", "base_url": "http://x"}]),
            encoding="utf-8",
        )
        gs = GatewaySettings(models_dir=str(tmp_path))
        loaded = gs.load_models_from_yaml()
        assert len(loaded) == 1
        assert loaded[0].id == "x"

    def test_save_and_load_models(self, tmp_path: Path):
        models = [
            ModelBackendConfig(
                id="s",
                name="S",
                type="remote",
                base_url="https://api.openai.com/v1",
                api_key="sk-test-secret",
            )
        ]
        gs = GatewaySettings(models_dir=str(tmp_path))
        gs.save_models_to_yaml(models)
        saved = (tmp_path / "models.yaml").read_text(encoding="utf-8")
        assert "sk-test-secret" not in saved
        assert "${OPENAI_API_KEY}" in saved
        loaded = gs.load_models_from_yaml()
        assert len(loaded) == 1
        assert loaded[0].id == "s"

    def test_load_policies_from_yaml(self, tmp_path: Path):
        f = tmp_path / "default.yaml"
        f.write_text(
            yaml.dump({"rules": [{"id": "r1", "name": "R1", "target_model": "gpt", "priority": 5}]}),
            encoding="utf-8",
        )
        gs = GatewaySettings(policies_dir=str(tmp_path))
        rules = gs.load_policies_from_yaml()
        assert len(rules) == 1
        assert rules[0].name == "R1"

    def test_save_policies(self, tmp_path: Path):
        rules = [PolicyRule(id="r1", name="R1", target_model="gpt")]
        gs = GatewaySettings(policies_dir=str(tmp_path))
        gs.save_policies_to_yaml(rules)
        content = yaml.safe_load((tmp_path / "default.yaml").read_text(encoding="utf-8"))
        assert len(content["rules"]) == 1
