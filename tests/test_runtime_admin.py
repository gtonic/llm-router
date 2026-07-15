from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from llm_router.config import GatewaySettings, ModelBackendConfig
from llm_router.pool.base import GenerateResult, HealthStatus, ModelBackend, UsageInfo
from llm_router.pool.pool import ModelPool
from llm_router.routing.round_robin import RoundRobinPolicy


class FakeBackend(ModelBackend):
    async def generate(self, messages, temperature=0.3, max_tokens=4096, tools=None, **kwargs):
        return GenerateResult(
            content="ok",
            model=self.config.model_name,
            usage=UsageInfo(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            finish_reason="stop",
        )

    async def generate_stream(self, messages, temperature=0.3, tools=None, max_tokens=None, **kwargs):
        if False:
            yield None

    async def health_check(self):
        return HealthStatus(healthy=True, latency_ms=0)


def test_runtime_config_round_trip_is_atomic(tmp_path: Path):
    path = tmp_path / "runtime.yaml"
    settings = GatewaySettings(runtime_config_path=str(path))
    settings.guardrails.pii_redact = False
    settings.guardrails.safety_categories = ["custom"]
    settings.rate_limit.enabled = False
    settings.save_runtime_config()

    restored = GatewaySettings(runtime_config_path=str(path))
    restored.load_runtime_config()
    assert restored.guardrails.pii_redact is False
    assert restored.guardrails.safety_categories == ["custom"]
    assert restored.rate_limit.enabled is False
    assert yaml.safe_load(path.read_text(encoding="utf-8"))["rate_limit"]["enabled"] is False


def test_runtime_config_can_rollback_previous_snapshot(tmp_path: Path):
    path = tmp_path / "runtime.yaml"
    settings = GatewaySettings(runtime_config_path=str(path))
    settings.guardrails.pii_enabled = True
    settings.save_runtime_config()
    settings.guardrails.pii_enabled = False
    settings.save_runtime_config()

    assert settings.rollback_runtime_config() is True
    assert settings.guardrails.pii_enabled is True


def test_pool_keeps_disabled_backends_but_excludes_them_from_routing(tmp_path: Path, monkeypatch):
    models_path = tmp_path / "models.yaml"
    models_path.write_text(
        yaml.safe_dump(
            [
                {"id": "enabled", "name": "Enabled", "type": "local", "base_url": "http://enabled"},
                {
                    "id": "disabled",
                    "name": "Disabled",
                    "type": "local",
                    "base_url": "http://disabled",
                    "enabled": False,
                },
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(ModelPool, "_create_backend", staticmethod(lambda config: FakeBackend(config)))
    pool = ModelPool(models_dir=str(tmp_path))

    assert pool.list_all_models() == ["enabled", "disabled"]
    assert pool.list_models() == ["enabled"]
    with pytest.raises(KeyError):
        pool.get("disabled")


def test_round_robin_updates_models():
    policy = RoundRobinPolicy(["one", "two"])
    policy.update_models(["three"])
    assert policy.model_ids == ["three"]
    assert policy._counter == 0


def test_model_config_to_dict_never_exposes_empty_secret():
    config = ModelBackendConfig(id="m", name="M", type="remote", base_url="http://m")
    assert config.to_dict()["api_key"] is None


def test_guardrail_flags_are_explicit():
    settings = GatewaySettings()
    settings.guardrails.pii_enabled = False
    settings.guardrails.abuse_enabled = False
    payload = settings.guardrails.to_dict()
    assert payload["pii_enabled"] is False
    assert payload["abuse_enabled"] is False
