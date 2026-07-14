"""Tests for pool/pool.py - ModelPool management."""

from __future__ import annotations

import pytest

from llm_router.config import ModelBackendConfig
from llm_router.pool.base import HealthStatus, ModelBackend
from llm_router.pool.pool import ModelPool


class DummyBackend(ModelBackend):
    """A minimal test double for ModelBackend."""

    async def generate(self, messages, temperature=0.3, max_tokens=4096, tools=None, **kwargs):
        raise NotImplementedError

    async def generate_stream(self, messages, temperature=0.3, **kwargs):
        raise NotImplementedError

    async def health_check(self) -> HealthStatus:
        return HealthStatus(healthy=True, latency_ms=1.0)


class TestModelPoolInit:
    def test_empty_pool(self, tmp_path):
        pool = ModelPool(models_dir=str(tmp_path))
        assert pool.list_models() == []

    def test_loads_yaml_files(self, tmp_path):
        """Pool loads YAML model configs on init."""
        (tmp_path / "models.yaml").write_text(
            "- id: model-a\n  name: Model A\n  type: local\n  base_url: http://localhost:8000\n  model_name: llama\n",
            encoding="utf-8",
        )
        pool = ModelPool(models_dir=str(tmp_path))
        assert "model-a" in pool.list_models()

    def test_skips_disabled_models(self, tmp_path):
        yaml_content = (
            "- id: disabled-model\n"
            "  name: Disabled\n"
            "  type: local\n"
            "  base_url: http://x\n"
            "  model_name: x\n"
            "  enabled: false\n"
        )
        (tmp_path / "models.yaml").write_text(yaml_content, encoding="utf-8")
        pool = ModelPool(models_dir=str(tmp_path))
        assert pool.list_models() == []

    def test_no_models_dir(self):
        pool = ModelPool(models_dir="/nonexistent/path/xyz")
        assert pool.list_models() == []


class TestModelPoolManagement:
    def test_add_backend(self):
        pool = ModelPool()
        cfg = ModelBackendConfig(id="test", name="Test", type="local", base_url="http://x")
        pool.add_backend(cfg, DummyBackend(cfg))
        assert pool.get("test") is not None
        assert pool.get("test").config.id == "test"

    def test_get_raises_missing(self):
        pool = ModelPool()
        with pytest.raises(KeyError, match="not found"):
            pool.get("missing")

    def test_get_or_default(self):
        pool = ModelPool()
        default_cfg = ModelBackendConfig(id="def", name="D", type="local", base_url="http://x")
        pool.add_backend(default_cfg, DummyBackend(default_cfg))
        default = pool.get_or_default("nonexistent", "def")
        assert default is not None
        assert default.config.id == "def"

    def test_get_or_default_falls_back_to_none(self):
        pool = ModelPool()
        result = pool.get_or_default("nonexistent", "nonexistent")
        assert result is None

    def test_list_models(self, tmp_path):
        pool = ModelPool(models_dir=str(tmp_path))
        ids = ["a", "b"]
        for mid in ids:
            cfg = ModelBackendConfig(id=mid, name=mid, type="local", base_url="http://x")
            pool.add_backend(cfg, DummyBackend(cfg))
        # pool.list_models() returns all registered IDs
        added = pool.list_models()
        assert "a" in added
        assert "b" in added

    def test_list_models_empty(self, tmp_path):
        pool = ModelPool(models_dir=str(tmp_path))
        assert pool.list_models() == []


class TestModelPoolHealth:
    @pytest.mark.asyncio
    async def test_health_check_all(self, tmp_path):
        pool = ModelPool(models_dir=str(tmp_path))
        cfg1 = ModelBackendConfig(id="h1", name="H1", type="local", base_url="http://x")
        cfg2 = ModelBackendConfig(id="h2", name="H2", type="local", base_url="http://y")
        pool.add_backend(cfg1, DummyBackend(cfg1))
        pool.add_backend(cfg2, DummyBackend(cfg2))
        results = await pool.health_check_all()
        assert len(results) == 2
        for mid, status in results.items():
            assert isinstance(status, HealthStatus)
            assert mid in results

    @pytest.mark.asyncio
    async def test_health_check_all_with_exception(self, tmp_path):
        """If a backend's health_check raises, result should be an error HealthStatus."""

        class FailingBackend(ModelBackend):
            async def generate(self, messages, temperature=0.3, max_tokens=4096, tools=None, **kwargs):
                raise NotImplementedError

            async def generate_stream(self, messages, temperature=0.3, **kwargs):
                raise NotImplementedError

            async def health_check(self) -> HealthStatus:
                raise RuntimeError("boom")

        pool = ModelPool(models_dir=str(tmp_path))
        cfg = ModelBackendConfig(id="fail", name="F", type="local", base_url="http://x")
        pool.add_backend(cfg, FailingBackend(cfg))
        results = await pool.health_check_all()
        assert "fail" in results
        assert results["fail"].healthy is False
        assert "boom" in results["fail"].error

    def test_get_healthy_models(self, tmp_path):
        pool = ModelPool(models_dir=str(tmp_path))
        cfg = ModelBackendConfig(id="healthy-1", name="H1", type="local", base_url="http://x")
        pool.add_backend(cfg, DummyBackend(cfg))
        healthy = pool.get_healthy_models()
        assert "healthy-1" in healthy


class TestModelPoolCreateBackend:
    def test_create_local_backend(self, tmp_path):
        _pool = ModelPool(models_dir=str(tmp_path))
        (tmp_path / "m.yaml").write_text(
            "- id: l1\n  name: L1\n  type: local\n  base_url: http://x\n  model_name: test\n",
            encoding="utf-8",
        )
        pool2 = ModelPool(models_dir=str(tmp_path))
        backend = pool2.get("l1")
        assert isinstance(backend, ModelBackend)

    def test_raises_on_unknown_type(self, tmp_path):
        """Backend with type 'unknown' should raise ValueError."""
        (tmp_path / "bad.yaml").write_text(
            "- id: bad\n  name: Bad\n  type: unknown\n  base_url: http://x\n  model_name: test\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="Unknown model type"):
            ModelPool(models_dir=str(tmp_path))
