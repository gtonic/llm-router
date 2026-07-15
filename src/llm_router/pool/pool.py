"""Model pool — manages all available model backends."""

from __future__ import annotations

import asyncio
import os
import re

import yaml

from llm_router.config import ModelBackendConfig
from llm_router.pool.base import HealthStatus, ModelBackend
from llm_router.pool.local import LlamaCPPBackend
from llm_router.pool.remote import RemoteBackend

# Pattern to match ${VAR_NAME} placeholders in strings
_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _substitute_env_vars(value):
    """Recursively substitute ${VAR_NAME} placeholders in strings with environment variable values."""
    if isinstance(value, str):

        def replacer(match):
            var_name = match.group(1)
            return os.environ.get(var_name, match.group(0))

        return _ENV_VAR_PATTERN.sub(replacer, value)
    elif isinstance(value, dict):
        return {k: _substitute_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_substitute_env_vars(item) for item in value]
    return value


class ModelPool:
    """Manages a collection of model backends with automatic loading."""

    def __init__(self, models_dir: str = "profiles") -> None:
        self._backends: dict[str, ModelBackend] = {}
        self._models_dir = models_dir
        self._load_configs()

    def _load_configs(self) -> None:
        """Load model configs from YAML files."""
        if not os.path.isdir(self._models_dir):
            return
        for filename in sorted(os.listdir(self._models_dir)):
            if filename.endswith(".yaml") or filename.endswith(".yml"):
                filepath = os.path.join(self._models_dir, filename)
                self._load_config_file(filepath)

    def _load_config_file(self, filepath: str) -> None:
        with open(filepath) as f:
            configs = yaml.safe_load(f)
        if not isinstance(configs, list):
            configs = [configs]
        for cfg_dict in configs:
            # Substitute environment variables in the config
            resolved = _substitute_env_vars(cfg_dict)
            cfg = ModelBackendConfig(**resolved)
            if cfg.enabled:
                backend = self._create_backend(cfg)
                self._backends[cfg.id] = backend

    @staticmethod
    def _create_backend(cfg: ModelBackendConfig) -> ModelBackend:
        if cfg.type == "local":
            return LlamaCPPBackend(cfg)
        elif cfg.type == "remote":
            return RemoteBackend(cfg)
        else:
            raise ValueError(f"Unknown model type: {cfg.type}")

    def add_backend(self, config: ModelBackendConfig, backend: ModelBackend) -> None:
        """Dynamically add a backend to the pool."""
        self._backends[config.id] = backend

    def get(self, model_id: str) -> ModelBackend:
        """Get a backend by its model ID."""
        if model_id not in self._backends:
            raise KeyError(f"Model '{model_id}' not found in pool. Available: {list(self._backends.keys())}")
        return self._backends[model_id]

    def get_or_default(self, model_id: str, default_id: str = "llama-local") -> ModelBackend:
        """Get backend by ID or fall back to default."""
        return self._backends.get(model_id, self._backends.get(default_id))

    def list_models(self) -> list[str]:
        """Return all available model IDs."""
        return list(self._backends.keys())

    async def health_check_all(self) -> dict[str, HealthStatus]:
        """Run health checks on all backends in parallel."""
        coros = {mid: backend.health_check() for mid, backend in self._backends.items()}
        results_list = await asyncio.gather(*coros.values(), return_exceptions=True)
        results: dict[str, HealthStatus] = dict(zip(coros.keys(), results_list, strict=True))
        # Convert exceptions to error HealthStatus
        for mid, result in results.items():
            if isinstance(result, Exception):
                results[mid] = HealthStatus(healthy=False, latency_ms=0, error=str(result))
        return results

    def get_healthy_models(self) -> list[str]:
        """Return IDs of healthy backends (lazy health check)."""
        import asyncio

        loop = asyncio.new_event_loop()
        try:
            results = loop.run_until_complete(self.health_check_all())
            return [mid for mid, status in results.items() if status.healthy]
        finally:
            loop.close()
