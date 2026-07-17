"""Model pool — manages all available model backends."""

from __future__ import annotations

import asyncio
import os
import re
import time
from urllib.parse import urlparse

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

    def __init__(self, models_dir: str = "profiles", strict_config: bool = False) -> None:
        self._backends: dict[str, ModelBackend] = {}
        self._models_dir = models_dir
        self._strict_config = strict_config
        self._health_cache: tuple[float, dict[str, HealthStatus]] | None = None
        self._load_configs()

    def reload(self) -> None:
        """Reload all backend configs from disk, replacing the current set."""
        self._backends = {}
        self._health_cache = None
        self._load_configs()

    def _load_configs(self) -> None:
        """Load model configs from YAML files."""
        if not os.path.isdir(self._models_dir):
            return
        filenames = sorted(
            filename
            for filename in os.listdir(self._models_dir)
            if filename.endswith(".yaml") or filename.endswith(".yml")
        )
        if "models.yaml" in filenames:
            filenames = ["models.yaml"]
        for filename in filenames:
            self._load_config_file(os.path.join(self._models_dir, filename))

    def _load_config_file(self, filepath: str) -> None:
        with open(filepath) as f:
            configs = yaml.safe_load(f)
        if not isinstance(configs, list):
            configs = [configs]
        for cfg_dict in configs:
            # Substitute environment variables in the config
            resolved = _substitute_env_vars(cfg_dict)
            if self._strict_config and any(isinstance(value, str) and "${" in value for value in resolved.values()):
                raise ValueError(f"Unresolved environment variable in {filepath}")
            cfg = ModelBackendConfig(**resolved)
            if self._strict_config and cfg.type == "remote" and not cfg.api_key:
                raise ValueError(f"Remote backend '{cfg.id}' requires an API key")
            parsed_url = urlparse(cfg.base_url)
            if self._strict_config and (parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc):
                raise ValueError(f"Backend '{cfg.id}' has invalid base_url")
            if cfg.id in self._backends:
                raise ValueError(f"Duplicate model backend id: {cfg.id}")
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
        backend = self._backends.get(model_id)
        if backend is None or not backend.config.enabled:
            raise KeyError(f"Model '{model_id}' not found or disabled. Available: {self.list_models()}")
        return backend

    def get_or_default(self, model_id: str, default_id: str = "llama-local") -> ModelBackend:
        """Get backend by ID or fall back to default."""
        requested = self._backends.get(model_id)
        if requested is not None and requested.config.enabled:
            return requested
        default = self._backends.get(default_id)
        return default if default is not None and default.config.enabled else None

    def list_models(self) -> list[str]:
        """Return all available model IDs."""
        return [model_id for model_id, backend in self._backends.items() if backend.config.enabled]

    def list_all_models(self) -> list[str]:
        """Return all backend IDs, including disabled configurations."""
        return list(self._backends)

    def replace_backend(self, config: ModelBackendConfig) -> None:
        """Rebuild and replace a backend after connection settings change."""
        self._backends[config.id] = self._create_backend(config)

    async def health_check_all(self, force: bool = False) -> dict[str, HealthStatus]:
        """Run health checks on all backends in parallel."""
        now = time.monotonic()
        if not force and self._health_cache and now - self._health_cache[0] < 15:
            return self._health_cache[1]
        coros = {mid: backend.health_check() for mid, backend in self._backends.items()}
        results_list = await asyncio.gather(*coros.values(), return_exceptions=True)
        results: dict[str, HealthStatus] = dict(zip(coros.keys(), results_list, strict=True))
        # Convert exceptions to error HealthStatus
        for mid, result in results.items():
            if isinstance(result, Exception):
                results[mid] = HealthStatus(healthy=False, latency_ms=0, error=str(result))
        self._health_cache = (now, results)
        return results

    async def get_healthy_models(self) -> list[str]:
        """Return IDs of healthy backends (lazy health check)."""
        results = await self.health_check_all()
        return [mid for mid, status in results.items() if status.healthy]
