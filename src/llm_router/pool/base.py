"""Abstract base class for all model backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
import json

from llm_router.config import ModelBackendConfig


def normalize_tool_calls(tool_calls: list[dict] | None) -> list[dict] | None:
    """Convert LangChain tool calls to OpenAI chat-completion tool calls."""
    if not tool_calls:
        return None
    normalized = []
    for index, call in enumerate(tool_calls):
        if call.get("type") == "function" and "function" in call:
            normalized.append(call)
            continue
        arguments = call.get("args", call.get("arguments", {}))
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, separators=(",", ":"))
        normalized.append(
            {
                "id": call.get("id") or f"call_{index}",
                "type": "function",
                "function": {"name": call.get("name", ""), "arguments": arguments},
            }
        )
    return normalized


@dataclass
class UsageInfo:
    """Token usage statistics."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: float = 0.0


@dataclass
class GenerateResult:
    """Result from a model generation call."""

    content: str
    model: str
    usage: UsageInfo
    finish_reason: str
    tool_calls: list[dict] | None = None
    latency_ms: float = 0.0


@dataclass
class HealthStatus:
    """Health check result for a model backend."""

    healthy: bool
    latency_ms: float
    error: str | None = None


class ModelBackend(ABC):
    """Abstract base class for all model backends.

    All concrete backends (local, remote, edge) must implement
    :meth:`generate`, :meth:`generate_stream`, and :meth:`health_check`.
    """

    def __init__(self, config: ModelBackendConfig) -> None:
        self.config = config

    # ── Abstract methods ──────────────────────

    @abstractmethod
    async def generate(
        self,
        messages: list[dict],
        temperature: float = 0.3,
        max_tokens: int = 4096,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> GenerateResult:
        """Run a non-streaming completion."""

    @abstractmethod
    async def generate_stream(
        self,
        messages: list[dict],
        temperature: float = 0.3,
        **kwargs,
    ) -> AsyncIterator[GenerateResult]:
        """Run a streaming completion."""

    @abstractmethod
    async def health_check(self) -> HealthStatus:
        """Check if the backend is reachable."""

    # ── Concrete helpers ──────────────────────

    async def generate_with_retry(
        self,
        messages: list[dict],
        max_retries: int = 3,
        **kwargs,
    ) -> GenerateResult:
        """Call :meth:`generate` with exponential-backoff retry."""
        import asyncio

        last_error: Exception | None = None
        for attempt in range(max_retries):
            try:
                return await self.generate(messages, **kwargs)
            except Exception as exc:
                last_error = exc
                if attempt < max_retries - 1:
                    delay = 0.2**attempt * 10
                    await asyncio.sleep(min(delay, 10))
        assert last_error is not None
        raise last_error
