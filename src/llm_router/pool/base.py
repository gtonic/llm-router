"""Abstract base class for all model backends."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass

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
        normalized_call = {
            "id": call.get("id") or f"call_{index}",
            "type": "function",
            "function": {"name": call.get("name") or "", "arguments": arguments},
        }
        if "index" in call:
            normalized_call["_stream_index"] = call["index"]
        normalized.append(normalized_call)
    return normalized


def merge_tool_calls(existing: list[dict], incoming: list[dict] | None) -> None:
    """Merge streamed tool-call fragments by explicit stream index when available."""
    if not incoming:
        return
    for position, call in enumerate(incoming):
        stream_index = call.get("_stream_index")
        if stream_index is None:
            target = existing[position] if position < len(existing) else None
        else:
            target = next(
                (item for item in existing if item.get("_stream_index") == stream_index),
                None,
            )
        if target is None:
            existing.append(call)
            continue
        target_function = target.setdefault("function", {})
        incoming_function = call.get("function", {})
        if not target.get("id") or target["id"] == f"call_{position}":
            target["id"] = call.get("id", target.get("id"))
        if incoming_function.get("name"):
            target_function["name"] = incoming_function["name"]
        target_function["arguments"] += incoming_function.get("arguments", "")


def to_langchain_tool_calls(tool_calls: list[dict] | None) -> list[dict]:
    """Convert OpenAI tool calls from a follow-up request to LangChain form."""
    converted = []
    for call in tool_calls or []:
        function = call.get("function", {})
        arguments = function.get("arguments", "{}")
        try:
            arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
        except json.JSONDecodeError:
            arguments = {}
        converted.append(
            {
                "name": function.get("name", ""),
                "args": arguments,
                "id": call.get("id", ""),
                "type": "tool_call",
            }
        )
    return converted


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


class EmptyResponseError(RuntimeError):
    """Raised when a backend reports success without assistant content."""


class ModelBackend(ABC):
    """Abstract base class for all model backends.

    All concrete backends (local, remote, edge) must implement
    :meth:`generate`, :meth:`generate_stream`, and :meth:`health_check`.
    """

    def __init__(self, config: ModelBackendConfig) -> None:
        self.config = config

    def _calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Estimate request cost (USD) from the backend's per-1M token prices."""
        input_cost = prompt_tokens / 1_000_000 * self.config.cost_per_1m_input
        output_cost = completion_tokens / 1_000_000 * self.config.cost_per_1m_output
        return round(input_cost + output_cost, 10)

    def _usage_chunk(self, usage_metadata: dict) -> GenerateResult:
        """Build a content-less terminal stream chunk carrying token usage + cost.

        ``usage_metadata`` is LangChain's dict form (input_tokens / output_tokens
        / total_tokens). Streaming callers read the usage; the text is empty.
        """
        prompt_tokens = int(usage_metadata.get("input_tokens", 0) or 0)
        completion_tokens = int(usage_metadata.get("output_tokens", 0) or 0)
        total_tokens = int(usage_metadata.get("total_tokens", 0) or 0) or (prompt_tokens + completion_tokens)
        return GenerateResult(
            content="",
            model=self.config.model_name,
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                cost=self._calculate_cost(prompt_tokens, completion_tokens),
            ),
            finish_reason="stop",
            latency_ms=0,
        )

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
        on_retry=None,
        **kwargs,
    ) -> GenerateResult:
        """Call :meth:`generate` with exponential-backoff retry.

        ``on_retry`` (if given) is called once before each backoff sleep, so the
        caller can record retry metrics without the pool layer importing them.
        """
        import asyncio

        last_error: Exception | None = None
        for attempt in range(max_retries):
            try:
                return await self.generate(messages, **kwargs)
            except Exception as exc:
                last_error = exc
                status_code = getattr(exc, "status_code", None)
                if status_code in {400, 401, 403, 404}:
                    break
                if attempt < max_retries - 1:
                    if on_retry is not None:
                        on_retry()
                    await asyncio.sleep(min(0.25 * (2**attempt), 5.0))
        assert last_error is not None
        raise last_error
