"""LlamaCPP / Ollama local backend.

Connects to a local LLM served via an OpenAI-compatible HTTP API
(e.g. llama.cpp server, Ollama, vLLM).
"""

from __future__ import annotations

import os
import time
from collections.abc import AsyncIterator

from llm_router.config import ModelBackendConfig
from llm_router.pool.base import (
    GenerateResult,
    HealthStatus,
    ModelBackend,
    UsageInfo,
    merge_tool_calls,
    normalize_tool_calls,
    to_langchain_tool_calls,
)


def _chat_template_kwargs(kwargs: dict) -> dict:
    """Chat-template kwargs for the local model.

    Defaults ``enable_thinking`` to False so Qwen3.x does not emit ``<think>``
    blocks (the local model is always served thinking-off in this deployment).
    ``reasoning_format: none`` only affects how reasoning is *returned*, not
    whether it is generated — only this template kwarg suppresses generation.
    Any client-supplied ``chat_template_kwargs`` (if threaded through) wins.
    """
    ctk = {"enable_thinking": False}
    client_ctk = kwargs.get("chat_template_kwargs")
    if isinstance(client_ctk, dict):
        ctk.update(client_ctk)
    return ctk


def _expose_reasoning() -> bool:
    """Whether to surface a model's ``reasoning_content`` as user-visible output.

    Off by default: a model that emits empty ``content`` with populated
    ``reasoning_content`` would otherwise leak its raw chain-of-thought as the
    answer. Set ``ROUTER_EXPOSE_REASONING=true`` for models that legitimately
    return the response only in the reasoning field.
    """
    return os.environ.get("ROUTER_EXPOSE_REASONING", "false").lower() == "true"


def _response_content(response) -> str:
    """Return assistant text from standard (or, if opted in, reasoning-only) responses."""
    content = getattr(response, "content", "")
    if isinstance(content, str) and content:
        return content

    if _expose_reasoning():
        for metadata in (
            getattr(response, "additional_kwargs", None),
            getattr(response, "response_metadata", None),
        ):
            if isinstance(metadata, dict):
                reasoning = metadata.get("reasoning_content")
                if isinstance(reasoning, str) and reasoning:
                    return reasoning
    return content if isinstance(content, str) else ""


class LlamaCPPBackend(ModelBackend):
    """Local LLM backend talking to Llama.cpp / Ollama.

    Uses ``langchain_openai.ChatOpenAI`` under the hood so any
    OpenAI-compatible local server works out of the box.
    """

    def __init__(self, config: ModelBackendConfig) -> None:
        super().__init__(config)
        self._client = None  # Lazy-init on first call
        self._http_client = None

    def _ensure_client(self):
        if self._client is None:
            import httpx
            from langchain_openai import ChatOpenAI

            self._client = ChatOpenAI(
                model=self.config.model_name,
                base_url=self.config.base_url,
                api_key=self.config.api_key or "sk-not-required",
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                timeout=httpx.Timeout(
                    self.config.read_timeout,
                    connect=self.config.connect_timeout,
                ),
            )

    async def generate(
        self,
        messages: list[dict],
        temperature: float = 0.3,
        max_tokens: int = 4096,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> GenerateResult:
        start = time.perf_counter()
        self._ensure_client()
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

        lc_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content") or ""
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "user":
                lc_messages.append(HumanMessage(content=content))
            elif role == "tool":
                lc_messages.append(ToolMessage(content=content, tool_call_id=msg.get("tool_call_id", "")))
            else:
                lc_messages.append(
                    AIMessage(content=content, tool_calls=to_langchain_tool_calls(msg.get("tool_calls")))
                )

        client = self._client.bind_tools(tools) if tools else self._client
        invoke_kwargs = {
            "timeout": self.config.timeout,
            "extra_body": {"reasoning_format": "none", "chat_template_kwargs": _chat_template_kwargs(kwargs)},
        }
        if max_tokens is not None:
            invoke_kwargs["max_tokens"] = max_tokens
        response = await client.ainvoke(lc_messages, **invoke_kwargs)

        elapsed = (time.perf_counter() - start) * 1000
        usage_data = getattr(response, "response_metadata", {}).get(
            "token_usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        )
        tool_calls = normalize_tool_calls(getattr(response, "tool_calls", None))
        return GenerateResult(
            content=_response_content(response),
            model=self.config.model_name,
            usage=UsageInfo(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
            ),
            finish_reason="tool_calls" if tool_calls else "stop",
            tool_calls=tool_calls,
            latency_ms=round(elapsed, 2),
        )

    async def generate_stream(
        self,
        messages: list[dict],
        temperature: float = 0.3,
        tools: list[dict] | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ) -> AsyncIterator[GenerateResult]:
        self._ensure_client()
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

        lc_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content") or ""
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "user":
                lc_messages.append(HumanMessage(content=content))
            elif role == "tool":
                lc_messages.append(ToolMessage(content=content, tool_call_id=msg.get("tool_call_id", "")))
            else:
                lc_messages.append(
                    AIMessage(content=content, tool_calls=to_langchain_tool_calls(msg.get("tool_calls")))
                )

        client = self._client.bind_tools(tools) if tools else self._client
        streamed_tool_calls = []
        stream_kwargs = {
            "timeout": self.config.timeout,
            "extra_body": {"reasoning_format": "none", "chat_template_kwargs": _chat_template_kwargs(kwargs)},
        }
        if max_tokens is not None:
            stream_kwargs["max_tokens"] = max_tokens
        async for chunk in client.astream(lc_messages, **stream_kwargs):
            content = _response_content(chunk)
            tool_calls = normalize_tool_calls(getattr(chunk, "tool_call_chunks", None))
            merge_tool_calls(streamed_tool_calls, tool_calls)
            yield GenerateResult(
                content=content if isinstance(content, str) else "",
                model=self.config.model_name,
                usage=UsageInfo(prompt_tokens=0, completion_tokens=0, total_tokens=0),
                finish_reason="incomplete",
                latency_ms=0,
            )
        if streamed_tool_calls:
            for tool_call in streamed_tool_calls:
                tool_call.pop("_stream_index", None)
            yield GenerateResult(
                content="",
                model=self.config.model_name,
                usage=UsageInfo(prompt_tokens=0, completion_tokens=0, total_tokens=0),
                finish_reason="tool_calls",
                tool_calls=streamed_tool_calls,
                latency_ms=0,
            )

    async def health_check(self) -> HealthStatus:
        start = time.perf_counter()
        try:
            import httpx

            async with httpx.AsyncClient(timeout=5) as http:
                base_url = self.config.base_url.rstrip("/")
                models_url = f"{base_url}/models" if base_url.endswith("/v1") else f"{base_url}/v1/models"
                resp = await http.get(models_url)
                if resp.status_code == 200:
                    latency = (time.perf_counter() - start) * 1000
                    return HealthStatus(healthy=True, latency_ms=round(latency, 2))
                return HealthStatus(healthy=False, latency_ms=0, error=f"HTTP {resp.status_code}")
        except Exception as exc:
            return HealthStatus(healthy=False, latency_ms=0, error=str(exc))
