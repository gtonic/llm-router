"""Remote model backend (OpenAI, Anthropic via litellm, etc.)."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator

from llm_router.config import ModelBackendConfig
from llm_router.pool.base import (
    EmptyResponseError,
    GenerateResult,
    HealthStatus,
    ModelBackend,
    UsageInfo,
    merge_tool_calls,
    normalize_tool_calls,
    to_langchain_tool_calls,
)


class RemoteBackend(ModelBackend):
    """Remote LLM backend via OpenAI-compatible API.

    Supports OpenAI, Azure OpenAI, Together AI, and any provider
    with an OpenAI-compatible endpoint.
    """

    def __init__(self, config: ModelBackendConfig) -> None:
        super().__init__(config)
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            import httpx
            from langchain_openai import ChatOpenAI

            self._client = ChatOpenAI(
                model=self.config.model_name,
                base_url=self.config.base_url,
                api_key=self.config.api_key,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                # Ask the API to include token usage on the final stream chunk so
                # streaming requests can be accounted (tokens + cost), not just
                # non-streaming ones.
                stream_usage=True,
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
        invoke_kwargs = {"timeout": self.config.timeout}
        if max_tokens is not None:
            invoke_kwargs["max_tokens"] = max_tokens
        response = await client.ainvoke(lc_messages, **invoke_kwargs)
        usage = response.response_metadata.get("token_usage", {})

        elapsed = (time.perf_counter() - start) * 1000
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        tool_calls = normalize_tool_calls(getattr(response, "tool_calls", None))
        if (not isinstance(response.content, str) or not response.content.strip()) and not tool_calls:
            raise EmptyResponseError(f"Backend '{self.config.id}' returned empty content")

        return GenerateResult(
            content=response.content if isinstance(response.content, str) else "",
            model=self.config.model_name,
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                cost=self._calculate_cost(prompt_tokens, completion_tokens),
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
        usage_metadata: dict | None = None
        stream_kwargs = {"timeout": self.config.timeout}
        if max_tokens is not None:
            stream_kwargs["max_tokens"] = max_tokens
        async for chunk in client.astream(lc_messages, **stream_kwargs):
            content = chunk.content if hasattr(chunk, "content") else ""
            tool_calls = normalize_tool_calls(getattr(chunk, "tool_call_chunks", None))
            merge_tool_calls(streamed_tool_calls, tool_calls)
            chunk_usage = getattr(chunk, "usage_metadata", None)
            if isinstance(chunk_usage, dict):
                usage_metadata = chunk_usage
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
        # Terminal accounting chunk with real token usage + cost, if the API
        # reported it (stream_usage). Content-less: consumers read usage, not text.
        if usage_metadata:
            yield self._usage_chunk(usage_metadata)

    async def health_check(self) -> HealthStatus:
        start = time.perf_counter()
        try:
            import httpx

            timeout = httpx.Timeout(
                self.config.health_timeout,
                connect=min(self.config.connect_timeout, self.config.health_timeout),
            )
            async with httpx.AsyncClient(timeout=timeout) as http:
                base_url = self.config.base_url.rstrip("/")
                models_url = f"{base_url}/models" if base_url.endswith("/v1") else f"{base_url}/v1/models"
                headers = {"Authorization": f"Bearer {self.config.api_key}"} if self.config.api_key else {}
                resp = await http.get(models_url, headers=headers)
                if resp.status_code == 200:
                    latency = (time.perf_counter() - start) * 1000
                    return HealthStatus(healthy=True, latency_ms=round(latency, 2))
                return HealthStatus(healthy=False, latency_ms=0, error=f"HTTP {resp.status_code}")
        except Exception as exc:
            return HealthStatus(healthy=False, latency_ms=0, error=str(exc))
