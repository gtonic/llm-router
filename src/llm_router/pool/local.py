"""LlamaCPP / Ollama local backend.

Connects to a local LLM served via an OpenAI-compatible HTTP API
(e.g. llama.cpp server, Ollama, vLLM).
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator

from llm_router.config import ModelBackendConfig
from llm_router.pool.base import GenerateResult, HealthStatus, ModelBackend, UsageInfo, normalize_tool_calls


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
            from langchain_openai import ChatOpenAI

            self._client = ChatOpenAI(
                model=self.config.model_name,
                base_url=self.config.base_url,
                api_key=self.config.api_key or "sk-not-required",
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                timeout=self.config.timeout,
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
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        lc_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "user":
                lc_messages.append(HumanMessage(content=content))
            else:
                lc_messages.append(AIMessage(content=content))

        client = self._client.bind_tools(tools) if tools else self._client
        response = await client.ainvoke(lc_messages, timeout=self.config.timeout)

        elapsed = (time.perf_counter() - start) * 1000
        usage_data = getattr(response, "response_metadata", {}).get(
            "token_usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        )
        tool_calls = normalize_tool_calls(getattr(response, "tool_calls", None))
        return GenerateResult(
            content=response.content if isinstance(response.content, str) else "",
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
        **kwargs,
    ) -> AsyncIterator[GenerateResult]:
        self._ensure_client()
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        lc_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "user":
                lc_messages.append(HumanMessage(content=content))
            else:
                lc_messages.append(AIMessage(content=content))

        client = self._client.bind_tools(tools) if tools else self._client
        async for chunk in client.astream(lc_messages, timeout=self.config.timeout):
            content = chunk.content if hasattr(chunk, "content") else ""
            tool_calls = normalize_tool_calls(getattr(chunk, "tool_call_chunks", None))
            yield GenerateResult(
                content=content if isinstance(content, str) else "",
                model=self.config.model_name,
                usage=UsageInfo(prompt_tokens=0, completion_tokens=0, total_tokens=0),
                finish_reason="tool_calls" if tool_calls else "incomplete",
                tool_calls=tool_calls,
                latency_ms=0,
            )

    async def health_check(self) -> HealthStatus:
        start = time.perf_counter()
        try:
            import httpx

            async with httpx.AsyncClient(timeout=5) as http:
                resp = await http.get(f"{self.config.base_url.rstrip('/')}/v1/models")
                if resp.status_code == 200:
                    latency = (time.perf_counter() - start) * 1000
                    return HealthStatus(healthy=True, latency_ms=round(latency, 2))
                return HealthStatus(healthy=False, latency_ms=0, error=f"HTTP {resp.status_code}")
        except Exception as exc:
            return HealthStatus(healthy=False, latency_ms=0, error=str(exc))
