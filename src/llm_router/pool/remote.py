"""Remote model backend (OpenAI, Anthropic via litellm, etc.)."""

from __future__ import annotations

import time
from typing import AsyncIterator

from llm_router.config import ModelBackendConfig
from llm_router.pool.base import GenerateResult, HealthStatus, ModelBackend, UsageInfo


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
            from langchain_openai import ChatOpenAI

            self._client = ChatOpenAI(
                model=self.config.model_name,
                base_url=self.config.base_url,
                api_key=self.config.api_key,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                timeout=self.config.timeout,
            )

    def _calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Estimate cost based on model configuration."""
        input_cost = prompt_tokens / 1_000_000 * self.config.cost_per_1m_input
        output_cost = completion_tokens / 1_000_000 * self.config.cost_per_1m_output
        return round(input_cost + output_cost, 10)

    async def generate(
        self,
        messages: list[dict],
        temperature: float = 0.3,
        max_tokens: int = 4096,
        **kwargs,
    ) -> GenerateResult:
        start = time.perf_counter()
        self._ensure_client()
        from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

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

        response = await self._client.ainvoke(lc_messages, timeout=self.config.timeout)
        usage = response.response_metadata.get("token_usage", {})

        elapsed = (time.perf_counter() - start) * 1000
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        return GenerateResult(
            content=response.content,
            model=self.config.model_name,
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                cost=self._calculate_cost(prompt_tokens, completion_tokens),
            ),
            finish_reason="stop",
            latency_ms=round(elapsed, 2),
        )

    async def generate_stream(
        self,
        messages: list[dict],
        temperature: float = 0.3,
        **kwargs,
    ) -> AsyncIterator[GenerateResult]:
        self._ensure_client()
        from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

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

        async for chunk in self._client.astream(lc_messages, timeout=self.config.timeout):
            content = chunk.content if hasattr(chunk, "content") else ""
            yield GenerateResult(
                content=content,
                model=self.config.model_name,
                usage=UsageInfo(prompt_tokens=0, completion_tokens=0, total_tokens=0),
                finish_reason="incomplete",
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
