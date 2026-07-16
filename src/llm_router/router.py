"""Main router engine — orchestrates guardrails, routing, and model calls."""

from __future__ import annotations

import inspect
import logging
import time
from collections.abc import AsyncIterator

from llm_router.config import GatewaySettings, RoutingStrategy
from llm_router.guardrails.abuse_filter import AbuseFilter
from llm_router.guardrails.content_safety import ContentSafety
from llm_router.guardrails.pii_filter import PiiFilter
from llm_router.guardrails.rate_limiter import RateLimiter
from llm_router.pool.base import GenerateResult
from llm_router.pool.pool import ModelPool
from llm_router.routing.complexity import ComplexityDetector
from llm_router.routing.hybrid import HybridRouter
from llm_router.routing.policy import PolicyMatcher
from llm_router.routing.round_robin import RoundRobinPolicy

logger = logging.getLogger("llm-router")


def _retry_attempts(backend) -> int:
    """Return a safe integer retry count for real and mocked backends."""
    value = getattr(backend.config, "retry_count", 1)
    return max(1, value) if isinstance(value, int) else 1


async def _generate_with_retry(backend, messages: list[dict], **kwargs):
    """Use retry support when supplied by a real backend implementation."""
    retry_method = getattr(backend, "generate_with_retry", None)
    if retry_method is not None and inspect.iscoroutinefunction(retry_method):
        return await retry_method(messages, max_retries=_retry_attempts(backend), **kwargs)
    return await backend.generate(messages, **kwargs)


def _message_text(message: dict) -> str:
    """Extract text from OpenAI string or content-part message formats."""
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            part.get("text", "") for part in content if isinstance(part, dict) and isinstance(part.get("text"), str)
        )
    return ""


def _redact_messages(messages: list[dict], pii_filter: PiiFilter) -> list[dict]:
    """Redact PII in string and OpenAI content-part messages before inference."""
    redacted = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            message = {**message, "content": pii_filter.redact_text(content)}
        elif isinstance(content, list):
            parts = [
                {
                    **part,
                    "text": pii_filter.redact_text(part["text"]),
                }
                if isinstance(part, dict) and isinstance(part.get("text"), str)
                else part
                for part in content
            ]
            message = {**message, "content": parts}
        redacted.append(message)
    return redacted


class RouterPolicyEngine:
    """Main router that orchestrates guardrails, routing, and model calls.

    Flow:
    1. Rate limit check
    2. Input PII filter
    3. Input abuse filter
    4. Policy routing decision
    5. Model call (local or remote)
    6. Output PII filter
    7. Output content safety
    8. Audit log
    """

    def __init__(
        self,
        pool: ModelPool,
        routing_strategy: RoutingStrategy,
        policy_matcher: PolicyMatcher,
        complexity_detector: ComplexityDetector,
        hybrid_router: HybridRouter,
        round_robin: RoundRobinPolicy,
        rate_limiter: RateLimiter,
        pii_filter: PiiFilter,
        abuse_filter: AbuseFilter,
        content_safety: ContentSafety,
        default_model: str = "llama-local",
        settings: GatewaySettings | None = None,
    ) -> None:
        self.pool = pool
        self.settings = settings or GatewaySettings()
        self.routing_strategy = routing_strategy
        self.policy_matcher = policy_matcher
        self.complexity_detector = complexity_detector
        self.hybrid_router = hybrid_router
        self.round_robin = round_robin
        self.rate_limiter = rate_limiter
        self.pii_filter = pii_filter
        self.abuse_filter = abuse_filter
        self.content_safety = content_safety
        self.default_model = default_model

    def sync_runtime_models(self) -> None:
        """Synchronize routing policies with currently enabled backends."""
        self.round_robin.update_models(self.pool.list_models())

    async def generate(
        self,
        messages: list[dict],
        user_id: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        tools: list[dict] | None = None,
        max_tokens: int | None = None,
    ) -> GenerateResult:
        """Run a non-streaming completion with full guardrails and tracing."""
        request_id = f"{user_id or 'anonymous'}:{int(time.time())}"

        # 1. Rate limit check
        rate_result = (
            await self.rate_limiter.check(client_id=request_id, tokens=100)
            if self.settings.rate_limit.enabled
            else None
        )
        if rate_result is not None and not rate_result.allowed:
            logger.warning("[%s] Rate limited: %s", request_id, rate_result.error)
            raise Exception(f"Rate limited: {rate_result.error}")

        # 2. Input PII filter
        full_text = " ".join(_message_text(msg) for msg in messages)
        pii_result = self.pii_filter.check(full_text, mode="input") if self.settings.guardrails.pii_enabled else None
        if pii_result is not None and pii_result.has_pii:
            logger.info("[%s] PII detected: %s", request_id, pii_result.patterns)
            messages = _redact_messages(messages, self.pii_filter)

        # 3. Input abuse filter
        abuse_result = self.abuse_filter.check(full_text) if self.settings.guardrails.abuse_enabled else None
        if abuse_result is not None and not abuse_result.safe:
            logger.warning(
                "[%s] Abuse detected (score=%.2f): %s",
                request_id,
                abuse_result.abuse_score,
                abuse_result.categories,
            )
            raise Exception(f"Abuse detected: {abuse_result.categories}")

        # 4. Routing decision
        strategy = self.routing_strategy
        if strategy == RoutingStrategy.POLICY:
            routing_result = await self.policy_matcher.route(messages)
        elif strategy == RoutingStrategy.COMPLEXITY:
            routing_result = await self.complexity_detector.route(messages)
        elif strategy == RoutingStrategy.ROUND_ROBIN:
            routing_result = await self.round_robin.route(messages)
        else:
            routing_result = await self.policy_matcher.route(messages)

        requested_model = None if model in {None, "auto", "router-auto"} else model
        selected_model = requested_model or routing_result.model_id or self.default_model
        if tools and requested_model is None:
            selected_model = self.default_model

        # 5. Model call
        try:
            backend = self.pool.get(selected_model)
            backend_kwargs = {"tools": tools} if tools else {}
            if max_tokens is not None:
                backend_kwargs["max_tokens"] = max_tokens
            result = await _generate_with_retry(backend, messages, **backend_kwargs)
            if self.pii_filter.redact:
                result.content = self.pii_filter.redact_text(result.content)
            logger.info(
                "[%s] Request complete: model=%s tokens=%d latency=%.0fms",
                request_id,
                selected_model,
                result.usage.total_tokens,
                result.latency_ms,
            )
            return result
        except Exception as exc:
            logger.error("[%s] Model call failed: %s", request_id, exc)
            fallback_model = self.settings.fallback_model
            if selected_model != fallback_model:
                fallback = self.pool.get(fallback_model)
                logger.warning("[%s] Falling back from %s to %s", request_id, selected_model, fallback_model)
                return await _generate_with_retry(fallback, messages, **backend_kwargs)
            raise

    async def generate_stream(
        self,
        messages: list[dict],
        user_id: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        tools: list[dict] | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[GenerateResult]:
        """Run a streaming completion with full guardrails and tracing."""
        request_id = f"{user_id or 'anonymous'}:{int(time.time())}"

        # 1. Rate limit check
        rate_result = (
            await self.rate_limiter.check(client_id=request_id, tokens=100)
            if self.settings.rate_limit.enabled
            else None
        )
        if rate_result is not None and not rate_result.allowed:
            raise Exception(f"Rate limited: {rate_result.error}")

        # 2. Input PII filter
        full_text = " ".join(_message_text(msg) for msg in messages)
        if self.settings.guardrails.pii_enabled:
            self.pii_filter.check(full_text, mode="input")

        # 3. Input abuse filter
        abuse_result = self.abuse_filter.check(full_text) if self.settings.guardrails.abuse_enabled else None
        if abuse_result is not None and not abuse_result.safe:
            raise Exception(f"Abuse detected: {abuse_result.categories}")

        # 4. Routing decision
        strategy = self.routing_strategy
        if strategy == RoutingStrategy.POLICY:
            routing_result = await self.policy_matcher.route(messages)
        elif strategy == RoutingStrategy.COMPLEXITY:
            routing_result = await self.complexity_detector.route(messages)
        elif strategy == RoutingStrategy.ROUND_ROBIN:
            routing_result = await self.round_robin.route(messages)
        else:
            routing_result = await self.policy_matcher.route(messages)

        requested_model = None if model in {None, "auto", "router-auto"} else model
        selected_model = requested_model or routing_result.model_id or self.default_model
        if tools and requested_model is None:
            selected_model = self.default_model

        # 5. Model call
        backend = self.pool.get(selected_model)
        backend_kwargs = {"tools": tools} if tools else {}
        if max_tokens is not None:
            backend_kwargs["max_tokens"] = max_tokens
        try:
            async for chunk in backend.generate_stream(messages, **backend_kwargs):
                if chunk.content or chunk.tool_calls:
                    yield chunk
        except Exception:
            fallback_model = self.settings.fallback_model
            if selected_model == fallback_model:
                raise
            fallback = self.pool.get(fallback_model)
            logger.warning("[%s] Streaming fallback from %s to %s", request_id, selected_model, fallback_model)
            async for chunk in fallback.generate_stream(messages, **backend_kwargs):
                if chunk.content or chunk.tool_calls:
                    yield chunk
