"""Main router engine — orchestrates guardrails, routing, and model calls."""

from __future__ import annotations

import logging
import time
from typing import AsyncIterator

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
    ) -> None:
        self.pool = pool
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

    async def generate(
        self,
        messages: list[dict],
        user_id: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ) -> GenerateResult:
        """Run a non-streaming completion with full guardrails and tracing."""
        request_id = f"{user_id or 'anonymous'}:{int(time.time())}"

        # 1. Rate limit check
        rate_result = await self.rate_limiter.check(client_id=request_id, tokens=100)
        if not rate_result.allowed:
            logger.warning("[%s] Rate limited: %s", request_id, rate_result.error)
            raise Exception(f"Rate limited: {rate_result.error}")

        # 2. Input PII filter
        full_text = " ".join(msg.get("content", "") for msg in messages)
        pii_result = self.pii_filter.check(full_text, mode="input")
        if pii_result.has_pii:
            logger.info("[%s] PII detected: %s", request_id, pii_result.patterns)

        # 3. Input abuse filter
        abuse_result = self.abuse_filter.check(full_text)
        if not abuse_result.safe:
            logger.warning("[%s] Abuse detected (score=%.2f): %s", request_id, abuse_result.abuse_score, abuse_result.categories)
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

        selected_model = model or routing_result.model_id or self.default_model

        # 5. Model call
        try:
            backend = self.pool.get(selected_model)
            result = await backend.generate(messages)
            logger.info(
                "[%s] Request complete: model=%s tokens=%d latency=%.0fms",
                request_id, selected_model, result.usage.total_tokens, result.latency_ms,
            )
            return result
        except Exception as exc:
            logger.error("[%s] Model call failed: %s", request_id, exc)
            raise

    async def generate_stream(
        self,
        messages: list[dict],
        user_id: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ) -> AsyncIterator[GenerateResult]:
        """Run a streaming completion with full guardrails and tracing."""
        request_id = f"{user_id or 'anonymous'}:{int(time.time())}"

        # 1. Rate limit check
        rate_result = await self.rate_limiter.check(client_id=request_id, tokens=100)
        if not rate_result.allowed:
            raise Exception(f"Rate limited: {rate_result.error}")

        # 2. Input PII filter
        full_text = " ".join(msg.get("content", "") for msg in messages)
        self.pii_filter.check(full_text, mode="input")

        # 3. Input abuse filter
        abuse_result = self.abuse_filter.check(full_text)
        if not abuse_result.safe:
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

        selected_model = model or routing_result.model_id or self.default_model

        # 5. Model call
        backend = self.pool.get(selected_model)
        async for chunk in backend.generate_stream(messages):
            yield chunk
