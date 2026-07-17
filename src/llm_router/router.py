"""Main router engine — orchestrates guardrails, routing, and model calls."""

from __future__ import annotations

import inspect
import logging
import time
from collections.abc import AsyncIterator

from llm_router.config import GatewaySettings, RoutingStrategy
from llm_router.guardrails.abuse_filter import AbuseFilter
from llm_router.guardrails.content_safety import ContentSafety, ContentSafetyBlockedError
from llm_router.guardrails.pii_filter import PiiFilter
from llm_router.guardrails.rate_limiter import RateLimiter
from llm_router.logging_setup import AuditEntry, AuditLogger
from llm_router.pool.base import EmptyResponseError, GenerateResult
from llm_router.pool.pool import ModelPool
from llm_router.routing.base import RoutingResult
from llm_router.routing.complexity import ComplexityDetector
from llm_router.routing.hybrid import HybridRouter
from llm_router.routing.policy import PolicyMatcher
from llm_router.routing.round_robin import RoundRobinPolicy

logger = logging.getLogger("llm-router")


class RateLimitExceededError(RuntimeError):
    """Raised when a client exceeds the configured rate limits."""


class AbuseDetectedError(RuntimeError):
    """Raised when input guardrails flag a request as abusive."""


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


async def _generate_stream_with_content(backend, messages: list[dict], **kwargs) -> AsyncIterator[GenerateResult]:
    """Yield meaningful stream chunks or fail so the caller can use its fallback."""
    yielded_content = False
    async for chunk in backend.generate_stream(messages, **kwargs):
        if chunk.content or chunk.tool_calls:
            yielded_content = True
            yield chunk
    if not yielded_content:
        raise EmptyResponseError(f"Backend '{backend.config.id}' returned an empty stream")


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
        self.audit_logger = AuditLogger(self.settings.log_dir)

    def sync_runtime_models(self) -> None:
        """Synchronize routing policies with currently enabled backends."""
        self.round_robin.update_models(self.pool.list_models())

    async def _route(self, messages: list[dict], routing_messages: list[dict]) -> RoutingResult:
        """Dispatch to the sub-router matching ``routing_strategy``.

        ``routing_messages`` is passed to the policy matcher specifically so it can see
        PII-bearing content even after ``messages`` has been redacted for the model call.
        """
        strategy = self.routing_strategy
        if strategy == RoutingStrategy.POLICY:
            return await self.policy_matcher.route(routing_messages)
        if strategy == RoutingStrategy.COMPLEXITY:
            return await self.complexity_detector.route(messages)
        if strategy == RoutingStrategy.HYBRID:
            return await self.hybrid_router.route(messages)
        if strategy == RoutingStrategy.ROUND_ROBIN:
            return await self.round_robin.route(messages)
        return await self.policy_matcher.route(messages)

    def _log_audit(
        self,
        request_id: str,
        user_id: str | None,
        model_selected: str,
        status: str,
        *,
        routing_result: RoutingResult | None = None,
        pii_result=None,
        abuse_result=None,
        result: GenerateResult | None = None,
        error: str | None = None,
        latency_ms: float = 0.0,
    ) -> None:
        """Best-effort JSONL audit entry. Never raises."""
        try:
            pii_patterns = getattr(pii_result, "patterns", None) or []
            abuse_score = getattr(abuse_result, "abuse_score", 0.0) or 0.0
            entry = AuditEntry(
                request_id=request_id,
                user_id=user_id,
                model_selected=model_selected,
                routing_strategy=self.routing_strategy.value,
                status=status,
                policy_matched=getattr(routing_result, "policy_matched", None),
                guardrail_pii_detected=len(pii_patterns),
                guardrail_abuse_score=abuse_score,
                prompt_tokens=result.usage.prompt_tokens if result else 0,
                completion_tokens=result.usage.completion_tokens if result else 0,
                total_tokens=result.usage.total_tokens if result else 0,
                cost=result.usage.cost if result else 0.0,
                latency_ms=result.latency_ms if result else latency_ms,
                error=error,
            )
            self.audit_logger.log(entry)
        except Exception:
            logger.debug("[%s] Failed to build audit entry", request_id, exc_info=True)

    async def generate(
        self,
        messages: list[dict],
        user_id: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        tools: list[dict] | None = None,
        max_tokens: int | None = None,
        client_ip: str | None = None,
    ) -> GenerateResult:
        """Run a non-streaming completion with full guardrails and tracing."""
        request_id = f"{user_id or 'anonymous'}:{int(time.time())}"
        # A stable identifier for rate limiting — request_id above is intentionally unique
        # per request (for logs/tracing) and must never be used as the rate-limit bucket key.
        client_id = user_id or api_key or client_ip or "anonymous"
        start = time.perf_counter()

        # 1. Rate limit check
        rate_result = (
            await self.rate_limiter.check(client_id=client_id, tokens=100) if self.settings.rate_limit.enabled else None
        )
        if rate_result is not None and not rate_result.allowed:
            logger.warning("[%s] Rate limited: %s", request_id, rate_result.error)
            self._log_audit(request_id, user_id, "", "rate_limited", error=rate_result.error)
            raise RateLimitExceededError(f"Rate limited: {rate_result.error}")

        # 2. Input PII filter
        full_text = " ".join(_message_text(msg) for msg in messages)
        routing_messages = messages
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
            self._log_audit(
                request_id,
                user_id,
                "",
                "blocked_abuse",
                pii_result=pii_result,
                abuse_result=abuse_result,
                error=f"Abuse detected: {abuse_result.categories}",
            )
            raise AbuseDetectedError(f"Abuse detected: {abuse_result.categories}")

        # 4. Routing decision
        routing_result = await self._route(messages, routing_messages)

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
            result = await _generate_with_retry(backend, messages, **backend_kwargs)
        except Exception as exc:
            logger.error("[%s] Model call failed: %s", request_id, exc)
            fallback_model = self.settings.fallback_model
            if selected_model == fallback_model:
                self._log_audit(
                    request_id,
                    user_id,
                    selected_model,
                    "error",
                    routing_result=routing_result,
                    pii_result=pii_result,
                    abuse_result=abuse_result,
                    error=str(exc),
                    latency_ms=(time.perf_counter() - start) * 1000,
                )
                raise
            logger.warning("[%s] Falling back from %s to %s", request_id, selected_model, fallback_model)
            try:
                fallback = self.pool.get(fallback_model)
                result = await _generate_with_retry(fallback, messages, **backend_kwargs)
                selected_model = fallback_model
            except Exception as fallback_exc:
                self._log_audit(
                    request_id,
                    user_id,
                    fallback_model,
                    "error",
                    routing_result=routing_result,
                    pii_result=pii_result,
                    abuse_result=abuse_result,
                    error=str(fallback_exc),
                    latency_ms=(time.perf_counter() - start) * 1000,
                )
                raise

        # 6. Output PII filter
        if self.pii_filter.redact:
            result.content = self.pii_filter.redact_text(result.content)

        # 7. Output content safety — runs outside the backend-call try/except above so a
        # policy block is never mistaken for a backend failure and retried on another model.
        if self.settings.guardrails.safety_enabled:
            safety_result = self.content_safety.check(result.content)
            if not safety_result.safe and safety_result.action == "block":
                logger.warning("[%s] Output blocked by content safety: %s", request_id, safety_result.categories)
                self._log_audit(
                    request_id,
                    user_id,
                    selected_model,
                    "blocked_content_safety",
                    routing_result=routing_result,
                    pii_result=pii_result,
                    abuse_result=abuse_result,
                    result=result,
                    error=f"Content safety blocked: {safety_result.categories}",
                )
                raise ContentSafetyBlockedError(
                    f"Response blocked by content safety policy: {safety_result.categories}"
                )
            if not safety_result.safe and safety_result.action == "mask":
                logger.info("[%s] Output masked by content safety: %s", request_id, safety_result.categories)
                result.content = "[CONTENT MASKED: policy violation]"

        logger.info(
            "[%s] Request complete: model=%s tokens=%d latency=%.0fms",
            request_id,
            selected_model,
            result.usage.total_tokens,
            result.latency_ms,
        )

        # 8. Audit log
        self._log_audit(
            request_id,
            user_id,
            selected_model,
            "success",
            routing_result=routing_result,
            pii_result=pii_result,
            abuse_result=abuse_result,
            result=result,
        )
        return result

    async def generate_stream(
        self,
        messages: list[dict],
        user_id: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        tools: list[dict] | None = None,
        max_tokens: int | None = None,
        client_ip: str | None = None,
    ) -> AsyncIterator[GenerateResult]:
        """Run a streaming completion with full guardrails and tracing."""
        request_id = f"{user_id or 'anonymous'}:{int(time.time())}"
        client_id = user_id or api_key or client_ip or "anonymous"

        # 1. Rate limit check
        rate_result = (
            await self.rate_limiter.check(client_id=client_id, tokens=100) if self.settings.rate_limit.enabled else None
        )
        if rate_result is not None and not rate_result.allowed:
            self._log_audit(request_id, user_id, "", "rate_limited", error=rate_result.error)
            raise RateLimitExceededError(f"Rate limited: {rate_result.error}")

        # 2. Input PII filter
        full_text = " ".join(_message_text(msg) for msg in messages)
        routing_messages = messages
        pii_result = self.pii_filter.check(full_text, mode="input") if self.settings.guardrails.pii_enabled else None
        if pii_result is not None and pii_result.has_pii:
            messages = _redact_messages(messages, self.pii_filter)

        # 3. Input abuse filter
        abuse_result = self.abuse_filter.check(full_text) if self.settings.guardrails.abuse_enabled else None
        if abuse_result is not None and not abuse_result.safe:
            self._log_audit(
                request_id,
                user_id,
                "",
                "blocked_abuse",
                pii_result=pii_result,
                abuse_result=abuse_result,
                error=f"Abuse detected: {abuse_result.categories}",
            )
            raise AbuseDetectedError(f"Abuse detected: {abuse_result.categories}")

        # 4. Routing decision
        routing_result = await self._route(messages, routing_messages)

        requested_model = None if model in {None, "auto", "router-auto"} else model
        selected_model = requested_model or routing_result.model_id or self.default_model
        if tools and requested_model is None:
            selected_model = self.default_model

        # 5. Model call
        backend = self.pool.get(selected_model)
        backend_kwargs = {"tools": tools} if tools else {}
        if max_tokens is not None:
            backend_kwargs["max_tokens"] = max_tokens

        safety_enabled = self.settings.guardrails.safety_enabled

        async def _safe_stream(source_backend) -> AsyncIterator[GenerateResult]:
            """Stream chunks, stopping (without a fallback retry) if output is blocked."""
            accumulated = ""
            async for chunk in _generate_stream_with_content(source_backend, messages, **backend_kwargs):
                if safety_enabled:
                    accumulated += chunk.content or ""
                    safety_result = self.content_safety.check(accumulated)
                    if not safety_result.safe and safety_result.action == "block":
                        logger.warning(
                            "[%s] Streaming output blocked by content safety: %s",
                            request_id,
                            safety_result.categories,
                        )
                        return
                yield chunk

        try:
            async for chunk in _safe_stream(backend):
                yield chunk
        except Exception:
            fallback_model = self.settings.fallback_model
            if selected_model == fallback_model:
                raise
            fallback = self.pool.get(fallback_model)
            logger.warning("[%s] Streaming fallback from %s to %s", request_id, selected_model, fallback_model)
            async for chunk in _safe_stream(fallback):
                yield chunk
