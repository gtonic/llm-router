"""Main router engine — orchestrates guardrails, routing, and model calls."""

from __future__ import annotations

import inspect
import logging
import time
from collections.abc import AsyncIterator

from opentelemetry import trace as otel_trace

from llm_router.config import GatewaySettings, RoutingStrategy
from llm_router.guardrails.abuse_filter import AbuseFilter
from llm_router.guardrails.content_safety import ContentSafety, ContentSafetyBlockedError
from llm_router.guardrails.pii_filter import PiiFilter
from llm_router.guardrails.rate_limiter import RateLimiter
from llm_router.logging_setup import AuditEntry, AuditLogger
from llm_router.pool.base import EmptyResponseError, GenerateResult, UsageInfo
from llm_router.pool.circuit_breaker import CircuitBreaker
from llm_router.pool.pool import ModelPool
from llm_router.routing.base import RoutingResult
from llm_router.routing.complexity import ComplexityDetector
from llm_router.routing.hybrid import HybridRouter
from llm_router.routing.policy import PolicyMatcher
from llm_router.routing.round_robin import RoundRobinPolicy
from llm_router.routing.session_affinity import SessionAffinityStore
from llm_router.tracing.otel_setup import get_tracer
from llm_router.tracing.span_attributes import SpanAttributes

logger = logging.getLogger("llm-router")


def _trace_id_hex(span) -> str | None:
    """Return the current span's trace ID as hex, or None if tracing is inactive."""
    ctx = span.get_span_context()
    if not ctx.is_valid:
        return None
    return format(ctx.trace_id, "032x")


class RateLimitExceededError(RuntimeError):
    """Raised when a client exceeds the configured rate limits."""


class AbuseDetectedError(RuntimeError):
    """Raised when input guardrails flag a request as abusive."""


class BackendUnavailableError(RuntimeError):
    """Raised when a backend is skipped because its circuit breaker is open."""


# HTTP status codes that indicate a client/request error — retrying or failing
# over to another backend won't help, so we surface them directly.
_NON_RETRYABLE_STATUS = {400, 401, 403, 404, 422}

# Sliding-window size (chars) for streaming content-safety scans. Must be well
# above the longest safety keyword so boundary-spanning matches are still caught.
_SAFETY_WINDOW_CHARS = 512


def _is_client_error(exc: Exception) -> bool:
    """Return True for non-retryable 4xx errors (bad request, auth, not found)."""
    status = getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
    return status in _NON_RETRYABLE_STATUS


def _emit(recorder: str, *args) -> None:
    """Best-effort Prometheus emission, deferred so the core has no module-load
    dependency on the server layer (avoids a circular import)."""
    try:
        from llm_router.server import app as server_app

        getattr(server_app, recorder)(*args)
    except Exception:
        logger.debug("metric %s emit failed", recorder, exc_info=True)


def _fallback_reason(exc: Exception) -> str:
    """Classify why a fallback happened, for the fallbacks_total 'reason' label."""
    return "circuit_open" if isinstance(exc, BackendUnavailableError) else type(exc).__name__


def _retry_attempts(backend) -> int:
    """Return a safe integer retry count for real and mocked backends."""
    value = getattr(backend.config, "retry_count", 1)
    return max(1, value) if isinstance(value, int) else 1


async def _generate_with_retry(backend, messages: list[dict], **kwargs):
    """Use retry support when supplied by a real backend implementation."""
    retry_method = getattr(backend, "generate_with_retry", None)
    if retry_method is not None and inspect.iscoroutinefunction(retry_method):
        model_id = getattr(backend.config, "id", "unknown")
        return await retry_method(
            messages,
            max_retries=_retry_attempts(backend),
            on_retry=lambda: _emit("record_retry", model_id),
            **kwargs,
        )
    return await backend.generate(messages, **kwargs)


async def _generate_stream_with_content(backend, messages: list[dict], **kwargs) -> AsyncIterator[GenerateResult]:
    """Yield meaningful stream chunks or fail so the caller can use its fallback.

    A content-less terminal chunk carrying token usage (for accounting) is still
    forwarded once real content has been produced, so callers can record
    tokens/cost — but an entirely empty stream still raises.
    """
    yielded_content = False
    async for chunk in backend.generate_stream(messages, **kwargs):
        if chunk.content or chunk.tool_calls:
            yielded_content = True
            yield chunk
        elif yielded_content and chunk.usage and chunk.usage.total_tokens:
            yield chunk
    if not yielded_content:
        raise EmptyResponseError(f"Backend '{backend.config.id}' returned an empty stream")


# max_tokens is a ceiling, not a prediction — cap its contribution to the TPM
# estimate so a large ceiling (e.g. 65536) doesn't spuriously exhaust the budget.
_TPM_OUTPUT_ESTIMATE_CAP = 4096


def _estimate_tokens(messages: list[dict], max_tokens: int | None) -> int:
    """Rough token estimate for TPM accounting (~4 chars/token + expected output).

    A real tokenizer would be exact; this cheap heuristic is enough to stop the
    rate limiter from treating a 50k-char prompt the same as a one-liner. The
    output term is bounded so a high ``max_tokens`` ceiling isn't billed as fully
    consumed.
    """
    prompt_chars = sum(len(_message_text(message)) for message in messages)
    output_tokens = min(max_tokens, _TPM_OUTPUT_ESTIMATE_CAP) if isinstance(max_tokens, int) and max_tokens > 0 else 256
    return max(1, prompt_chars // 4 + output_tokens)


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


def _bounded_guardrail_text(messages: list[dict], max_tokens: int) -> str:
    """Concatenate the most-recent message text up to ~``max_tokens`` (4 chars/token).

    Guardrail (PII/abuse) scanning is O(text length) and runs synchronously in the
    async request handler; scanning a full 74k-token context is ~70 ms of CPU that
    blocks the event loop. Bounding to the recent tail keeps it cheap — older turns
    were already scanned on their own request, and the current turn is at the end.
    """
    budget = max(1, max_tokens) * 4
    parts: list[str] = []
    total = 0
    for message in reversed(messages):
        text = _message_text(message)
        parts.append(text)
        total += len(text) + 1
        if total >= budget:
            break
    return " ".join(reversed(parts))[-budget:]


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
        self.audit_logger = AuditLogger(self.settings.log_dir, retention_days=self.settings.audit_retention_days)
        # Per-backend circuit breakers, created lazily on first use.
        self._breakers: dict[str, CircuitBreaker] = {}
        # Session → backend affinity ("stickiness") store.
        self._affinity = SessionAffinityStore(
            ttl=self.settings.session_affinity_ttl,
            max_entries=self.settings.session_affinity_max_entries,
        )
        # EWMA of observed request latency per backend, for the 'latency' strategy.
        self._latency_ewma: dict[str, float] = {}

    def _observe_latency(self, model_id: str, latency_ms: float) -> None:
        """Fold an observed request latency into the per-backend EWMA."""
        if not latency_ms or latency_ms <= 0:
            return
        prev = self._latency_ewma.get(model_id)
        self._latency_ewma[model_id] = latency_ms if prev is None else 0.3 * latency_ms + 0.7 * prev

    # ── Session affinity ──────────────────────

    def _session_key(self, session_id: str | None, user_id: str | None) -> str | None:
        """Derive the affinity key: X-Session-Id, else user_id, else none (off)."""
        if not self.settings.session_affinity_enabled:
            return None
        if session_id:
            return f"sid:{session_id}"
        if user_id:
            return f"uid:{user_id}"
        return None

    def _affinity_model(self, session_key: str | None) -> str | None:
        """Return the pinned model for a session if it's still usable, else None."""
        if session_key is None:
            return None
        model = self._affinity.get(session_key)
        if model and model in self._enabled_models() and not self._is_unavailable(model):
            return model
        return None

    def _affinity_remember(self, session_key: str | None, model: str) -> None:
        """Pin the model that actually served this session (refreshes the TTL)."""
        if session_key is not None and model:
            self._affinity.set(session_key, model)

    def sync_runtime_models(self) -> None:
        """Synchronize routing policies with currently enabled backends."""
        self.round_robin.update_models(self.pool.list_models())

    # ── Circuit breaking / availability ───────

    def _breaker(self, model_id: str) -> CircuitBreaker:
        """Return the circuit breaker for ``model_id``, creating it on demand."""
        breaker = self._breakers.get(model_id)
        if breaker is None:
            breaker = CircuitBreaker(
                threshold=self.settings.circuit_breaker_threshold,
                cooldown=self.settings.circuit_breaker_cooldown,
            )
            self._breakers[model_id] = breaker
        return breaker

    def _is_unavailable(self, model_id: str) -> bool:
        """True if the model's breaker is open (read-only, no probe consumed)."""
        breaker = self._breakers.get(model_id)
        return breaker is not None and breaker.blocked

    def _prefer_available(self, model_id: str) -> str:
        """Reroute away from a circuit-open backend at selection time.

        If the routed model's breaker is open and a healthy alternative exists
        (preferring the configured fallback), return that instead so the audit
        log and metrics attribute the request to the backend actually used.
        Fail-open: any uncertainty returns ``model_id`` unchanged.
        """
        if not self.settings.circuit_breaker_enabled or not self._is_unavailable(model_id):
            return model_id
        fallback = self.settings.fallback_model
        list_models = getattr(self.pool, "list_models", None)
        others = list(list_models()) if callable(list_models) else []
        for candidate in [fallback, *others]:
            if candidate and candidate != model_id and not self._is_unavailable(candidate):
                logger.warning("Rerouting from circuit-open '%s' to '%s'", model_id, candidate)
                return candidate
        return model_id

    async def _attempt(self, model_id: str, messages: list[dict], backend_kwargs: dict) -> GenerateResult:
        """Call a backend through its circuit breaker, recording the outcome."""
        breaker = self._breaker(model_id) if self.settings.circuit_breaker_enabled else None
        if breaker is not None and not breaker.allow():
            raise BackendUnavailableError(f"Circuit breaker open for '{model_id}'")
        backend = self.pool.get(model_id)
        try:
            result = await _generate_with_retry(backend, messages, **backend_kwargs)
        except Exception:
            if breaker is not None and breaker.record_failure():
                _emit("record_circuit_open", model_id)
            raise
        if breaker is not None:
            breaker.record_success()
        return result

    @staticmethod
    def _stream_result(model_id: str, usage: UsageInfo | None) -> GenerateResult:
        """Wrap streamed token usage in a GenerateResult for the audit log."""
        return GenerateResult(content="", model=model_id, usage=usage or UsageInfo(0, 0, 0), finish_reason="stop")

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
        if strategy == RoutingStrategy.LATENCY:
            return await self._route_latency()
        if strategy == RoutingStrategy.COST:
            return self._route_cost()
        return await self.policy_matcher.route(routing_messages)

    def _enabled_models(self) -> list[str]:
        """Best-effort list of enabled backend IDs (empty if the pool can't list)."""
        list_models = getattr(self.pool, "list_models", None)
        try:
            return list(list_models()) if callable(list_models) else []
        except Exception:
            return []

    def _resolve_model(self, model: str | None) -> str | None:
        """Normalize a requested model id to a routing target, or None for 'auto'.

        Tolerates a ``provider/model`` prefix some OpenAI clients add (e.g.
        ``router/gpt-5.4-nano`` → ``gpt-5.4-nano``) so it resolves to a real
        backend instead of KeyError-ing into the fallback path. Unknown names are
        returned unchanged (handled by the normal routing/fallback flow).
        """
        if model is None or model in {"auto", "router-auto"}:
            return None
        available = set(self._enabled_models())
        if model in available:
            return model
        suffix = model.rsplit("/", 1)[-1]  # strip a leading "provider/" prefix
        if suffix in {"auto", "router-auto"}:
            return None
        return suffix if suffix in available else model

    def _route_cost(self) -> RoutingResult:
        """Route to the cheapest enabled backend by configured per-1M token price."""
        cheapest, best_price = None, None
        for model_id in self._enabled_models():
            cfg = self.pool.get(model_id).config
            price = getattr(cfg, "cost_per_1m_input", 0.0) + getattr(cfg, "cost_per_1m_output", 0.0)
            if best_price is None or price < best_price:
                best_price, cheapest = price, model_id
        return RoutingResult(
            model_id=cheapest or self.default_model, strategy="cost", metadata={"price_per_1m": best_price}
        )

    async def _route_latency(self) -> RoutingResult:
        """Route to the fastest healthy backend, preferring observed request latency
        (EWMA) over the health-check ping — a 2-min-prefill backend pings fast but
        generates slowly, so the ping alone would pick the worst one."""
        try:
            health = await self.pool.health_check_all()
        except Exception:
            health = {}
        healthy = [mid for mid, status in (health or {}).items() if getattr(status, "healthy", False)]
        if not healthy:
            return RoutingResult(model_id=self.default_model, strategy="latency", metadata={})

        def _score(model_id: str) -> float:
            observed = self._latency_ewma.get(model_id)
            if observed is not None:
                return observed
            ping = getattr(health[model_id], "latency_ms", None)
            return ping if ping is not None else float("inf")

        fastest = min(healthy, key=_score)
        return RoutingResult(
            model_id=fastest, strategy="latency", metadata={"latency_ms": self._latency_ewma.get(fastest)}
        )

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
        session_key: str | None = None,
        affinity: str | None = None,
        span=None,
    ) -> None:
        """Best-effort JSONL audit entry, correlated with the OTEL trace ID. Never raises.

        Also finalizes ``span`` (sets status/attributes and ends it) when provided —
        this is the single place every request path terminates, streaming excepted.
        """
        try:
            pii_patterns = getattr(pii_result, "patterns", None) or []
            abuse_score = getattr(abuse_result, "abuse_score", 0.0) or 0.0
            trace_id = _trace_id_hex(span) if span is not None else None
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
                trace_id=trace_id,
                session_key=session_key,
                affinity=affinity,
            )
            self.audit_logger.log(entry)
            if span is not None:
                span.set_attribute(SpanAttributes.GATEWAY_REQUEST_ID, request_id)
                span.set_attribute(SpanAttributes.MODEL_SELECTED, model_selected or "")
                span.set_attribute(SpanAttributes.AUDIT_STATUS, status)
                span.set_attribute(SpanAttributes.AUDIT_LOGGED, True)
                span.set_attribute(SpanAttributes.GUARDRAIL_PII_DETECTED, bool(pii_patterns))
                span.set_attribute(SpanAttributes.GUARDRAIL_ABUSE_SCORE, float(abuse_score))
                if routing_result is not None and routing_result.policy_matched:
                    span.set_attribute(SpanAttributes.ROUTING_POLICY, routing_result.policy_matched)
                if result is not None:
                    span.set_attribute(SpanAttributes.TOKEN_PROMPT, result.usage.prompt_tokens)
                    span.set_attribute(SpanAttributes.TOKEN_COMPLETION, result.usage.completion_tokens)
                    span.set_attribute(SpanAttributes.TOKEN_TOTAL, result.usage.total_tokens)
                    span.set_attribute(SpanAttributes.TOKEN_COST, result.usage.cost)
                    span.set_attribute(SpanAttributes.LATENCY_MS, result.latency_ms)
                if error:
                    span.set_attribute(SpanAttributes.RESPONSE_ERROR, error)
                    span.set_status(otel_trace.Status(otel_trace.StatusCode.ERROR, error))
                else:
                    span.set_status(otel_trace.Status(otel_trace.StatusCode.OK))
        except Exception:
            logger.debug("[%s] Failed to build audit entry", request_id, exc_info=True)
        finally:
            if span is not None:
                span.end()

    async def generate(
        self,
        messages: list[dict],
        user_id: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        tools: list[dict] | None = None,
        max_tokens: int | None = None,
        client_ip: str | None = None,
        session_id: str | None = None,
    ) -> GenerateResult:
        """Run a non-streaming completion with full guardrails and tracing."""
        request_id = f"{user_id or 'anonymous'}:{int(time.time())}"
        # A stable identifier for rate limiting — request_id above is intentionally unique
        # per request (for logs/tracing) and must never be used as the rate-limit bucket key.
        client_id = user_id or api_key or client_ip or "anonymous"
        start = time.perf_counter()

        span = get_tracer("llm-router").start_span("gateway.chat_completion")
        span.set_attribute(SpanAttributes.GATEWAY_REQUEST_ID, request_id)
        span.set_attribute(SpanAttributes.GATEWAY_USER_ID, user_id or "anonymous")
        span.set_attribute(SpanAttributes.ROUTING_STRATEGY, self.routing_strategy.value)

        # 1. Rate limit check
        rate_result = (
            await self.rate_limiter.check(client_id=client_id, tokens=_estimate_tokens(messages, max_tokens))
            if self.settings.rate_limit.enabled
            else None
        )
        if rate_result is not None and not rate_result.allowed:
            logger.warning("[%s] Rate limited: %s", request_id, rate_result.error)
            span.set_attribute(SpanAttributes.GUARDRAIL_RATE_LIMITED, True)
            self._log_audit(request_id, user_id, "", "rate_limited", error=rate_result.error, span=span)
            raise RateLimitExceededError(f"Rate limited: {rate_result.error}")

        # 2. Input PII filter
        pii_enabled = self.settings.guardrails.pii_enabled
        abuse_enabled = self.settings.guardrails.abuse_enabled
        routing_messages = messages
        # Scan only a bounded recent window (skip entirely when both are off).
        full_text = (
            _bounded_guardrail_text(messages, self.settings.guardrails.pii_max_tokens)
            if (pii_enabled or abuse_enabled)
            else ""
        )
        pii_result = self.pii_filter.check(full_text, mode="input") if pii_enabled else None
        if pii_result is not None and pii_result.has_pii:
            logger.info("[%s] PII detected: %s", request_id, pii_result.patterns)
            messages = _redact_messages(messages, self.pii_filter)

        # 3. Input abuse filter
        abuse_result = self.abuse_filter.check(full_text) if abuse_enabled else None
        if abuse_result is not None and not abuse_result.safe:
            logger.warning(
                "[%s] Abuse detected (score=%.2f): %s",
                request_id,
                abuse_result.abuse_score,
                abuse_result.categories,
            )
            span.set_attribute(SpanAttributes.GUARDRAIL_ABUSE_SAFE, False)
            self._log_audit(
                request_id,
                user_id,
                "",
                "blocked_abuse",
                pii_result=pii_result,
                abuse_result=abuse_result,
                error=f"Abuse detected: {abuse_result.categories}",
                span=span,
            )
            raise AbuseDetectedError(f"Abuse detected: {abuse_result.categories}")

        # 4. Routing decision — session affinity first, else the strategy.
        # Affinity applies only to auto-routed, tool-less turns; a tools/explicit
        # turn must NOT pin the session (that would hijack later auto turns).
        requested_model = self._resolve_model(model)
        session_key = self._session_key(session_id, user_id)
        affinity_eligible = session_key is not None and requested_model is None and not tools
        sticky_model = self._affinity_model(session_key) if affinity_eligible else None
        if sticky_model is not None:
            routing_result = RoutingResult(
                model_id=sticky_model, strategy=self.routing_strategy.value, metadata={"affinity": "hit"}
            )
            affinity = "hit"
            _emit("record_affinity", "hit")
        else:
            routing_result = await self._route(messages, routing_messages)
            affinity = "store" if affinity_eligible else None

        selected_model = requested_model or routing_result.model_id or self.default_model
        if tools and requested_model is None:
            selected_model = self.default_model

        # 5. Model call — circuit-breaker aware, with a single fallback hop
        selected_model = self._prefer_available(selected_model)
        backend_kwargs = {"tools": tools} if tools else {}
        if max_tokens is not None:
            backend_kwargs["max_tokens"] = max_tokens
        try:
            result = await self._attempt(selected_model, messages, backend_kwargs)
        except Exception as exc:
            logger.error("[%s] Model call failed: %s", request_id, exc)
            fallback_model = self.settings.fallback_model
            # Don't fail over on non-retryable client errors (4xx) — a second
            # backend won't fix a bad request — nor when we're already on fallback.
            if selected_model == fallback_model or _is_client_error(exc):
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
                    span=span,
                )
                raise
            logger.warning("[%s] Falling back from %s to %s", request_id, selected_model, fallback_model)
            _emit("record_fallback", selected_model, fallback_model, _fallback_reason(exc))
            try:
                result = await self._attempt(fallback_model, messages, backend_kwargs)
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
                    span=span,
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
                    span=span,
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

        # 8. Record latency, pin the session (eligible turns only), and audit.
        self._observe_latency(selected_model, result.latency_ms)
        if affinity == "store":
            _emit("record_affinity", "store")
        if affinity_eligible:
            self._affinity_remember(session_key, selected_model)
        self._log_audit(
            request_id,
            user_id,
            selected_model,
            "success",
            routing_result=routing_result,
            pii_result=pii_result,
            abuse_result=abuse_result,
            result=result,
            session_key=session_key,
            affinity=affinity,
            span=span,
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
        session_id: str | None = None,
    ) -> AsyncIterator[GenerateResult]:
        """Run a streaming completion with full guardrails and tracing."""
        request_id = f"{user_id or 'anonymous'}:{int(time.time())}"
        client_id = user_id or api_key or client_ip or "anonymous"
        stream_start = time.perf_counter()

        span = get_tracer("llm-router").start_span("gateway.chat_completion_stream")
        span.set_attribute(SpanAttributes.GATEWAY_REQUEST_ID, request_id)
        span.set_attribute(SpanAttributes.GATEWAY_USER_ID, user_id or "anonymous")
        span.set_attribute(SpanAttributes.ROUTING_STRATEGY, self.routing_strategy.value)

        # 1. Rate limit check
        rate_result = (
            await self.rate_limiter.check(client_id=client_id, tokens=_estimate_tokens(messages, max_tokens))
            if self.settings.rate_limit.enabled
            else None
        )
        if rate_result is not None and not rate_result.allowed:
            span.set_attribute(SpanAttributes.GUARDRAIL_RATE_LIMITED, True)
            self._log_audit(request_id, user_id, "", "rate_limited", error=rate_result.error, span=span)
            raise RateLimitExceededError(f"Rate limited: {rate_result.error}")

        # 2. Input PII filter
        pii_enabled = self.settings.guardrails.pii_enabled
        abuse_enabled = self.settings.guardrails.abuse_enabled
        routing_messages = messages
        # Scan only a bounded recent window (skip entirely when both are off).
        full_text = (
            _bounded_guardrail_text(messages, self.settings.guardrails.pii_max_tokens)
            if (pii_enabled or abuse_enabled)
            else ""
        )
        pii_result = self.pii_filter.check(full_text, mode="input") if pii_enabled else None
        if pii_result is not None and pii_result.has_pii:
            messages = _redact_messages(messages, self.pii_filter)

        # 3. Input abuse filter
        abuse_result = self.abuse_filter.check(full_text) if abuse_enabled else None
        if abuse_result is not None and not abuse_result.safe:
            span.set_attribute(SpanAttributes.GUARDRAIL_ABUSE_SAFE, False)
            self._log_audit(
                request_id,
                user_id,
                "",
                "blocked_abuse",
                pii_result=pii_result,
                abuse_result=abuse_result,
                error=f"Abuse detected: {abuse_result.categories}",
                span=span,
            )
            raise AbuseDetectedError(f"Abuse detected: {abuse_result.categories}")

        # 4. Routing decision — session affinity first, else the strategy.
        # Affinity applies only to auto-routed, tool-less turns (see generate()).
        requested_model = self._resolve_model(model)
        session_key = self._session_key(session_id, user_id)
        affinity_eligible = session_key is not None and requested_model is None and not tools
        sticky_model = self._affinity_model(session_key) if affinity_eligible else None
        if sticky_model is not None:
            routing_result = RoutingResult(
                model_id=sticky_model, strategy=self.routing_strategy.value, metadata={"affinity": "hit"}
            )
            affinity = "hit"
            _emit("record_affinity", "hit")
        else:
            routing_result = await self._route(messages, routing_messages)
            affinity = "store" if affinity_eligible else None

        selected_model = requested_model or routing_result.model_id or self.default_model
        if tools and requested_model is None:
            selected_model = self.default_model
        span.set_attribute(SpanAttributes.MODEL_SELECTED, selected_model)
        if routing_result.policy_matched:
            span.set_attribute(SpanAttributes.ROUTING_POLICY, routing_result.policy_matched)

        # 5. Model call — circuit-breaker aware. Never restart on another model
        # once bytes have been streamed to the client (that would concatenate
        # partial + full output into a garbled response).
        selected_model = self._prefer_available(selected_model)
        backend_kwargs = {"tools": tools} if tools else {}
        if max_tokens is not None:
            backend_kwargs["max_tokens"] = max_tokens

        safety_enabled = self.settings.guardrails.safety_enabled
        safety_blocked = False

        async def _safe_stream(source_backend) -> AsyncIterator[GenerateResult]:
            """Stream chunks, stopping (and flagging a content-safety block) rather
            than falling back. Content safety scans a bounded sliding window (not
            the whole growing buffer every chunk), keeping this O(n) not O(n^2);
            the window overlaps chunk boundaries so split keywords are still caught.
            """
            nonlocal safety_blocked
            window = ""
            async for chunk in _generate_stream_with_content(source_backend, messages, **backend_kwargs):
                if safety_enabled and chunk.content:
                    window = (window + chunk.content)[-_SAFETY_WINDOW_CHARS:]
                    safety_result = self.content_safety.check(window)
                    if not safety_result.safe and safety_result.action == "block":
                        logger.warning(
                            "[%s] Streaming output blocked by content safety: %s",
                            request_id,
                            safety_result.categories,
                        )
                        safety_blocked = True
                        return
                yield chunk

        async def _run(model_id: str) -> AsyncIterator[GenerateResult]:
            """Stream one backend through its circuit breaker, recording the outcome."""
            breaker = self._breaker(model_id) if self.settings.circuit_breaker_enabled else None
            if breaker is not None and not breaker.allow():
                raise BackendUnavailableError(f"Circuit breaker open for '{model_id}'")
            source = self.pool.get(model_id)
            try:
                async for chunk in _safe_stream(source):
                    yield chunk
            except Exception:
                if breaker is not None and breaker.record_failure():
                    _emit("record_circuit_open", model_id)
                raise
            if breaker is not None:
                breaker.record_success()

        yielded_to_client = False
        final_usage: UsageInfo | None = None
        span_ended = False

        def _capture(chunk: GenerateResult) -> None:
            nonlocal final_usage
            if chunk.usage and chunk.usage.total_tokens:
                final_usage = chunk.usage

        def _audit_error(model_id: str, exc: Exception) -> None:
            """Write an error audit entry (parity with the non-streaming path)."""
            self._log_audit(
                request_id,
                user_id,
                model_id,
                "error",
                routing_result=routing_result,
                pii_result=pii_result,
                abuse_result=abuse_result,
                error=str(exc),
                session_key=session_key,
                affinity=affinity,
                span=span,
            )

        try:
            try:
                async for chunk in _run(selected_model):
                    yielded_to_client = True
                    _capture(chunk)
                    yield chunk
            except Exception as exc:
                fallback_model = self.settings.fallback_model
                if yielded_to_client:
                    logger.error("[%s] Stream failed after partial output; not falling back: %s", request_id, exc)
                    _audit_error(selected_model, exc)
                    span_ended = True
                    raise
                if selected_model == fallback_model or _is_client_error(exc):
                    _audit_error(selected_model, exc)
                    span_ended = True
                    raise
                logger.warning("[%s] Streaming fallback from %s to %s", request_id, selected_model, fallback_model)
                _emit("record_fallback", selected_model, fallback_model, _fallback_reason(exc))
                selected_model = fallback_model
                try:
                    async for chunk in _run(fallback_model):
                        _capture(chunk)
                        yield chunk
                except Exception as fallback_exc:
                    _audit_error(fallback_model, fallback_exc)
                    span_ended = True
                    raise
            if safety_blocked:
                # Cut for policy — tell the client (OpenAI-standard content_filter)
                # and audit as a block, not a success.
                yield GenerateResult(
                    content="",
                    model=selected_model,
                    usage=final_usage or UsageInfo(0, 0, 0),
                    finish_reason="content_filter",
                )
                self._log_audit(
                    request_id,
                    user_id,
                    selected_model,
                    "blocked_content_safety",
                    routing_result=routing_result,
                    pii_result=pii_result,
                    abuse_result=abuse_result,
                    result=self._stream_result(selected_model, final_usage),
                    error="Response blocked by content safety policy",
                    session_key=session_key,
                    affinity=affinity,
                    span=span,
                )
            else:
                # Success — record latency, pin the session (eligible turns only) + audit.
                self._observe_latency(selected_model, (time.perf_counter() - stream_start) * 1000)
                if affinity == "store":
                    _emit("record_affinity", "store")
                if affinity_eligible:
                    self._affinity_remember(session_key, selected_model)
                self._log_audit(
                    request_id,
                    user_id,
                    selected_model,
                    "success",
                    routing_result=routing_result,
                    pii_result=pii_result,
                    abuse_result=abuse_result,
                    result=self._stream_result(selected_model, final_usage),
                    session_key=session_key,
                    affinity=affinity,
                    span=span,
                )
            span_ended = True
        finally:
            if not span_ended:
                span.end()
