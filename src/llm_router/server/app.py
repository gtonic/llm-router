"""FastAPI application factory for the LLM Router & Gateway."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from llm_router.config import GatewaySettings
from llm_router.guardrails.abuse_filter import AbuseFilter
from llm_router.guardrails.content_safety import ContentSafety
from llm_router.guardrails.pii_filter import PiiFilter
from llm_router.guardrails.rate_limiter import RateLimiter
from llm_router.pool.pool import ModelPool
from llm_router.router import RouterPolicyEngine
from llm_router.routing.complexity import ComplexityDetector
from llm_router.routing.hybrid import HybridRouter
from llm_router.routing.policy import PolicyMatcher
from llm_router.routing.round_robin import RoundRobinPolicy
from llm_router.tracing.otel_setup import setup_otel

logger = logging.getLogger("llm-router")

# Global router instance (initialized in lifespan)
router_engine: RouterPolicyEngine | None = None

# ───────────────────────────────────────────
# Prometheus Metrics
# ───────────────────────────────────────────

try:
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

    PROMETHEUS_ENABLED = True

    # Request counters
    REQUESTS_TOTAL = Counter(
        "llm_router_requests_total",
        "Total number of requests",
        ["model", "strategy", "status"],
    )

    ERRORS_TOTAL = Counter(
        "llm_router_errors_total",
        "Total number of errors",
        ["model", "error_type"],
    )

    # Latency histogram
    REQUEST_DURATION = Histogram(
        "llm_router_request_duration_seconds",
        "Request latency in seconds",
        ["model", "strategy"],
        buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
    )

    # Cost tracking
    COST_TOTAL = Counter(
        "llm_router_cost_total",
        "Total cost in USD",
        ["model"],
    )

    # Token usage
    TOKENS_TOTAL = Counter(
        "llm_router_tokens_total",
        "Total tokens processed",
        ["type", "model"],  # type: prompt|completion
    )

    # Guardrails
    PII_DETECTED = Counter(
        "llm_router_pii_detected_total",
        "Total PII detections",
        ["sensitivity"],
    )

    ABUSE_BLOCKED = Counter(
        "llm_router_abuse_blocked_total",
        "Total abuse requests blocked",
        [],
    )

    RATE_LIMITED = Counter(
        "llm_router_rate_limited_total",
        "Total rate-limited requests",
        [],
    )

    # Route decisions
    ROUTE_DECISIONS = Counter(
        "llm_router_route_decisions_total",
        "Total routing decisions",
        ["strategy"],
    )

    # Response status codes
    RESPONSE_STATUS = Counter(
        "llm_router_response_status_total",
        "Response status codes",
        ["status"],
    )

    ADMIN_ACTIONS = Counter(
        "llm_router_admin_actions_total",
        "Runtime administration actions",
        ["action", "status"],
    )

    # Current active requests
    ACTIVE_REQUESTS = Gauge(
        "llm_router_active_requests",
        "Number of active requests",
        [],
    )

    def record_request(
        model: str,
        strategy: str,
        status: str,
        duration: float,
        cost: float = 0.0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ):
        """Record metrics for a completed request."""
        REQUESTS_TOTAL.labels(model=model, strategy=strategy, status=status).inc()
        REQUEST_DURATION.labels(model=model, strategy=strategy).observe(duration)
        COST_TOTAL.labels(model=model).inc(cost)
        TOKENS_TOTAL.labels(type="prompt", model=model).inc(prompt_tokens)
        TOKENS_TOTAL.labels(type="completion", model=model).inc(completion_tokens)
        RESPONSE_STATUS.labels(status=status).inc()

    def record_error(model: str, error_type: str):
        """Record an error."""
        ERRORS_TOTAL.labels(model=model, error_type=error_type).inc()
        RESPONSE_STATUS.labels(status="error").inc()

    def record_pii():
        """Record PII detection."""
        PII_DETECTED.labels(sensitivity="high").inc()

    def record_abuse():
        """Record abuse detection."""
        ABUSE_BLOCKED.inc()

    def record_rate_limit():
        """Record rate limiting."""
        RATE_LIMITED.inc()

    def record_route_decision(strategy: str):
        """Record a routing decision."""
        ROUTE_DECISIONS.labels(strategy=strategy).inc()

    def record_admin_action(action: str, status: str = "success"):
        """Record a runtime administration action without request details."""
        ADMIN_ACTIONS.labels(action=action, status=status).inc()

    def metrics_endpoint():
        """Prometheus metrics endpoint."""
        from starlette.responses import Response

        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

except ImportError:
    PROMETHEUS_ENABLED = False

    def record_request(*args, **kwargs):
        pass

    def record_error(*args, **kwargs):
        pass

    def record_pii():
        pass

    def record_abuse():
        pass

    def record_rate_limit():
        pass

    def record_route_decision(*args, **kwargs):
        pass

    def record_admin_action(*args, **kwargs):
        pass

    def metrics_endpoint():
        from starlette.responses import PlainTextResponse

        return PlainTextResponse("Prometheus client not installed", status_code=501)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application startup and shutdown."""
    # Startup
    global router_engine
    settings = GatewaySettings()
    settings.load_runtime_config()

    # Initialize OpenTelemetry
    if settings.otlp_enabled:
        setup_otel(
            service_name="llm-router",
            otlp_enabled=True,
            otlp_endpoint=settings.otlp_endpoint,
            otlp_protocol=settings.otlp_protocol,
        )

    # Initialize model pool
    model_pool = ModelPool(models_dir=settings.models_dir)
    logger.info("Loaded models: %s", model_pool.list_models())

    # Initialize guardrails
    rate_limiter = RateLimiter(rpm=settings.rate_limit_rpm, tpm=settings.rate_limit_tpm)
    pii_filter = PiiFilter(redact=settings.guardrails.pii_redact, custom_patterns=settings.guardrails.pii_patterns)
    abuse_filter = AbuseFilter(block_threshold=settings.guardrails.abuse_block_threshold)
    from llm_router.guardrails.content_safety import SafetyLevel

    content_safety = ContentSafety(
        block_threshold=SafetyLevel.MEDIUM if settings.guardrails.safety_enabled else SafetyLevel.CRITICAL
    )
    content_safety._keywords[SafetyLevel.MEDIUM] = {
        keyword.lower() for keyword in settings.guardrails.safety_categories
    }

    # Initialize routing strategies
    policy_matcher = PolicyMatcher(policies_dir=settings.policies_dir, default_policy=settings.default_policy)
    complexity_detector = ComplexityDetector()
    hybrid_router = HybridRouter()
    round_robin = RoundRobinPolicy(model_pool.list_models())

    # Create the main router engine
    router_engine = RouterPolicyEngine(
        pool=model_pool,
        settings=settings,
        routing_strategy=settings.default_strategy,
        policy_matcher=policy_matcher,
        complexity_detector=complexity_detector,
        hybrid_router=hybrid_router,
        round_robin=round_robin,
        rate_limiter=rate_limiter,
        pii_filter=pii_filter,
        abuse_filter=abuse_filter,
        content_safety=content_safety,
        default_model=settings.default_model,
    )

    yield


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    from llm_router.server.routes import admin_router, skill_router
    from llm_router.server.routes import router as routes_router

    app = FastAPI(
        title="LLM Router & Gateway",
        description="Policy-based LLM request routing with guardrails and OpenTelemetry tracing.",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routes
    app.include_router(routes_router)
    app.include_router(admin_router)
    app.include_router(skill_router)

    @app.get("/health")
    async def health():
        return {"status": "ok", "version": "0.1.0"}

    @app.get("/metrics")
    async def metrics():
        """Prometheus metrics endpoint."""
        return metrics_endpoint()

    return app


# Module-level app instance so uvicorn can import via "llm_router.server.app:app"
app = create_app()


def main():
    """CLI entry point for the FastAPI server."""
    import uvicorn

    uvicorn.run(
        "llm_router.server.app:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False,
    )


if __name__ == "__main__":
    main()
