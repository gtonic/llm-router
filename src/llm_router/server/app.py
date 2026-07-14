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


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application startup and shutdown."""
    # Startup
    global router_engine
    settings = GatewaySettings()

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
    pii_filter = PiiFilter(redact=settings.pii_redact)
    abuse_filter = AbuseFilter()
    content_safety = ContentSafety()

    # Initialize routing strategies
    policy_matcher = PolicyMatcher(policies_dir=settings.policies_dir, default_policy=settings.default_policy)
    complexity_detector = ComplexityDetector()
    hybrid_router = HybridRouter()
    round_robin = RoundRobinPolicy(model_pool.list_models())

    # Create the main router engine
    router_engine = RouterPolicyEngine(
        pool=model_pool,
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

    @app.get("/health")
    async def health():
        return {"status": "ok", "version": "0.1.0"}

    return app


def main():
    """CLI entry point for the FastAPI server."""
    import uvicorn

    create_app()
    uvicorn.run(
        "llm_router.server.app:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False,
    )


if __name__ == "__main__":
    main()
