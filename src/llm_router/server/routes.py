"""API routes for the LLM Router & Gateway.

Implements OpenAI-compatible endpoints plus gateway-specific routes.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncGenerator
from dataclasses import asdict
from datetime import UTC

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from llm_router.models import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChunkChoice,
    MessageRole,
    ModelInfo,
    ModelListResponse,
)
from llm_router.models import (
    UsageInfo as ModelsUsageInfo,
)
from llm_router.server.app import (
    PROMETHEUS_ENABLED,
    record_error,
    record_request,
    router_engine,
)
from llm_router.server.system_manifest import (
    BackendHealth,
    build_manifest,
)

logger = logging.getLogger("llm-router")
router = APIRouter(prefix="/v1", tags=["llm-router"])


def _normalize_messages(messages: list[dict]) -> list[dict]:
    """Convert OpenAI text content parts to the router's internal string form."""
    normalized = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            text = " ".join(
                part.get("text", "") for part in content if isinstance(part, dict) and isinstance(part.get("text"), str)
            )
            message = {**message, "content": text}
        normalized.append(message)
    return normalized


def _serialize_guardrail_result(result: object) -> dict:
    """Serialize guardrail results across dataclass and Pydantic implementations."""
    if hasattr(result, "model_dump"):
        return result.model_dump()
    if hasattr(result, "__dataclass_fields__"):
        return asdict(result)
    return {
        "safe": result.safe,
        "abuse_score": result.abuse_score,
        "categories": result.categories,
        "details": result.details,
    }


@router.post("/chat/completions")
async def chat_completions(request: Request, body: ChatCompletionRequest):
    """OpenAI-compatible chat completion endpoint.

    Routes the request through guardrails, policy router, and model backend.
    Supports both streaming and non-streaming responses.
    """
    from llm_router.server.app import router_engine

    if router_engine is None:
        raise HTTPException(status_code=503, detail="Router not initialized")

    request_id = str(uuid.uuid4())[:8]
    logger.info("[%s] %s %s", request_id, request.client.host, "POST /v1/chat/completions")

    # Non-streaming
    if not body.stream:
        try:
            start = time.perf_counter()
            result = await router_engine.generate(
                messages=_normalize_messages(body.model_dump()["messages"]),
                user_id=body.user_id,
                api_key=body.api_key,
                model=body.model,
                tools=body.tools,
                max_tokens=body.max_tokens,
            )
            elapsed = time.perf_counter() - start
            elapsed_ms = elapsed * 1000

            response = ChatCompletionResponse(
                model=result.model,
                choices=[
                    {
                        "index": 0,
                        "message": {
                            "role": MessageRole.assistant,
                            "content": result.content,
                            "tool_calls": result.tool_calls,
                        },
                        "finish_reason": result.finish_reason,
                    }
                ],
                usage=ModelsUsageInfo(
                    prompt_tokens=result.usage.prompt_tokens,
                    completion_tokens=result.usage.completion_tokens,
                    total_tokens=result.usage.total_tokens,
                ),
            )
            response.usage.cost = 0.0  # Will be set by router

            # Record Prometheus metrics
            if PROMETHEUS_ENABLED:
                strategy = router_engine.routing_strategy.value if router_engine else "unknown"
                record_request(
                    model=result.model,
                    strategy=strategy,
                    status="success",
                    duration=elapsed,
                    cost=getattr(result.usage, "cost", 0.0) or 0.0,
                    prompt_tokens=result.usage.prompt_tokens,
                    completion_tokens=result.usage.completion_tokens,
                )

            logger.info("[%s] %s %s %.1fms model=%s", request_id, request.client.host, "OK", elapsed_ms, result.model)
            return response

        except Exception as exc:
            logger.error("[%s] Error: %s", request_id, exc)
            if PROMETHEUS_ENABLED:
                record_error(model=body.model or "unknown", error_type=type(exc).__name__)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    # Streaming
    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            last_finish_reason = "stop"
            async for chunk in router_engine.generate_stream(
                messages=_normalize_messages(body.model_dump()["messages"]),
                user_id=body.user_id,
                api_key=body.api_key,
                model=body.model,
                tools=body.tools,
                max_tokens=body.max_tokens,
            ):
                delta = {"content": chunk.content, "role": "assistant"}
                if chunk.tool_calls:
                    delta["tool_calls"] = chunk.tool_calls
                    last_finish_reason = "tool_calls"
                chunk_data = ChatCompletionChunk(
                    model=chunk.model,
                    choices=[
                        ChunkChoice(
                            delta=delta,
                            finish_reason="tool_calls" if chunk.finish_reason == "tool_calls" else None,
                        )
                    ],
                )
                yield f"data: {chunk_data.model_dump_json()}\n\n"
            final_chunk = ChatCompletionChunk(
                model=body.model,
                choices=[ChunkChoice(delta={}, finish_reason=last_finish_reason)],
            )
            yield f"data: {final_chunk.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as exc:
            logger.error("[%s] Stream error: %s", request_id, exc)
            error_chunk = ChatCompletionChunk(
                model="error",
                choices=[ChunkChoice(delta={"content": str(exc)}, finish_reason="error")],
            )
            yield f"data: {error_chunk.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/models")
async def list_models():
    """List available models (OpenAI-compatible)."""
    from llm_router.server.app import router_engine

    if router_engine is None:
        raise HTTPException(status_code=503, detail="Router not initialized")

    model_list = ModelListResponse(
        data=[ModelInfo(id="router-auto")] + [ModelInfo(id=model_id) for model_id in router_engine.pool.list_models()]
    )
    return model_list


@router.get("/guardrails/pii/patterns")
async def get_pii_patterns():
    """Return configured PII detection patterns."""
    from llm_router.server.app import router_engine

    if router_engine is None:
        raise HTTPException(status_code=503, detail="Router not initialized")

    return {
        "patterns": [name for _, name in router_engine.pii_filter.PATTERNS],
        "redact_enabled": router_engine.pii_filter.redact,
    }


@router.post("/guardrails/check")
async def check_guardrails(request: Request):
    """Manually trigger a guardrail check."""
    from llm_router.server.app import router_engine

    if router_engine is None:
        raise HTTPException(status_code=503, detail="Router not initialized")

    try:
        data = await request.json()
        text = data.get("text", "")
        mode = data.get("mode", "input")

        if mode == "pii":
            result = router_engine.pii_filter.check(text)
            return {"has_pii": result.has_pii, "patterns": result.patterns}
        elif mode == "abuse":
            result = router_engine.abuse_filter.check(text)
            return _serialize_guardrail_result(result)
        elif mode == "safety":
            result = router_engine.content_safety.check(text)
            return result.__dict__
        else:
            return {"error": "Invalid mode. Use 'pii', 'abuse', or 'safety'."}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/admin/reload")
async def reload_config():
    """Reload policies and models from disk."""
    from llm_router.server.app import router_engine

    if router_engine is None:
        raise HTTPException(status_code=503, detail="Router not initialized")

    try:
        router_engine.policy_matcher._load_policies()
        # TODO: Also reload model configs
        return {"status": "reloaded", "policies": len(router_engine.policy_matcher.rules)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ───────────────────────────────────────────
# System Manifest Endpoints (DevOps Monitoring)
# ───────────────────────────────────────────

# Module-level start time for uptime calculation
_start_time: str = ""


@router.get("/system/manifest")
async def get_system_manifest():
    """Return the full system self-description.

    This endpoint enables DevOps monitoring agents to discover:
      - System metadata (version, runtime, models)
      - Health status of all backends
      - Allowed lifecycle actions
      - Log patterns for anomaly detection
      - Monitoring thresholds
      - API endpoint catalog
    """
    from datetime import datetime

    if router_engine is None:
        raise HTTPException(status_code=503, detail="Router not initialized")

    # Set start time on first call
    global _start_time
    if not _start_time:
        _start_time = datetime.now(UTC).isoformat()

    # Get backend health
    health_results = await router_engine.pool.health_check_all()
    backend_health = {}
    model_configs = []
    for mid, backend in router_engine.pool._backends.items():
        status = health_results.get(mid)
        if status is None:
            # health_results is a dict of HealthStatus, not iterable as zip
            continue
        backend_health[mid] = BackendHealth(
            id=mid,
            name=backend.config.name,
            type=backend.config.type,
            healthy=status.healthy,
            latency_ms=status.latency_ms,
            error=status.error,
        )
        model_configs.append(backend.config)

    # Calculate uptime
    uptime = (datetime.now(UTC) - datetime.fromisoformat(_start_time)).total_seconds()

    manifest = build_manifest(
        settings=router_engine.pool._settings if hasattr(router_engine.pool, "_settings") else None,
        start_time=_start_time,
        uptime_seconds=uptime,
        models=model_configs,
        health_status=backend_health,
    )

    return manifest.to_dict()


@router.get("/system/health")
async def get_system_health():
    """Detailed health check — backends, routing, guardrails."""
    if router_engine is None:
        raise HTTPException(status_code=503, detail="Router not initialized")

    health_results = await router_engine.pool.health_check_all()

    backends = {}
    all_healthy = True
    for mid, backend in router_engine.pool._backends.items():
        status = health_results.get(mid)
        if status is None:
            continue
        backends[mid] = {
            "id": mid,
            "name": backend.config.name,
            "type": backend.config.type,
            "healthy": status.healthy,
            "latency_ms": round(status.latency_ms, 1),
            "error": status.error,
        }
        if not status.healthy:
            all_healthy = False

    return {
        "status": "healthy" if all_healthy else "degraded",
        "version": "0.1.0",
        "backends": backends,
        "routing_strategy": router_engine.routing_strategy.value,
        "guardrails": {
            "pii_enabled": True,
            "abuse_enabled": True,
            "content_safety_enabled": True,
            "rate_limiting_enabled": True,
        },
        "policies_loaded": len(router_engine.policy_matcher.rules),
    }


@router.get("/system/metrics")
async def get_system_metrics():
    """Current metrics snapshot for Prometheus-compatible consumption.

    Returns a JSON snapshot of all counters, histograms, and gauges.
    Also exposes the raw Prometheus text format at /metrics.
    """
    if router_engine is None:
        raise HTTPException(status_code=503, detail="Router not initialized")

    # Get backend health for metrics
    health_results = await router_engine.pool.health_check_all()

    metrics = {
        "backends": {
            mid: {
                "healthy": status.healthy,
                "latency_ms": round(status.latency_ms, 1),
            }
            for mid, status in health_results.items()
        },
        "routing_strategy": router_engine.routing_strategy.value,
        "policies_count": len(router_engine.policy_matcher.rules),
        "prometheus_enabled": PROMETHEUS_ENABLED,
        "prometheus_endpoint": "/metrics",
    }

    return metrics


@router.get("/system/capabilities")
async def get_system_capabilities():
    """Return allowed lifecycle actions.

    This tells DevOps agents what operations they can perform on this system.
    """
    if router_engine is None:
        raise HTTPException(status_code=503, detail="Router not initialized")

    return {
        "system": "llm-router",
        "version": "0.1.0",
        "allowed_actions": [
            {
                "action": "health_check",
                "description": "Check system and backend health",
                "method": "GET",
                "endpoint": "/system/health",
                "auth_required": False,
            },
            {
                "action": "get_manifest",
                "description": "Get system self-description",
                "method": "GET",
                "endpoint": "/system/manifest",
                "auth_required": False,
            },
            {
                "action": "get_metrics",
                "description": "Fetch current metrics",
                "method": "GET",
                "endpoint": "/system/metrics",
                "auth_required": False,
            },
            {
                "action": "reload_config",
                "description": "Reload policies and models from disk",
                "method": "POST",
                "endpoint": "/admin/reload",
                "auth_required": True,
            },
            {
                "action": "check_guardrails",
                "description": "Manually trigger a guardrail check",
                "method": "POST",
                "endpoint": "/guardrails/check",
                "auth_required": True,
            },
        ],
        "lifecycle": {
            "restart_on_critical": True,
            "max_restarts": 3,
            "restart_window": "5m",
        },
    }
