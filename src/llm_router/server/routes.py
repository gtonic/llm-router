"""API routes for the LLM Router & Gateway.

Implements OpenAI-compatible endpoints plus gateway-specific routes.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncGenerator
from dataclasses import asdict

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
