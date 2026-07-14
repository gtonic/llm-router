"""API routes for the LLM Router & Gateway.

Implements OpenAI-compatible endpoints plus gateway-specific routes.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncGenerator

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

logger = logging.getLogger("llm-router")
router = APIRouter(prefix="/v1", tags=["llm-router"])


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
                messages=body.model_dump()["messages"],
                user_id=body.user_id,
                api_key=body.api_key,
                model=body.model,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000

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
            logger.info("[%s] %s %s %.1fms model=%s", request_id, request.client.host, "OK", elapsed_ms, result.model)
            return response

        except Exception as exc:
            logger.error("[%s] Error: %s", request_id, exc)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    # Streaming
    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            async for chunk in router_engine.generate_stream(
                messages=body.model_dump()["messages"],
                user_id=body.user_id,
                api_key=body.api_key,
                model=body.model,
            ):
                chunk_data = ChatCompletionChunk(
                    model=chunk.model,
                    choices=[
                        ChunkChoice(
                            delta={"content": chunk.content, "role": "assistant"},
                            finish_reason=None,
                        )
                    ],
                )
                yield f"data: {chunk_data.model_dump_json()}\n\n"
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

    model_list = ModelListResponse(data=[ModelInfo(id=model_id) for model_id in router_engine.pool.list_models()])
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
            return result.model_dump()
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
