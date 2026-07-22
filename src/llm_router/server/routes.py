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

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from llm_router.guardrails.content_safety import ContentSafetyBlockedError
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
from llm_router.pool.pool import ModelPool
from llm_router.router import AbuseDetectedError, RateLimitExceededError
from llm_router.server.app import (
    PROMETHEUS_ENABLED,
    record_admin_action,
    record_backend_health,
    record_empty_response,
    record_error,
    record_first_frame,
    record_request,
    record_route_decision,
    record_ttft,
)
from llm_router.server.skill_manifest import (
    get_skill_index,
    get_skill_md,
)
from llm_router.server.system_manifest import (
    BackendHealth,
    build_manifest,
)

logger = logging.getLogger("llm-router")


def require_admin(x_admin_token: str | None = Header(default=None)) -> None:
    """Require the configured admin token for runtime mutation endpoints."""
    import secrets

    from llm_router.server.app import router_engine

    configured = router_engine.settings.admin_token if router_engine else ""
    if not configured:
        raise HTTPException(status_code=503, detail="Admin authentication is not configured")
    if not x_admin_token or not secrets.compare_digest(x_admin_token, configured):
        raise HTTPException(status_code=401, detail="Invalid admin token")


def require_api_key(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> str:
    """Authenticate a data-plane request; return a stable, non-secret principal.

    Returns ``"anonymous"`` when no keys are configured (auth disabled). When
    ``ROUTER_API_KEYS`` is set, a valid key is required via
    ``Authorization: Bearer <key>`` or the ``X-API-Key`` header, and a hashed
    principal id (never the raw key) is returned for use as the tenant identity.
    """
    import hashlib
    import secrets

    from llm_router.server.app import router_engine

    keys = router_engine.settings.data_plane_keys() if router_engine else set()
    if not keys:
        return "anonymous"
    presented: str | None = None
    if authorization and authorization.lower().startswith("bearer "):
        presented = authorization[7:].strip()
    elif x_api_key:
        presented = x_api_key.strip()
    if presented and any(secrets.compare_digest(presented, key) for key in keys):
        return "key-" + hashlib.sha256(presented.encode()).hexdigest()[:12]
    raise HTTPException(status_code=401, detail="Missing or invalid API key")


router = APIRouter(prefix="/v1", tags=["llm-router"])
admin_router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(require_admin)])
skill_router = APIRouter(tags=["agent-skills"])
system_router = APIRouter(tags=["system"])


# ───────────────────────────────────────────
# Pydantic request/response models for admin API
# ───────────────────────────────────────────


class ModelBackendCreate(BaseModel):
    """Request body for adding a new model backend."""

    id: str = Field(..., description="Unique backend identifier")
    name: str = Field(..., description="Human-readable name")
    type: str = Field(..., description="Backend type: 'local' or 'remote'")
    base_url: str = Field(..., description="API base URL")
    api_key: str = Field(default="", description="API key (empty for local)")
    model_name: str = Field(default="", description="Model name for the backend")
    enabled: bool = Field(default=True, description="Whether backend is enabled")
    temperature: float = Field(default=0.3, description="Default temperature")
    max_tokens: int = Field(default=4096, description="Max output tokens")
    timeout: int = Field(default=60, description="Request timeout in seconds")
    retry_count: int = Field(default=3, description="Retry count on failure")
    cost_per_1m_input: float = Field(default=0.0, description="Cost per 1M input tokens (USD)")
    cost_per_1m_output: float = Field(default=0.0, description="Cost per 1M output tokens (USD)")
    tags: list[str] = Field(default_factory=list, description="Tags for routing")


class ModelBackendUpdate(BaseModel):
    """Request body for updating an existing model backend."""

    name: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    model_name: str | None = None
    enabled: bool | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    timeout: int | None = None
    retry_count: int | None = None
    cost_per_1m_input: float | None = None
    cost_per_1m_output: float | None = None
    tags: list[str] | None = None


class GuardrailConfigUpdate(BaseModel):
    """Request body for updating guardrail configuration."""

    pii_enabled: bool | None = None
    pii_redact: bool | None = None
    pii_max_tokens: int | None = None
    pii_patterns: list[str] | None = None
    abuse_block_threshold: float | None = None
    abuse_enabled: bool | None = None
    safety_enabled: bool | None = None
    safety_categories: list[str] | None = None


class PiiPatternCreate(BaseModel):
    """Request body for adding a custom PII pattern."""

    name: str = Field(..., description="Human-readable pattern name")
    pattern: str = Field(..., description="Regex pattern string")
    sensitivity: str = Field(default="high", description="Sensitivity level: low|medium|high")


class AdminResponse(BaseModel):
    """Standard admin response envelope."""

    status: str
    message: str
    details: dict | None = None


def _settings():
    """Return the live gateway settings used by the running engine."""
    from llm_router.server.app import router_engine as live_engine

    if live_engine is None:
        raise HTTPException(status_code=503, detail="Router not initialized")
    return live_engine.settings


def _persist_models() -> None:
    """Persist the current in-memory backend configurations."""
    from llm_router.server.app import router_engine as live_engine

    settings = _settings()
    settings.save_models_to_yaml([backend.config for backend in live_engine.pool._backends.values()])


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
async def chat_completions(
    request: Request,
    body: ChatCompletionRequest,
    principal: str = Depends(require_api_key),
    x_session_id: str | None = Header(default=None, alias="X-Session-Id"),
):
    """OpenAI-compatible chat completion endpoint.

    Routes the request through guardrails, policy router, and model backend.
    Supports both streaming and non-streaming responses.
    """
    from llm_router.server.app import router_engine

    if router_engine is None:
        raise HTTPException(status_code=503, detail="Router not initialized")

    request_id = str(uuid.uuid4())[:8]
    logger.info("[%s] %s %s", request_id, request.client.host, "POST /v1/chat/completions")

    client_ip = request.client.host if request.client else None
    # Bind the rate-limit / audit identity to the authenticated key when auth is
    # enabled; the body's user_id/api_key are client-controlled and spoofable.
    if principal != "anonymous":
        eff_user_id, eff_api_key = principal, None
    else:
        eff_user_id, eff_api_key = body.user_id, body.api_key

    # Non-streaming
    if not body.stream:
        try:
            start = time.perf_counter()
            result = await router_engine.generate(
                messages=_normalize_messages(body.model_dump()["messages"]),
                user_id=eff_user_id,
                api_key=eff_api_key,
                model=body.model,
                tools=body.tools,
                max_tokens=body.max_tokens,
                client_ip=client_ip,
                session_id=x_session_id,
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
                    cost=result.usage.cost,
                ),
            )

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
                record_route_decision(strategy)

            logger.info("[%s] %s %s %.1fms model=%s", request_id, request.client.host, "OK", elapsed_ms, result.model)
            return response

        except RateLimitExceededError as exc:
            logger.warning("[%s] Rate limited: %s", request_id, exc)
            if PROMETHEUS_ENABLED:
                record_error(model=body.model or "unknown", error_type=type(exc).__name__)
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        except (AbuseDetectedError, ContentSafetyBlockedError) as exc:
            logger.warning("[%s] Blocked: %s", request_id, exc)
            if PROMETHEUS_ENABLED:
                record_error(model=body.model or "unknown", error_type=type(exc).__name__)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.error("[%s] Error: %s", request_id, exc)
            if PROMETHEUS_ENABLED:
                record_error(model=body.model or "unknown", error_type=type(exc).__name__)
                if type(exc).__name__ == "EmptyResponseError":
                    record_empty_response(body.model or "unknown")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    # Streaming
    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            start = time.perf_counter()
            last_finish_reason = "stop"
            streamed_model = body.model
            streamed_usage = None
            ttft_recorded = False
            first_frame_recorded = False
            async for chunk in router_engine.generate_stream(
                messages=_normalize_messages(body.model_dump()["messages"]),
                user_id=eff_user_id,
                api_key=eff_api_key,
                model=body.model,
                tools=body.tools,
                max_tokens=body.max_tokens,
                client_ip=client_ip,
                session_id=x_session_id,
            ):
                streamed_model = chunk.model or streamed_model
                # Keep the real token usage from the terminal accounting chunk
                # (content chunks carry zeros); don't let it be overwritten back.
                if chunk.usage and chunk.usage.total_tokens:
                    streamed_usage = chunk.usage
                # The terminal accounting chunk has no content/tool_calls — capture
                # its usage above but don't emit an empty delta to the client.
                if not chunk.content and not chunk.tool_calls:
                    continue
                if PROMETHEUS_ENABLED:
                    now = time.perf_counter()
                    # First frame: first streamed chunk of any kind (prefill + connect).
                    if not first_frame_recorded:
                        record_first_frame(streamed_model or "unknown", now - start)
                        first_frame_recorded = True
                    # First token: first chunk with user-visible content. The gap
                    # first_token - first_frame is time spent on pre-content tokens.
                    if not ttft_recorded and (chunk.content or chunk.tool_calls):
                        record_ttft(streamed_model or "unknown", now - start)
                        ttft_recorded = True
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
            if PROMETHEUS_ENABLED:
                usage = streamed_usage or ModelsUsageInfo(prompt_tokens=0, completion_tokens=0, total_tokens=0)
                record_request(
                    model=streamed_model or "unknown",
                    strategy=router_engine.routing_strategy.value,
                    status="success",
                    duration=time.perf_counter() - start,
                    cost=getattr(usage, "cost", 0.0) or 0.0,
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                )
                record_route_decision(router_engine.routing_strategy.value)
            final_chunk = ChatCompletionChunk(
                model=streamed_model or body.model,
                choices=[ChunkChoice(delta={}, finish_reason=last_finish_reason)],
            )
            yield f"data: {final_chunk.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as exc:
            logger.error("[%s] Stream error: %s", request_id, exc)
            if PROMETHEUS_ENABLED:
                record_error(model=body.model or "unknown", error_type=type(exc).__name__)
                if type(exc).__name__ == "EmptyResponseError":
                    record_empty_response(body.model or "unknown")
            # Don't echo raw exception text (may leak internals) — log it above,
            # send the client a generic message correlated by request_id.
            error_chunk = ChatCompletionChunk(
                model="error",
                choices=[
                    ChunkChoice(
                        delta={"content": f"Internal error during streaming (request {request_id})"},
                        finish_reason="error",
                    )
                ],
            )
            yield f"data: {error_chunk.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/models", dependencies=[Depends(require_api_key)])
async def list_models():
    """List available models (OpenAI-compatible)."""
    from llm_router.server.app import router_engine

    if router_engine is None:
        raise HTTPException(status_code=503, detail="Router not initialized")

    return ModelListResponse(
        data=[ModelInfo(id="router-auto")] + [ModelInfo(id=model_id) for model_id in router_engine.pool.list_models()]
    )


@router.get("/guardrails/pii/patterns", dependencies=[Depends(require_api_key)])
async def get_pii_patterns():
    """Return configured PII detection patterns."""
    from llm_router.server.app import router_engine

    if router_engine is None:
        raise HTTPException(status_code=503, detail="Router not initialized")

    return {
        "patterns": [name for _, name in router_engine.pii_filter.PATTERNS],
        "redact_enabled": router_engine.pii_filter.redact,
    }


@router.post("/guardrails/check", dependencies=[Depends(require_api_key)])
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


@admin_router.post("/reload")
async def reload_config():
    """Reload policies and models from disk."""
    from llm_router.server.app import router_engine

    if router_engine is None:
        raise HTTPException(status_code=503, detail="Router not initialized")

    try:
        router_engine.pool.reload()
        router_engine.policy_matcher._load_policies()
        router_engine.sync_runtime_models()
        record_admin_action("reload")
        return {
            "status": "reloaded",
            "policies": len(router_engine.policy_matcher.rules),
            "models": len(router_engine.pool.list_models()),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ───────────────────────────────────────────
# Admin API — Runtime Configuration (P0)
# ───────────────────────────────────────────


@admin_router.get("/models")
async def list_admin_models():
    """List all model backends (enabled + disabled) with full config."""
    from llm_router.server.app import router_engine

    if router_engine is None:
        raise HTTPException(status_code=503, detail="Router not initialized")

    models = []
    for _mid, backend in router_engine.pool._backends.items():
        config = backend.config
        models.append(
            {
                "id": config.id,
                "name": config.name,
                "type": config.type,
                "base_url": config.base_url,
                "enabled": config.enabled,
                "is_local": config.is_local,
                "is_remote": config.is_remote,
                "tags": config.tags,
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
                "timeout": config.timeout,
                "retry_count": config.retry_count,
                "cost_per_1m_input": config.cost_per_1m_input,
                "cost_per_1m_output": config.cost_per_1m_output,
            }
        )

    return {
        "status": "ok",
        "count": len(models),
        "models": models,
    }


@admin_router.post("/models", status_code=201)
async def create_model_backend(body: ModelBackendCreate):
    """Add a new model backend at runtime.

    Creates the backend, adds it to the pool, and persists to YAML.
    The backend becomes immediately available for routing.
    """
    from llm_router.config import ModelBackendConfig
    from llm_router.server.app import router_engine

    if router_engine is None:
        raise HTTPException(status_code=503, detail="Router not initialized")

    # Check for duplicate ID
    if body.id in router_engine.pool._backends:
        raise HTTPException(
            status_code=409,
            detail=f"Backend with id '{body.id}' already exists. Available: {router_engine.pool.list_models()}",
        )

    # Validate type
    if body.type not in ("local", "remote"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid type '{body.type}'. Must be 'local' or 'remote'.",
        )

    # Create config and backend
    cfg = ModelBackendConfig(
        id=body.id,
        name=body.name,
        type=body.type,
        base_url=body.base_url,
        api_key=body.api_key,
        model_name=body.model_name or body.id,
        enabled=body.enabled,
        temperature=body.temperature,
        max_tokens=body.max_tokens,
        timeout=body.timeout,
        retry_count=body.retry_count,
        cost_per_1m_input=body.cost_per_1m_input,
        cost_per_1m_output=body.cost_per_1m_output,
        tags=body.tags,
    )

    try:
        backend = ModelPool._create_backend(cfg)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to create backend: {exc}") from exc

    # Add to pool
    router_engine.pool.add_backend(cfg, backend)

    # Persist to YAML
    try:
        _persist_models()
        router_engine.sync_runtime_models()
    except Exception as persist_err:
        logger.warning("Failed to persist model config to YAML: %s", persist_err)
        # Don't fail the request — backend is still live in memory

    logger.info("Admin: Added backend '%s' (type=%s, enabled=%s)", body.id, body.type, body.enabled)
    record_admin_action("model_create")

    return AdminResponse(
        status="created",
        message=f"Backend '{body.id}' added successfully",
        details={"id": body.id, "enabled": body.enabled},
    )


@admin_router.get("/models/{model_id}")
async def get_model_backend(model_id: str):
    """Get configuration for a specific model backend."""
    from llm_router.server.app import router_engine

    if router_engine is None:
        raise HTTPException(status_code=503, detail="Router not initialized")

    if model_id not in router_engine.pool._backends:
        raise HTTPException(
            status_code=404,
            detail=f"Backend '{model_id}' not found. Available: {router_engine.pool.list_models()}",
        )

    backend = router_engine.pool._backends[model_id]
    config = backend.config

    return {
        "id": config.id,
        "name": config.name,
        "type": config.type,
        "base_url": config.base_url,
        "enabled": config.enabled,
        "is_local": config.is_local,
        "is_remote": config.is_remote,
        "tags": config.tags,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "timeout": config.timeout,
        "retry_count": config.retry_count,
        "cost_per_1m_input": config.cost_per_1m_input,
        "cost_per_1m_output": config.cost_per_1m_output,
    }


@admin_router.put("/models/{model_id}")
async def update_model_backend(model_id: str, body: ModelBackendUpdate):
    """Update an existing model backend's configuration.

    Only provided fields are updated (partial update).
    Changes are persisted to YAML.
    """
    from llm_router.server.app import router_engine

    if router_engine is None:
        raise HTTPException(status_code=503, detail="Router not initialized")

    if model_id not in router_engine.pool._backends:
        raise HTTPException(
            status_code=404,
            detail=f"Backend '{model_id}' not found. Available: {router_engine.pool.list_models()}",
        )

    backend = router_engine.pool._backends[model_id]
    config = backend.config

    # Apply partial update and rebuild the live client for connection changes.
    updates = body.model_dump(exclude_unset=True)
    candidate = type(config)(**{**config.__dict__, **updates})
    if {"base_url", "api_key", "model_name"}.intersection(updates):
        try:
            router_engine.pool.replace_backend(candidate)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to rebuild backend: {exc}") from exc
    else:
        for key, value in updates.items():
            if hasattr(config, key):
                setattr(config, key, value)

    # Persist to YAML
    try:
        _persist_models()
        router_engine.sync_runtime_models()
    except Exception as persist_err:
        logger.warning("Failed to persist model config to YAML: %s", persist_err)

    logger.info("Admin: Updated backend '%s': %s", model_id, list(updates.keys()))
    record_admin_action("model_update")

    return AdminResponse(
        status="updated",
        message=f"Backend '{model_id}' updated",
        details={"updated_fields": list(updates.keys())},
    )


@admin_router.delete("/models/{model_id}", status_code=200)
async def delete_model_backend(model_id: str):
    """Remove a model backend at runtime.

    The backend is removed from the pool and the config is persisted.
    Cannot remove the last remaining backend.
    """
    from llm_router.server.app import router_engine

    if router_engine is None:
        raise HTTPException(status_code=503, detail="Router not initialized")

    if model_id not in router_engine.pool._backends:
        raise HTTPException(
            status_code=404,
            detail=f"Backend '{model_id}' not found. Available: {router_engine.pool.list_models()}",
        )

    if len(router_engine.pool._backends) <= 1:
        raise HTTPException(
            status_code=400,
            detail="Cannot remove the last remaining backend. Add another first.",
        )

    # Remove from pool
    del router_engine.pool._backends[model_id]

    # Persist to YAML
    try:
        _persist_models()
        router_engine.sync_runtime_models()
    except Exception as persist_err:
        logger.warning("Failed to persist model config to YAML: %s", persist_err)

    logger.info("Admin: Removed backend '%s'", model_id)
    record_admin_action("model_delete")

    return AdminResponse(
        status="deleted",
        message=f"Backend '{model_id}' removed",
        details={"remaining": router_engine.pool.list_models()},
    )


@admin_router.patch("/models/{model_id}/toggle")
async def toggle_model_backend(model_id: str):
    """Toggle a backend's enabled/disabled state.

    Disabled backends remain in the pool but won't receive traffic.
    """
    from llm_router.server.app import router_engine

    if router_engine is None:
        raise HTTPException(status_code=503, detail="Router not initialized")

    if model_id not in router_engine.pool._backends:
        raise HTTPException(
            status_code=404,
            detail=f"Backend '{model_id}' not found. Available: {router_engine.pool.list_models()}",
        )

    backend = router_engine.pool._backends[model_id]
    config = backend.config
    new_state = not config.enabled
    config.enabled = new_state

    # Persist to YAML
    try:
        _persist_models()
        router_engine.sync_runtime_models()
    except Exception as persist_err:
        logger.warning("Failed to persist model config to YAML: %s", persist_err)

    logger.info("Admin: Toggled backend '%s' to %s", model_id, "enabled" if new_state else "disabled")
    record_admin_action("model_toggle")

    return AdminResponse(
        status="toggled",
        message=f"Backend '{model_id}' is now {'enabled' if new_state else 'disabled'}",
        details={"id": model_id, "enabled": new_state},
    )


# ───────────────────────────────────────────
# Guardrail Hot-Toggle (P0)
# ───────────────────────────────────────────


@admin_router.get("/guardrails")
async def get_guardrails():
    """Get current guardrail configuration and status."""
    from llm_router.server.app import router_engine

    if router_engine is None:
        raise HTTPException(status_code=503, detail="Router not initialized")

    return {
        "status": "ok",
        "guardrails": {
            "pii": {
                "enabled": router_engine.settings.guardrails.pii_enabled,
                "redact": router_engine.pii_filter.redact,
                "patterns_count": len(router_engine.pii_filter.PATTERNS),
                "max_tokens": router_engine.settings.guardrails.pii_max_tokens,
            },
            "abuse": {
                "enabled": router_engine.settings.guardrails.abuse_enabled,
                "block_threshold": router_engine.settings.guardrails.abuse_block_threshold,
            },
            "content_safety": {
                "enabled": (
                    router_engine.content_safety.block_threshold.value
                    if hasattr(router_engine.content_safety, "block_threshold")
                    else "medium"
                ),
            },
            "rate_limiting": {
                "enabled": router_engine.settings.rate_limit.enabled,
                "rpm": router_engine.rate_limiter.rpm,
                "tpm": router_engine.rate_limiter.tpm,
            },
        },
    }


@admin_router.post("/rollback")
async def rollback_runtime_config():
    """Restore the previous persisted guardrail/runtime configuration."""
    from llm_router.guardrails.content_safety import SafetyLevel
    from llm_router.server.app import router_engine

    if router_engine is None:
        raise HTTPException(status_code=503, detail="Router not initialized")
    if not router_engine.settings.rollback_runtime_config():
        raise HTTPException(status_code=404, detail="No runtime configuration snapshot available")

    settings = router_engine.settings
    router_engine.pii_filter._redact = settings.guardrails.pii_redact
    router_engine.abuse_filter.block_threshold = settings.guardrails.abuse_block_threshold
    router_engine.content_safety.block_threshold = (
        SafetyLevel.MEDIUM if settings.guardrails.safety_enabled else SafetyLevel.CRITICAL
    )
    router_engine.rate_limiter.rpm = settings.rate_limit.rpm
    router_engine.rate_limiter.tpm = settings.rate_limit.tpm
    record_admin_action("config_rollback")
    return AdminResponse(status="rolled_back", message="Runtime configuration restored")


@admin_router.put("/guardrails")
async def update_guardrails(body: GuardrailConfigUpdate):
    """Update guardrail configuration at runtime.

    All changes take effect immediately for subsequent requests.
    Changes are persisted to the gateway settings.
    """
    from llm_router.server.app import router_engine

    if router_engine is None:
        raise HTTPException(status_code=503, detail="Router not initialized")

    updates = body.model_dump(exclude_unset=True)
    applied = []

    # PII settings
    if "pii_redact" in updates:
        router_engine.pii_filter._redact = updates["pii_redact"]
        applied.append("pii_redact")

    if "pii_enabled" in updates:
        router_engine.settings.guardrails.pii_enabled = updates["pii_enabled"]
        applied.append("pii_enabled")

    if "pii_max_tokens" in updates:
        router_engine.settings.guardrails.pii_max_tokens = updates["pii_max_tokens"]
        applied.append("pii_max_tokens")

    if "pii_patterns" in updates:
        # Replace only the custom patterns (built-ins are left untouched) — mirrors
        # PiiFilter.__init__'s custom_patterns handling, not a positional overwrite.
        import re as _re

        builtin_patterns = [
            (name, compiled)
            for name, compiled in router_engine.pii_filter.PATTERNS
            if name not in router_engine.pii_filter._custom_patterns
        ]
        new_custom = []
        new_custom_names = set()
        for index, raw_pattern in enumerate(updates["pii_patterns"]):
            try:
                compiled = _re.compile(raw_pattern)
            except _re.error:
                continue
            name = f"custom_{index}"
            new_custom.append((name, compiled))
            new_custom_names.add(name)
        router_engine.pii_filter.PATTERNS = new_custom + builtin_patterns
        router_engine.pii_filter._custom_patterns = new_custom_names
        applied.append("pii_patterns")

    # Abuse settings
    if "abuse_block_threshold" in updates:
        router_engine.settings.guardrails.abuse_block_threshold = updates["abuse_block_threshold"]
        router_engine.abuse_filter.block_threshold = updates["abuse_block_threshold"]
        applied.append("abuse_block_threshold")

    if "abuse_enabled" in updates:
        router_engine.settings.guardrails.abuse_enabled = updates["abuse_enabled"]
        applied.append("abuse_enabled")

    # Content safety settings
    if "safety_enabled" in updates:
        from llm_router.guardrails.content_safety import SafetyLevel

        if updates["safety_enabled"]:
            router_engine.content_safety.block_threshold = SafetyLevel.MEDIUM
        else:
            router_engine.content_safety.block_threshold = SafetyLevel.CRITICAL
        applied.append("safety_enabled")

    if "safety_categories" in updates:
        # Update the dangerous keywords dynamically
        from llm_router.guardrails.content_safety import SafetyLevel

        router_engine.content_safety._keywords[SafetyLevel.MEDIUM] = {
            keyword.lower() for keyword in updates["safety_categories"]
        }
        applied.append("safety_categories")

    # Persist guardrail config
    try:
        settings = router_engine.settings
        settings.guardrails.pii_redact = router_engine.pii_filter.redact
        settings.guardrails.abuse_block_threshold = router_engine.settings.guardrails.abuse_block_threshold
        if "safety_enabled" in updates:
            settings.guardrails.safety_enabled = updates["safety_enabled"]
        if "safety_categories" in updates:
            settings.guardrails.safety_categories = updates["safety_categories"]
        settings.save_runtime_config()
    except Exception as persist_err:
        logger.warning("Failed to persist guardrail config: %s", persist_err)

    logger.info("Admin: Updated guardrails: %s", applied)
    record_admin_action("guardrail_update")

    return AdminResponse(
        status="updated",
        message="Guardrails updated",
        details={"applied": applied},
    )


@admin_router.patch("/guardrails/{guardrail_name}/toggle")
async def toggle_guardrail(guardrail_name: str):
    """Toggle a specific guardrail on/off.

    Supported: pii, abuse, content_safety, rate_limiting
    """
    from llm_router.server.app import router_engine

    if router_engine is None:
        raise HTTPException(status_code=503, detail="Router not initialized")

    guardrail_name = guardrail_name.lower()

    if guardrail_name == "pii":
        new_state = not router_engine.settings.guardrails.pii_enabled
        router_engine.settings.guardrails.pii_enabled = new_state
        status_msg = f"PII filter {'enabled' if new_state else 'disabled'}"
    elif guardrail_name == "abuse":
        # Abuse filter is always on at the engine level; toggle via threshold
        if router_engine.settings is not None:
            current = router_engine.settings.guardrails.abuse_block_threshold
            new_threshold = 1.0 if current < 1.0 else 0.8  # 1.0 = effectively disabled
            router_engine.settings.guardrails.abuse_block_threshold = new_threshold
            router_engine.abuse_filter.block_threshold = new_threshold
            new_state = new_threshold < 1.0
            router_engine.settings.guardrails.abuse_enabled = new_state
            status_msg = f"Abuse filter {'enabled' if new_state else 'disabled'} (threshold={new_threshold})"
        else:
            raise HTTPException(status_code=500, detail="Cannot toggle abuse: settings not available")
    elif guardrail_name == "content_safety":
        from llm_router.guardrails.content_safety import SafetyLevel

        if router_engine.content_safety.block_threshold >= SafetyLevel.CRITICAL:
            router_engine.content_safety.block_threshold = SafetyLevel.MEDIUM
            new_state = True
        else:
            router_engine.content_safety.block_threshold = SafetyLevel.CRITICAL
            new_state = False
        status_msg = f"Content safety {'enabled' if new_state else 'disabled'}"
    elif guardrail_name == "rate_limiting":
        if not router_engine.settings.rate_limit.enabled:
            router_engine.settings.rate_limit.enabled = True
            new_state = True
        else:
            router_engine.settings.rate_limit.enabled = False
            new_state = False
        status_msg = f"Rate limiting {'enabled' if new_state else 'disabled'}"
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown guardrail '{guardrail_name}'. Supported: pii, abuse, content_safety, rate_limiting",
        )

    logger.info("Admin: Toggled guardrail '%s' → %s", guardrail_name, status_msg)
    record_admin_action("guardrail_toggle")

    try:
        router_engine.settings.guardrails.pii_redact = router_engine.pii_filter.redact
        router_engine.settings.guardrails.safety_enabled = (
            router_engine.content_safety.block_threshold.value != "critical"
        )
        router_engine.settings.save_runtime_config()
    except Exception as persist_err:
        logger.warning("Failed to persist guardrail toggle: %s", persist_err)

    return AdminResponse(
        status="toggled",
        message=status_msg,
        details={"guardrail": guardrail_name, "enabled": new_state},
    )


# ───────────────────────────────────────────
# PII Pattern Management
# ───────────────────────────────────────────


@admin_router.post("/guardrails/pii/patterns")
async def add_pii_pattern(body: PiiPatternCreate):
    """Add a custom PII detection pattern.

    The pattern is compiled and added to the active PII filter.
    """
    import re

    from llm_router.server.app import router_engine

    if router_engine is None:
        raise HTTPException(status_code=503, detail="Router not initialized")

    try:
        compiled = re.compile(body.pattern)
    except re.error as exc:
        raise HTTPException(status_code=400, detail=f"Invalid regex pattern: {exc}") from exc

    # Add to PATTERNS (prepend so it's checked first)
    new_patterns = [(body.name, compiled)] + list(router_engine.pii_filter.PATTERNS)
    router_engine.pii_filter.PATTERNS = new_patterns
    router_engine.pii_filter._custom_patterns.add(body.name)
    router_engine.settings.guardrails.pii_patterns.append(body.pattern)
    router_engine.settings.save_runtime_config()

    logger.info("Admin: Added PII pattern '%s' (%d total patterns)", body.name, len(new_patterns))

    return AdminResponse(
        status="created",
        message=f"PII pattern '{body.name}' added",
        details={"name": body.name, "total_patterns": len(new_patterns)},
    )


@admin_router.delete("/guardrails/pii/patterns/{pattern_name}")
async def remove_pii_pattern(pattern_name: str):
    """Remove a PII pattern by name.

    Built-in patterns cannot be removed. Only custom patterns can be deleted.
    """
    from llm_router.server.app import router_engine

    if router_engine is None:
        raise HTTPException(status_code=503, detail="Router not initialized")

    # Find and remove the pattern
    original_count = len(router_engine.pii_filter.PATTERNS)
    if pattern_name not in router_engine.pii_filter._custom_patterns:
        raise HTTPException(status_code=400, detail=f"Built-in pattern '{pattern_name}' cannot be removed")

    new_patterns = [(name, pat) for name, pat in router_engine.pii_filter.PATTERNS if name != pattern_name]

    if len(new_patterns) == original_count:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Pattern '{pattern_name}' not found. Available: {[n for n, _ in router_engine.pii_filter.PATTERNS]}"
            ),
        )

    router_engine.pii_filter.PATTERNS = new_patterns
    router_engine.pii_filter._custom_patterns.remove(pattern_name)
    router_engine.settings.save_runtime_config()
    logger.info("Admin: Removed PII pattern '%s' (%d remaining)", pattern_name, len(new_patterns))

    return AdminResponse(
        status="deleted",
        message=f"PII pattern '{pattern_name}' removed",
        details={"remaining": len(new_patterns)},
    )


# ───────────────────────────────────────────
# System Manifest Endpoints (DevOps Monitoring)
# ───────────────────────────────────────────

# Module-level start time for uptime calculation
_start_time: str = ""


@router.get("/system/manifest", dependencies=[Depends(require_api_key)])
@system_router.get("/system/manifest", dependencies=[Depends(require_api_key)])
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

    from llm_router.server.app import router_engine

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
        settings=router_engine.settings,
        start_time=_start_time,
        uptime_seconds=uptime,
        models=model_configs,
        health_status=backend_health,
    )

    return manifest.to_dict()


@router.get("/system/health")
@system_router.get("/system/health")
async def get_system_health():
    """Detailed health check — backends, routing, guardrails."""
    from llm_router.server.app import router_engine

    if router_engine is None:
        raise HTTPException(status_code=503, detail="Router not initialized")

    health_results = await router_engine.pool.health_check_all()

    backends = {}
    all_healthy = True
    for mid, backend in router_engine.pool._backends.items():
        status = health_results.get(mid)
        if status is None:
            continue
        if PROMETHEUS_ENABLED:
            record_backend_health(mid, backend.config.type, status.healthy)
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
            "pii_enabled": router_engine.settings.guardrails.pii_enabled,
            "abuse_enabled": router_engine.settings.guardrails.abuse_enabled,
            "content_safety_enabled": router_engine.settings.guardrails.safety_enabled,
            "rate_limiting_enabled": router_engine.settings.rate_limit.enabled,
        },
        "policies_loaded": len(router_engine.policy_matcher.rules),
    }


@router.get("/system/metrics", dependencies=[Depends(require_api_key)])
@system_router.get("/system/metrics", dependencies=[Depends(require_api_key)])
async def get_system_metrics():
    """Current metrics snapshot for Prometheus-compatible consumption.

    Returns a JSON snapshot of all counters, histograms, and gauges.
    Also exposes the raw Prometheus text format at /metrics.
    """
    from llm_router.server.app import router_engine

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
@system_router.get("/system/capabilities")
async def get_system_capabilities():
    """Return allowed lifecycle actions.

    This tells DevOps agents what operations they can perform on this system.
    """
    from llm_router.server.app import router_engine

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
                "auth_required": True,
            },
            {
                "action": "get_metrics",
                "description": "Fetch current metrics",
                "method": "GET",
                "endpoint": "/system/metrics",
                "auth_required": True,
            },
            {
                "action": "manage_models",
                "description": "Create, update, toggle, or remove model backends",
                "method": "GET|POST|PUT|PATCH|DELETE",
                "endpoint": "/admin/models",
                "auth_required": True,
            },
            {
                "action": "manage_guardrails",
                "description": "Inspect and change runtime guardrails",
                "method": "GET|PUT|PATCH",
                "endpoint": "/admin/guardrails",
                "auth_required": True,
            },
            {
                "action": "rollback_runtime_config",
                "description": "Restore the previous runtime configuration snapshot",
                "method": "POST",
                "endpoint": "/admin/rollback",
                "auth_required": True,
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


# ───────────────────────────────────────────
# Agent Skill Discovery (Cloudflare RFC v0.2.0)
# ───────────────────────────────────────────


@skill_router.get("/.well-known/agent-skills/index.json")
async def get_agent_skills_index():
    """Return the agent skills discovery index.

    Implements the Cloudflare Agent Skills Discovery RFC (v0.2.0 draft).
    Agents fetch this to discover available skills without prior configuration.

    Returns a JSON index with skill metadata (name, description, url, digest).
    """
    return get_skill_index()


@skill_router.get("/.well-known/agent-skills/llm-router-gateway/SKILL.md")
async def get_agent_skill():
    """Return the llm-router-gateway Agent Skill artifact.

    This is the SKILL.md file that agents load when the skill is activated.
    The skill is generic and agent-agnostic (works with Claude Code, Codex CLI,
    OpenClaw, Cursor, Gemini CLI, Hermes Agent, etc.).
    """
    from fastapi.responses import PlainTextResponse

    return PlainTextResponse(
        content=get_skill_md(),
        media_type="text/markdown",
    )
