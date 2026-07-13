"""OpenAI-compatible request / response models plus router-specific types.

All models follow the OpenAI Chat Completions API format so any OpenAI SDK
client can talk to the router without changes.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, Field


# ───────────────────────────────────────────
# Enums
# ───────────────────────────────────────────

class MessageRole(str, Enum):
    system = "system"
    user = "user"
    assistant = "assistant"
    tool = "tool"


class FinishReason(str, Enum):
    stop = "stop"
    tool_calls = "tool_calls"
    length = "length"
    error = "error"


# ───────────────────────────────────────────
# Request Models
# ───────────────────────────────────────────

class FunctionCall(BaseModel):
    name: str
    arguments: str  # JSON string


class ToolCall(BaseModel):
    id: str = Field(default_factory=lambda: f"call_{uuid.uuid4().hex[:16]}")
    type: Literal["function"] = "function"
    function: FunctionCall


class ChatMessage(BaseModel):
    """A single message in a chat turn."""
    role: MessageRole
    content: str | None = None
    name: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""
    model: str
    messages: list[ChatMessage]
    tools: list[dict] | None = None
    stream: bool = False
    temperature: float = 0.3
    top_p: float = 1.0
    max_tokens: int | None = None
    stop: str | list[str] | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    seed: int | None = None
    response_format: dict | None = None

    # Router-specific (opaque to the client)
    user_id: str | None = None
    api_key: str | None = None


# ───────────────────────────────────────────
# Response Models
# ───────────────────────────────────────────

class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: float = 0.0


class ChunkChoice(BaseModel):
    delta: dict[str, Any]
    finish_reason: FinishReason | None = None
    index: int = 0


class ChatCompletionChunk(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(datetime.utcnow().timestamp()))
    model: str
    choices: list[ChunkChoice]


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: FinishReason = FinishReason.stop


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(datetime.utcnow().timestamp()))
    model: str
    choices: list[ChatCompletionChoice]
    usage: UsageInfo


# ───────────────────────────────────────────
# Model Registry
# ───────────────────────────────────────────

class ModelInfo(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int = Field(default_factory=lambda: int(datetime.utcnow().timestamp()))
    owned_by: str = "llm-router"
    permission: list[dict] = Field(default_factory=list)


class ModelListResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelInfo] = Field(default_factory=list)


# ───────────────────────────────────────────
# Router-Specific Models
# ───────────────────────────────────────────

class PiiResult(BaseModel):
    has_pii: bool
    patterns: list[str] = Field(default_factory=list)
    redacted_text: str | None = None


class AbuseResult(BaseModel):
    safe: bool = True
    abuse_score: float = 0.0
    categories: list[str] = Field(default_factory=list)
    details: str = ""


class GuardrailCheckRequest(BaseModel):
    text: str
    mode: Literal["input", "output"] = "input"


class GuardrailCheckResponse(BaseModel):
    safe: bool
    violations: list[str] = Field(default_factory=list)
    redacted_text: str | None = None
    abuse_score: float = 0.0
    pii_patterns: list[str] = Field(default_factory=list)


class RoutingDecision(BaseModel):
    model_id: str
    strategy: str
    policy_matched: str | None = None
    latency_ms: float = 0.0
