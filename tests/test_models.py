"""Tests for request-model DoS/cost guard rails."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from llm_router.models import (
    MAX_MESSAGES,
    MAX_OUTPUT_TOKENS,
    MAX_TOTAL_PROMPT_CHARS,
    ChatCompletionRequest,
)


def _msg(content="hi"):
    return {"role": "user", "content": content}


def test_valid_request_passes():
    req = ChatCompletionRequest(model="m", messages=[_msg()], max_tokens=128)
    assert req.max_tokens == 128
    assert len(req.messages) == 1


def test_rejects_empty_messages():
    with pytest.raises(ValidationError):
        ChatCompletionRequest(model="m", messages=[])


def test_rejects_too_many_messages():
    with pytest.raises(ValidationError):
        ChatCompletionRequest(model="m", messages=[_msg() for _ in range(MAX_MESSAGES + 1)])


def test_rejects_max_tokens_over_ceiling():
    with pytest.raises(ValidationError):
        ChatCompletionRequest(model="m", messages=[_msg()], max_tokens=MAX_OUTPUT_TOKENS + 1)


def test_rejects_non_positive_max_tokens():
    with pytest.raises(ValidationError):
        ChatCompletionRequest(model="m", messages=[_msg()], max_tokens=0)


def test_rejects_oversized_prompt_content():
    huge = "x" * (MAX_TOTAL_PROMPT_CHARS + 1)
    with pytest.raises(ValidationError):
        ChatCompletionRequest(model="m", messages=[_msg(huge)])


def test_content_part_text_counts_toward_size_limit():
    part = {"type": "text", "text": "x" * (MAX_TOTAL_PROMPT_CHARS + 1)}
    with pytest.raises(ValidationError):
        ChatCompletionRequest(model="m", messages=[{"role": "user", "content": [part]}])
