"""Tests for pool/base.py — GenerateResult, UsageInfo, HealthStatus, ModelBackend ABC."""

from __future__ import annotations

from abc import ABC

import pytest

from llm_router.pool.base import (
    GenerateResult,
    HealthStatus,
    ModelBackend,
    UsageInfo,
    merge_tool_calls,
    normalize_tool_calls,
)


class TestUsageInfo:
    def test_defaults(self):
        info = UsageInfo(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        assert info.prompt_tokens == 10
        assert info.completion_tokens == 20
        assert info.total_tokens == 30
        assert info.cost == 0.0

    def test_custom_cost(self):
        info = UsageInfo(prompt_tokens=100, completion_tokens=50, total_tokens=150, cost=0.003)
        assert info.cost == 0.003

    def test_dataclass_equality(self):
        a = UsageInfo(prompt_tokens=1, completion_tokens=1, total_tokens=2)
        b = UsageInfo(prompt_tokens=1, completion_tokens=1, total_tokens=2)
        assert a == b


class TestGenerateResult:
    def test_minimal(self):
        r = GenerateResult(
            content="hello",
            model="test-model",
            usage=UsageInfo(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            finish_reason="stop",
        )
        assert r.content == "hello"
        assert r.model == "test-model"
        assert r.finish_reason == "stop"
        assert r.tool_calls is None
        assert r.latency_ms == 0.0


class TestToolCallMerging:
    def test_merges_streamed_fragments(self):
        calls = []
        merge_tool_calls(
            calls,
            [
                {
                    "id": "call_abc",
                    "type": "function",
                    "function": {"name": "bash", "arguments": '{"command":'},
                }
            ],
        )
        merge_tool_calls(
            calls,
            [
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {"name": "", "arguments": '"pwd"}'},
                }
            ],
        )

        assert calls == [
            {
                "id": "call_abc",
                "type": "function",
                "function": {"name": "bash", "arguments": '{"command":"pwd"}'},
            }
        ]

    def test_keeps_parallel_tool_calls_separate_by_stream_index(self):
        calls = []
        merge_tool_calls(
            calls,
            normalize_tool_calls(
                [
                    {
                        "index": 0,
                        "id": "call_skill",
                        "name": "skill_view",
                        "args": '{"name":"llm-wiki"}',
                    },
                    {
                        "index": 1,
                        "id": "call_code",
                        "name": "execute_code",
                        "args": '{"code":"print(1)"}',
                    },
                ]
            ),
        )

        assert [call["function"] for call in calls] == [
            {"name": "skill_view", "arguments": '{"name":"llm-wiki"}'},
            {"name": "execute_code", "arguments": '{"code":"print(1)"}'},
        ]

    def test_full(self):
        r = GenerateResult(
            content="world",
            model="gpt-4",
            usage=UsageInfo(prompt_tokens=50, completion_tokens=100, total_tokens=150, cost=0.005),
            finish_reason="tool_calls",
            tool_calls=[{"name": "search", "arguments": "{}"}],
            latency_ms=42.5,
        )
        assert r.tool_calls == [{"name": "search", "arguments": "{}"}]
        assert r.latency_ms == 42.5

    def test_finish_reason_default(self):
        r = GenerateResult(
            content="",
            model="m",
            usage=UsageInfo(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            finish_reason="stop",
        )
        assert r.finish_reason == "stop"


class TestHealthStatus:
    def test_healthy(self):
        s = HealthStatus(healthy=True, latency_ms=12.3)
        assert s.healthy is True
        assert s.latency_ms == 12.3
        assert s.error is None

    def test_unhealthy(self):
        s = HealthStatus(healthy=False, latency_ms=0, error="connection refused")
        assert s.healthy is False
        assert s.error == "connection refused"


class TestModelBackendABC:
    def test_is_abstract(self):
        """ModelBackend must be an ABC — cannot be instantiated directly."""
        with pytest.raises(TypeError):
            ModelBackend()  # type: ignore[abstract]

    def test_is_subclass(self):
        assert issubclass(ModelBackend, ABC)

    def test_abstract_methods_exist(self):
        """Verify the expected abstract methods are present."""
        abstract_methods = getattr(ModelBackend, "__abstractmethods__", set())
        expected = {"generate", "generate_stream", "health_check"}
        assert expected.issubset(abstract_methods), f"Missing abstract methods: {expected - abstract_methods}"

    def test_generate_with_retry_exists(self):
        """generate_with_retry should be a concrete method on the base class."""
        assert hasattr(ModelBackend, "generate_with_retry")
        assert callable(ModelBackend.generate_with_retry)
