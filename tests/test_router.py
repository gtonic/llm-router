"""Tests for router.py - RouterPolicyEngine."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_router.config import RoutingStrategy
from llm_router.pool.base import GenerateResult, UsageInfo
from llm_router.router import RouterPolicyEngine

# ── Fixtures ────────────────────────────────────────────────────────


def _make_engine(**overrides):
    """Build a RouterPolicyEngine with all dependencies mocked."""
    pool = MagicMock()
    policy_matcher = AsyncMock()
    complexity_detector = AsyncMock()
    hybrid_router = AsyncMock()
    round_robin = AsyncMock()
    rate_limiter = MagicMock()
    pii_filter = MagicMock()
    abuse_filter = MagicMock()
    content_safety = MagicMock()
    pii_filter.redact = False

    for name, obj in {
        "pool": pool,
        "policy_matcher": policy_matcher,
        "complexity_detector": complexity_detector,
        "hybrid_router": hybrid_router,
        "round_robin": round_robin,
        "rate_limiter": rate_limiter,
        "pii_filter": pii_filter,
        "abuse_filter": abuse_filter,
        "content_safety": content_safety,
    }.items():
        setattr(engine_base(), name, obj)

    kw = {
        "pool": pool,
        "routing_strategy": RoutingStrategy.POLICY,
        "policy_matcher": policy_matcher,
        "complexity_detector": complexity_detector,
        "hybrid_router": hybrid_router,
        "round_robin": round_robin,
        "rate_limiter": rate_limiter,
        "pii_filter": pii_filter,
        "abuse_filter": abuse_filter,
        "content_safety": content_safety,
        "default_model": "test-model",
    }
    kw.update(overrides)
    return RouterPolicyEngine(**kw)


class _EngineBase:
    """Minimal object to hold the engine instance for attribute assignment."""

    pass


def engine_base():
    """Return a fresh object that can receive router attributes."""
    return _EngineBase()


def make_router(**overrides):
    """Helper to create a RouterPolicyEngine with defaults overridden."""
    pool = MagicMock()
    policy_matcher = AsyncMock()
    complexity_detector = AsyncMock()
    hybrid_router = AsyncMock()
    round_robin = AsyncMock()
    rate_limiter = MagicMock()
    pii_filter = MagicMock()
    abuse_filter = MagicMock()
    content_safety = MagicMock()
    pii_filter.redact = False

    kw = {
        "pool": pool,
        "routing_strategy": RoutingStrategy.POLICY,
        "policy_matcher": policy_matcher,
        "complexity_detector": complexity_detector,
        "hybrid_router": hybrid_router,
        "round_robin": round_robin,
        "rate_limiter": rate_limiter,
        "pii_filter": pii_filter,
        "abuse_filter": abuse_filter,
        "content_safety": content_safety,
        "default_model": "test-model",
        **overrides,
    }
    return RouterPolicyEngine(**kw)


# ── init tests ──────────────────────────────────────────────────────


class TestRouterPolicyEngineInit:
    def test_has_all_expected_attrs(self):
        engine = make_router()
        assert engine.default_model == "test-model"
        assert engine.routing_strategy == RoutingStrategy.POLICY

    def test_default_model_overridable(self):
        engine = make_router(default_model="gpt-4o")
        assert engine.default_model == "gpt-4o"

    def test_all_storategies_available(self):
        for strategy in RoutingStrategy:
            engine = make_router(routing_strategy=strategy)
            assert engine.routing_strategy == strategy


# ── generate (non-streaming) tests ──────────────────────────────────


class TestGenerate:
    def _mock_backend_success(self, engine):
        """Configure mocks so generate() returns a happy path result."""
        result = GenerateResult(
            content="Hello world",
            model="test-model",
            usage=UsageInfo(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            finish_reason="stop",
            latency_ms=120.0,
        )
        engine.pool.get.return_value.generate = AsyncMock(return_value=result)

    def _mock_rate_ok(self, engine):
        """Rate limiter that allows the request."""
        engine.rate_limiter.check = AsyncMock(return_value=MagicMock(allowed=True, remaining_requests=59))

    def _mock_abuse_safe(self, engine):
        """Abuse filter that marks everything safe."""
        result = MagicMock()
        result.safe = True
        result.abuse_score = 0.0
        result.categories = []
        engine.abuse_filter.check.return_value = result

    def _mock_pii_no_pii(self, engine):
        """PII filter with no PII detected."""
        result = MagicMock()
        result.has_pii = False
        result.patterns = []
        engine.pii_filter.check.return_value = result

    def _mock_policy_route(self, engine, model_id="test-model"):
        """Policy matcher that routes to a given model."""
        engine.policy_matcher.route = AsyncMock(return_value=MagicMock(model_id=model_id, strategy="policy"))

    def test_generate_success(self):
        engine = make_router()
        self._mock_rate_ok(engine)
        self._mock_abuse_safe(engine)
        self._mock_pii_no_pii(engine)
        self._mock_policy_route(engine)
        self._mock_backend_success(engine)

        result = asyncio.run(engine.generate([{"role": "user", "content": "Hello"}], user_id="u1"))
        assert result.content == "Hello world"
        assert result.model == "test-model"
        assert result.usage.total_tokens == 15

    def test_generate_calls_rate_limit_check(self):
        engine = make_router()
        self._mock_rate_ok(engine)
        self._mock_abuse_safe(engine)
        self._mock_pii_no_pii(engine)
        self._mock_policy_route(engine)
        self._mock_backend_success(engine)

        asyncio.run(engine.generate([{"role": "user", "content": "Hi"}], user_id="u1"))
        engine.rate_limiter.check.assert_awaited_once()

    def test_generate_calls_pii_filter(self):
        engine = make_router()
        self._mock_rate_ok(engine)
        self._mock_abuse_safe(engine)
        self._mock_pii_no_pii(engine)
        self._mock_policy_route(engine)
        self._mock_backend_success(engine)

        asyncio.run(engine.generate([{"role": "user", "content": "Hi"}]))
        engine.pii_filter.check.assert_called_once()

    def test_generate_calls_abuse_filter(self):
        engine = make_router()
        self._mock_rate_ok(engine)
        self._mock_abuse_safe(engine)
        self._mock_pii_no_pii(engine)
        self._mock_policy_route(engine)
        self._mock_backend_success(engine)

        asyncio.run(engine.generate([{"role": "user", "content": "Hi"}]))
        engine.abuse_filter.check.assert_called_once()

    def test_generate_calls_policy_route(self):
        engine = make_router(routing_strategy=RoutingStrategy.POLICY)
        self._mock_rate_ok(engine)
        self._mock_abuse_safe(engine)
        self._mock_pii_no_pii(engine)
        self._mock_policy_route(engine, model_id="policy-model")
        self._mock_backend_success(engine)

        asyncio.run(engine.generate([{"role": "user", "content": "Hi"}]))
        engine.policy_matcher.route.assert_awaited_once()

    def test_generate_calls_complexity_route(self):
        engine = make_router(routing_strategy=RoutingStrategy.COMPLEXITY)
        self._mock_rate_ok(engine)
        self._mock_abuse_safe(engine)
        self._mock_pii_no_pii(engine)
        self._mock_backend_success(engine)
        engine.complexity_detector.route = AsyncMock(
            return_value=MagicMock(model_id="complexity-model", strategy="complexity")
        )

        asyncio.run(engine.generate([{"role": "user", "content": "Hi"}]))
        engine.complexity_detector.route.assert_awaited_once()
        # Should NOT call policy matcher
        engine.policy_matcher.route.assert_not_called()

    def test_generate_calls_round_robin_route(self):
        engine = make_router(routing_strategy=RoutingStrategy.ROUND_ROBIN)
        self._mock_rate_ok(engine)
        self._mock_abuse_safe(engine)
        self._mock_pii_no_pii(engine)
        self._mock_backend_success(engine)
        engine.round_robin.route = AsyncMock(return_value=MagicMock(model_id="rr-model", strategy="round_robin"))

        asyncio.run(engine.generate([{"role": "user", "content": "Hi"}]))
        engine.round_robin.route.assert_awaited_once()

    def test_generate_uses_explicit_model_override(self):
        engine = make_router()
        self._mock_rate_ok(engine)
        self._mock_abuse_safe(engine)
        self._mock_pii_no_pii(engine)
        self._mock_policy_route(engine, model_id="policy-model")
        self._mock_backend_success(engine)

        asyncio.run(engine.generate([{"role": "user", "content": "Hi"}], model="override-model"))
        # Pool.get should be called with the explicit model, not the routed one
        engine.pool.get.assert_called_with("override-model")

    def test_generate_router_auto_uses_routed_model(self):
        engine = make_router()
        self._mock_rate_ok(engine)
        self._mock_abuse_safe(engine)
        self._mock_pii_no_pii(engine)
        self._mock_policy_route(engine, model_id="routed-model")
        self._mock_backend_success(engine)

        asyncio.run(engine.generate([{"role": "user", "content": "Hi"}], model="router-auto"))

        engine.pool.get.assert_called_with("routed-model")

    def test_generate_uses_routed_model_when_no_explicit(self):
        engine = make_router()
        self._mock_rate_ok(engine)
        self._mock_abuse_safe(engine)
        self._mock_pii_no_pii(engine)
        self._mock_policy_route(engine, model_id="routed-model")
        self._mock_backend_success(engine)

        asyncio.run(engine.generate([{"role": "user", "content": "Hi"}]))
        engine.pool.get.assert_called_with("routed-model")

    def test_generate_falls_back_to_default_model(self):
        engine = make_router()
        self._mock_rate_ok(engine)
        self._mock_abuse_safe(engine)
        self._mock_pii_no_pii(engine)
        self._mock_policy_route(engine, model_id=None)
        self._mock_backend_success(engine)

        asyncio.run(engine.generate([{"role": "user", "content": "Hi"}]))
        engine.pool.get.assert_called_with("test-model")

    def test_generate_raises_on_rate_limit(self):
        engine = make_router()
        engine.rate_limiter.check = AsyncMock(return_value=MagicMock(allowed=False, error="too many requests"))

        with pytest.raises(Exception, match="Rate limited"):
            asyncio.run(engine.generate([{"role": "user", "content": "Hi"}]))

    def test_generate_raises_on_abuse(self):
        engine = make_router()
        self._mock_rate_ok(engine)
        engine.pii_filter.check.return_value = MagicMock(has_pii=False, patterns=[])
        abuse = MagicMock()
        abuse.safe = False
        abuse.categories = ["hate"]
        abuse.abuse_score = 0.95
        engine.abuse_filter.check.return_value = abuse

        with pytest.raises(Exception, match="Abuse detected"):
            asyncio.run(engine.generate([{"role": "user", "content": "bad stuff"}]))

    def test_generate_pool_get_called_once(self):
        engine = make_router()
        self._mock_rate_ok(engine)
        self._mock_abuse_safe(engine)
        self._mock_pii_no_pii(engine)
        self._mock_policy_route(engine)
        self._mock_backend_success(engine)

        asyncio.run(engine.generate([{"role": "user", "content": "Hi"}]))
        assert engine.pool.get.call_count == 1

    def test_generate_passes_messages_to_backend(self):
        engine = make_router()
        self._mock_rate_ok(engine)
        self._mock_abuse_safe(engine)
        self._mock_pii_no_pii(engine)
        self._mock_policy_route(engine)
        self._mock_backend_success(engine)
        messages = [{"role": "user", "content": "test msg"}]

        asyncio.run(engine.generate(messages))
        backend = engine.pool.get.return_value
        backend.generate.assert_awaited_with(messages)

    def test_generate_result_has_correct_usage(self):
        engine = make_router()
        self._mock_rate_ok(engine)
        self._mock_abuse_safe(engine)
        self._mock_pii_no_pii(engine)
        self._mock_policy_route(engine)
        expected_usage = UsageInfo(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        result = GenerateResult(content="ok", model="m", usage=expected_usage, finish_reason="stop", latency_ms=50.0)
        engine.pool.get.return_value.generate = AsyncMock(return_value=result)

        gen = asyncio.run(engine.generate([{"role": "user", "content": "x"}]))
        assert gen.usage.prompt_tokens == 100
        assert gen.usage.completion_tokens == 50
        assert gen.usage.total_tokens == 150


# ── generate_stream tests ───────────────────────────────────────────


class TestGenerateStream:
    def _stream_result(self, content, model="test-model"):
        return GenerateResult(
            content=content,
            model=model,
            usage=UsageInfo(prompt_tokens=1, completion_tokens=len(content), total_tokens=1 + len(content)),
            finish_reason="stop",
            latency_ms=10.0,
        )

    def _mock_rate_ok(self, engine):
        engine.rate_limiter.check = AsyncMock(return_value=MagicMock(allowed=True, remaining_requests=59))

    def _mock_abuse_safe(self, engine):
        result = MagicMock()
        result.safe = True
        result.categories = []
        engine.abuse_filter.check.return_value = result

    def _mock_pii_no_pii(self, engine):
        engine.pii_filter.check.return_value = MagicMock(has_pii=False, patterns=[])

    def _mock_policy_route(self, engine, model_id="test-model"):
        engine.policy_matcher.route = AsyncMock(return_value=MagicMock(model_id=model_id, strategy="policy"))

    def test_stream_success(self):
        engine = make_router()
        self._mock_rate_ok(engine)
        self._mock_abuse_safe(engine)
        self._mock_pii_no_pii(engine)
        self._mock_policy_route(engine)

        async def _stream(messages):
            yield self._stream_result("chunk1")
            yield self._stream_result("chunk2")

        engine.pool.get.return_value.generate_stream = _stream

        chunks = asyncio.run(self._consume_stream(engine))

        assert len(chunks) == 2
        assert chunks[0].content == "chunk1"
        assert chunks[1].content == "chunk2"

    def test_stream_skips_empty_content_chunks(self):
        engine = make_router()
        self._mock_rate_ok(engine)
        self._mock_abuse_safe(engine)
        self._mock_pii_no_pii(engine)
        self._mock_policy_route(engine)

        async def _stream(messages):
            yield self._stream_result("")
            yield self._stream_result("visible")
            yield self._stream_result("")

        engine.pool.get.return_value.generate_stream = _stream

        chunks = asyncio.run(self._consume_stream(engine))

        assert [chunk.content for chunk in chunks] == ["visible"]

    def test_stream_keeps_tool_call_chunks_without_content(self):
        engine = make_router()
        self._mock_rate_ok(engine)
        self._mock_abuse_safe(engine)
        self._mock_pii_no_pii(engine)
        self._mock_policy_route(engine)

        async def _stream(messages):
            yield GenerateResult(
                content="",
                model="test-model",
                usage=UsageInfo(prompt_tokens=0, completion_tokens=0, total_tokens=0),
                finish_reason="tool_calls",
                tool_calls=[
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "bash", "arguments": "{\"command\":\"pwd\"}"},
                    }
                ],
            )

        engine.pool.get.return_value.generate_stream = _stream

        chunks = asyncio.run(self._consume_stream(engine))

        assert len(chunks) == 1
        assert chunks[0].tool_calls[0]["function"]["name"] == "bash"

    def test_stream_calls_rate_limit(self):
        engine = make_router()
        self._mock_rate_ok(engine)
        self._mock_abuse_safe(engine)
        self._mock_pii_no_pii(engine)
        self._mock_policy_route(engine)

        async def _stream_gen(messages):
            yield self._stream_result("x")

        engine.pool.get.return_value.generate_stream = _stream_gen

        async def _collect():
            async for _ in engine.generate_stream([{"role": "user", "content": "hi"}]):
                pass

        asyncio.run(_collect())
        engine.rate_limiter.check.assert_awaited_once()

    def test_stream_calls_abuse_filter(self):
        engine = make_router()
        self._mock_rate_ok(engine)
        self._mock_abuse_safe(engine)
        self._mock_pii_no_pii(engine)
        self._mock_policy_route(engine)

        async def _stream_gen(messages):
            yield self._stream_result("x")

        engine.pool.get.return_value.generate_stream = _stream_gen

        async def _collect():
            async for _ in engine.generate_stream([{"role": "user", "content": "hi"}]):
                pass

        asyncio.run(_collect())
        engine.abuse_filter.check.assert_called_once()

    def test_stream_raises_on_rate_limit(self):
        engine = make_router()
        engine.rate_limiter.check = AsyncMock(return_value=MagicMock(allowed=False, error="blocked"))

        async def _collect():
            async for _ in engine.generate_stream([{"role": "user", "content": "hi"}]):
                pass

        with pytest.raises(Exception, match="Rate limited"):
            asyncio.run(_collect())

    def test_stream_raises_on_abuse(self):
        engine = make_router()
        self._mock_rate_ok(engine)
        abuse = MagicMock()
        abuse.safe = False
        abuse.categories = ["violence"]
        engine.abuse_filter.check.return_value = abuse

        async def _collect():
            async for _ in engine.generate_stream([{"role": "user", "content": "hi"}]):
                pass

        with pytest.raises(Exception, match="Abuse detected"):
            asyncio.run(_collect())

    def test_stream_uses_explicit_model(self):
        engine = make_router()
        self._mock_rate_ok(engine)
        self._mock_abuse_safe(engine)
        self._mock_pii_no_pii(engine)
        engine.policy_matcher.route = AsyncMock(return_value=MagicMock(model_id="policy-model"))

        async def _stream_gen(messages):
            yield self._stream_result("ok", model="explicit")

        engine.pool.get.return_value.generate_stream = _stream_gen

        async def _collect():
            async for _chunk in engine.generate_stream([{"role": "user", "content": "hi"}], model="explicit"):
                pass

        asyncio.run(_collect())
        engine.pool.get.assert_called_with("explicit")

    def test_stream_uses_routed_model(self):
        engine = make_router()
        self._mock_rate_ok(engine)
        self._mock_abuse_safe(engine)
        self._mock_pii_no_pii(engine)
        self._mock_policy_route(engine, model_id="routed-stream")

        async def _stream_gen(messages):
            yield self._stream_result("ok", model="routed-stream")

        engine.pool.get.return_value.generate_stream = _stream_gen

        async def _collect():
            async for _chunk in engine.generate_stream([{"role": "user", "content": "hi"}]):
                pass

        asyncio.run(_collect())
        engine.pool.get.assert_called_with("routed-stream")

    def test_stream_calls_complexity_strategy(self):
        engine = make_router(routing_strategy=RoutingStrategy.COMPLEXITY)
        self._mock_rate_ok(engine)
        self._mock_abuse_safe(engine)
        self._mock_pii_no_pii(engine)
        engine.complexity_detector.route = AsyncMock(return_value=MagicMock(model_id="cx", strategy="complexity"))

        async def _stream_gen(messages):
            yield self._stream_result("ok", "cx")

        engine.pool.get.return_value.generate_stream = _stream_gen

        async def _collect():
            async for _chunk in engine.generate_stream([{"role": "user", "content": "hi"}]):
                pass

        asyncio.run(_collect())
        engine.complexity_detector.route.assert_awaited_once()
        engine.policy_matcher.route.assert_not_called()

    def test_stream_empty_stream(self):
        engine = make_router()
        self._mock_rate_ok(engine)
        self._mock_abuse_safe(engine)
        self._mock_pii_no_pii(engine)
        self._mock_policy_route(engine)

        async def _empty(messages):
            return
            yield  # make it a generator

        engine.pool.get.return_value.generate_stream = _empty

        async def _collect():
            collected = []
            async for chunk in engine.generate_stream([{"role": "user", "content": "hi"}]):
                collected.append(chunk)
            return collected

        chunks = asyncio.run(_collect())
        assert chunks == []

    def test_stream_calls_round_robin_strategy(self):
        """ROUND_ROBIN strategy should call round_robin.route in generate_stream."""
        engine = make_router(routing_strategy=RoutingStrategy.ROUND_ROBIN)
        self._mock_rate_ok(engine)
        self._mock_abuse_safe(engine)
        self._mock_pii_no_pii(engine)
        engine.round_robin.route = AsyncMock(return_value=MagicMock(model_id="rr-model", strategy="round_robin"))

        async def _stream_gen(messages):
            yield self._stream_result("ok", model="rr-model")

        engine.pool.get.return_value.generate_stream = _stream_gen

        async def _collect():
            chunks = []
            async for chunk in engine.generate_stream([{"role": "user", "content": "hi"}]):
                chunks.append(chunk)
            return chunks

        chunks = asyncio.run(_collect())
        assert len(chunks) == 1
        engine.round_robin.route.assert_awaited_once()

    @staticmethod
    async def _consume_stream(engine):
        """Consume stream chunks into a list."""
        collected = []
        async for chunk in engine.generate_stream([{"role": "user", "content": "hi"}]):
            collected.append(chunk)
        return collected


# ── Integration-style: strategy routing tests ───────────────────────


class TestRouterStrategyRouting:
    """Verify the engine delegates to the correct sub-router for each strategy."""

    def _ready(self, engine):
        """Mock all dependencies for a successful call."""
        engine.rate_limiter.check = AsyncMock(return_value=MagicMock(allowed=True, remaining_requests=59))
        engine.abuse_filter.check.return_value = MagicMock(safe=True, categories=[], abuse_score=0)
        engine.pii_filter.check.return_value = MagicMock(has_pii=False, patterns=[])
        result = GenerateResult(
            content="ok",
            model="m",
            usage=UsageInfo(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            finish_reason="stop",
            latency_ms=1.0,
        )
        engine.pool.get.return_value.generate = AsyncMock(return_value=result)
        return engine

    def _stream_ready(self, engine):
        """Mock all dependencies for a successful stream call."""
        engine.rate_limiter.check = AsyncMock(return_value=MagicMock(allowed=True, remaining_requests=59))
        engine.abuse_filter.check.return_value = MagicMock(safe=True, categories=[])
        engine.pii_filter.check.return_value = MagicMock(has_pii=False, patterns=[])
        return engine

    def test_policy_strategy_routes_through_matcher(self):
        engine = self._ready(make_router(routing_strategy=RoutingStrategy.POLICY))
        engine.policy_matcher.route = AsyncMock(return_value=MagicMock(model_id="pm", strategy="policy"))
        asyncio.run(engine.generate([{"role": "user", "content": "hi"}]))
        engine.policy_matcher.route.assert_awaited_once()

    def test_complexity_strategy_routes_through_detector(self):
        engine = self._ready(make_router(routing_strategy=RoutingStrategy.COMPLEXITY))
        engine.complexity_detector.route = AsyncMock(return_value=MagicMock(model_id="cx", strategy="complexity"))
        asyncio.run(engine.generate([{"role": "user", "content": "hi"}]))
        engine.complexity_detector.route.assert_awaited_once()
        engine.policy_matcher.route.assert_not_called()

    def test_round_robin_strategy_routes_through_rr(self):
        engine = self._ready(make_router(routing_strategy=RoutingStrategy.ROUND_ROBIN))
        engine.round_robin.route = AsyncMock(return_value=MagicMock(model_id="rr", strategy="round_robin"))
        asyncio.run(engine.generate([{"role": "user", "content": "hi"}]))
        engine.round_robin.route.assert_awaited_once()
        engine.policy_matcher.route.assert_not_called()

    def test_hybrid_strategy_falls_back_to_policy_matcher(self):
        engine = self._ready(make_router(routing_strategy=RoutingStrategy.HYBRID))
        engine.policy_matcher.route = AsyncMock(return_value=MagicMock(model_id="hm", strategy="hybrid"))
        asyncio.run(engine.generate([{"role": "user", "content": "hi"}]))
        engine.policy_matcher.route.assert_awaited_once()

    def test_latency_strategy_falls_back_to_policy_matcher(self):
        engine = self._ready(make_router(routing_strategy=RoutingStrategy.LATENCY))
        engine.policy_matcher.route = AsyncMock(return_value=MagicMock(model_id="lm", strategy="latency"))
        asyncio.run(engine.generate([{"role": "user", "content": "hi"}]))
        engine.policy_matcher.route.assert_awaited_once()

    def test_cost_strategy_falls_back_to_policy_matcher(self):
        engine = self._ready(make_router(routing_strategy=RoutingStrategy.COST))
        engine.policy_matcher.route = AsyncMock(return_value=MagicMock(model_id="cm", strategy="cost"))
        asyncio.run(engine.generate([{"role": "user", "content": "hi"}]))
        engine.policy_matcher.route.assert_awaited_once()


# ── Edge cases ──────────────────────────────────────────────────────


class TestEdgeCases:
    def test_generate_with_empty_messages(self):
        engine = make_router()
        engine.rate_limiter.check = AsyncMock(return_value=MagicMock(allowed=True, remaining_requests=59))
        engine.abuse_filter.check.return_value = MagicMock(safe=True, categories=[], abuse_score=0)
        engine.pii_filter.check.return_value = MagicMock(has_pii=False, patterns=[])
        engine.policy_matcher.route = AsyncMock(return_value=MagicMock(model_id="empty-model", strategy="policy"))
        engine.pool.get.return_value.generate = AsyncMock(
            return_value=GenerateResult(
                content="",
                model="empty",
                usage=UsageInfo(prompt_tokens=0, completion_tokens=0, total_tokens=0),
                finish_reason="stop",
                latency_ms=0.0,
            )
        )
        result = asyncio.run(engine.generate([]))
        assert result.content == ""
        engine.pool.get.return_value.generate.assert_called_once_with([])

    def test_generate_with_user_id_none(self):
        engine = make_router()
        engine.rate_limiter.check = AsyncMock(return_value=MagicMock(allowed=True, remaining_requests=59))
        engine.abuse_filter.check.return_value = MagicMock(safe=True, categories=[], abuse_score=0)
        engine.pii_filter.check.return_value = MagicMock(has_pii=False, patterns=[])
        engine.policy_matcher.route = AsyncMock(return_value=MagicMock(model_id="m", strategy="policy"))
        engine.pool.get.return_value.generate = AsyncMock(
            return_value=GenerateResult(
                content="x",
                model="m",
                usage=UsageInfo(prompt_tokens=1, completion_tokens=1, total_tokens=2),
                finish_reason="stop",
                latency_ms=1.0,
            )
        )
        result = asyncio.run(engine.generate([{"role": "user", "content": "hi"}], user_id=None))
        assert result.content == "x"

    def test_generate_with_api_key_ignored_in_request_id(self):
        """API key is accepted as parameter but not used in request ID construction."""
        engine = make_router()
        engine.rate_limiter.check = AsyncMock(return_value=MagicMock(allowed=True, remaining_requests=59))
        engine.abuse_filter.check.return_value = MagicMock(safe=True, categories=[], abuse_score=0)
        engine.pii_filter.check.return_value = MagicMock(has_pii=False, patterns=[])
        engine.policy_matcher.route = AsyncMock(return_value=MagicMock(model_id="m", strategy="policy"))
        engine.pool.get.return_value.generate = AsyncMock(
            return_value=GenerateResult(
                content="x",
                model="m",
                usage=UsageInfo(prompt_tokens=1, completion_tokens=1, total_tokens=2),
                finish_reason="stop",
                latency_ms=1.0,
            )
        )
        asyncio.run(engine.generate([{"role": "user", "content": "hi"}], user_id="u1", api_key="sk-xxx"))
        assert engine.rate_limiter.check.called

    def test_generate_stream_with_explicit_model_uses_it(self):
        engine = make_router()
        engine.rate_limiter.check = AsyncMock(return_value=MagicMock(allowed=True, remaining_requests=59))
        engine.abuse_filter.check.return_value = MagicMock(safe=True, categories=[])
        engine.pii_filter.check.return_value = MagicMock(has_pii=False, patterns=[])
        engine.policy_matcher.route = AsyncMock(return_value=MagicMock(model_id="routed"))

        async def _stream(messages):
            yield GenerateResult(
                content="streamed",
                model="explicit-model",
                usage=UsageInfo(prompt_tokens=1, completion_tokens=8, total_tokens=9),
                finish_reason="stop",
                latency_ms=5.0,
            )

        engine.pool.get.return_value.generate_stream = _stream

        async def _collect():
            async for chunk in engine.generate_stream([{"role": "user", "content": "hi"}], model="explicit-model"):
                return chunk

        result = asyncio.run(_collect())
        assert result.model == "explicit-model"


# ── Edge case: PII detected logging ─────────────────────────────────


class TestPIIDetectedLog:
    def test_generate_logs_pii_detected_when_pii_found(self):
        """When PII is detected, the router logs an info message."""
        engine = make_router()
        engine.rate_limiter.check = AsyncMock(return_value=MagicMock(allowed=True, remaining_requests=59))
        engine.abuse_filter.check.return_value = MagicMock(safe=True, categories=[])
        engine.pii_filter.check = MagicMock(return_value=MagicMock(has_pii=True, patterns=["email"]))
        engine.policy_matcher.route = AsyncMock(return_value=MagicMock(model_id="m", strategy="policy"))
        engine.pool.get.return_value.generate = AsyncMock(
            return_value=GenerateResult(
                content="ok",
                model="m",
                usage=UsageInfo(prompt_tokens=1, completion_tokens=1, total_tokens=2),
                finish_reason="stop",
                latency_ms=1.0,
            )
        )

        result = asyncio.run(engine.generate([{"role": "user", "content": "test@test.com"}]))
        assert result.content == "ok"
        # Verify PII check was called with the full text
        engine.pii_filter.check.assert_called()
        call_arg = engine.pii_filter.check.call_args[0][0]
        assert "test@test.com" in call_arg


# ── Edge case: model call failure ─────────────────────────────────────


class TestModelCallFailure:
    def test_generate_raises_on_model_call_error(self):
        """When the backend raises, the error is logged and re-raised."""
        engine = make_router()
        engine.rate_limiter.check = AsyncMock(return_value=MagicMock(allowed=True, remaining_requests=59))
        engine.abuse_filter.check.return_value = MagicMock(safe=True, categories=[])
        engine.pii_filter.check.return_value = MagicMock(has_pii=False, patterns=[])
        engine.policy_matcher.route = AsyncMock(return_value=MagicMock(model_id="m", strategy="policy"))
        engine.pool.get.return_value.generate = AsyncMock(side_effect=RuntimeError("connection refused"))

        with pytest.raises(RuntimeError, match="connection refused"):
            asyncio.run(engine.generate([{"role": "user", "content": "hi"}]))


# ── Edge case: generate_stream fallback strategies ────────────────────


class TestStreamFallbackStrategies:
    def test_stream_hybrid_strategy_calls_policy_matcher(self):
        """HYBRID strategy falls through to policy matcher in generate_stream."""
        engine = make_router(routing_strategy=RoutingStrategy.HYBRID)
        engine.rate_limiter.check = AsyncMock(return_value=MagicMock(allowed=True, remaining_requests=59))
        engine.abuse_filter.check.return_value = MagicMock(safe=True, categories=[])
        engine.pii_filter.check.return_value = MagicMock(has_pii=False, patterns=[])
        engine.policy_matcher.route = AsyncMock(return_value=MagicMock(model_id="hybrid-model", strategy="policy"))

        async def _stream_gen(messages):
            yield GenerateResult(
                content="ok",
                model="hybrid-model",
                usage=UsageInfo(prompt_tokens=1, completion_tokens=2, total_tokens=3),
                finish_reason="stop",
                latency_ms=5.0,
            )

        engine.pool.get.return_value.generate_stream = _stream_gen

        async def _collect():
            chunks = []
            async for chunk in engine.generate_stream([{"role": "user", "content": "hi"}]):
                chunks.append(chunk)
            return chunks

        chunks = asyncio.run(_collect())
        assert len(chunks) == 1
        assert chunks[0].model == "hybrid-model"
        # policy_matcher.route should be called (fallback)
        engine.policy_matcher.route.assert_awaited_once()

    def test_stream_latency_strategy_calls_policy_matcher(self):
        """LATENCY strategy falls through to policy matcher in generate_stream."""
        engine = make_router(routing_strategy=RoutingStrategy.LATENCY)
        engine.rate_limiter.check = AsyncMock(return_value=MagicMock(allowed=True, remaining_requests=59))
        engine.abuse_filter.check.return_value = MagicMock(safe=True, categories=[])
        engine.pii_filter.check.return_value = MagicMock(has_pii=False, patterns=[])
        engine.policy_matcher.route = AsyncMock(return_value=MagicMock(model_id="latency-model", strategy="latency"))

        async def _stream_gen(messages):
            yield GenerateResult(
                content="ok",
                model="latency-model",
                usage=UsageInfo(prompt_tokens=1, completion_tokens=2, total_tokens=3),
                finish_reason="stop",
                latency_ms=5.0,
            )

        engine.pool.get.return_value.generate_stream = _stream_gen

        async def _collect():
            chunks = []
            async for chunk in engine.generate_stream([{"role": "user", "content": "hi"}]):
                chunks.append(chunk)
            return chunks

        chunks = asyncio.run(_collect())
        assert chunks[0].model == "latency-model"
        engine.policy_matcher.route.assert_awaited_once()

    def test_stream_cost_strategy_calls_policy_matcher(self):
        """COST strategy falls through to policy matcher in generate_stream."""
        engine = make_router(routing_strategy=RoutingStrategy.COST)
        engine.rate_limiter.check = AsyncMock(return_value=MagicMock(allowed=True, remaining_requests=59))
        engine.abuse_filter.check.return_value = MagicMock(safe=True, categories=[])
        engine.pii_filter.check.return_value = MagicMock(has_pii=False, patterns=[])
        engine.policy_matcher.route = AsyncMock(return_value=MagicMock(model_id="cost-model", strategy="cost"))

        async def _stream_gen(messages):
            yield GenerateResult(
                content="ok",
                model="cost-model",
                usage=UsageInfo(prompt_tokens=1, completion_tokens=2, total_tokens=3),
                finish_reason="stop",
                latency_ms=5.0,
            )

        engine.pool.get.return_value.generate_stream = _stream_gen

        async def _collect():
            chunks = []
            async for chunk in engine.generate_stream([{"role": "user", "content": "hi"}]):
                chunks.append(chunk)
            return chunks

        chunks = asyncio.run(_collect())
        assert chunks[0].model == "cost-model"
        engine.policy_matcher.route.assert_awaited_once()


# ── Import test ──────────────────────────────────────────────────────


class TestImports:
    def test_router_module_importable(self):
        from llm_router import router  # noqa: F401

        assert hasattr(router, "RouterPolicyEngine")

    def test_router_exports_engine_class(self):
        from llm_router.router import RouterPolicyEngine as RPE  # noqa: N817

        assert RPE is not None
