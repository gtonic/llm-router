"""Tests for router.py - RouterPolicyEngine."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_router.config import RoutingStrategy
from llm_router.pool.base import EmptyResponseError, GenerateResult, UsageInfo
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

    def test_generate_routes_pii_on_raw_messages_and_sends_redacted_messages(self):
        engine = make_router()
        self._mock_rate_ok(engine)
        self._mock_abuse_safe(engine)
        engine.pii_filter.redact = True
        engine.pii_filter.check.return_value = MagicMock(has_pii=True, patterns=["email"])
        engine.pii_filter.redact_text.side_effect = lambda text: text.replace("jane@example.com", "[REDACTED]")
        self._mock_policy_route(engine, model_id="llama-local")
        self._mock_backend_success(engine)
        messages = [{"role": "user", "content": "Contact jane@example.com"}]

        asyncio.run(engine.generate(messages, model="router-auto"))

        engine.policy_matcher.route.assert_awaited_once_with(messages)
        engine.pool.get.return_value.generate.assert_awaited_with([{"role": "user", "content": "Contact [REDACTED]"}])

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

        def _empty():
            # Real backends emit content-less body chunks with zero usage.
            return GenerateResult(
                content="", model="test-model", usage=UsageInfo(0, 0, 0), finish_reason="stop", latency_ms=0.0
            )

        async def _stream(messages):
            yield _empty()
            yield self._stream_result("visible")
            yield _empty()

        engine.pool.get.return_value.generate_stream = _stream

        chunks = asyncio.run(self._consume_stream(engine))

        assert [chunk.content for chunk in chunks] == ["visible"]

    def test_stream_forwards_terminal_usage_chunk(self):
        """The content-less terminal chunk carrying token usage is forwarded so
        the route layer can account tokens/cost for streaming requests."""
        engine = make_router()
        self._mock_rate_ok(engine)
        self._mock_abuse_safe(engine)
        self._mock_pii_no_pii(engine)
        self._mock_policy_route(engine)

        async def _stream(messages):
            yield GenerateResult(
                content="hi", model="test-model", usage=UsageInfo(0, 0, 0), finish_reason="incomplete", latency_ms=0
            )
            yield GenerateResult(
                content="",
                model="test-model",
                usage=UsageInfo(prompt_tokens=3, completion_tokens=5, total_tokens=8, cost=0.01),
                finish_reason="stop",
                latency_ms=0,
            )

        engine.pool.get.return_value.generate_stream = _stream

        chunks = asyncio.run(self._consume_stream(engine))

        assert [c.content for c in chunks] == ["hi", ""]
        assert chunks[-1].usage.total_tokens == 8
        assert chunks[-1].usage.cost == 0.01

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
                        "function": {"name": "bash", "arguments": '{"command":"pwd"}'},
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

        with pytest.raises(EmptyResponseError):
            asyncio.run(_collect())

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

    def test_hybrid_strategy_routes_through_hybrid_router(self):
        engine = self._ready(make_router(routing_strategy=RoutingStrategy.HYBRID))
        engine.hybrid_router.route = AsyncMock(return_value=MagicMock(model_id="hm", strategy="hybrid"))
        asyncio.run(engine.generate([{"role": "user", "content": "hi"}]))
        engine.hybrid_router.route.assert_awaited_once()
        engine.policy_matcher.route.assert_not_called()

    def test_latency_strategy_selects_lowest_latency_healthy_backend(self):
        from llm_router.pool.base import HealthStatus

        engine = make_router(routing_strategy=RoutingStrategy.LATENCY)
        engine.pool.health_check_all = AsyncMock(
            return_value={
                "slow": HealthStatus(healthy=True, latency_ms=200.0),
                "fast": HealthStatus(healthy=True, latency_ms=20.0),
                "down": HealthStatus(healthy=False, latency_ms=0.0),
            }
        )
        result = asyncio.run(engine._route([{"role": "user", "content": "hi"}], [{"role": "user", "content": "hi"}]))
        assert result.model_id == "fast"
        assert result.strategy == "latency"
        engine.policy_matcher.route.assert_not_called()

    def test_cost_strategy_selects_cheapest_backend(self):
        engine = make_router(routing_strategy=RoutingStrategy.COST)
        cheap = MagicMock()
        cheap.config = MagicMock(cost_per_1m_input=0.0, cost_per_1m_output=0.0)
        pricey = MagicMock()
        pricey.config = MagicMock(cost_per_1m_input=1.0, cost_per_1m_output=2.0)
        engine.pool.list_models.return_value = ["pricey", "cheap"]
        engine.pool.get.side_effect = {"pricey": pricey, "cheap": cheap}.get
        result = asyncio.run(engine._route([{"role": "user", "content": "hi"}], [{"role": "user", "content": "hi"}]))
        assert result.model_id == "cheap"
        assert result.strategy == "cost"
        engine.policy_matcher.route.assert_not_called()


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

    def test_generate_uses_api_key_for_rate_limit_when_no_user_id(self):
        """When user_id is absent, api_key is used as the rate-limit client identifier."""
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
        asyncio.run(engine.generate([{"role": "user", "content": "hi"}], user_id=None, api_key="sk-xxx"))
        engine.rate_limiter.check.assert_awaited_once()
        assert engine.rate_limiter.check.await_args.kwargs["client_id"] == "sk-xxx"

    def test_generate_uses_user_id_for_rate_limit_over_api_key(self):
        """user_id takes precedence over api_key as the rate-limit client identifier."""
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
        engine.rate_limiter.check.assert_awaited_once()
        assert engine.rate_limiter.check.await_args.kwargs["client_id"] == "u1"

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
    def test_empty_stream_falls_back_to_configured_model(self):
        engine = make_router()
        engine.rate_limiter.check = AsyncMock(return_value=MagicMock(allowed=True, remaining_requests=59))
        engine.abuse_filter.check.return_value = MagicMock(safe=True, categories=[])
        engine.pii_filter.check.return_value = MagicMock(has_pii=False, patterns=[])
        engine.policy_matcher.route = AsyncMock(return_value=MagicMock(model_id="primary", strategy="policy"))
        engine.settings.fallback_model = "fallback"
        primary = MagicMock()
        fallback = MagicMock()

        async def _empty_stream(messages, **kwargs):
            if False:
                yield None

        async def _fallback_stream(messages, **kwargs):
            yield GenerateResult(
                content="fallback response",
                model="fallback",
                usage=UsageInfo(prompt_tokens=1, completion_tokens=2, total_tokens=3),
                finish_reason="stop",
                latency_ms=5.0,
            )

        primary.generate_stream = _empty_stream
        fallback.generate_stream = _fallback_stream
        engine.pool.get.side_effect = {"primary": primary, "fallback": fallback}.get

        async def _collect():
            chunks = []
            async for chunk in engine.generate_stream([{"role": "user", "content": "hi"}]):
                chunks.append(chunk)
            return chunks

        chunks = asyncio.run(_collect())

        assert [chunk.content for chunk in chunks] == ["fallback response"]
        assert engine.pool.get.call_args_list[0].args == ("primary",)
        assert engine.pool.get.call_args_list[1].args == ("fallback",)

    def test_stream_hybrid_strategy_routes_through_hybrid_router(self):
        """HYBRID strategy routes through hybrid_router in generate_stream."""
        engine = make_router(routing_strategy=RoutingStrategy.HYBRID)
        engine.rate_limiter.check = AsyncMock(return_value=MagicMock(allowed=True, remaining_requests=59))
        engine.abuse_filter.check.return_value = MagicMock(safe=True, categories=[])
        engine.pii_filter.check.return_value = MagicMock(has_pii=False, patterns=[])
        engine.hybrid_router.route = AsyncMock(return_value=MagicMock(model_id="hybrid-model", strategy="hybrid"))

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
        engine.hybrid_router.route.assert_awaited_once()
        engine.policy_matcher.route.assert_not_called()

    def test_stream_cost_strategy_routes_to_cheapest_backend(self):
        """COST strategy in generate_stream streams from the cheapest backend."""
        engine = make_router(routing_strategy=RoutingStrategy.COST)
        engine.rate_limiter.check = AsyncMock(return_value=MagicMock(allowed=True, remaining_requests=59))
        engine.abuse_filter.check.return_value = MagicMock(safe=True, categories=[])
        engine.pii_filter.check.return_value = MagicMock(has_pii=False, patterns=[])

        cheap = MagicMock()
        cheap.config = MagicMock(cost_per_1m_input=0.0, cost_per_1m_output=0.0)
        pricey = MagicMock()
        pricey.config = MagicMock(cost_per_1m_input=5.0, cost_per_1m_output=5.0)

        async def _stream_gen(messages, **kwargs):
            yield GenerateResult(
                content="ok",
                model="cheap",
                usage=UsageInfo(prompt_tokens=1, completion_tokens=2, total_tokens=3),
                finish_reason="stop",
                latency_ms=5.0,
            )

        cheap.generate_stream = _stream_gen
        engine.pool.list_models.return_value = ["pricey", "cheap"]
        engine.pool.get.side_effect = {"pricey": pricey, "cheap": cheap}.get

        async def _collect():
            chunks = []
            async for chunk in engine.generate_stream([{"role": "user", "content": "hi"}]):
                chunks.append(chunk)
            return chunks

        chunks = asyncio.run(_collect())
        assert chunks[0].content == "ok"
        engine.policy_matcher.route.assert_not_called()
        assert any(call.args == ("cheap",) for call in engine.pool.get.call_args_list)


# ── Resilience: circuit breaker + no mid-stream double-emission ───────


class TestResilience:
    def _base(self, engine, routed="primary"):
        engine.rate_limiter.check = AsyncMock(return_value=MagicMock(allowed=True, remaining_requests=59))
        engine.abuse_filter.check.return_value = MagicMock(safe=True, categories=[], abuse_score=0.0)
        engine.pii_filter.check.return_value = MagicMock(has_pii=False, patterns=[])
        engine.policy_matcher.route = AsyncMock(return_value=MagicMock(model_id=routed, strategy="policy"))
        engine.settings.fallback_model = "fallback"

    def test_no_fallback_on_client_error(self):
        """A 4xx from the primary is surfaced directly — no wasted fallback call."""
        engine = make_router()
        self._base(engine)
        primary = MagicMock()
        err = RuntimeError("bad request")
        err.status_code = 400
        primary.generate = AsyncMock(side_effect=err)
        fallback = MagicMock()
        fallback.generate = AsyncMock()
        engine.pool.get.side_effect = {"primary": primary, "fallback": fallback}.get

        with pytest.raises(RuntimeError, match="bad request"):
            asyncio.run(engine.generate([{"role": "user", "content": "hi"}]))
        assert engine.pool.get.call_count == 1  # fallback never fetched
        fallback.generate.assert_not_called()

    def test_open_circuit_reroutes_before_calling_dead_backend(self):
        """Once the primary's breaker is open, selection reroutes to a healthy model."""
        engine = make_router()
        self._base(engine)
        breaker = engine._breaker("primary")
        for _ in range(engine.settings.circuit_breaker_threshold):
            breaker.record_failure()
        assert breaker.blocked

        primary = MagicMock()
        primary.generate = AsyncMock()
        fallback = MagicMock()
        fallback.generate = AsyncMock(
            return_value=GenerateResult(
                content="ok", model="fallback", usage=UsageInfo(1, 1, 2), finish_reason="stop", latency_ms=1.0
            )
        )
        engine.pool.list_models.return_value = ["primary", "fallback"]
        engine.pool.get.side_effect = {"primary": primary, "fallback": fallback}.get

        result = asyncio.run(engine.generate([{"role": "user", "content": "hi"}]))
        assert result.model == "fallback"
        primary.generate.assert_not_called()  # dead backend never hit

    def test_token_estimate_scales_with_prompt(self):
        """The TPM estimate reflects prompt size + expected output, not a constant."""
        from llm_router.router import _estimate_tokens

        small = _estimate_tokens([{"role": "user", "content": "hi"}], None)
        large = _estimate_tokens([{"role": "user", "content": "x" * 4000}], None)
        assert large > small
        # ~4 chars/token for the prompt, plus the default output allowance
        assert large >= 4000 // 4

    def test_fallback_emits_fallback_metric(self):
        engine = make_router()
        self._base(engine)
        primary = MagicMock()
        primary.generate = AsyncMock(side_effect=RuntimeError("boom"))
        fallback = MagicMock()
        fallback.generate = AsyncMock(
            return_value=GenerateResult(
                content="ok", model="fallback", usage=UsageInfo(1, 1, 2), finish_reason="stop", latency_ms=1.0
            )
        )
        engine.pool.get.side_effect = {"primary": primary, "fallback": fallback}.get

        with patch("llm_router.router._emit") as emit:
            result = asyncio.run(engine.generate([{"role": "user", "content": "hi"}]))

        assert result.model == "fallback"
        assert any(c.args == ("record_fallback", "primary", "fallback", "RuntimeError") for c in emit.call_args_list)

    def _sticky_engine(self):
        engine = make_router()
        self._base(engine, routed="model-a")
        engine.settings.session_affinity_enabled = True
        engine.pool.list_models.return_value = ["model-a", "model-b"]
        engine.pool.get.return_value.generate = AsyncMock(
            return_value=GenerateResult(
                content="ok", model="served", usage=UsageInfo(1, 1, 2), finish_reason="stop", latency_ms=1.0
            )
        )
        return engine

    def test_session_sticks_to_first_backend(self):
        engine = self._sticky_engine()
        # 1st request routes via strategy to model-a and pins the session.
        asyncio.run(engine.generate([{"role": "user", "content": "hi"}], user_id="u", session_id="s1"))
        # strategy now WOULD pick model-b, but the session is pinned to model-a.
        engine.policy_matcher.route = AsyncMock(return_value=MagicMock(model_id="model-b", strategy="policy"))
        engine.pool.get.reset_mock()
        asyncio.run(engine.generate([{"role": "user", "content": "again"}], user_id="u", session_id="s1"))
        assert any(c.args == ("model-a",) for c in engine.pool.get.call_args_list)
        assert all(c.args != ("model-b",) for c in engine.pool.get.call_args_list)
        engine.policy_matcher.route.assert_not_awaited()  # strategy skipped on the sticky hit

    def test_no_stickiness_when_disabled(self):
        engine = self._sticky_engine()
        engine.settings.session_affinity_enabled = False
        asyncio.run(engine.generate([{"role": "user", "content": "hi"}], user_id="u", session_id="s1"))
        engine.policy_matcher.route = AsyncMock(return_value=MagicMock(model_id="model-b", strategy="policy"))
        engine.pool.get.reset_mock()
        asyncio.run(engine.generate([{"role": "user", "content": "again"}], user_id="u", session_id="s1"))
        assert any(c.args == ("model-b",) for c in engine.pool.get.call_args_list)  # re-routed, not sticky

    def test_sticky_falls_back_to_strategy_when_pinned_model_unavailable(self):
        engine = self._sticky_engine()
        asyncio.run(engine.generate([{"role": "user", "content": "hi"}], user_id="u", session_id="s1"))
        # model-a's breaker opens → the pin is no longer usable.
        breaker = engine._breaker("model-a")
        for _ in range(engine.settings.circuit_breaker_threshold):
            breaker.record_failure()
        engine.policy_matcher.route = AsyncMock(return_value=MagicMock(model_id="model-b", strategy="policy"))
        engine.pool.get.reset_mock()
        asyncio.run(engine.generate([{"role": "user", "content": "again"}], user_id="u", session_id="s1"))
        engine.policy_matcher.route.assert_awaited_once()  # strategy re-consulted
        assert any(c.args == ("model-b",) for c in engine.pool.get.call_args_list)

    def test_no_fallback_after_partial_stream_output(self):
        """If the primary streams content then errors, the client is NOT given a
        second model's full answer concatenated onto the partial output."""
        engine = make_router()
        self._base(engine)
        primary = MagicMock()
        fallback = MagicMock()

        async def _partial_then_error(messages, **kwargs):
            yield GenerateResult(
                content="partial", model="primary", usage=UsageInfo(0, 0, 0), finish_reason="incomplete", latency_ms=0
            )
            raise RuntimeError("connection reset")

        async def _fallback_stream(messages, **kwargs):
            yield GenerateResult(
                content="SHOULD-NOT-APPEAR",
                model="fallback",
                usage=UsageInfo(0, 0, 0),
                finish_reason="stop",
                latency_ms=0,
            )

        primary.generate_stream = _partial_then_error
        fallback.generate_stream = _fallback_stream
        engine.pool.get.side_effect = {"primary": primary, "fallback": fallback}.get

        chunks: list = []

        async def _collect():
            async for chunk in engine.generate_stream([{"role": "user", "content": "hi"}]):
                chunks.append(chunk)

        with pytest.raises(RuntimeError, match="connection reset"):
            asyncio.run(_collect())
        assert [c.content for c in chunks] == ["partial"]  # only primary's partial output
        assert engine.pool.get.call_count == 1  # fallback stream never started


# ── Import test ──────────────────────────────────────────────────────


class TestImports:
    def test_router_module_importable(self):
        from llm_router import router  # noqa: F401

        assert hasattr(router, "RouterPolicyEngine")

    def test_router_exports_engine_class(self):
        from llm_router.router import RouterPolicyEngine as RPE  # noqa: N817

        assert RPE is not None
