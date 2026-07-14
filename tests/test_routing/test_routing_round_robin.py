"""Tests for routing/round_robin.py - RoundRobinPolicy."""

from __future__ import annotations

import pytest

from llm_router.routing.round_robin import RoundRobinPolicy


class TestRoundRobinPolicy:
    @pytest.mark.asyncio
    async def test_round_robin_cycle(self):
        """Verify round-robin distributes evenly."""
        models = ["m1", "m2", "m3"]
        router = RoundRobinPolicy(models)

        results = []
        for _ in range(6):
            result = await router.route([{"role": "user", "content": "x"}])
            results.append(result.model_id)

        assert results == ["m1", "m2", "m3", "m1", "m2", "m3"]

    @pytest.mark.asyncio
    async def test_single_model(self):
        router = RoundRobinPolicy(["only-one"])
        for _ in range(10):
            result = await router.route([{"role": "user", "content": "x"}])
            assert result.model_id == "only-one"
            assert result.strategy == "round_robin"

    @pytest.mark.asyncio
    async def test_returns_routing_result(self):
        router = RoundRobinPolicy(["a", "b"])
        result = await router.route([{"role": "user", "content": "Hi"}])
        assert hasattr(result, "model_id")
        assert hasattr(result, "strategy")
        assert result.strategy == "round_robin"

    @pytest.mark.asyncio
    async def test_concurrent_access(self):
        """Ensure thread safety with concurrent calls."""
        import asyncio
        models = ["m1", "m2", "m3"]
        router = RoundRobinPolicy(models)

        async def route_one():
            return await router.route([{"role": "user", "content": "x"}])

        results = await asyncio.gather(*[route_one() for _ in range(100)])
        model_ids = [r.model_id for r in results]
        # Should only use models from the list
        for mid in model_ids:
            assert mid in models
