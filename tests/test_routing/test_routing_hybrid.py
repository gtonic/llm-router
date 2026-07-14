"""Tests for routing/hybrid.py - HybridRouter."""

from __future__ import annotations

import pytest

from llm_router.routing.hybrid import HybridRouter, HybridPlan, HybridStep


class TestHybridStep:
    def test_minimal(self):
        step = HybridStep(model="m1", task="t1")
        assert step.model == "m1"
        assert step.task == "t1"
        assert step.description == ""
        assert step.depends_on is None

    def test_full(self):
        step = HybridStep(model="m2", task="t2", description="Desc", depends_on="s1")
        assert step.model == "m2"
        assert step.description == "Desc"
        assert step.depends_on == "s1"


class TestHybridPlan:
    def test_empty_plan(self):
        plan = HybridPlan()
        assert plan.steps == []
        assert plan.is_hybrid is False

    def test_with_steps(self):
        plan = HybridPlan(steps=[HybridStep("m1", "t1")], is_hybrid=True)
        assert len(plan.steps) == 1
        assert plan.is_hybrid is True


class TestHybridRouter:
    def test_route_returns_first_step_model(self):
        import asyncio
        router = HybridRouter()
        result = asyncio.run(router.route([{"role": "user", "content": "Test"}]))
        assert result.model_id == "small_local"  # First step model
        assert result.strategy == "hybrid"

    def test_route_contains_plan_metadata(self):
        import asyncio
        router = HybridRouter()
        result = asyncio.run(router.route([{"role": "user", "content": "Test"}]))
        plan = result.metadata.get("plan", [])
        assert len(plan) == 3  # DEFAULT_PLAN has 3 steps
        assert "extract_entities" in plan[0]

    def test_default_plan_steps(self):
        router = HybridRouter()
        steps = router.DEFAULT_PLAN
        assert len(steps) == 3
        assert steps[0].model == "small_local"
        assert steps[1].model == "remote_top"
        assert steps[2].model == "small_local"
