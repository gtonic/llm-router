"""Hybrid routing — splits complex requests into multi-step plans."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llm_router.pool.base import ModelBackend

from llm_router.routing.base import PolicyBase, RoutingResult


@dataclass
class HybridStep:
    model: str
    task: str
    description: str = ""
    depends_on: str | None = None


@dataclass
class HybridPlan:
    steps: list[HybridStep] = field(default_factory=list)
    is_hybrid: bool = False


class HybridRouter(PolicyBase):
    """Splits complex requests into multi-step plans for different models."""

    # Default hybrid plan template — can be extended with LLM-based planning
    DEFAULT_PLAN = [
        HybridStep(model="small_local", task="extract_entities", description="Extract entities from input"),
        HybridStep(model="remote_top", task="analyze", description="Perform complex analysis", depends_on="extract_entities"),
        HybridStep(model="small_local", task="format", description="Format and sanitize output", depends_on="analyze"),
    ]

    async def route(
        self,
        messages: list[dict],
        user_id: str | None = None,
        api_key: str | None = None,
        available_models: list[ModelBackend] | None = None,
    ) -> RoutingResult:
        """Return the first model from the default hybrid plan."""
        plan = HybridPlan(steps=self.DEFAULT_PLAN, is_hybrid=True)
        return RoutingResult(
            model_id=plan.steps[0].model,
            strategy="hybrid",
            metadata={"plan": [s.model + ": " + s.task for s in plan.steps], "plan_description": plan.steps[0].description},
        )
