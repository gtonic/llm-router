"""Hybrid routing — splits complex requests into multi-step plans."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llm_router.pool.base import ModelBackend

from llm_router.routing.base import PolicyBase, RoutingResult
from llm_router.routing.complexity import ComplexityDetector

# Complexity levels that warrant escalating to the remote (more capable) model.
_ESCALATION_LEVELS = {"high", "critical"}


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

    def __init__(
        self,
        local_model: str = "llama-local",
        remote_model: str = "gpt-5.4-nano",
        complexity_detector: ComplexityDetector | None = None,
    ) -> None:
        """Use ``local_model``/``remote_model`` (real backend IDs) for the plan's steps."""
        self.local_model = local_model
        self.remote_model = remote_model
        self._detector = complexity_detector or ComplexityDetector()

    def build_plan(self) -> HybridPlan:
        """Build the multi-step plan using the configured local/remote model IDs."""
        return HybridPlan(
            steps=[
                HybridStep(model=self.local_model, task="extract_entities", description="Extract entities from input"),
                HybridStep(
                    model=self.remote_model,
                    task="analyze",
                    description="Perform complex analysis",
                    depends_on="extract_entities",
                ),
                HybridStep(
                    model=self.local_model,
                    task="format",
                    description="Format and sanitize output",
                    depends_on="analyze",
                ),
            ],
            is_hybrid=True,
        )

    async def route(
        self,
        messages: list[dict],
        user_id: str | None = None,
        api_key: str | None = None,
        available_models: list[ModelBackend] | None = None,
    ) -> RoutingResult:
        """Local-first, escalate to the remote model for complex requests."""
        complexity = self._detector.analyze(messages)
        escalate = complexity.level in _ESCALATION_LEVELS
        model_id = self.remote_model if escalate else self.local_model
        plan = self.build_plan()
        return RoutingResult(
            model_id=model_id,
            strategy="hybrid",
            metadata={
                "complexity": complexity.level,
                "escalated": escalate,
                "plan": [s.model + ": " + s.task for s in plan.steps],
                "plan_description": plan.steps[0].description,
            },
        )
