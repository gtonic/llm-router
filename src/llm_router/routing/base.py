"""Abstract base for all routing strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llm_router.pool.base import ModelBackend


@dataclass
class RoutingResult:
    """Result from a routing decision."""
    model_id: str
    strategy: str
    policy_matched: str | None = None
    metadata: dict = field(default_factory=dict)


class PolicyBase(ABC):
    """Abstract base class for all routing strategies."""

    @abstractmethod
    async def route(
        self,
        messages: list[dict],
        user_id: str | None = None,
        api_key: str | None = None,
        available_models: list[ModelBackend] | None = None,
    ) -> RoutingResult:
        """Select the best model for the given request."""
        ...
