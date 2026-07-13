"""Round-robin load balancing across model backends."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llm_router.pool.base import ModelBackend

from llm_router.routing.base import PolicyBase, RoutingResult


class RoundRobinPolicy(PolicyBase):
    """Distributes requests evenly across a list of model IDs."""

    def __init__(self, model_ids: list[str]) -> None:
        self.model_ids = model_ids
        self._counter = 0
        self._lock = asyncio.Lock()

    async def route(
        self,
        messages: list[dict],
        user_id: str | None = None,
        api_key: str | None = None,
        available_models: list[ModelBackend] | None = None,
    ) -> RoutingResult:
        async with self._lock:
            model = self.model_ids[self._counter % len(self.model_ids)]
            self._counter += 1
            return RoutingResult(model_id=model, strategy="round_robin")
