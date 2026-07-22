"""Session → backend affinity store.

Remembers which backend served a session so follow-up turns of the same
conversation stick to it — for output consistency and to keep the backend's
prefix/KV cache warm (lower TTFT and prefill cost). Bounded (LRU) with a
per-entry TTL. In-memory / single-process: across replicas you'd back this with
a shared store, same caveat as the rate limiter.
"""

from __future__ import annotations

import time
from collections import OrderedDict


class SessionAffinityStore:
    """Bounded, TTL'd map of ``session_key -> model_id`` with LRU eviction."""

    def __init__(self, ttl: float = 1800.0, max_entries: int = 10000, *, now=time.monotonic) -> None:
        self._ttl = max(0.0, float(ttl))
        self._max = max(1, int(max_entries))
        self._now = now
        # key -> (model_id, expires_at)
        self._entries: OrderedDict[str, tuple[str, float]] = OrderedDict()

    def get(self, key: str) -> str | None:
        """Return the sticky model for ``key`` if present and unexpired, else None."""
        entry = self._entries.get(key)
        if entry is None:
            return None
        model, expires_at = entry
        if self._now() >= expires_at:
            self._entries.pop(key, None)
            return None
        self._entries.move_to_end(key)  # refresh LRU position
        return model

    def set(self, key: str, model: str) -> None:
        """Pin ``key`` to ``model`` and (re)start its TTL, evicting the oldest if full."""
        self._entries[key] = (model, self._now() + self._ttl)
        self._entries.move_to_end(key)
        while len(self._entries) > self._max:
            self._entries.popitem(last=False)

    def __len__(self) -> int:
        return len(self._entries)
