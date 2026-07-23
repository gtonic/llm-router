"""Rate limiter using token-bucket algorithm."""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass


@dataclass
class RateLimitResult:
    allowed: bool
    remaining_requests: int
    reset_at: float | None = None
    error: str = ""


class RateLimiter:
    """Token-bucket rate limiter per client/API key.

    Supports both requests-per-minute and tokens-per-minute limits.

    Per-client state is bounded by an LRU cap (``max_clients``): each distinct
    client_id costs memory, so an attacker who can vary the identity (e.g. an
    unauthenticated caller rotating source IPs) cannot grow the maps without
    bound. Least-recently-seen clients are evicted once the cap is exceeded;
    eviction only drops accounting history, never bypasses an active limit.
    """

    def __init__(self, rpm: int = 60, tpm: int = 60000, max_clients: int = 10000) -> None:
        self.rpm = rpm
        self.tpm = tpm
        self.max_clients = max(1, max_clients)
        # client_id -> list of request timestamps (LRU-ordered, oldest first)
        self._requests: OrderedDict[str, list[float]] = OrderedDict()
        # client_id -> list of (timestamp, token_count)
        self._tokens: OrderedDict[str, list[tuple[float, int]]] = OrderedDict()
        self._cleanup_interval = 300  # seconds
        self._last_cleanup: float = time.monotonic()

    async def check(self, client_id: str, tokens: int = 0) -> RateLimitResult:
        """Check if a request is within rate limits.

        Args:
            client_id: Unique client identifier (user_id, API key, IP).
            tokens: Estimated token count for this request (for TPM limit).

        Returns:
            RateLimitResult with allowed/denied status.
        """
        now = time.monotonic()

        # Periodic cleanup of old entries
        if now - self._last_cleanup > self._cleanup_interval:
            self._cleanup(now)
            self._last_cleanup = now

        # Check RPM limit
        rpm_allowed, remaining = self._check_rpm(client_id, now)
        if not rpm_allowed:
            self._touch(client_id)
            return RateLimitResult(
                allowed=False,
                remaining_requests=remaining,
                error="Rate limit exceeded: too many requests per minute",
            )

        # Check TPM limit
        if tokens > 0:
            tpm_allowed = self._check_tpm(client_id, tokens, now)
            if not tpm_allowed:
                self._touch(client_id)
                return RateLimitResult(
                    allowed=False,
                    remaining_requests=remaining,
                    error="Rate limit exceeded: too many tokens per minute",
                )

        # All checks passed — record this request
        self._requests.setdefault(client_id, []).append(now)
        if tokens > 0:
            self._tokens.setdefault(client_id, []).append((now, tokens))

        self._touch(client_id)
        self._evict()
        return RateLimitResult(allowed=True, remaining_requests=max(0, remaining - 1))

    def _check_rpm(self, client_id: str, now: float) -> tuple[bool, int]:
        """Check requests-per-minute limit."""
        cutoff = now - 60
        # Count requests in the last 60 seconds
        recent = [t for t in self._requests.get(client_id, []) if t > cutoff]
        if recent or client_id in self._requests:
            self._requests[client_id] = recent
        remaining = max(0, self.rpm - len(recent))
        return len(recent) < self.rpm, remaining

    def _check_tpm(self, client_id: str, tokens: int, now: float) -> bool:
        """Check tokens-per-minute limit."""
        cutoff = now - 60
        recent = [(t, n) for t, n in self._tokens.get(client_id, []) if t > cutoff]
        if recent or client_id in self._tokens:
            self._tokens[client_id] = recent
        used_tokens = sum(n for _, n in recent)
        return used_tokens + tokens <= self.tpm

    def _touch(self, client_id: str) -> None:
        """Mark ``client_id`` as most-recently-seen for LRU ordering."""
        if client_id in self._requests:
            self._requests.move_to_end(client_id)
        if client_id in self._tokens:
            self._tokens.move_to_end(client_id)

    def _evict(self) -> None:
        """Drop least-recently-seen clients once the cap is exceeded."""
        while len(self._requests) > self.max_clients:
            old_id, _ = self._requests.popitem(last=False)
            self._tokens.pop(old_id, None)

    def _cleanup(self, now: float) -> None:
        """Remove stale entries older than 10 minutes."""
        cutoff = now - 600
        for client_id in list(self._requests.keys()):
            self._requests[client_id] = [t for t in self._requests[client_id] if t > cutoff]
            if not self._requests[client_id]:
                del self._requests[client_id]
        for client_id in list(self._tokens.keys()):
            self._tokens[client_id] = [(t, n) for t, n in self._tokens[client_id] if t > cutoff]
            if not self._tokens[client_id]:
                del self._tokens[client_id]
