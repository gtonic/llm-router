"""Per-backend circuit breaker.

Stops the router from hammering a backend that is currently failing: after a
run of consecutive failures the breaker OPENs and rejects calls for a cooldown
window, then allows a single HALF_OPEN probe to decide whether to close again.
Passive (driven by real call outcomes) rather than relying on an active health
ping, so it reflects what requests actually experience.
"""

from __future__ import annotations

import time
from enum import StrEnum


class CircuitState(StrEnum):
    """Lifecycle states of a :class:`CircuitBreaker`."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """A minimal per-backend circuit breaker.

    * CLOSED — calls flow; consecutive failures are counted.
    * OPEN — calls are rejected until ``cooldown`` seconds have elapsed.
    * HALF_OPEN — one probe is allowed; success closes, failure re-opens.
    """

    def __init__(self, threshold: int = 5, cooldown: float = 30.0, *, now=time.monotonic) -> None:
        self._threshold = max(1, int(threshold))
        self._cooldown = max(0.0, float(cooldown))
        self._now = now
        self._failures = 0
        self._opened_at: float | None = None
        self._half_open = False

    @property
    def state(self) -> CircuitState:
        if self._opened_at is None:
            return CircuitState.CLOSED
        if self._half_open:
            return CircuitState.HALF_OPEN
        return CircuitState.OPEN

    @property
    def blocked(self) -> bool:
        """True if the breaker is open and still within its cooldown.

        Read-only: unlike :meth:`allow` it does not consume the half-open
        probe, so it is safe to call during backend selection.
        """
        if self._opened_at is None:
            return False
        return (self._now() - self._opened_at) < self._cooldown

    def allow(self) -> bool:
        """Return True if a call may be attempted now (may open a probe)."""
        if self._opened_at is None:
            return True
        if self._now() - self._opened_at >= self._cooldown:
            # Cooldown elapsed — permit a single half-open probe.
            self._half_open = True
            return True
        return False

    def record_success(self) -> None:
        """Reset the breaker to CLOSED."""
        self._failures = 0
        self._opened_at = None
        self._half_open = False

    def record_failure(self) -> bool:
        """Count a failure; open (or re-open) the breaker at the threshold.

        Returns ``True`` if this failure transitioned the breaker to OPEN
        (either a threshold trip or a failed half-open probe), else ``False``.
        """
        self._failures += 1
        if self._half_open:
            # Probe failed — re-open for a fresh cooldown.
            self._opened_at = self._now()
            self._half_open = False
            return True
        if self._opened_at is None and self._failures >= self._threshold:
            self._opened_at = self._now()
            return True
        return False
