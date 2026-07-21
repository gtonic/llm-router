"""Tests for the per-backend circuit breaker."""

from __future__ import annotations

from llm_router.pool.circuit_breaker import CircuitBreaker, CircuitState


class _Clock:
    """Deterministic monotonic clock for tests."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def test_starts_closed_and_allows():
    cb = CircuitBreaker(threshold=3, cooldown=10.0, now=_Clock())
    assert cb.state is CircuitState.CLOSED
    assert cb.allow() is True
    assert cb.blocked is False


def test_opens_after_threshold_consecutive_failures():
    cb = CircuitBreaker(threshold=3, cooldown=10.0, now=_Clock())
    cb.record_failure()
    cb.record_failure()
    assert cb.state is CircuitState.CLOSED  # still under threshold
    cb.record_failure()
    assert cb.state is CircuitState.OPEN
    assert cb.blocked is True
    assert cb.allow() is False


def test_success_resets_failure_count():
    cb = CircuitBreaker(threshold=3, cooldown=10.0, now=_Clock())
    cb.record_failure()
    cb.record_failure()
    cb.record_success()
    cb.record_failure()
    cb.record_failure()
    assert cb.state is CircuitState.CLOSED  # counter reset, only 2 since success


def test_half_open_probe_after_cooldown_then_close_on_success():
    clock = _Clock()
    cb = CircuitBreaker(threshold=1, cooldown=10.0, now=clock)
    cb.record_failure()
    assert cb.blocked is True
    clock.advance(10.0)
    # blocked is read-only and false once cooldown elapsed
    assert cb.blocked is False
    # allow() opens a half-open probe
    assert cb.allow() is True
    assert cb.state is CircuitState.HALF_OPEN
    cb.record_success()
    assert cb.state is CircuitState.CLOSED


def test_half_open_probe_failure_reopens():
    clock = _Clock()
    cb = CircuitBreaker(threshold=1, cooldown=10.0, now=clock)
    cb.record_failure()
    clock.advance(10.0)
    assert cb.allow() is True  # half-open probe
    cb.record_failure()  # probe fails
    assert cb.state is CircuitState.OPEN
    assert cb.blocked is True  # re-opened for a fresh cooldown
    assert cb.allow() is False


def test_blocked_does_not_consume_the_probe():
    clock = _Clock()
    cb = CircuitBreaker(threshold=1, cooldown=10.0, now=clock)
    cb.record_failure()
    clock.advance(10.0)
    # Reading blocked repeatedly must not flip the breaker to half-open.
    assert cb.blocked is False
    assert cb.state is CircuitState.OPEN
    assert cb.blocked is False
    assert cb.state is CircuitState.OPEN
