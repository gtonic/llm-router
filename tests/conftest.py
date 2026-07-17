"""Shared pytest fixtures."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True, scope="session")
def _shutdown_otel_tracer_provider():
    """Flush and stop any TracerProvider's background export thread before teardown.

    Some tests exercise the real app lifespan (`setup_otel`), which registers a
    global BatchSpanProcessor. Without an explicit shutdown, its worker thread can
    try to export queued spans after pytest has already closed the captured
    stdout/stderr streams, producing a noisy "I/O operation on closed file" error.
    """
    yield
    from opentelemetry import trace

    provider = trace.get_tracer_provider()
    shutdown = getattr(provider, "shutdown", None)
    if callable(shutdown):
        shutdown()
