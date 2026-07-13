"""OpenTelemetry tracing setup for the LLM Router & Gateway."""

from llm_router.tracing.otel_setup import setup_otel
from llm_router.tracing.span_attributes import SpanAttributes

__all__ = ["SpanAttributes", "get_tracer", "setup_otel"]
