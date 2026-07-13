"""OpenTelemetry initialization and tracer management."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from opentelemetry.trace import Tracer


_tracer = None


def setup_otel(
    service_name: str = "llm-router",
    otlp_enabled: bool = True,
    otlp_endpoint: str = "http://localhost:4318/v1/traces",
    otlp_protocol: str = "http/protobuf",
) -> None:
    """Initialize OpenTelemetry with OTLP exporter.

    Sets up the global TracerProvider with OTLP HTTP exporter.
    Use this at application startup.

    Args:
        service_name: Name of the service for trace metadata.
        otlp_enabled: Whether to enable OTLP exporting.
        otlp_endpoint: OTLP collector endpoint URL.
        otlp_protocol: Transport protocol (http/protobuf or grpc).
    """
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

    # Create resource
    resource = Resource.create({"service.name": service_name})

    # Create provider with resource
    provider = TracerProvider(resource=resource)

    if otlp_enabled:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.exporter import OTLPSpanExporter
            from opentelemetry.exporter.otlp.proto.http.exporter import OTLPSpanExporter as HTTPExporter
            from opentelemetry.exporter.otlp.proto.grpc.exporter import OTLPSpanExporter

            exporter = OTLPSpanExporter(
                endpoint=otlp_endpoint,
                timeout=5,
            )
            processor = BatchSpanProcessor(exporter)
            provider.add_span_processor(processor)
        except ImportError:
            # Fallback: console exporter
            exporter = ConsoleSpanExporter()
            processor = BatchSpanProcessor(exporter)
            provider.add_span_processor(processor)
    else:
        # Development: console exporter
        exporter = ConsoleSpanExporter()
        processor = BatchSpanProcessor(exporter)
        provider.add_span_processor(processor)

    trace.set_tracer_provider(provider)


def get_tracer(name: str = "llm-router") -> Tracer:
    """Get the configured tracer instance.

    Args:
        name: Name for the tracer.

    Returns:
        Configured OpenTelemetry tracer.
    """
    from opentelemetry import trace
    return trace.get_tracer(name)
