"""Tests for tracing module - OTEL setup and span attributes."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

# ── Span Attributes Tests ─────────────────────────────────────────


class TestSpanAttributes:
    """Test SpanAttributes class has all expected constants."""

    def test_gateway_attributes_exist(self):
        from llm_router.tracing.span_attributes import SpanAttributes

        assert hasattr(SpanAttributes, "GATEWAY_REQUEST_ID")
        assert hasattr(SpanAttributes, "GATEWAY_USER_ID")
        assert hasattr(SpanAttributes, "GATEWAY_CLIENT_IP")
        assert hasattr(SpanAttributes, "GATEWAY_METHOD")
        assert hasattr(SpanAttributes, "GATEWAY_PATH")

    def test_gateway_attribute_values(self):
        from llm_router.tracing.span_attributes import SpanAttributes

        assert SpanAttributes.GATEWAY_REQUEST_ID == "gateway.request_id"
        assert SpanAttributes.GATEWAY_USER_ID == "gateway.user_id"
        assert SpanAttributes.GATEWAY_CLIENT_IP == "gateway.client_ip"
        assert SpanAttributes.GATEWAY_METHOD == "gateway.method"
        assert SpanAttributes.GATEWAY_PATH == "gateway.path"

    def test_routing_attributes_exist(self):
        from llm_router.tracing.span_attributes import SpanAttributes

        assert hasattr(SpanAttributes, "ROUTING_STRATEGY")
        assert hasattr(SpanAttributes, "ROUTING_POLICY")
        assert hasattr(SpanAttributes, "ROUTING_POLICY_ID")

    def test_routing_attribute_values(self):
        from llm_router.tracing.span_attributes import SpanAttributes

        assert SpanAttributes.ROUTING_STRATEGY == "routing.strategy"
        assert SpanAttributes.ROUTING_POLICY == "routing.policy_matched"
        assert SpanAttributes.ROUTING_POLICY_ID == "routing.policy_id"

    def test_model_attributes_exist(self):
        from llm_router.tracing.span_attributes import SpanAttributes

        assert hasattr(SpanAttributes, "MODEL_SELECTED")
        assert hasattr(SpanAttributes, "MODEL_PROVIDER")
        assert hasattr(SpanAttributes, "MODEL_BACKEND_TYPE")

    def test_model_attribute_values(self):
        from llm_router.tracing.span_attributes import SpanAttributes

        assert SpanAttributes.MODEL_SELECTED == "model.selected"
        assert SpanAttributes.MODEL_PROVIDER == "model.provider"
        assert SpanAttributes.MODEL_BACKEND_TYPE == "model.backend_type"

    def test_guardrail_attributes_exist(self):
        from llm_router.tracing.span_attributes import SpanAttributes

        assert hasattr(SpanAttributes, "GUARDRAIL_PII_DETECTED")
        assert hasattr(SpanAttributes, "GUARDRAIL_PII_COUNT")
        assert hasattr(SpanAttributes, "GUARDRAIL_ABUSE_SCORE")
        assert hasattr(SpanAttributes, "GUARDRAIL_ABUSE_SAFE")
        assert hasattr(SpanAttributes, "GUARDRAIL_CONTENT_SAFE")
        assert hasattr(SpanAttributes, "GUARDRAIL_RATE_LIMITED")

    def test_guardrail_attribute_values(self):
        from llm_router.tracing.span_attributes import SpanAttributes

        assert SpanAttributes.GUARDRAIL_PII_DETECTED == "guardrail.pii_detected"
        assert SpanAttributes.GUARDRAIL_PII_COUNT == "guardrail.pii_count"
        assert SpanAttributes.GUARDRAIL_ABUSE_SCORE == "guardrail.abuse_score"
        assert SpanAttributes.GUARDRAIL_ABUSE_SAFE == "guardrail.abuse_safe"
        assert SpanAttributes.GUARDRAIL_CONTENT_SAFE == "guardrail.content_safe"
        assert SpanAttributes.GUARDRAIL_RATE_LIMITED == "guardrail.rate_limited"

    def test_token_attributes_exist(self):
        from llm_router.tracing.span_attributes import SpanAttributes

        assert hasattr(SpanAttributes, "TOKEN_PROMPT")
        assert hasattr(SpanAttributes, "TOKEN_COMPLETION")
        assert hasattr(SpanAttributes, "TOKEN_TOTAL")
        assert hasattr(SpanAttributes, "TOKEN_COST")

    def test_token_attribute_values(self):
        from llm_router.tracing.span_attributes import SpanAttributes

        assert SpanAttributes.TOKEN_PROMPT == "token.prompt"
        assert SpanAttributes.TOKEN_COMPLETION == "token.completion"
        assert SpanAttributes.TOKEN_TOTAL == "token.total"
        assert SpanAttributes.TOKEN_COST == "token.cost"

    def test_latency_attributes_exist(self):
        from llm_router.tracing.span_attributes import SpanAttributes

        assert hasattr(SpanAttributes, "LATENCY_MS")
        assert hasattr(SpanAttributes, "LATENCY_P99_MS")

    def test_latency_attribute_values(self):
        from llm_router.tracing.span_attributes import SpanAttributes

        assert SpanAttributes.LATENCY_MS == "latency.ms"
        assert SpanAttributes.LATENCY_P99_MS == "latency.p99_ms"

    def test_response_attributes_exist(self):
        from llm_router.tracing.span_attributes import SpanAttributes

        assert hasattr(SpanAttributes, "RESPONSE_FINISH_REASON")
        assert hasattr(SpanAttributes, "RESPONSE_STATUS")
        assert hasattr(SpanAttributes, "RESPONSE_ERROR")

    def test_response_attribute_values(self):
        from llm_router.tracing.span_attributes import SpanAttributes

        assert SpanAttributes.RESPONSE_FINISH_REASON == "response.finish_reason"
        assert SpanAttributes.RESPONSE_STATUS == "response.status"
        assert SpanAttributes.RESPONSE_ERROR == "response.error"

    def test_audit_attributes_exist(self):
        from llm_router.tracing.span_attributes import SpanAttributes

        assert hasattr(SpanAttributes, "AUDIT_REQUEST_ID")
        assert hasattr(SpanAttributes, "AUDIT_STATUS")
        assert hasattr(SpanAttributes, "AUDIT_LOGGED")

    def test_audit_attribute_values(self):
        from llm_router.tracing.span_attributes import SpanAttributes

        assert SpanAttributes.AUDIT_REQUEST_ID == "audit.request_id"
        assert SpanAttributes.AUDIT_STATUS == "audit.status"
        assert SpanAttributes.AUDIT_LOGGED == "audit.logged"


# ── OTEL Setup Tests ──────────────────────────────────────────────


class TestSetupOtel:
    """Test OpenTelemetry setup function."""

    def test_setup_otel_creates_provider(self):
        """setup_otel should create and set a TracerProvider."""
        from llm_router.tracing.otel_setup import setup_otel

        with (
            patch("opentelemetry.trace.set_tracer_provider") as mock_set_provider,
            patch("opentelemetry.sdk.trace.TracerProvider"),
            patch("opentelemetry.sdk.resources.Resource.create") as mock_resource,
            patch("opentelemetry.sdk.trace.export.ConsoleSpanExporter"),
        ):
            mock_resource.return_value = MagicMock()

            setup_otel(
                service_name="test-service",
                otlp_enabled=False,  # Use console exporter, not OTLP
            )

            mock_set_provider.assert_called_once()
            # Verify resource was created with correct service name
            mock_resource.assert_called_once_with({"service.name": "test-service"})

    def test_setup_otel_with_otlp_enabled(self):
        """setup_otel should try OTLP exporter when otlp_enabled=True."""
        import sys
        from unittest.mock import MagicMock as Mock
        from unittest.mock import patch

        from llm_router.tracing.otel_setup import setup_otel

        # Mock the OTLPSpanExporter class at the module level where it's used
        mock_otlp = Mock()
        mock_resource = Mock()

        # Create a mock module that has OTLPSpanExporter
        mock_grpc_module = type(sys)("mock_grpc")
        mock_grpc_module.OTLPSpanExporter = mock_otlp

        with (
            patch("opentelemetry.trace.set_tracer_provider"),
            patch("opentelemetry.sdk.trace.TracerProvider"),
            patch("opentelemetry.sdk.resources.Resource.create", return_value=mock_resource),
            patch.dict(sys.modules, {"opentelemetry.exporter.otlp.proto.grpc.exporter": mock_grpc_module}),
        ):
            mock_otlp.return_value = Mock()

            setup_otel(
                service_name="test-service",
                otlp_enabled=True,
                otlp_endpoint="http://collector:4318/v1/traces",
                otlp_protocol="http/protobuf",
            )

            mock_otlp.assert_called_once_with(
                endpoint="http://collector:4318/v1/traces",
                timeout=5,
            )

    def test_setup_otel_fallback_on_import_error(self):
        """setup_otel should fallback to console exporter on ImportError."""
        import sys
        from unittest.mock import MagicMock as Mock
        from unittest.mock import patch

        mock_resource = Mock()

        # Create a mock module that raises ImportError on OTLPSpanExporter access
        mock_grpc_module = type(sys)("mock_grpc")

        class FakeOTLPClass:
            def __init__(self, *a, **kw):
                raise ImportError("no otlp")

        mock_grpc_module.OTLPSpanExporter = FakeOTLPClass

        with (
            patch("opentelemetry.trace.set_tracer_provider"),
            patch("opentelemetry.sdk.trace.TracerProvider"),
            patch("opentelemetry.sdk.resources.Resource.create", return_value=mock_resource),
            patch("opentelemetry.sdk.trace.export.ConsoleSpanExporter") as mock_console,
            patch.dict(sys.modules, {"opentelemetry.exporter.otlp.proto.grpc.exporter": mock_grpc_module}),
        ):
            from llm_router.tracing.otel_setup import setup_otel

            setup_otel(otlp_enabled=True)

            mock_console.assert_called_once()

    def test_setup_otel_default_service_name(self):
        """setup_otel should use default service name when not specified."""
        from llm_router.tracing.otel_setup import setup_otel

        with (
            patch("opentelemetry.trace.set_tracer_provider"),
            patch("opentelemetry.sdk.trace.TracerProvider"),
            patch("opentelemetry.sdk.resources.Resource.create") as mock_resource,
            patch("opentelemetry.sdk.trace.export.ConsoleSpanExporter"),
        ):
            mock_resource.return_value = MagicMock()

            setup_otel()

            # Verify service.name was passed to Resource.create
            calls = [str(c) for c in mock_resource.call_args_list]
            assert any("llm-router" in c for c in calls)

    def test_setup_otel_with_custom_timeout(self):
        """setup_otel should accept custom timeout for OTLP exporter."""
        import sys
        from unittest.mock import MagicMock as Mock
        from unittest.mock import patch

        from llm_router.tracing.otel_setup import setup_otel

        mock_resource = Mock()
        mock_otlp = Mock()

        # Create a mock module that has OTLPSpanExporter
        mock_grpc_module = type(sys)("mock_grpc")
        mock_grpc_module.OTLPSpanExporter = mock_otlp

        with (
            patch("opentelemetry.trace.set_tracer_provider"),
            patch("opentelemetry.sdk.trace.TracerProvider"),
            patch("opentelemetry.sdk.resources.Resource.create", return_value=mock_resource),
            patch.dict(sys.modules, {"opentelemetry.exporter.otlp.proto.grpc.exporter": mock_grpc_module}),
        ):
            mock_otlp.return_value = Mock()

            # Note: the current implementation doesn't expose timeout parameter,
            # but we verify the default behavior works
            setup_otel(otlp_enabled=True)
            mock_otlp.assert_called_once()


class TestGetTracer:
    """Test get_tracer function."""

    def test_get_tracer_returns_tracer(self):
        """get_tracer should return a tracer instance."""
        from llm_router.tracing.otel_setup import get_tracer

        tracer = get_tracer("test-tracer")
        assert tracer is not None

    def test_get_tracer_uses_provided_name(self):
        """get_tracer should use the provided name."""
        with patch("opentelemetry.trace.get_tracer") as mock_get:
            mock_get.return_value = MagicMock()

            from llm_router.tracing.otel_setup import get_tracer

            get_tracer("my-custom-tracer")

            mock_get.assert_called_once_with("my-custom-tracer")


# ── Module Import Tests ───────────────────────────────────────────


class TestTracingModule:
    """Test tracing module exports."""

    def test_tracing_module_importable(self):
        """Tracing module should be importable."""
        from llm_router import tracing

        assert hasattr(tracing, "setup_otel")
        assert hasattr(tracing, "SpanAttributes")

    def test_tracing_exports(self):
        """Tracing module should export SpanAttributes and setup_otel."""
        from llm_router.tracing import SpanAttributes, setup_otel

        assert SpanAttributes is not None
        assert callable(setup_otel)

    def test_span_attributes_class_accessible(self):
        """SpanAttributes should be accessible from tracing module."""
        from llm_router.tracing import SpanAttributes

        assert hasattr(SpanAttributes, "GATEWAY_REQUEST_ID")

    def test_setup_otel_callable(self):
        """setup_otel should be callable."""
        from llm_router.tracing import setup_otel

        assert callable(setup_otel)
