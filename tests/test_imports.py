"""Basic import tests to verify the package loads correctly."""

from llm_router import __version__


def test_version():
    """Verify version is set."""
    assert __version__ == "0.1.0"


def test_import_config():
    """Verify config module loads."""
    from llm_router.config import GatewaySettings, ModelBackendConfig, PolicyRule, RoutingStrategy

    assert issubclass(RoutingStrategy, str)


def test_import_models():
    """Verify models module loads."""
    from llm_router.models import (
        ChatCompletionRequest,
        ChatCompletionResponse,
        MessageRole,
        UsageInfo,
    )

    assert MessageRole.user == "user"


def test_import_guardrails():
    """Verify guardrails modules load."""
    from llm_router.guardrails import AbuseFilter, ContentSafety, PiiFilter, RateLimiter

    assert issubclass(PiiFilter, object)


def test_import_routing():
    """Verify routing modules load."""
    from llm_router.routing import PolicyBase, PolicyMatcher, RoundRobinPolicy

    assert issubclass(PolicyBase, object)


def test_import_pool():
    """Verify pool modules load."""
    from llm_router.pool import ModelBackend, ModelPool

    assert issubclass(ModelBackend, object)


def test_import_tracing():
    """Verify tracing modules load."""
    from llm_router.tracing import SpanAttributes, get_tracer, setup_otel

    assert hasattr(SpanAttributes, "GATEWAY_REQUEST_ID")
