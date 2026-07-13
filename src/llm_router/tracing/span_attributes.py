"""Span attribute constants for OTEL traces."""


class SpanAttributes:
    """Constants for all span attributes used in the LLM Router & Gateway."""

    # ── Gateway-level ───────────────────────
    GATEWAY_REQUEST_ID = "gateway.request_id"
    GATEWAY_USER_ID = "gateway.user_id"
    GATEWAY_CLIENT_IP = "gateway.client_ip"
    GATEWAY_METHOD = "gateway.method"
    GATEWAY_PATH = "gateway.path"

    # ── Routing ─────────────────────────────
    ROUTING_STRATEGY = "routing.strategy"
    ROUTING_POLICY = "routing.policy_matched"
    ROUTING_POLICY_ID = "routing.policy_id"

    # ── Model selection ─────────────────────
    MODEL_SELECTED = "model.selected"
    MODEL_PROVIDER = "model.provider"
    MODEL_BACKEND_TYPE = "model.backend_type"

    # ── Guardrails ──────────────────────────
    GUARDRAIL_PII_DETECTED = "guardrail.pii_detected"
    GUARDRAIL_PII_COUNT = "guardrail.pii_count"
    GUARDRAIL_ABUSE_SCORE = "guardrail.abuse_score"
    GUARDRAIL_ABUSE_SAFE = "guardrail.abuse_safe"
    GUARDRAIL_CONTENT_SAFE = "guardrail.content_safe"
    GUARDRAIL_RATE_LIMITED = "guardrail.rate_limited"

    # ── Token usage ─────────────────────────
    TOKEN_PROMPT = "token.prompt"
    TOKEN_COMPLETION = "token.completion"
    TOKEN_TOTAL = "token.total"
    TOKEN_COST = "token.cost"

    # ── Latency ─────────────────────────────
    LATENCY_MS = "latency.ms"
    LATENCY_P99_MS = "latency.p99_ms"

    # ── Response ────────────────────────────
    RESPONSE_FINISH_REASON = "response.finish_reason"
    RESPONSE_STATUS = "response.status"
    RESPONSE_ERROR = "response.error"

    # ── Audit ───────────────────────────────
    AUDIT_REQUEST_ID = "audit.request_id"
    AUDIT_STATUS = "audit.status"
    AUDIT_LOGGED = "audit.logged"
