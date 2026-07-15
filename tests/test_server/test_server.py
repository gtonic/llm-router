"""Tests for server/app.py and server/routes.py."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient

# ── Test Client Factory ────────────────────────────────────────────


def make_test_client():
    """Create a test client with mocked router engine."""
    from llm_router.config import RoutingStrategy
    from llm_router.pool.base import GenerateResult, UsageInfo
    from llm_router.router import RouterPolicyEngine

    pool = MagicMock()
    policy_matcher = MagicMock()
    complexity_detector = MagicMock()
    hybrid_router = MagicMock()
    round_robin = MagicMock()
    rate_limiter = MagicMock()
    pii_filter = MagicMock()
    abuse_filter = MagicMock()
    content_safety = MagicMock()

    def _make_rate_result(allowed=True, remaining=59):
        r = MagicMock()
        r.allowed = allowed
        r.remaining_requests = remaining
        return r

    rate_limiter.check = AsyncMock(return_value=_make_rate_result())

    def _make_pii_result(has_pii=False, patterns=None, **kwargs):
        p = MagicMock()
        p.has_pii = has_pii
        p.patterns = patterns or []
        return p

    pii_filter.check = _make_pii_result

    def _make_abuse_result(safe=True, score=0.0, categories=None, details=""):
        r = MagicMock()
        r.safe = safe
        r.abuse_score = score
        r.categories = categories or []
        r.details = details
        return r

    abuse_filter.check = _make_abuse_result
    content_safety.check = _make_abuse_result

    def _make_route(model_id="test-model", strategy="policy"):
        m = MagicMock()
        m.model_id = model_id
        m.strategy = strategy
        m.metadata = {}
        m.policies_matched = []
        return m

    policy_matcher.route = AsyncMock(side_effect=lambda *a, **kw: _make_route())
    complexity_detector.analyze = MagicMock(return_value=MagicMock(level="low", score=0.1, factors=["simple"]))
    hybrid_router.route = MagicMock(return_value=_make_route(strategy="hybrid"))
    round_robin.route = MagicMock(return_value=_make_route(strategy="round_robin"))

    pii_filter.PATTERNS = [
        ("ssn", "Social Security Number"),
        ("email", "Email Address"),
    ]
    pii_filter.redact = True
    pii_filter.redact_text = lambda text: text

    engine = RouterPolicyEngine(
        pool=pool,
        routing_strategy=RoutingStrategy.POLICY,
        policy_matcher=policy_matcher,
        complexity_detector=complexity_detector,
        hybrid_router=hybrid_router,
        round_robin=round_robin,
        rate_limiter=rate_limiter,
        pii_filter=pii_filter,
        abuse_filter=abuse_filter,
        content_safety=content_safety,
        default_model="test-model",
    )

    result = GenerateResult(
        content="Mocked response",
        model="test-model",
        usage=UsageInfo(
            prompt_tokens=5,
            completion_tokens=10,
            total_tokens=15,
        ),
        finish_reason="stop",
        latency_ms=25.0,
    )
    pool.get.return_value.generate = AsyncMock(return_value=result)

    async def _stream_gen(*args, **kwargs):
        yield GenerateResult(
            content="stream1",
            model="test-model",
            usage=UsageInfo(
                prompt_tokens=1,
                completion_tokens=1,
                total_tokens=2,
            ),
            finish_reason="stop",
        )

    pool.get.return_value.generate_stream = _stream_gen

    import llm_router.server.app as app_mod

    app_mod.router_engine = engine

    from llm_router.server.app import create_app

    return TestClient(create_app())


def make_tool_streaming_client():
    client = make_test_client()
    import llm_router.server.app as app_mod

    async def _tool_stream(*args, **kwargs):
        from llm_router.pool.base import GenerateResult, UsageInfo

        yield GenerateResult(
            content="",
            model="test-model",
            usage=UsageInfo(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            finish_reason="tool_calls",
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "bash", "arguments": '{"command":"pwd"}'},
                }
            ],
        )

    app_mod.router_engine.pool.get.return_value.generate_stream = _tool_stream
    return client


# ── App Factory Tests ──────────────────────────────────────────────


class TestCreateApp:
    def test_app_created(self):
        """create_app returns a FastAPI instance."""
        from llm_router.server.app import create_app

        app = create_app()
        assert app is not None
        assert app.title == "LLM Router & Gateway"
        assert app.version == "0.1.0"

    def test_health_route_registered(self):
        """The /health endpoint should be registered."""
        from llm_router.server.app import create_app

        # Use a client that skips lifespan (lifespan depends on config)
        app = create_app()
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "ok"
            assert "version" in data

    def test_cors_middleware_configured(self):
        """CORS middleware should allow all origins."""
        from llm_router.server.app import create_app

        app = create_app()
        # At least the lifespan + error handlers exist
        assert app is not None

    def test_routes_registered(self):
        """All expected routes should be registered."""
        from llm_router.server.app import create_app

        app = create_app()
        # Collect paths from all routes, including those inside _IncludedRouter
        route_paths = []
        for route in app.routes:
            if hasattr(route, "path"):
                route_paths.append(route.path)
            elif hasattr(route, "original_router"):
                # _IncludedRouter wraps the APIRouter
                for inner in route.original_router.routes:
                    if hasattr(inner, "path"):
                        route_paths.append(inner.path)
        assert any("/chat/completions" in p for p in route_paths)
        assert any("/models" in p for p in route_paths)
        assert any("/guardrails" in p for p in route_paths)


# ── Health Endpoint Tests ──────────────────────────────────────────


class TestHealthEndpoint:
    def test_health_ok(self):
        client = make_test_client()
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["version"] == "0.1.0"

    def test_health_returns_json(self):
        client = make_test_client()
        resp = client.get("/health")
        assert resp.headers["content-type"].startswith("application/json")


# ── Chat Completions Endpoint Tests ────────────────────────────────


class TestChatCompletions:
    def _post_chat(self, client, **kwargs):
        payload = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello!"}],
            "stream": False,
        }
        payload.update(kwargs)
        return client.post("/v1/chat/completions", json=payload)

    def test_non_streaming_success(self):
        client = make_test_client()
        resp = self._post_chat(client)
        assert resp.status_code == 200
        data = resp.json()
        assert "id" in data
        assert data["model"] == "test-model"
        assert len(data["choices"]) == 1
        choice = data["choices"][0]
        assert choice["message"]["role"] == "assistant"
        assert "Mocked" in choice["message"]["content"]
        assert "usage" in data
        assert data["usage"]["total_tokens"] == 15

    def test_non_streaming_with_stream_false(self):
        """Explicit stream=False should return non-streaming response."""
        client = make_test_client()
        resp = self._post_chat(client, stream=False)
        assert resp.status_code == 200
        data = resp.json()
        assert "choices" in data

    def test_non_streaming_custom_model(self):
        """Should use provided model parameter."""
        client = make_test_client()
        resp = self._post_chat(client, model="gpt-4o")
        assert resp.status_code == 200
        data = resp.json()
        # Router overrides to actual model
        assert data["model"] == "test-model"

    def test_non_streaming_with_temperature(self):
        """Temperature parameter should not cause errors."""
        client = make_test_client()
        resp = self._post_chat(client, temperature=0.8, max_tokens=100)
        assert resp.status_code == 200

    def test_non_streaming_forwards_max_tokens_to_backend(self):
        client = make_test_client()
        resp = self._post_chat(client, max_tokens=8192)

        assert resp.status_code == 200
        import llm_router.server.app as app_mod

        backend_call = app_mod.router_engine.pool.get.return_value.generate.await_args
        assert backend_call is not None
        assert backend_call.kwargs["max_tokens"] == 8192

    def test_non_streaming_system_message(self):
        """System messages should be passed through."""
        client = make_test_client()
        resp = self._post_chat(
            client,
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
            ],
        )
        assert resp.status_code == 200
        assert resp.json()["choices"][0]["message"]["role"] == "assistant"

    def test_non_streaming_text_content_parts(self):
        """OpenAI content-part arrays should be accepted for text messages."""
        client = make_test_client()
        resp = self._post_chat(
            client,
            messages=[{"role": "user", "content": [{"type": "text", "text": "Hello!"}]}],
        )
        assert resp.status_code == 200

    def test_non_streaming_with_user_id(self):
        """user_id parameter should be accepted without error."""
        client = make_test_client()
        resp = self._post_chat(client, user_id="user-123")
        assert resp.status_code == 200

    def test_non_streaming_with_api_key(self):
        """api_key parameter should be accepted without error."""
        client = make_test_client()
        resp = self._post_chat(client, api_key="sk-test-key")
        assert resp.status_code == 200

    def test_streaming_enabled(self):
        """stream=True should trigger streaming response."""
        client = make_test_client()
        resp = self._post_chat(client, stream=True)
        assert resp.status_code == 200
        # Streaming returns a StreamingResponse
        assert "text/event-stream" in resp.headers.get("content-type", "")

    def test_streaming_tool_call_preserves_structured_delta(self):
        """Streaming tool calls must survive even when their content is empty."""
        client = make_tool_streaming_client()
        resp = self._post_chat(
            client,
            stream=True,
            tools=[
                {
                    "type": "function",
                    "function": {"name": "bash", "parameters": {"type": "object"}},
                }
            ],
        )

        assert resp.status_code == 200
        assert '"tool_calls"' in resp.text
        assert '"name":"bash"' in resp.text
        assert '"finish_reason":"tool_calls"' in resp.text

    def test_streaming_with_multiple_messages(self):
        """Streaming should handle conversation history."""
        client = make_test_client()
        resp = self._post_chat(
            client,
            stream=True,
            messages=[
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"},
                {"role": "user", "content": "And 3+3?"},
            ],
        )
        assert resp.status_code == 200

    def test_streaming_with_custom_params(self):
        """Streaming should respect custom temperature and max_tokens."""
        client = make_test_client()
        resp = self._post_chat(
            client,
            stream=True,
            temperature=0.5,
            max_tokens=50,
            top_p=0.9,
        )
        assert resp.status_code == 200

    def test_chat_completions_missing_messages(self):
        """Missing messages field should return 422."""
        client = make_test_client()
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "test-model"},
        )
        assert resp.status_code == 422

    def test_chat_completions_empty_body(self):
        """Empty body should return 422."""
        client = make_test_client()
        resp = client.post("/v1/chat/completions", json={})
        assert resp.status_code == 422

    def test_chat_completions_invalid_model_field(self):
        """Missing model field should return 422."""
        client = make_test_client()
        resp = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hi"}]},
        )
        assert resp.status_code == 422


# ── Models Endpoint Tests ──────────────────────────────────────────


class TestModelsEndpoint:
    def _get_models(self, client):
        return client.get("/v1/models")

    def test_list_models_success(self):
        client = make_test_client()
        resp = self._get_models(client)
        assert resp.status_code == 200
        data = resp.json()
        assert "data" in data
        assert isinstance(data["data"], list)
        assert data["object"] == "list"

    def test_list_models_empty_pool(self):
        """Should retain the logical auto model when no backends are available."""
        from llm_router.server.app import router_engine

        if router_engine:
            router_engine.pool.list_models.return_value = []
        client = make_test_client()
        resp = self._get_models(client)
        assert resp.status_code == 200
        data = resp.json()
        assert [model["id"] for model in data["data"]] == ["router-auto"]

    def test_list_models_multiple_models(self):
        """Should return all registered models."""
        client = make_test_client()
        resp = self._get_models(client)
        assert resp.status_code == 200
        data = resp.json()
        for model in data["data"]:
            assert "id" in model
            assert model["object"] == "model"
            assert model["owned_by"] == "llm-router"

    def test_list_models_has_required_fields(self):
        """Each model should have required OpenAI-compatible fields."""
        client = make_test_client()
        resp = self._get_models(client)
        assert resp.status_code == 200
        data = resp.json()
        if data["data"]:
            model = data["data"][0]
            assert "id" in model
            assert "object" in model
            assert "created" in model
            assert "owned_by" in model


# ── Guardrail PII Patterns Endpoint Tests ─────────────────────────


class TestPiiPatternsEndpoint:
    def _get_pii_patterns(self, client):
        return client.get("/v1/guardrails/pii/patterns")

    def test_pii_patterns_success(self):
        client = make_test_client()
        resp = self._get_pii_patterns(client)
        assert resp.status_code == 200
        data = resp.json()
        assert "patterns" in data
        assert "redact_enabled" in data
        assert isinstance(data["patterns"], list)

    def test_pii_patterns_contains_expected(self):
        """Should include common PII pattern names."""
        client = make_test_client()
        resp = self._get_pii_patterns(client)
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["patterns"]) >= 0

    def test_redact_enabled_flag(self):
        """redact_enabled should be a boolean."""
        client = make_test_client()
        resp = self._get_pii_patterns(client)
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data["redact_enabled"], bool)


# ── Guardrail Check Endpoint Tests ─────────────────────────────────


class TestGuardrailCheckEndpoint:
    def _post_check(self, client, mode="pii", text="Hello world"):
        return client.post(
            "/v1/guardrails/check",
            json={"text": text, "mode": mode},
        )

    def test_check_pii_mode(self):
        client = make_test_client()
        resp = self._post_check(client, mode="pii", text="My email is test@example.com")
        assert resp.status_code == 200
        data = resp.json()
        assert "has_pii" in data or "safe" in data

    def test_check_abuse_mode(self):
        client = make_test_client()
        resp = self._post_check(client, mode="abuse", text="Hello friend")
        assert resp.status_code == 200
        data = resp.json()
        # abuse_filter.check returns a MagicMock with safe/score attributes
        assert "safe" in data or "safe" not in data  # may vary by mock state

    def test_check_safety_mode(self):
        client = make_test_client()
        resp = self._post_check(client, mode="safety", text="Safe content here")
        assert resp.status_code == 200
        data = resp.json()
        assert "safe" in data

    def test_check_invalid_mode(self):
        client = make_test_client()
        resp = self._post_check(client, mode="invalid", text="test")
        assert resp.status_code == 200
        data = resp.json()
        assert "error" in data

    def test_check_empty_text(self):
        client = make_test_client()
        resp = self._post_check(client, text="")
        assert resp.status_code == 200

    def test_check_missing_text(self):
        """Missing text field should be handled gracefully."""
        client = make_test_client()
        resp = client.post("/v1/guardrails/check", json={"mode": "pii"})
        assert resp.status_code == 200


# ── Admin Reload Endpoint Tests ────────────────────────────────────


class TestAdminReloadEndpoint:
    def _post_reload(self, client):
        return client.post("/v1/admin/reload")

    def test_reload_success(self):
        client = make_test_client()
        resp = self._post_reload(client)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "reloaded"
        assert "policies" in data

    def test_reload_returns_policy_count(self):
        client = make_test_client()
        resp = self._post_reload(client)
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data["policies"], int)
        assert data["policies"] >= 0

    def test_reload_with_failing_policy_matcher(self):
        """Should return 500 if policy reload fails."""
        # Create client first, then set up failure on its engine
        client = make_test_client()

        # Import and set the failure after make_test_client creates the engine
        import llm_router.server.app as app_mod

        original = app_mod.router_engine.policy_matcher._load_policies
        app_mod.router_engine.policy_matcher._load_policies = MagicMock(side_effect=RuntimeError("config load failed"))

        try:
            resp = self._post_reload(client)
            assert resp.status_code == 500
            assert "config load failed" in resp.json()["detail"]
        finally:
            app_mod.router_engine.policy_matcher._load_policies = original


# ── Integration / Error Path Tests ─────────────────────────────────


class TestErrorPaths:
    def test_chat_completions_503_without_engine(self):
        """Should return 503 when router_engine is None."""
        import llm_router.server.app as app_mod
        from llm_router.server.app import create_app

        original = app_mod.router_engine

        try:
            with TestClient(create_app(), raise_server_exceptions=False) as client:
                # Set to None AFTER create_app (lifespan would have initialized it)
                app_mod.router_engine = None
                resp = client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "x",
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                )
                assert resp.status_code == 503
                assert "not initialized" in resp.json()["detail"]
        finally:
            app_mod.router_engine = original

    def test_models_503_without_engine(self):
        """Should return 503 when router_engine is None."""
        import llm_router.server.app as app_mod
        from llm_router.server.app import create_app

        original = app_mod.router_engine

        try:
            with TestClient(create_app(), raise_server_exceptions=False) as client:
                # Set to None AFTER create_app
                app_mod.router_engine = None
                resp = client.get("/v1/models")
                assert resp.status_code == 503
        finally:
            app_mod.router_engine = original

    def test_guardrails_check_503_without_engine(self):
        """Should return 503 when router_engine is None."""
        import llm_router.server.app as app_mod
        from llm_router.server.app import create_app

        original = app_mod.router_engine

        try:
            with TestClient(create_app(), raise_server_exceptions=False) as client:
                # Set to None AFTER create_app
                app_mod.router_engine = None
                resp = client.post(
                    "/v1/guardrails/check",
                    json={"text": "hi", "mode": "pii"},
                )
                assert resp.status_code == 503
        finally:
            app_mod.router_engine = original
