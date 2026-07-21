"""Tests for pool/local.py - LlamaCPPBackend."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_router.config import ModelBackendConfig
from llm_router.pool.base import GenerateResult
from llm_router.pool.local import LlamaCPPBackend


class AsyncIteratorMock:
    def __init__(self, items):
        self.items = items
        self.idx = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.idx >= len(self.items):
            raise StopAsyncIteration
        item = self.items[self.idx]
        self.idx += 1
        return item


class TestLlamaCPPBackendInit:
    def test_initialization(self):
        cfg = ModelBackendConfig(
            id="lc",
            name="Llama",
            type="local",
            base_url="http://localhost:8080/v1",
        )
        backend = LlamaCPPBackend(cfg)
        assert backend.config.id == "lc"
        assert backend._client is None
        assert backend._http_client is None


class TestLlamaCPPBackendGenerate:
    @pytest.mark.asyncio
    async def test_generate(self):
        cfg = ModelBackendConfig(
            id="llama",
            name="Llama",
            type="local",
            base_url="http://localhost:8080/v1",
            model_name="llama-3.1-8b",
        )
        backend = LlamaCPPBackend(cfg)

        mock_response = MagicMock()
        mock_response.content = "Hello from Llama!"
        usage_meta = {
            "token_usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            }
        }
        mock_response.response_metadata = usage_meta

        backend._ensure_client = MagicMock()
        backend._client = MagicMock()
        backend._client.ainvoke = AsyncMock(return_value=mock_response)

        messages = [{"role": "user", "content": "Hi"}]
        result = await backend.generate(messages)

        assert isinstance(result, GenerateResult)
        assert result.content == "Hello from Llama!"
        assert result.model == "llama-3.1-8b"
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 5
        assert result.finish_reason == "stop"
        assert result.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_generate_calls_ainvoke(self):
        """Verify generate passes messages through correctly."""
        cfg = ModelBackendConfig(
            id="llama",
            name="Llama",
            type="local",
            base_url="http://localhost:8080/v1",
            model_name="llama",
        )
        backend = LlamaCPPBackend(cfg)

        mock_response = MagicMock()
        mock_response.content = "ok"
        mock_response.response_metadata = {"token_usage": {}}
        backend._ensure_client = MagicMock()
        backend._client = MagicMock()
        backend._client.ainvoke = AsyncMock(return_value=mock_response)

        messages = [
            {"role": "system", "content": "Be nice"},
            {"role": "user", "content": "Hello"},
        ]
        await backend.generate(messages)

        backend._ensure_client.assert_called_once()
        backend._client.ainvoke.assert_called_once()
        call_args = backend._client.ainvoke.call_args
        lc_messages = call_args[0][0]
        assert len(lc_messages) == 2
        assert call_args.kwargs["extra_body"] == {
            "reasoning_format": "none",
            "chat_template_kwargs": {"enable_thinking": False},
        }

    @pytest.mark.asyncio
    async def test_uses_reasoning_content_only_when_opted_in(self, monkeypatch):
        monkeypatch.setenv("ROUTER_EXPOSE_REASONING", "true")
        cfg = ModelBackendConfig(
            id="llama",
            name="Llama",
            type="local",
            base_url="http://localhost:8080/v1",
            model_name="llama",
        )
        backend = LlamaCPPBackend(cfg)
        mock_response = MagicMock(content="", response_metadata={"reasoning_content": "Recovered answer"})
        backend._ensure_client = MagicMock()
        backend._client = MagicMock()
        backend._client.ainvoke = AsyncMock(return_value=mock_response)

        result = await backend.generate([{"role": "user", "content": "Hi"}])

        assert result.content == "Recovered answer"

    @pytest.mark.asyncio
    async def test_reasoning_content_not_leaked_by_default(self, monkeypatch):
        monkeypatch.delenv("ROUTER_EXPOSE_REASONING", raising=False)
        cfg = ModelBackendConfig(
            id="llama",
            name="Llama",
            type="local",
            base_url="http://localhost:8080/v1",
            model_name="llama",
        )
        backend = LlamaCPPBackend(cfg)
        mock_response = MagicMock(content="", response_metadata={"reasoning_content": "internal chain of thought"})
        backend._ensure_client = MagicMock()
        backend._client = MagicMock()
        backend._client.ainvoke = AsyncMock(return_value=mock_response)

        result = await backend.generate([{"role": "user", "content": "Hi"}])

        assert result.content == ""  # reasoning is not surfaced as the answer


class TestLlamaCPPBackendStream:
    @pytest.mark.asyncio
    async def test_generate_stream(self):
        cfg = ModelBackendConfig(
            id="llama",
            name="Llama",
            type="local",
            base_url="http://localhost:8080/v1",
            model_name="llama",
        )
        backend = LlamaCPPBackend(cfg)

        mock_chunk = MagicMock()
        mock_chunk.content = "chunk"

        backend._ensure_client = MagicMock()
        backend._client = MagicMock()
        backend._client.astream = lambda *a, **k: AsyncIteratorMock([mock_chunk])

        messages = [{"role": "user", "content": "Hi"}]
        chunks = [c async for c in backend.generate_stream(messages)]

        assert len(chunks) == 1
        assert chunks[0].content == "chunk"
        assert chunks[0].model == "llama"


class TestLlamaCPPBackendHealth:
    def _make_httpx_mock(self, status_code: int = 200):
        """Create a fake httpx module mock."""
        mock_client = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.status_code = status_code
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_resp)

        mock_httpx = MagicMock()
        mock_httpx.AsyncClient = lambda **kwargs: mock_client
        return mock_httpx

    @pytest.mark.asyncio
    async def test_health_check_healthy(self):
        cfg = ModelBackendConfig(
            id="llama",
            name="Llama",
            type="local",
            base_url="http://localhost:8080/v1",
        )
        backend = LlamaCPPBackend(cfg)

        mock_httpx = self._make_httpx_mock(200)
        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            result = await backend.health_check()
        assert result.healthy is True
        assert result.latency_ms >= 0

        mock_client = mock_httpx.AsyncClient()
        mock_client.get.assert_awaited_once_with("http://localhost:8080/v1/models")

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self):
        cfg = ModelBackendConfig(
            id="llama",
            name="Llama",
            type="local",
            base_url="http://localhost:8080/v1",
        )
        backend = LlamaCPPBackend(cfg)

        mock_httpx = self._make_httpx_mock(503)
        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            result = await backend.health_check()
        assert result.healthy is False
        assert "503" in result.error

    @pytest.mark.asyncio
    async def test_health_check_exception(self):
        cfg = ModelBackendConfig(
            id="llama",
            name="Llama",
            type="local",
            base_url="http://localhost:8080/v1",
        )
        backend = LlamaCPPBackend(cfg)

        mock_httpx = MagicMock()
        mock_httpx.AsyncClient.side_effect = Exception("connection refused")
        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            result = await backend.health_check()
        assert result.healthy is False
        assert "connection refused" in result.error
