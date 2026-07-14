"""Tests for pool/remote.py - RemoteBackend."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_router.config import ModelBackendConfig
from llm_router.pool.base import GenerateResult
from llm_router.pool.remote import RemoteBackend


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


class TestRemoteBackendInit:
    def test_initialization(self):
        cfg = ModelBackendConfig(
            id="gpt",
            name="GPT-4",
            type="remote",
            base_url="https://api.openai.com/v1",
            api_key="sk-test",
        )
        backend = RemoteBackend(cfg)
        assert backend.config.id == "gpt"
        assert backend._client is None


class TestRemoteBackendCost:
    def test_calculate_cost_zero(self):
        cfg = ModelBackendConfig(
            id="gpt",
            name="GPT-4",
            type="remote",
            base_url="https://api.openai.com/v1",
            api_key="sk",
            cost_per_1m_input=0.03,
            cost_per_1m_output=0.06,
        )
        backend = RemoteBackend(cfg)
        cost = backend._calculate_cost(0, 0)
        assert cost == 0.0

    def test_calculate_cost_nonzero(self):
        cfg = ModelBackendConfig(
            id="gpt",
            name="GPT-4",
            type="remote",
            base_url="https://api.openai.com/v1",
            api_key="sk",
            cost_per_1m_input=0.03,
            cost_per_1m_output=0.06,
        )
        backend = RemoteBackend(cfg)
        cost = backend._calculate_cost(1_000_000, 1_000_000)
        assert cost == 0.09

    def test_calculate_cost_fractional(self):
        cfg = ModelBackendConfig(
            id="gpt",
            name="GPT-4",
            type="remote",
            base_url="https://api.openai.com/v1",
            api_key="sk",
            cost_per_1m_input=0.03,
            cost_per_1m_output=0.06,
        )
        backend = RemoteBackend(cfg)
        cost = backend._calculate_cost(100, 200)
        assert cost > 0.0


class TestRemoteBackendGenerate:
    @pytest.mark.asyncio
    async def test_generate(self):
        cfg = ModelBackendConfig(
            id="gpt",
            name="GPT-4",
            type="remote",
            base_url="https://api.openai.com/v1",
            api_key="sk-test",
            model_name="gpt-4",
            cost_per_1m_input=0.03,
            cost_per_1m_output=0.06,
        )
        backend = RemoteBackend(cfg)

        mock_response = MagicMock()
        mock_response.content = "GPT answer"
        usage_meta = {
            "token_usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            }
        }
        mock_response.response_metadata = usage_meta

        backend._ensure_client = MagicMock()
        backend._client = MagicMock()
        backend._client.ainvoke = AsyncMock(return_value=mock_response)

        messages = [{"role": "user", "content": "Hello"}]
        result = await backend.generate(messages)

        assert isinstance(result, GenerateResult)
        assert result.content == "GPT answer"
        assert result.model == "gpt-4"
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 20
        assert result.usage.total_tokens == 30
        assert result.usage.cost > 0

    @pytest.mark.asyncio
    async def test_generate_missing_token_usage(self):
        """Should not crash if response_metadata lacks token_usage."""
        cfg = ModelBackendConfig(
            id="gpt",
            name="GPT-4",
            type="remote",
            base_url="https://api.openai.com/v1",
            api_key="sk-test",
            model_name="gpt-4",
        )
        backend = RemoteBackend(cfg)

        mock_response = MagicMock()
        mock_response.content = "fallback"
        mock_response.response_metadata = {}

        backend._ensure_client = MagicMock()
        backend._client = MagicMock()
        backend._client.ainvoke = AsyncMock(return_value=mock_response)

        messages = [{"role": "user", "content": "Hi"}]
        result = await backend.generate(messages)

        assert result.content == "fallback"
        assert result.usage.prompt_tokens == 0
        assert result.usage.completion_tokens == 0


class TestRemoteBackendStream:
    @pytest.mark.asyncio
    async def test_generate_stream(self):
        cfg = ModelBackendConfig(
            id="gpt",
            name="GPT-4",
            type="remote",
            base_url="https://api.openai.com/v1",
            api_key="sk-test",
            model_name="gpt-4",
        )
        backend = RemoteBackend(cfg)

        mock_chunk = MagicMock()
        mock_chunk.content = "streamed"

        backend._ensure_client = MagicMock()
        backend._client = MagicMock()
        backend._client.astream = lambda *a, **k: AsyncIteratorMock([mock_chunk])

        messages = [{"role": "user", "content": "Stream me"}]
        chunks = [c async for c in backend.generate_stream(messages)]

        assert len(chunks) == 1
        assert chunks[0].content == "streamed"


class TestRemoteBackendHealth:
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
            id="gpt",
            name="GPT",
            type="remote",
            base_url="https://api.openai.com/v1",
            api_key="sk",
        )
        backend = RemoteBackend(cfg)

        mock_httpx = self._make_httpx_mock(200)
        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            result = await backend.health_check()

        assert result.healthy is True

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self):
        cfg = ModelBackendConfig(
            id="gpt",
            name="GPT",
            type="remote",
            base_url="https://api.openai.com/v1",
            api_key="sk",
        )
        backend = RemoteBackend(cfg)

        mock_httpx = self._make_httpx_mock(500)
        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            result = await backend.health_check()

        assert result.healthy is False
