"""Tests for guardrails/rate_limiter.py - RateLimiter."""

from __future__ import annotations

import pytest

from llm_router.guardrails.rate_limiter import RateLimiter, RateLimitResult


class TestRateLimitResult:
    def test_defaults(self):
        r = RateLimitResult(allowed=True, remaining_requests=5)
        assert r.allowed is True
        assert r.remaining_requests == 5
        assert r.error == ""
        assert r.reset_at is None

    def test_blocked(self):
        r = RateLimitResult(allowed=False, remaining_requests=0, error="Rate limited")
        assert r.allowed is False
        assert r.error == "Rate limited"


class TestRateLimiter:
    @pytest.mark.asyncio
    async def test_allows_first_requests(self):
        limiter = RateLimiter(rpm=5, tpm=10000)
        result = await limiter.check("user1", tokens=100)
        assert result.allowed is True
        assert result.remaining_requests >= 0

    @pytest.mark.asyncio
    async def test_blocks_after_rpm_limit(self):
        limiter = RateLimiter(rpm=3, tpm=10000)
        for _i in range(3):
            result = await limiter.check("user2", tokens=10)
            assert result.allowed is True

        result = await limiter.check("user2", tokens=10)
        assert result.allowed is False
        assert "rate limit" in result.error.lower() or "exceeded" in result.error.lower()

    @pytest.mark.asyncio
    async def test_different_clients_independent(self):
        limiter = RateLimiter(rpm=2, tpm=10000)
        await limiter.check("client_a", tokens=10)
        await limiter.check("client_a", tokens=10)
        # client_a is now at limit
        result = await limiter.check("client_a", tokens=10)
        assert result.allowed is False
        # client_b should still be allowed
        result_b = await limiter.check("client_b", tokens=10)
        assert result_b.allowed is True

    @pytest.mark.asyncio
    async def test_tpm_limit(self):
        limiter = RateLimiter(rpm=1000, tpm=500)
        result = await limiter.check("user4", tokens=400)
        assert result.allowed is True
        result = await limiter.check("user4", tokens=200)
        assert result.allowed is False
        assert "too many tokens" in result.error.lower() or "rate limit" in result.error.lower()

    @pytest.mark.asyncio
    async def test_check_with_zero_tokens(self):
        limiter = RateLimiter(rpm=100, tpm=10000)
        result = await limiter.check("user5", tokens=0)
        assert result.allowed is True

    @pytest.mark.asyncio
    async def test_remaining_includes_count(self):
        limiter = RateLimiter(rpm=5, tpm=10000)
        for _i in range(3):
            result = await limiter.check("incr", tokens=10)
            assert result.allowed is True
            assert result.remaining_requests >= 0
