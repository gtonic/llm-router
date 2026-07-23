"""Tests for security helpers: SSRF URL validation, ReDoS pattern checks, and
the rate limiter's bounded (LRU) client state."""

from __future__ import annotations

import asyncio

import pytest

from llm_router.guardrails.rate_limiter import RateLimiter
from llm_router.security import (
    UnsafeBackendURLError,
    is_safe_regex_pattern,
    validate_backend_url,
)


class TestValidateBackendURL:
    def test_allows_public_host(self):
        assert validate_backend_url("https://api.openai.com/v1") == "https://api.openai.com/v1"

    def test_allows_loopback_literal(self):
        # Legitimate local backends live on loopback/private ranges.
        assert validate_backend_url("http://127.0.0.1:8080/v1")

    def test_allows_private_literal(self):
        assert validate_backend_url("http://192.168.1.10:8888/v1")
        assert validate_backend_url("http://10.0.0.5/v1")

    def test_rejects_link_local_metadata(self):
        # The classic cloud-metadata SSRF target.
        with pytest.raises(UnsafeBackendURLError):
            validate_backend_url("http://169.254.169.254/latest/meta-data/")

    def test_rejects_ipv6_link_local(self):
        with pytest.raises(UnsafeBackendURLError):
            validate_backend_url("http://[fe80::1]/v1")

    def test_rejects_non_http_scheme(self):
        with pytest.raises(UnsafeBackendURLError):
            validate_backend_url("file:///etc/passwd")
        with pytest.raises(UnsafeBackendURLError):
            validate_backend_url("gopher://169.254.169.254/")

    def test_rejects_missing_host(self):
        with pytest.raises(UnsafeBackendURLError):
            validate_backend_url("http:///v1")

    def test_unresolvable_host_is_allowed(self):
        # Don't block on transient/unknown DNS — the request just fails later.
        assert validate_backend_url("http://does-not-resolve.invalid/v1")


class TestIsSafeRegexPattern:
    def test_allows_normal_patterns(self):
        assert is_safe_regex_pattern(r"\b\d{3}-\d{2}-\d{4}\b")[0]
        assert is_safe_regex_pattern(r"(abc)+")[0]
        assert is_safe_regex_pattern(r"(\d{3})+")[0]  # bounded inner quantifier is fine

    def test_rejects_nested_unbounded_quantifier(self):
        assert not is_safe_regex_pattern(r"(a+)+$")[0]
        assert not is_safe_regex_pattern(r"(a*)*")[0]
        assert not is_safe_regex_pattern(r"(.*)+")[0]
        assert not is_safe_regex_pattern(r"(\d+)+")[0]

    def test_rejects_overlong_pattern(self):
        assert not is_safe_regex_pattern("a" * 500)[0]


class TestRateLimiterBounded:
    def test_evicts_lru_clients_over_cap(self):
        limiter = RateLimiter(rpm=1000, tpm=1_000_000, max_clients=10)
        for i in range(100):
            asyncio.run(limiter.check(client_id=f"client-{i}", tokens=1))
        # State is bounded regardless of how many distinct clients appeared.
        assert len(limiter._requests) <= 10
        assert len(limiter._tokens) <= 10

    def test_still_enforces_limit_for_active_client(self):
        limiter = RateLimiter(rpm=3, tpm=1_000_000, max_clients=10)
        results = [asyncio.run(limiter.check(client_id="steady")) for _ in range(5)]
        assert results[0].allowed and results[2].allowed
        assert not results[3].allowed  # 4th request in the window is blocked
