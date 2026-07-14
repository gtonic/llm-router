"""Tests for guardrails/pii_filter.py - PiiFilter."""

from __future__ import annotations

import pytest

from llm_router.guardrails.pii_filter import PiiFilter, PiiResult


class TestPiiFilterInit:
    def test_default_redact_true(self):
        f = PiiFilter()
        assert f.redact is True

    def test_redact_false(self):
        f = PiiFilter(redact=False)
        assert f.redact is False


class TestPiiFilterCheck:
    def setup_method(self):
        self.filter = PiiFilter()

    def test_no_pii(self):
        result = self.filter.check("Hello, how are you today?")
        assert result.has_pii is False
        assert result.patterns == []
        assert result.redacted_text is None

    def test_detects_email(self):
        result = self.filter.check("Contact me at user@example.com")
        assert result.has_pii is True
        assert "email" in result.patterns

    def test_detects_phone(self):
        result = self.filter.check("Call me at +49 123 456 7890")
        assert result.has_pii is True
        assert "phone" in result.patterns

    def test_detects_api_key(self):
        result = self.filter.check("Use key sk_abcdefghijklmnopqrst1234")
        assert result.has_pii is True
        assert "api_key" in result.patterns

    def test_detects_jwt(self):
        result = self.filter.check("Token: eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.abc123")
        assert result.has_pii is True
        assert "jwt" in result.patterns

    def test_detects_ssn(self):
        result = self.filter.check("SSN: 123-45-6789")
        assert result.has_pii is True
        assert "ssn" in result.patterns

    def test_detects_health_data(self):
        result = self.filter.check("I have diabetes and take medication")
        assert result.has_pii is True
        assert "health_data" in result.patterns

    def test_multiple_patterns(self):
        text = "Email: user@example.com, Phone: +49 123 456 7890"
        result = self.filter.check(text)
        assert result.has_pii is True
        assert "email" in result.patterns
        assert "phone" in result.patterns

    def test_redacts_pii(self):
        result = self.filter.check("My email is user@example.com")
        assert result.redacted_text is not None
        assert "user@example.com" not in result.redacted_text
        assert "[REDACTED]" in result.redacted_text

    def test_no_redact_when_disabled(self):
        f = PiiFilter(redact=False)
        result = f.check("Contact me at user@example.com")
        assert result.redacted_text is None


class TestPiiFilterRedactText:
    def test_redacts_text(self):
        f = PiiFilter()
        result = f.redact_text("Email: user@example.com")
        assert "[REDACTED]" in result
        assert "user@example.com" not in result

    def test_health_data_redacted(self):
        f = PiiFilter()
        result = f.redact_text("I have diabetes")
        assert "[HEALTH_REDACTED]" in result
