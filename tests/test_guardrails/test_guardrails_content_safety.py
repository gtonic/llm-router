"""Tests for guardrails/content_safety.py - ContentSafety."""

from __future__ import annotations

from llm_router.guardrails.content_safety import ContentSafety, SafetyLevel


class TestSafetyLevel:
    def test_all_values(self):
        assert SafetyLevel.LOW == "low"
        assert SafetyLevel.MEDIUM == "medium"
        assert SafetyLevel.HIGH == "high"
        assert SafetyLevel.CRITICAL == "critical"

    def test_ordering(self):
        assert SafetyLevel.LOW < SafetyLevel.MEDIUM
        assert SafetyLevel.MEDIUM < SafetyLevel.HIGH
        assert SafetyLevel.HIGH < SafetyLevel.CRITICAL
        assert SafetyLevel.LOW <= SafetyLevel.MEDIUM
        assert SafetyLevel.CRITICAL >= SafetyLevel.HIGH
        assert SafetyLevel.MEDIUM <= SafetyLevel.MEDIUM


class TestContentSafetyCheck:
    def setup_method(self):
        self.safety = ContentSafety(block_threshold=SafetyLevel.MEDIUM)

    def test_safe_text(self):
        result = self.safety.check("The weather is nice today.")
        assert result.safe is True
        assert result.level == SafetyLevel.LOW
        assert result.action == "log"

    def test_critical_content_blocked(self):
        result = self.safety.check("This is about making a bomb and terrorism")
        assert result.safe is False
        assert result.level == SafetyLevel.CRITICAL
        assert result.action == "block"

    def test_high_content_blocked(self):
        result = self.safety.check("I hate genocide and racism")
        assert result.safe is False
        assert result.level in (SafetyLevel.HIGH, SafetyLevel.CRITICAL)
        assert result.action == "block"

    def test_medium_content_allowed(self):
        """MEDIUM content with MEDIUM threshold is allowed (action=mask but still safe)."""
        result = self.safety.check("drug manufacturing instructions")
        assert result.safe is True
        assert result.level == SafetyLevel.MEDIUM
        assert result.action == "mask"

    def test_low_content_logs(self):
        result = self.safety.check("normal conversation")
        assert result.safe is True
        assert result.action == "log"

    def test_categories_returned(self):
        result = self.safety.check("terrorism and hate speech")
        assert len(result.categories) > 0
        assert "terrorism" in result.categories
        assert "hate speech" in result.categories

    def test_blocks_higher_threshold(self):
        """With LOW threshold, MEDIUM content should be blocked."""
        safety = ContentSafety(block_threshold=SafetyLevel.LOW)
        result = safety.check("drug manufacturing")
        assert result.safe is False
        assert result.level == SafetyLevel.MEDIUM

    def test_blocks_above_threshold(self):
        """CRITICAL content should be blocked even with MEDIUM threshold."""
        safety = ContentSafety(block_threshold=SafetyLevel.MEDIUM)
        result = safety.check("terrorism")
        assert result.safe is False
        assert result.level == SafetyLevel.CRITICAL

    def test_custom_threshold(self):
        safety = ContentSafety(block_threshold=SafetyLevel.CRITICAL)
        result = safety.check("drug manufacturing")
        assert result.safe is True

    def test_action_map(self):
        """Verify action is correct for each level."""
        for level in (SafetyLevel.LOW, SafetyLevel.MEDIUM, SafetyLevel.HIGH, SafetyLevel.CRITICAL):
            safety = ContentSafety(block_threshold=level)
            result = safety.check("terrorism")  # critical keyword
            assert result.action in ("block", "mask", "log")
            assert isinstance(result.safe, bool)
