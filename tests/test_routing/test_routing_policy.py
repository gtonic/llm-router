"""Tests for routing/policy.py - PolicyMatcher."""

from __future__ import annotations

import tempfile
from pathlib import Path

from llm_router.routing.policy import PolicyMatcher, _classify_task_type, _text_has_pii


class TestPIIPatterns:
    def test_detects_email(self):
        assert _text_has_pii("contact user@example.com please") is True

    def test_detects_phone(self):
        assert _text_has_pii("call me at 123-456-7890") is True

    def test_detects_api_key(self):
        assert _text_has_pii("key is sk-abcdefghijklmnopqrst1234") is True

    def test_detects_jwt(self):
        assert _text_has_pii("token eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxIn0.abc") is True

    def test_no_pii(self):
        assert _text_has_pii("just a regular message with no sensitive data") is False


class TestTaskClassification:
    def test_code_classification(self):
        assert _classify_task_type("def hello(): print('world')") == "code"

    def test_german_code_classification(self):
        assert _classify_task_type("Schreibe eine Python-Funktion zum Addieren") == "code"

    def test_analysis_classification(self):
        assert _classify_task_type("analyze and compare these results") == "analysis"

    def test_creative_classification(self):
        assert _classify_task_type("write a creative story") == "creative"

    def test_general_classification(self):
        assert _classify_task_type("what is the weather today") == "general"


class TestPolicyMatcher:
    def setup_method(self):
        """Create a temporary directory with a test policy file."""
        self.tmpdir = Path(tempfile.mkdtemp())
        policy_file = self.tmpdir / "default.yaml"
        policy_file.write_text(
            """
rules:
  - id: pii-rule
    name: "Block PII"
    conditions:
      contains_pii: true
    target_model: "gpt-4o"
    priority: 10
    enabled: true
  - id: general-rule
    name: "General"
    conditions:
      task_type: general
    target_model: "llama-3.1-8b"
    priority: 5
    enabled: true
""",
            encoding="utf-8",
        )

    def teardown_method(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_loads_policies(self):
        matcher = PolicyMatcher(policies_dir=str(self.tmpdir))
        assert len(matcher.rules) == 2
        assert matcher.rules[0].id == "pii-rule"  # Higher priority first
        assert matcher.rules[1].id == "general-rule"

    def test_route_matches_pii_rule(self):
        matcher = PolicyMatcher(policies_dir=str(self.tmpdir))
        import asyncio

        result = asyncio.run(matcher.route([{"role": "user", "content": "My email is test@example.com"}]))
        assert result.model_id == "gpt-4o"
        assert result.policy_matched == "pii-rule"
        assert result.strategy == "policy"

    def test_route_matches_general_rule(self):
        matcher = PolicyMatcher(policies_dir=str(self.tmpdir))
        import asyncio

        result = asyncio.run(matcher.route([{"role": "user", "content": "What is 2+2?"}]))
        assert result.model_id == "llama-3.1-8b"
        assert result.strategy == "policy"

        def test_route_matches_complexity_rule(self):
                policy_file = self.tmpdir / "default.yaml"
                policy_file.write_text(
                        """
rules:
    - id: complex-rule
        name: "Complex"
        conditions:
            complexity: high
        target_model: "remote-model"
        priority: 10
        enabled: true
""",
                        encoding="utf-8",
                )
                matcher = PolicyMatcher(policies_dir=str(self.tmpdir))
                import asyncio

                long_prompt = "word " * 501
                result = asyncio.run(matcher.route([{"role": "user", "content": long_prompt}]))
                assert result.model_id == "remote-model"
                assert result.policy_matched == "complex-rule"

    def test_route_no_match_returns_default(self):
        self.tmpdir2 = Path(tempfile.mkdtemp())
        try:
            policy_file = self.tmpdir2 / "empty.yaml"
            policy_file.write_text("rules: []", encoding="utf-8")
            matcher = PolicyMatcher(policies_dir=str(self.tmpdir2))
            import asyncio

            result = asyncio.run(matcher.route([{"role": "user", "content": "Hello"}]))
            assert result.model_id == "default"
            assert result.metadata.get("reason") == "no_match"
        finally:
            import shutil

            shutil.rmtree(self.tmpdir2, ignore_errors=True)

    def test_disabled_rules_ignored(self):
        policy_file = self.tmpdir / "default.yaml"
        policy_file.write_text(
            """
rules:
  - id: disabled-rule
    name: "Disabled"
    conditions: {}
    target_model: "gpt"
    priority: 100
    enabled: false
  - id: active-rule
    name: "Active"
    conditions:
      task_type: general
    target_model: "llama"
    priority: 1
    enabled: true
""",
            encoding="utf-8",
        )
        matcher = PolicyMatcher(policies_dir=str(self.tmpdir))
        assert len(matcher.rules) == 1
        assert matcher.rules[0].id == "active-rule"

    def test_no_policies_dir(self):
        matcher = PolicyMatcher(policies_dir="/nonexistent/dir/xyz")
        assert len(matcher.rules) == 0

    def test_invalid_yaml_file_ignored(self):
        policy_file = self.tmpdir / "bad.yaml"
        policy_file.write_text("just some random text", encoding="utf-8")
        matcher = PolicyMatcher(policies_dir=str(self.tmpdir))
        # Should not crash, just have no rules from this file
        assert isinstance(matcher.rules, list)

    def test_policy_matched_in_metadata(self):
        matcher = PolicyMatcher(policies_dir=str(self.tmpdir))
        import asyncio

        result = asyncio.run(matcher.route([{"role": "user", "content": "test@example.com"}]))
        assert result.metadata.get("rule_name") == "Block PII"
