"""Tests for routing/complexity.py - ComplexityDetector."""

from __future__ import annotations

from llm_router.routing.complexity import COMPLEXITY_TO_MODEL, ComplexityDetector, ComplexityScore


class TestComplexityDetector:
    def setup_method(self):
        self.detector = ComplexityDetector()

    def test_simple_greeting(self):
        result = self.detector.analyze([{"role": "user", "content": "Hi"}])
        assert result.level == "low"
        assert result.score < 0.25
        assert "simple_greeting" in result.factors

    def test_code_request(self):
        content = "def hello(): print('world') "
        content += "please code something"
        result = self.detector.analyze([{"role": "user", "content": content}])
        assert result.level in ("low", "medium")
        assert "code_related" in result.factors

    def test_analysis_request(self):
        result = self.detector.analyze([{"role": "user", "content": "Please analyze and compare these two approaches"}])
        assert "analysis_related" in result.factors

    def test_creative_request(self):
        result = self.detector.analyze([{"role": "user", "content": "Write a creative story about dragons"}])
        assert "creative_related" in result.factors

    def test_long_prompt(self):
        long_text = "This is a test sentence. " * 200
        result = self.detector.analyze([{"role": "user", "content": long_text}])
        assert result.level in ("high", "critical")
        # With 1000 words (>500) → very_long_prompt factor
        assert "very_long_prompt" in result.factors or "long_prompt" in result.factors

    def test_very_long_prompt(self):
        long_text = "Word " * 600
        result = self.detector.analyze([{"role": "user", "content": long_text}])
        # 2400 words → score = 0.7 → level "high" (0.5 <= 0.7 < 0.75)
        assert result.level in ("high", "critical")
        assert "very_long_prompt" in result.factors

    def test_multi_line(self):
        lines = "\n".join([f"Line {i}" for i in range(10)])
        result = self.detector.analyze([{"role": "user", "content": lines}])
        assert "multi_line" in result.factors

    def test_tool_calls_increase_complexity(self):
        tool_call = {
            "id": "call_1",
            "function": {"name": "test", "arguments": "{}"},
        }
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [tool_call],
            },
            {"role": "user", "content": "Next step"},
        ]
        result = self.detector.analyze(messages)
        assert "has_tool_calls" in result.factors

    def test_score_clamped_0_to_1(self):
        for text in ["Hi", "x" * 1000]:
            result = self.detector.analyze([{"role": "user", "content": text}])
            assert 0.0 <= result.score <= 1.0

    def test_score_rounded_to_2_decimals(self):
        result = self.detector.analyze([{"role": "user", "content": "Hi"}])
        assert result.score == round(result.score, 2)


class TestComplexityToModel:
    def test_all_levels_mapped(self):
        for level in ("low", "medium", "high", "critical"):
            assert level in COMPLEXITY_TO_MODEL or COMPLEXITY_TO_MODEL.get(level) is not None

    def test_all_values_are_strings(self):
        for level, model_id in COMPLEXITY_TO_MODEL.items():
            assert isinstance(level, str)
            assert isinstance(model_id, str)

    async def test_routes_local_for_low_complexity(self):
        result = await ComplexityDetector().route([{"role": "user", "content": "Hi"}])
        assert result.model_id == "llama-local"

    async def test_routes_remote_for_high_complexity(self):
        result = await ComplexityDetector().route([{"role": "user", "content": "word " * 600}])
        assert result.model_id == "gpt-5.6-luna"

    async def test_unknown_level_falls_back_to_local_model(self, monkeypatch):
        detector = ComplexityDetector()
        monkeypatch.setattr(
            detector,
            "analyze",
            lambda messages: ComplexityScore(level="unknown", score=0.0, factors=[]),
        )
        result = await detector.route([{"role": "user", "content": "Hi"}])
        assert result.model_id == "llama-local"
