"""Complexity-based routing strategy."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llm_router.pool.base import ModelBackend

from llm_router.routing.base import PolicyBase, RoutingResult

# Complexity thresholds and keyword mappings
SIMPLE_KEYWORDS = {"hello", "hi", "yes", "no", "ok", "thanks", "bye"}
CODE_KEYWORDS = {"def ", "function", "class ", "import ", "const ", "let ", "var ", "print(", "console.log"}
ANALYSIS_KEYWORDS = {"analyze", "compare", "evaluate", "summarize", "summarize", "explain", "why"}
CREATIVE_KEYWORDS = {"write", "create", "design", "compose", "story", "poem", "song"}

# Mapping complexity levels to model IDs (overridable via config)
COMPLEXITY_TO_MODEL = {
    "low": "smallest_local",
    "medium": "medium_local",
    "high": "large_local",
    "critical": "remote_top",
}


@dataclass
class ComplexityScore:
    level: str  # low, medium, high, critical
    score: float  # 0.0 - 1.0
    factors: list[str]


class ComplexityDetector:
    """Detects request complexity from message content."""

    def analyze(self, messages: list[dict]) -> ComplexityScore:
        """Analyze messages and return complexity score."""
        full_text = " ".join(msg.get("content", "") for msg in messages)
        words = full_text.split()
        word_count = len(words)
        has_newlines = full_text.count("\n")

        score = 0.0
        factors = []

        # Factor 1: Length
        if word_count < 20:
            score += 0.1
        elif word_count < 100:
            score += 0.3
        elif word_count < 500:
            score += 0.5
            factors.append("long_prompt")
        else:
            score += 0.7
            factors.append("very_long_prompt")

        if has_newlines > 5:
            score += 0.1
            factors.append("multi_line")

        # Factor 2: Keywords
        text_lower = full_text.lower()
        if any(kw in text_lower for kw in CODE_KEYWORDS):
            score += 0.2
            factors.append("code_related")
        if any(kw in text_lower for kw in ANALYSIS_KEYWORDS):
            score += 0.2
            factors.append("analysis_related")
        if any(kw in text_lower for kw in CREATIVE_KEYWORDS):
            score += 0.1
            factors.append("creative_related")
        if any(kw in text_lower for kw in SIMPLE_KEYWORDS):
            score -= 0.1
            factors.append("simple_greeting")

        # Factor 3: Tool usage
        if any("tool_calls" in msg for msg in messages):
            score += 0.2
            factors.append("has_tool_calls")

        # Normalize and determine level
        score = max(0.0, min(1.0, score))
        if score < 0.25:
            level = "low"
        elif score < 0.5:
            level = "medium"
        elif score < 0.75:
            level = "high"
        else:
            level = "critical"

        return ComplexityScore(level=level, score=round(score, 2), factors=factors)

    async def route(
        self,
        messages: list[dict],
        user_id: str | None = None,
        api_key: str | None = None,
        available_models: list[ModelBackend] | None = None,
    ) -> RoutingResult:
        complexity = self.analyze(messages)
        model_id = COMPLEXITY_TO_MODEL.get(complexity.level, "large_local")
        return RoutingResult(
            model_id=model_id,
            strategy="complexity",
            metadata={"complexity": complexity.level, "score": complexity.score, "factors": complexity.factors},
        )
