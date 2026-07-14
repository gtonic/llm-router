"""Content safety filtering for LLM outputs.

Filters dangerous/unwanted content in model outputs:
- Violence, hate speech, illegal content
- Self-harm, terrorism, deepfakes
- Weapon code, drug manufacturing
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class SafetyLevel(StrEnum):
    """Severity levels for content safety violations."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def priority(self) -> int:
        """Numeric priority for ordering (LOW=0 … CRITICAL=3)."""
        return {"low": 0, "medium": 1, "high": 2, "critical": 3}.get(self.value, -1)

    def __le__(self, other):
        if isinstance(other, SafetyLevel):
            return self.priority <= other.priority
        return NotImplemented

    def __lt__(self, other):
        if isinstance(other, SafetyLevel):
            return self.priority < other.priority
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, SafetyLevel):
            return self.priority >= other.priority
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, SafetyLevel):
            return self.priority > other.priority
        return NotImplemented


@dataclass
class SafetyCheckResult:
    safe: bool
    level: SafetyLevel
    categories: list[str]
    action: str  # "log", "warn", "mask", "block"


class ContentSafety:
    """Filters dangerous content in LLM outputs."""

    DANGEROUS_KEYWORDS = {
        SafetyLevel.CRITICAL: [
            "child pornography",
            "pedophilia",
            "CSAM",
            "terrorism",
            "jihadist",
            "suicide method",
            "making a bomb",
            "making a nuclear weapon",
            "chemical weapon",
        ],
        SafetyLevel.HIGH: [
            "hate speech",
            "genocide",
            "holocaust denial",
            "racist",
            "antisemitic",
            "white supremacy",
            "child sexual abuse",
            "bestiality",
            "ransomware",
            "exploit",
            "malware",
        ],
        SafetyLevel.MEDIUM: [
            "drug manufacturing",
            "drug production",
            "meth",
            "cocaine",
            "weapon building",
            "weapon construction",
            "gun modification",
            "self-harm",
            "self harm",
            "cutting",
            "anorexia",
            "bulimia",
        ],
    }

    def __init__(self, block_threshold: SafetyLevel = SafetyLevel.MEDIUM) -> None:
        self.block_threshold = block_threshold
        # Compile keyword sets for fast lookup
        self._keywords: dict[str, set[str]] = {}
        for level, words in self.DANGEROUS_KEYWORDS.items():
            self._keywords[level] = {w.lower() for w in words}

    def check(self, text: str) -> SafetyCheckResult:
        """Check text for dangerous content.

        Args:
            text: Text to check.

        Returns:
            SafetyCheckResult with classification.
        """
        text_lower = text.lower()
        max_level = SafetyLevel.LOW
        found_categories: list[str] = []

        for level in [SafetyLevel.CRITICAL, SafetyLevel.HIGH, SafetyLevel.MEDIUM]:
            for keyword in self._keywords[level]:
                if keyword in text_lower and level != SafetyLevel.LOW:
                    max_level = level
                    found_categories.append(keyword)

        # Determine action based on severity
        action_map = {
            SafetyLevel.CRITICAL: "block",
            SafetyLevel.HIGH: "block",
            SafetyLevel.MEDIUM: "mask",
            SafetyLevel.LOW: "log",
        }
        action = action_map[max_level]

        return SafetyCheckResult(
            safe=max_level <= self.block_threshold,
            level=max_level,
            categories=found_categories,
            action=action,
        )
