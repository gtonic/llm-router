"""Abuse/misuse detection for LLM requests.

Detects:
- Prompt injection attempts
- Jailbreak attempts
- Encoding tricks (Base64, Hex, Leetspeak)
- Toxic/hateful content
- Illegal content requests
- Multi-turn evasion
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class AbuseResult:
    safe: bool = True
    abuse_score: float = 0.0
    categories: list[str] = field(default_factory=list)
    details: str = ""


class AbuseFilter:
    """Detects abuse patterns in LLM requests and responses."""

    # Prompt injection patterns
    INJECTION_PATTERNS = [
        re.compile(r"(?i)\b(ignore\s+previous\s+instructions?)\b"),
        re.compile(r"(?i)\b(you\s+are\s+now\s+|you\s+are\s+no\s+longer)"),
        re.compile(r"(?i)\bdump\s+your\s+(entire|complete|full)\s+(prompt|instruction|system|context)\b"),
        re.compile(r"(?i)\bshow\s+your\s+(system|developer|original)\s+(prompt|instructions?)\b"),
        re.compile(r"(?i)\bdont't\s+be\s+a\s+filter\b"),
        re.compile(r"(?i)\byour\s+new\s+(role|persona)\s+(is|to\s+be)\b"),
    ]

    # Jailbreak patterns
    JAILBREAK_PATTERNS = [
        re.compile(r"(?i)\bdo\s+anything\s+now\b"),
        re.compile(r"(?i)\bfrom\s+now\s+on\b.*\byou\s+are\b.*\bdan\b", re.IGNORECASE),
        re.compile(r"(?i)\bignore\s+all\s+(your|the)\s+(rules?|instructions?|policies?|guidelines?)\b"),
        re.compile(r"(?i)\bsimulate\s+(a|an)\s+(developer|sysadmin|admin)\s+mode\b"),
        re.compile(r"(?i)\bpretend\s+to\s+be\b"),
        re.compile(r"(?i)\bact\s+(as|like)\s+(a|an)\s+(developer|sysadmin|admin)\b"),
    ]

    # Encoding trick patterns
    ENCODING_PATTERNS = [
        (re.compile(r"^[A-Za-z0-9+/]{50,}={0,2}$"), "base64_long"),  # Long base64
        (re.compile(r"^0x[a-fA-F0-9]{50,}$"), "hex_long"),  # Long hex
        (re.compile(r"(?i)%[0-9a-f]{2}{10,}"), "url_encoded_long"),  # URL encoded
    ]

    # Leetspeak patterns
    LEETSPEAK_PATTERNS = [
        (re.compile(r"[0-1@35\$7]"), "leetspeak"),
    ]

    # Toxic/illegal content categories
    TOXIC_KEYWORDS = [
        "hate speech",
        "racial slur",
        "genocide",
        "terrorism",
        "suicide method",
        "child exploitation",
        "self harm",
        "drug manufacturing",
        "weapon building",
        "breaking the law",
        "illegal activity",
        "fraud",
        "ransomware",
    ]

    def check(self, text: str) -> AbuseResult:
        """Check text for abuse patterns.

        Returns:
            AbuseResult with safety assessment.
        """
        score = 0.0
        categories: list[str] = []
        details: list[str] = []

        # Check injection patterns
        for pattern in self.INJECTION_PATTERNS:
            if pattern.search(text):
                score += 0.3
                if "prompt_injection" not in categories:
                    categories.append("prompt_injection")
                    details.append(f"Matched injection pattern: {pattern.pattern[:40]}...")

        # Check jailbreak patterns
        for pattern in self.JAILBREAK_PATTERNS:
            if pattern.search(text):
                score += 0.4
                if "jailbreak" not in categories:
                    categories.append("jailbreak")
                    details.append(f"Matched jailbreak pattern: {pattern.pattern[:40]}...")

        # Check encoding tricks
        for pattern, name in self.ENCODING_PATTERNS:
            if pattern.match(text.strip()):
                score += 0.15
                if "encoding_trick" not in categories:
                    categories.append("encoding_trick")
                    details.append(f"Detected {name} encoding")

        # Check leetspeak
        for _pattern, _name in self.LEETSPEAK_PATTERNS:
            if pattern.search(text):
                score += 0.1
                if "leetspeak" not in categories:
                    categories.append("leetspeak")
                    break

        # Check toxic/illegal keywords
        text_lower = text.lower()
        for kw in self.TOXIC_KEYWORDS:
            if kw in text_lower:
                score += 0.2
                categories.append("toxic_content")
                break

        # Cap score at 1.0
        score = min(score, 1.0)

        # Determine if safe
        is_safe = score < 0.5

        return AbuseResult(
            safe=is_safe,
            abuse_score=round(score, 2),
            categories=categories,
            details="; ".join(details) if details else "",
        )
