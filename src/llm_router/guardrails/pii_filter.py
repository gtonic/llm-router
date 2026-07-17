"""PII detection and redaction for input/output filtering.

Supports detection and redaction of:
- Email addresses
- Phone numbers
- IBAN (DE/AT/CH)
- Credit card numbers
- API keys
- JWTs
- SSN (US)
- Post addresses (DE pattern)
- Birth dates
- Vehicle registration numbers
- Tax IDs (DE)
- Health data keywords
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class PiiResult:
    has_pii: bool
    patterns: list[str]
    redacted_text: str | None = None


class PiiFilter:
    """Detects and redacts PII in text."""

    # PII patterns: (name, compiled_regex, action)
    PATTERNS = [
        ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")),
        # At least one separator between digit groups is required — otherwise this
        # matches any bare 8+ digit run (order numbers, IDs, ...) as a "phone number".
        ("phone", re.compile(r"\b(?:\+49|0049|0)?(?:[1-9][0-9]{1,4})[-.\s][0-9]{3,}[-.\s]?[0-9]{3,}\b")),
        ("iban_de", re.compile(r"\b(DE\d{2}\s?\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4})\b")),
        ("credit_card", re.compile(r"\b(?:4[0-9]{3}|5[1-5][0-9]{2}|3[47][0-9]{2}|6(?:011|5[0-9]{2}))[0-9]{12}\b")),
        ("api_key", re.compile(r"\b(?:sk|ghp|gho|ghu|ghs|github_pat)_[a-zA-Z0-9]{20,}\b")),
        ("jwt", re.compile(r"\beyJ[a-zA-Z0-9_-]*\.eyJ[a-zA-Z0-9_-]*\.[a-zA-Z0-9_-]*\b")),
        ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
        # Requires the "DE" country prefix (real USt-IdNr. format: DE + 9 digits).
        # A bare 11-12 digit number alone isn't distinguishable from an order/invoice
        # number, so we don't match on digits without the prefix.
        ("vat_id", re.compile(r"\bDE\d{9}\b")),
        ("post_code_city", re.compile(r"\b\d{5}[\s-]+[A-ZÜA-Zäöß]{1}[A-Za-zäöß]{1,}\b")),
        (
            "vehicle_number",
            re.compile(r"\b([A-Z]{1,3}[\s-])?[A-ZÜÄÖ]{1,2}\s*[A-ZÜÄÖ]{1,2}[\s-]?\d{1,4}[\s-]?\d{1,4}\b"),
        ),
        (
            "health_data",
            re.compile(r"\b(?:Diabetes|Chemotherapie|HIV|AIDS|Krebs|Alkoholkrankheit|Psychopharmaka)\b", re.IGNORECASE),
        ),
        ("date_of_birth", re.compile(r"\b(?:(?:0[1-9]|[12]\d|3[01])\.(?:0[1-9]|1[02])\.\d{4})\b")),
    ]

    # Health-related keywords for broader detection (superset of the "health_data"
    # regex above, including German variants not covered there).
    HEALTH_KEYWORDS = [
        "diabetes",
        "chemotherapie",
        "hiv",
        "aids",
        "krebs",
        "alkoholkrankheit",
        "psychopharmaka",
        "krebserkrankung",
        "suchterkrankung",
        "depression",
        "bipolare störung",
        "schizophrenie",
        "autismus",
        "adhs",
    ]

    # Word-bounded, case-insensitive — a naive substring match would flag ordinary
    # words like "said"/"raid"/"archive" as health data (they contain "aid"/"hiv").
    _HEALTH_KEYWORD_PATTERN = re.compile(
        r"\b(?:" + "|".join(re.escape(kw) for kw in HEALTH_KEYWORDS) + r")\b",
        re.IGNORECASE,
    )

    def __init__(self, redact: bool = True, custom_patterns: list[str] | None = None) -> None:
        self._redact = redact
        self.PATTERNS = list(self.PATTERNS)
        self._custom_patterns: set[str] = set()
        for index, pattern in enumerate(custom_patterns or []):
            try:
                name = f"custom_{index}"
                self.PATTERNS.insert(0, (name, re.compile(pattern)))
                self._custom_patterns.add(name)
            except re.error:
                continue

    @property
    def redact(self) -> bool:
        """Whether PII should be redacted."""
        return self._redact

    def check(self, text: str, mode: str = "input") -> PiiResult:
        """Check text for PII patterns.

        Args:
            text: Text to check.
            mode: "input" or "output" (for logging).

        Returns:
            PiiResult with detection details.
        """
        found_patterns: list[str] = []
        result_text = text

        # Check each pattern
        for name, pattern in self.PATTERNS:
            matches = pattern.findall(text)
            if matches:
                found_patterns.append(name)
                if self._redact and name not in ("health_data",):
                    result_text = pattern.sub(" [REDACTED]", result_text)

        # Also check health keywords (word-bounded — see _HEALTH_KEYWORD_PATTERN)
        if self._HEALTH_KEYWORD_PATTERN.search(text):
            if "health_data" not in found_patterns:
                found_patterns.append("health_data")
            if self._redact:
                result_text = self._HEALTH_KEYWORD_PATTERN.sub("[HEALTH_REDACTED]", result_text)

        return PiiResult(
            has_pii=bool(found_patterns),
            patterns=found_patterns,
            redacted_text=result_text if self._redact and found_patterns else None,
        )

    def redact_text(self, text: str) -> str:
        """Redact PII from text.

        Args:
            text: Input text.

        Returns:
            Text with PII replaced by [REDACTED].
        """
        result = text
        for name, pattern in self.PATTERNS:
            if name in ("health_data",):
                continue
            result = pattern.sub(" [REDACTED]", result)
        return self._HEALTH_KEYWORD_PATTERN.sub("[HEALTH_REDACTED]", result)
