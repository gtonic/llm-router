"""Policy-based routing: matches requests against YAML-defined rules."""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llm_router.pool.base import ModelBackend

from llm_router.config import PolicyRule
from llm_router.routing.base import PolicyBase, RoutingResult

# PII detection patterns (simplified — full implementation in guardrails/)
PII_PATTERNS = [
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # email
    r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",  # phone
    r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b",  # credit card
    r"\b[A-Z]{2}\d{2}\s*\d{4}\s*\d{4}\s*\d{4}\s*\d{1,18}\b",  # IBAN
    r"sk-[a-zA-Z0-9]{20,}",  # API key
    r"eyJ[a-zA-Z0-9_-]*\.eyJ[a-zA-Z0-9_-]*\.[a-zA-Z0-9_-]*",  # JWT
]


def _text_has_pii(text: str) -> bool:
    """Check if text contains PII patterns."""
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in PII_PATTERNS)


def _classify_task_type(text: str) -> str:
    """Classify the task type from prompt content."""
    text_lower = text.lower()
    code_kw = {"def ", "function", "class ", "import ", "const ", "let ", "var "}
    analysis_kw = {"analyze", "compare", "evaluate", "summarize", "explain", "why", "how"}
    creative_kw = {"write", "create", "design", "compose", "story", "poem"}

    if any(kw in text_lower for kw in code_kw):
        return "code"
    if any(kw in text_lower for kw in analysis_kw):
        return "analysis"
    if any(kw in text_lower for kw in creative_kw):
        return "creative"
    return "general"


class PolicyMatcher(PolicyBase):
    """Matches requests against YAML policy rules."""

    def __init__(self, policies_dir: str = "agent-policies", default_policy: str = "default") -> None:
        self.policies_dir = policies_dir
        self.default_policy = default_policy
        self.rules: list[PolicyRule] = []
        self._load_policies()

    def _load_policies(self) -> None:
        """Load all YAML policy files from policies_dir."""
        if not os.path.isdir(self.policies_dir):
            return
        for filename in os.listdir(self.policies_dir):
            if not (filename.endswith(".yaml") or filename.endswith(".yml")):
                continue
            filepath = os.path.join(self.policies_dir, filename)
            self._load_single_policy(filepath)

    def _load_single_policy(self, filepath: str) -> None:
        import yaml

        with open(filepath) as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict) or "rules" not in data:
            return
        for rule_data in data["rules"]:
            rule = PolicyRule(**rule_data)
            if rule.enabled:
                self.rules.append(rule)
        # Sort by priority (highest first)
        self.rules.sort(key=lambda r: r.priority, reverse=True)

    def _evaluate_conditions(self, conditions: dict, messages: list[dict]) -> bool:
        """Evaluate a single rule's conditions against messages."""
        full_text = " ".join(msg.get("content", "") for msg in messages)
        full_text.lower()

        for key, value in conditions.items():
            if key == "contains_pii":
                if value and not _text_has_pii(full_text):
                    return False
            elif key == "task_type":
                task_type = _classify_task_type(full_text)
                if value != task_type:
                    return False
            elif key == "privacy":
                # Placeholder — full implementation in guardrails/
                pass
            elif key == "all":
                if not all(self._evaluate_conditions(sub, messages) for sub in value):
                    return False
            elif key == "any" and not any(self._evaluate_conditions(sub, messages) for sub in value):
                return False
        return True

    async def route(
        self,
        messages: list[dict],
        user_id: str | None = None,
        api_key: str | None = None,
        available_models: list[ModelBackend] | None = None,
    ) -> RoutingResult:
        """Find the highest-priority matching rule for the request."""
        for rule in self.rules:
            if self._evaluate_conditions(rule.conditions, messages):
                return RoutingResult(
                    model_id=rule.target_model,
                    strategy="policy",
                    policy_matched=rule.id,
                    metadata={"rule_name": rule.name},
                )
        return RoutingResult(
            model_id="default",
            strategy="policy",
            policy_matched=None,
            metadata={"reason": "no_match"},
        )
