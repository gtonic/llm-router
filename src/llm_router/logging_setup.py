"""JSONL audit logging for gateway requests.

Writes one JSON line per request to ``<log_dir>/audit_YYYYMMDD.jsonl``. File
rotation/retention is left to external tooling (logrotate, a sidecar, etc.)
rather than reimplemented here.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger("llm-router")


@dataclass
class AuditEntry:
    """A single audit log record for one gateway request."""

    request_id: str
    user_id: str | None
    model_selected: str
    routing_strategy: str
    status: str
    policy_matched: str | None = None
    guardrail_pii_detected: int = 0
    guardrail_abuse_score: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0
    latency_ms: float = 0.0
    error: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict:
        """Serialise to a plain dict for JSON encoding."""
        return {
            "timestamp": self.timestamp,
            "request_id": self.request_id,
            "user_id": self.user_id,
            "model_selected": self.model_selected,
            "routing_strategy": self.routing_strategy,
            "policy_matched": self.policy_matched,
            "guardrail_pii_detected": self.guardrail_pii_detected,
            "guardrail_abuse_score": self.guardrail_abuse_score,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost": self.cost,
            "latency_ms": self.latency_ms,
            "error": self.error,
            "status": self.status,
        }


class AuditLogger:
    """Appends one JSONL audit entry per request under ``log_dir``."""

    def __init__(self, log_dir: str = "logs") -> None:
        self.log_dir = Path(log_dir)

    def log(self, entry: AuditEntry) -> None:
        """Write ``entry`` to today's audit file. Never raises on I/O failure."""
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            date_str = datetime.now(UTC).strftime("%Y%m%d")
            path = self.log_dir / f"audit_{date_str}.jsonl"
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.warning("Failed to write audit log entry: %s", exc)
