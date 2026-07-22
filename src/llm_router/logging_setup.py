"""JSONL audit logging for gateway requests.

Writes one JSON line per request to ``<log_dir>/audit_YYYYMMDD.jsonl``. File
rotation/retention is left to external tooling (logrotate, a sidecar, etc.)
rather than reimplemented here.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
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
    trace_id: str | None = None
    session_key: str | None = None
    affinity: str | None = None  # "hit" | "store" | None
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict:
        """Serialise to a plain dict for JSON encoding."""
        return {
            "timestamp": self.timestamp,
            "request_id": self.request_id,
            "trace_id": self.trace_id,
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
            "session_key": self.session_key,
            "affinity": self.affinity,
        }


class AuditLogger:
    """Appends one JSONL audit entry per request under ``log_dir``.

    Files older than ``retention_days`` are pruned opportunistically (checked at
    most once per hour on the write path) rather than by a separate scheduler.
    """

    def __init__(self, log_dir: str = "logs", retention_days: int = 30) -> None:
        self.log_dir = Path(log_dir)
        self.retention_days = retention_days
        self._cleanup_interval = 3600.0  # seconds
        self._last_cleanup: float = 0.0

    def log(self, entry: AuditEntry) -> None:
        """Write ``entry`` to today's audit file. Never raises on I/O failure."""
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._cleanup_if_due()
            date_str = datetime.now(UTC).strftime("%Y%m%d")
            path = self.log_dir / f"audit_{date_str}.jsonl"
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.warning("Failed to write audit log entry: %s", exc)

    def _cleanup_if_due(self) -> None:
        """Delete audit files older than ``retention_days``, at most once per hour."""
        now = time.monotonic()
        if now - self._last_cleanup < self._cleanup_interval:
            return
        self._last_cleanup = now
        cutoff = datetime.now(UTC) - timedelta(days=self.retention_days)
        for path in self.log_dir.glob("audit_*.jsonl"):
            date_str = path.stem.removeprefix("audit_")
            try:
                file_date = datetime.strptime(date_str, "%Y%m%d").replace(tzinfo=UTC)
            except ValueError:
                continue
            if file_date < cutoff:
                try:
                    path.unlink()
                except OSError as exc:
                    logger.warning("Failed to remove expired audit log %s: %s", path, exc)
