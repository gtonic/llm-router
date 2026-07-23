"""Security helpers: backend URL (SSRF) validation and regex safety checks.

These guard the runtime admin surface — backends and PII patterns can be added
at runtime via the admin API, so their inputs are validated here before they can
be persisted and acted upon (a fetch to the backend URL / a regex run on every
request).
"""

from __future__ import annotations

import ipaddress
import re
import socket
from urllib.parse import urlparse


class UnsafeBackendURLError(ValueError):
    """Raised when a backend base_url targets a disallowed (SSRF) address."""


# Ranges an SSRF attacker would target. Private and loopback ranges are
# intentionally ALLOWED — legitimate local backends (host.docker.internal,
# localhost, 10.x / 172.16.x / 192.168.x) live there, and blocking them would
# break the primary use case. Link-local covers the cloud metadata endpoint
# (169.254.169.254 / fd00:ec2::) which is the highest-value SSRF target.
_BLOCKED_NETWORKS = (
    ipaddress.ip_network("169.254.0.0/16"),  # IPv4 link-local (incl. 169.254.169.254 IMDS)
    ipaddress.ip_network("fe80::/10"),  # IPv6 link-local
    ipaddress.ip_network("fd00:ec2::/32"),  # AWS IMDSv2 IPv6
)


def _is_blocked_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Return True when ``ip`` is a link-local/metadata/multicast/reserved target."""
    if ip.is_multicast or ip.is_reserved or ip.is_unspecified:
        return True
    for net in _BLOCKED_NETWORKS:
        if ip.version == net.version and ip in net:
            return True
    # Normalise IPv4-mapped IPv6 (::ffff:a.b.c.d) to IPv4 and re-check.
    mapped = getattr(ip, "ipv4_mapped", None)
    return mapped is not None and _is_blocked_ip(mapped)


def validate_backend_url(base_url: str) -> str:
    """Validate a backend base URL against SSRF-prone targets; return it on success.

    Rejects non-http(s) schemes and hosts that resolve to link-local / cloud
    metadata / multicast / reserved addresses. Loopback and private ranges are
    allowed so legitimate local backends keep working.

    Note: this checks the address resolved *now* — a DNS-rebinding host could
    still resolve elsewhere at request time (documented residual risk).
    """
    if not base_url or not isinstance(base_url, str):
        raise UnsafeBackendURLError("base_url must be a non-empty string")

    parsed = urlparse(base_url)
    if parsed.scheme not in ("http", "https"):
        raise UnsafeBackendURLError(f"base_url scheme must be http or https, got '{parsed.scheme}'")
    host = parsed.hostname
    if not host:
        raise UnsafeBackendURLError("base_url must include a host")

    # IP literal → check directly without DNS.
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None
    if literal is not None:
        if _is_blocked_ip(literal):
            raise UnsafeBackendURLError(f"base_url host {host} is a disallowed (link-local/metadata) address")
        return base_url

    # Hostname → resolve and check every returned address.
    try:
        infos = socket.getaddrinfo(host, parsed.port or None, proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        # Cannot resolve at validation time; don't block on transient DNS — the
        # request would simply fail later. The link-local block above still
        # applies to any literal target.
        return base_url
    for info in infos:
        addr = info[4][0]
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError:
            continue
        if _is_blocked_ip(ip):
            raise UnsafeBackendURLError(f"base_url host '{host}' resolves to disallowed address {addr}")
    return base_url


# ── Regex (ReDoS) safety ───────────────────────────────────────────────
MAX_PATTERN_LENGTH = 200

# A group whose body contains an unbounded quantifier and is itself quantified —
# the classic catastrophic-backtracking shape: (a+)+, (a*)*, (.*)+, (\d+)+.
_QUANTIFIED_GROUP = re.compile(r"\(([^()]*)\)[*+]")


def is_safe_regex_pattern(pattern: str) -> tuple[bool, str]:
    """Heuristically reject catastrophic-backtracking (ReDoS) regex patterns.

    Returns ``(ok, reason)``. This is a heuristic, not a proof — it catches the
    common nested-unbounded-quantifier shapes and over-long patterns. Patterns
    are admin-supplied and run on every request, so this is defense-in-depth
    against a careless or compromised admin token.
    """
    if not isinstance(pattern, str):
        return False, "pattern must be a string"
    if len(pattern) > MAX_PATTERN_LENGTH:
        return False, f"pattern too long (>{MAX_PATTERN_LENGTH} chars)"
    for match in _QUANTIFIED_GROUP.finditer(pattern):
        body = match.group(1)
        if "*" in body or "+" in body:
            return False, "nested unbounded quantifier (catastrophic-backtracking risk)"
    return True, ""
