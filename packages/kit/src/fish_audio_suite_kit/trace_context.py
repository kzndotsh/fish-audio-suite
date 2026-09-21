"""W3C Trace Context header parsing. No OpenTelemetry."""

from __future__ import annotations

import re
import secrets
from collections.abc import Mapping

_TRACEPARENT_RE = re.compile(
    r"^([0-9a-f]{2})-([0-9a-f]{32})-([0-9a-f]{16})-([0-9a-f]{2})$",
    re.IGNORECASE,
)
_ZERO_TRACE = "0" * 32
_ZERO_SPAN = "0" * 16
_TRACESTATE_MAX = 512


def canonical_traceparent(value: str) -> str | None:
    """Return lowercase `00-…` or None if the header is invalid."""
    m = _TRACEPARENT_RE.fullmatch(value.strip())
    if m is None:
        return None
    ver, trace, span, flags = (g.lower() for g in m.groups())
    if trace == _ZERO_TRACE or span == _ZERO_SPAN:
        return None
    return f"{ver}-{trace}-{span}-{flags}"


def trace_id_of(traceparent: str) -> str | None:
    parsed = canonical_traceparent(traceparent)
    if parsed is None:
        return None
    return parsed.split("-")[1]


def make_traceparent(*, trace_id: str | None = None) -> str:
    """Sampled `traceparent`. Reuse `trace_id` for sibling Fish calls in one workflow."""
    tid = (trace_id or "").strip().lower()
    if len(tid) != 32 or any(c not in "0123456789abcdef" for c in tid) or tid == _ZERO_TRACE:
        tid = secrets.token_hex(16)
    span = secrets.token_hex(8)
    while span == _ZERO_SPAN:
        span = secrets.token_hex(8)
    return f"00-{tid}-{span}-01"


def w3c_trace_headers(incoming: Mapping[str, str]) -> dict[str, str]:
    """Copy a valid `traceparent` and optional `tracestate`. Empty if none."""
    parent = canonical_traceparent(_header(incoming, "traceparent"))
    if parent is None:
        return {}
    out = {"traceparent": parent}
    state = _header(incoming, "tracestate")
    if state and len(state) <= _TRACESTATE_MAX and "\n" not in state and "\r" not in state:
        out["tracestate"] = state
    return out


def _header(incoming: Mapping[str, str], name: str) -> str:
    want = name.lower()
    for key, value in incoming.items():
        if str(key).lower() == want:
            return str(value).strip()
    return ""
