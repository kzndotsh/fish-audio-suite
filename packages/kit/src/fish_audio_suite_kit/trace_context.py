"""W3C Trace Context header parsing. No OpenTelemetry."""

from __future__ import annotations

import re
import secrets
from collections.abc import Mapping

_TRACE_ID_BYTES = 16
_SPAN_ID_BYTES = 8
_TRACE_HEX = _TRACE_ID_BYTES * 2
_SPAN_HEX = _SPAN_ID_BYTES * 2
_TRACEPARENT_RE = re.compile(
    rf"^([0-9a-f]{{2}})-([0-9a-f]{{{_TRACE_HEX}}})-([0-9a-f]{{{_SPAN_HEX}}})-([0-9a-f]{{2}})$",
    re.IGNORECASE,
)
_HEX32_RE = re.compile(rf"^[0-9a-f]{{{_TRACE_HEX}}}$")
_ZERO_TRACE = "0" * _TRACE_HEX
_ZERO_SPAN = "0" * _SPAN_HEX
_TRACESTATE_MAX = 512
_TRACE_VERSION = "00"
_SAMPLED_FLAGS = "01"


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
    """Return the 32-hex trace id from a valid ``traceparent``.

    Parameters
    ----------
    traceparent : str
        A W3C ``traceparent`` header value.

    Returns
    -------
    str or None
        The trace id, or None when the header is invalid or all-zero.
    """
    parsed = canonical_traceparent(traceparent)
    if parsed is None:
        return None
    return parsed.split("-")[1]


def make_traceparent(*, trace_id: str | None = None) -> str:
    """Build a sampled traceparent. Reuse `trace_id` for sibling Fish calls in one workflow."""
    tid = (trace_id or "").strip().lower()
    if tid == _ZERO_TRACE or _HEX32_RE.fullmatch(tid) is None:
        tid = secrets.token_hex(_TRACE_ID_BYTES)
    span = secrets.token_hex(_SPAN_ID_BYTES)
    while span == _ZERO_SPAN:
        span = secrets.token_hex(_SPAN_ID_BYTES)
    return f"{_TRACE_VERSION}-{tid}-{span}-{_SAMPLED_FLAGS}"


def ensure_trace_headers(incoming: Mapping[str, str]) -> dict[str, str]:
    """Forward a valid W3C header, or mint a sampled `traceparent`."""
    found = w3c_trace_headers(incoming)
    if found:
        return found
    return {"traceparent": make_traceparent()}


def w3c_trace_headers(incoming: Mapping[str, str]) -> dict[str, str]:
    """Copy a valid `traceparent` and optional `tracestate`. Empty if none."""
    parent = canonical_traceparent(_header(incoming, "traceparent"))
    if parent is None:
        return {}
    out = {"traceparent": parent}
    state = _header(incoming, "tracestate")
    if _tracestate_ok(state):
        out["tracestate"] = state
    return out


def _tracestate_ok(state: str) -> bool:
    if not state or len(state) > _TRACESTATE_MAX:
        return False
    return "\n" not in state and "\r" not in state


def _header(incoming: Mapping[str, str], name: str) -> str:
    want = name.lower()
    for key, value in incoming.items():
        if str(key).lower() == want:
            return str(value).strip()
    return ""
