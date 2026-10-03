"""W3C Trace Context header parsing. No OpenTelemetry."""

from __future__ import annotations

import re
import secrets
from collections.abc import Mapping
from typing import Final

__all__ = [
    "canonical_traceparent",
    "ensure_trace_headers",
    "make_traceparent",
    "trace_id_of",
]

_TRACE_ID_BYTES: Final = 16
_SPAN_ID_BYTES: Final = 8
_TRACE_HEX: Final = _TRACE_ID_BYTES * 2
_SPAN_HEX: Final = _SPAN_ID_BYTES * 2
_TRACEPARENT_RE = re.compile(
    rf"^([0-9a-f]{{2}})-([0-9a-f]{{{_TRACE_HEX}}})-([0-9a-f]{{{_SPAN_HEX}}})-([0-9a-f]{{2}})$",
    re.IGNORECASE,
)
_HEX32_RE = re.compile(rf"^[0-9a-f]{{{_TRACE_HEX}}}$")
_ZERO_TRACE = "0" * _TRACE_HEX
_ZERO_SPAN = "0" * _SPAN_HEX
_TRACESTATE_MAX: Final = 512
_TRACE_VERSION: Final = "00"
_SAMPLED_FLAGS: Final = "01"


def canonical_traceparent(value: str) -> str | None:
    """Validate a W3C ``traceparent`` and put it in canonical form.

    Parameters
    ----------
    value : str
        A ``traceparent`` header value. Surrounding space and upper case are fine.

    Returns
    -------
    str or None
        The lowercase ``00-…`` value, or None when it is malformed, uses version
        ``ff``, or has an all-zero trace or span id.
    """
    m = _TRACEPARENT_RE.fullmatch(value.strip())
    if m is None:
        return None
    ver, trace, span, flags = (g.lower() for g in m.groups())
    # W3C forbids version ff. All-zero ids are not a real trace.
    if ver == "ff" or trace == _ZERO_TRACE or span == _ZERO_SPAN:
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
    """Build a sampled ``traceparent`` with a fresh span id.

    Parameters
    ----------
    trace_id : str or None, optional
        A 32-hex trace id to reuse, so sibling Fish calls in one workflow share a
        trace. A missing, malformed or all-zero value gets a new random id.

    Returns
    -------
    str
        ``00-<trace id>-<span id>-01``. The span id is random on every call.

    Examples
    --------
    >>> parent = make_traceparent(trace_id="0AF7651916CD43DD8448EB211C80319C")
    >>> parent.startswith("00-0af7651916cd43dd8448eb211c80319c-")
    True
    >>> parent.endswith("-01")
    True
    >>> len(make_traceparent())
    55
    """
    tid = (trace_id or "").strip().lower()
    if tid == _ZERO_TRACE or _HEX32_RE.fullmatch(tid) is None:
        tid = secrets.token_hex(_TRACE_ID_BYTES)
    span = secrets.token_hex(_SPAN_ID_BYTES)
    while span == _ZERO_SPAN:
        span = secrets.token_hex(_SPAN_ID_BYTES)
    return f"{_TRACE_VERSION}-{tid}-{span}-{_SAMPLED_FLAGS}"


def ensure_trace_headers(incoming: Mapping[str, str]) -> dict[str, str]:
    """Forward a valid W3C header, or mint a sampled ``traceparent``.

    Parameters
    ----------
    incoming : Mapping of str to str
        Request headers. Names are matched without regard to case.

    Returns
    -------
    dict of str to str
        A valid incoming ``traceparent`` (lowercased) plus a usable ``tracestate``,
        or a freshly minted sampled ``traceparent`` when none was valid.
    """
    found = _forwarded_headers(incoming)
    if found:
        return found
    return {"traceparent": make_traceparent()}


def _forwarded_headers(incoming: Mapping[str, str]) -> dict[str, str]:
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
    # Header values are ASCII. A control character splits the line, and a
    # surrogate makes the HTTP client refuse the request.
    return all(32 <= ord(ch) < 127 for ch in state)


def _header(incoming: Mapping[str, str], name: str) -> str:
    want = name.lower()
    for key, value in incoming.items():
        if str(key).lower() == want:
            return str(value).strip()
    return ""
