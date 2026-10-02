from __future__ import annotations

from fish_audio_suite_kit import (
    ensure_trace_headers,
    make_traceparent,
    trace_id_of,
)
from fish_audio_suite_kit.trace_context import canonical_traceparent, w3c_trace_headers

_SAMPLE_PARENT = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"


def test_canonical_traceparent() -> None:
    assert canonical_traceparent(_SAMPLE_PARENT) == _SAMPLE_PARENT
    assert canonical_traceparent(_SAMPLE_PARENT.upper()) == _SAMPLE_PARENT
    assert canonical_traceparent("not-a-trace") is None
    assert canonical_traceparent("00-" + "0" * 32 + "-" + "0" * 16 + "-01") is None
    forbidden = "ff-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
    assert canonical_traceparent(forbidden) is None
    assert w3c_trace_headers({"traceparent": forbidden}) == {}


def test_w3c_trace_headers_and_mint() -> None:
    headers = w3c_trace_headers({"Traceparent": _SAMPLE_PARENT, "tracestate": "congo=t61rcWkgMzE"})
    assert headers["traceparent"] == _SAMPLE_PARENT
    assert headers["tracestate"] == "congo=t61rcWkgMzE"
    injected = w3c_trace_headers(
        {"traceparent": _SAMPLE_PARENT, "tracestate": "congo=ok\r\nX-Injected: 1"}
    )
    assert injected == {"traceparent": _SAMPLE_PARENT}
    nulled = w3c_trace_headers({"traceparent": _SAMPLE_PARENT, "tracestate": "congo=ok\x00"})
    assert "tracestate" not in nulled
    surrogate = w3c_trace_headers({"traceparent": _SAMPLE_PARENT, "tracestate": "vendor=\ud800"})
    assert surrogate == {"traceparent": _SAMPLE_PARENT}
    assert w3c_trace_headers({}) == {}
    forwarded = ensure_trace_headers({"traceparent": _SAMPLE_PARENT})
    assert forwarded["traceparent"] == _SAMPLE_PARENT
    minted_out = ensure_trace_headers({})
    assert canonical_traceparent(minted_out["traceparent"]) == minted_out["traceparent"]
    minted = make_traceparent()
    assert canonical_traceparent(minted) == minted
    child = make_traceparent(trace_id="4bf92f3577b34da6a3ce929d0e0e4736")
    assert trace_id_of(child) == "4bf92f3577b34da6a3ce929d0e0e4736"
    assert child != minted
