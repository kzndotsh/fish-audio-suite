from __future__ import annotations

import pytest

from fish_audio_suite_kit import (
    ensure_trace_headers,
    make_traceparent,
    trace_id_of,
)

_SAMPLE_PARENT = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
_SAMPLE_TRACE = "4bf92f3577b34da6a3ce929d0e0e4736"


def test_a_valid_traceparent_is_forwarded_in_lowercase() -> None:
    assert ensure_trace_headers({"traceparent": _SAMPLE_PARENT}) == {"traceparent": _SAMPLE_PARENT}
    upper = ensure_trace_headers({"traceparent": f" {_SAMPLE_PARENT.upper()} "})
    assert upper == {"traceparent": _SAMPLE_PARENT}
    assert trace_id_of(_SAMPLE_PARENT.upper()) == _SAMPLE_TRACE


@pytest.mark.parametrize(
    "bad",
    [
        "not-a-trace",
        "00-" + "0" * 32 + "-00f067aa0ba902b7-01",
        "00-" + _SAMPLE_TRACE + "-" + "0" * 16 + "-01",
        "ff-" + _SAMPLE_TRACE + "-00f067aa0ba902b7-01",
    ],
)
def test_an_invalid_traceparent_is_replaced_with_a_fresh_one(bad: str) -> None:
    assert trace_id_of(bad) is None
    minted = ensure_trace_headers({"traceparent": bad, "tracestate": "congo=ok"})
    assert set(minted) == {"traceparent"}
    assert minted["traceparent"] != bad
    assert trace_id_of(minted["traceparent"]) not in {None, _SAMPLE_TRACE}


def test_w3c_trace_headers_and_mint() -> None:
    headers = ensure_trace_headers(
        {"Traceparent": _SAMPLE_PARENT, "tracestate": "congo=t61rcWkgMzE"}
    )
    assert headers["traceparent"] == _SAMPLE_PARENT
    assert headers["tracestate"] == "congo=t61rcWkgMzE"
    injected = ensure_trace_headers(
        {"traceparent": _SAMPLE_PARENT, "tracestate": "congo=ok\r\nX-Injected: 1"}
    )
    assert injected == {"traceparent": _SAMPLE_PARENT}
    nulled = ensure_trace_headers({"traceparent": _SAMPLE_PARENT, "tracestate": "congo=ok\x00"})
    assert "tracestate" not in nulled
    surrogate = ensure_trace_headers({"traceparent": _SAMPLE_PARENT, "tracestate": "vendor=\ud800"})
    assert surrogate == {"traceparent": _SAMPLE_PARENT}
    # A minted header is already canonical, so it is forwarded unchanged.
    minted_out = ensure_trace_headers({})
    assert ensure_trace_headers(minted_out) == minted_out
    minted = make_traceparent()
    assert ensure_trace_headers({"traceparent": minted}) == {"traceparent": minted}
    child = make_traceparent(trace_id=_SAMPLE_TRACE)
    assert trace_id_of(child) == _SAMPLE_TRACE
    assert child != minted
