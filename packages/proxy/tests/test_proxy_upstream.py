from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient
from proxy_helpers import (
    FakeUpstream,
    run_fish_send,
)
from starlette.responses import JSONResponse

from fish_audio_suite_kit import (
    ensure_trace_headers,
)
from fish_audio_suite_proxy.server import app


def test_fish_client_timeout_and_user_agent() -> None:
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        http = app.state.http
        assert isinstance(http, httpx.AsyncClient)
        assert http.timeout.connect == 10.0
        assert http.timeout.read == 120.0
        assert http.timeout.write == 120.0
        assert http.timeout.pool == 5.0
        assert http.headers["User-Agent"].startswith("fish-audio-suite-proxy/")


def test_upstream_trace_headers_forward_or_mint() -> None:
    sample = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
    forwarded = ensure_trace_headers({"traceparent": sample, "tracestate": "vendor=1"})
    assert forwarded["traceparent"] == sample
    assert forwarded["tracestate"] == "vendor=1"
    minted = ensure_trace_headers({})
    assert minted["traceparent"].startswith("00-")


def test_fish_send_retries_429_then_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    out, client, sleeps = run_fish_send(
        monkeypatch,
        [
            FakeUpstream(429, b'{"message": "slow down", "status": 429}'),
            FakeUpstream(200),
        ],
    )
    assert isinstance(out, FakeUpstream)
    assert out.status_code == 200
    assert client.sends == 2
    sleeps.assert_awaited_once()


def test_fish_send_does_not_treat_a_redirect_as_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out, client, sleeps = run_fish_send(
        monkeypatch,
        [FakeUpstream(302, b"<html>moved</html>")],
    )
    assert isinstance(out, JSONResponse)
    assert out.status_code == 302
    assert client.sends == 1
    sleeps.assert_not_awaited()


def test_fish_send_does_not_retry_401(monkeypatch: pytest.MonkeyPatch) -> None:
    out, client, sleeps = run_fish_send(
        monkeypatch,
        [FakeUpstream(401, b'{"message": "Invalid Token", "status": 401}')],
    )
    assert out.status_code == 401
    assert client.sends == 1
    sleeps.assert_not_awaited()


def test_fish_send_retries_a_connect_timeout_then_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    out, client, sleeps = run_fish_send(
        monkeypatch,
        [httpx.ConnectTimeout("timed out"), FakeUpstream(200)],
    )
    assert isinstance(out, FakeUpstream)
    assert out.status_code == 200
    assert client.sends == 2
    sleeps.assert_awaited_once()
