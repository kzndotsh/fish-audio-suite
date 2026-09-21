from __future__ import annotations

import asyncio
from typing import Any, cast
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi.testclient import TestClient

from fish_audio_suite_kit import is_asr_hallucination, is_tts_junk
from fish_audio_suite_proxy.server import (
    _chunk_length_hi,
    _fish_send,
    _pick_reference_id,
    _resolve_asr_model,
    _upstream_trace_headers,
    _uvicorn_run_kwargs,
    app,
    prepare_tts_text,
)


class _FakeUpstream:
    def __init__(self, status_code: int, body: bytes = b"") -> None:
        self.status_code = status_code
        self._body = body

    async def aread(self) -> bytes:
        return self._body

    async def aclose(self) -> None:
        return None


class _FakeFishClient:
    def __init__(self, outcomes: list[Exception | _FakeUpstream]) -> None:
        self._outcomes = list(outcomes)
        self.sends = 0

    def build_request(self, **request_kwargs: Any) -> dict[str, Any]:
        return request_kwargs

    async def send(self, _req: Any, stream: bool = False) -> _FakeUpstream:
        del stream
        self.sends += 1
        item = self._outcomes.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def test_prepare_tts_normalizes_cues() -> None:
    spoken = prepare_tts_text("Excited, hello there friend", dialogue_only=False)
    assert spoken.startswith("[excited]")
    assert "hello" in spoken


def test_junk_unclosed_cue() -> None:
    spoken = prepare_tts_text("[warm,", dialogue_only=False)
    assert is_tts_junk(spoken)


def test_whisper_alias_on_speech_path() -> None:
    spoken = prepare_tts_text("[whispers] keep this quiet now", dialogue_only=False)
    assert "[whispering]" in spoken


def test_prepare_keeps_stacked_cues_and_speaker_token() -> None:
    spoken = prepare_tts_text(
        "<|speaker:0|> [sad][whispering] I miss you so much",
        dialogue_only=False,
    )
    assert "<|speaker:0|>" in spoken
    assert "[sad]" in spoken
    assert "[whispering]" in spoken


def test_asr_model_whisper_remap() -> None:
    assert _resolve_asr_model("whisper-1", "transcribe-1") == "transcribe-1"


def test_asr_hallucination_without_network() -> None:
    assert is_asr_hallucination("谢谢观看")
    assert not is_asr_hallucination("你好")
    assert not is_asr_hallucination("hello there friend")


def test_chunk_length_hi_cloud_vs_self_host() -> None:
    assert _chunk_length_hi("https://api.fish.audio") == 300
    assert _chunk_length_hi("http://127.0.0.1:8080") == 1000


def test_pick_reference_id_list() -> None:
    assert _pick_reference_id({"reference_id": ["a", "b"]}) == ["a", "b"]
    assert _pick_reference_id({"voice": "solo"}) == "solo"


def test_empty_transcription_is_fish_error_shape() -> None:
    with TestClient(app) as client:
        r = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("a.wav", b"", "audio/wav")},
        )
        assert r.status_code == 400
        assert r.json() == {"message": "empty audio upload", "status": 400}


def test_speech_without_key_is_401(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FISH_API_KEY", raising=False)
    with TestClient(app) as client:
        r = client.post("/v1/audio/speech", json={"input": "[clear] Hello there friend"})
        assert r.status_code == 401
        body = r.json()
        assert body["status"] == 401
        assert "message" in body


def test_health_without_api_key() -> None:
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["defaults"]["asr_strip_speakers"] is False
        assert body["defaults"]["tts_dialogue_only"] is False
        models = client.get("/v1/models")
        ids = {m["id"] for m in models.json()["data"]}
        assert "s2.1-pro-free" in ids
        assert "drama-3-preview" in ids


def test_upstream_trace_headers_forward_or_mint() -> None:
    sample = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
    forwarded = _upstream_trace_headers({"traceparent": sample, "tracestate": "vendor=1"})
    assert forwarded["traceparent"] == sample
    assert forwarded["tracestate"] == "vendor=1"
    minted = _upstream_trace_headers({})
    assert minted["traceparent"].startswith("00-")


def test_uvicorn_run_kwargs_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FISH_PROXY_WORKERS", raising=False)
    monkeypatch.delenv("WEB_CONCURRENCY", raising=False)
    monkeypatch.delenv("FISH_PROXY_LIMIT_CONCURRENCY", raising=False)
    monkeypatch.delenv("FISH_PROXY_GRACEFUL_SHUTDOWN", raising=False)
    monkeypatch.delenv("FISH_PROXY_PORT", raising=False)
    kw = _uvicorn_run_kwargs()
    assert kw["workers"] == 1
    assert kw["port"] == 8849
    assert kw["loop"] == "auto"
    assert kw["http"] == "auto"
    assert kw["ws"] == "none"
    assert kw["timeout_graceful_shutdown"] == 120
    assert kw["timeout_keep_alive"] == 5
    assert "limit_concurrency" not in kw


def test_uvicorn_run_kwargs_workers_and_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FISH_PROXY_WORKERS", "4")
    monkeypatch.setenv("FISH_PROXY_LIMIT_CONCURRENCY", "32")
    monkeypatch.setenv("FISH_PROXY_GRACEFUL_SHUTDOWN", "90")
    kw = _uvicorn_run_kwargs()
    assert kw["workers"] == 4
    assert kw["limit_concurrency"] == 32
    assert kw["timeout_graceful_shutdown"] == 90


def test_fish_send_retries_429_then_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps = AsyncMock()
    monkeypatch.setattr("fish_audio_suite_proxy.server.asyncio.sleep", sleeps)
    client = _FakeFishClient(
        [
            _FakeUpstream(429, b'{"message": "slow down", "status": 429}'),
            _FakeUpstream(200),
        ]
    )

    async def _run() -> None:
        out = await _fish_send(
            cast(httpx.AsyncClient, client),
            stream=False,
            method="POST",
            url="https://api.fish.audio/v1/tts",
        )
        assert isinstance(out, _FakeUpstream)
        assert out.status_code == 200

    asyncio.run(_run())
    assert client.sends == 2
    sleeps.assert_awaited_once()


def test_fish_send_does_not_retry_401(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps = AsyncMock()
    monkeypatch.setattr("fish_audio_suite_proxy.server.asyncio.sleep", sleeps)
    client = _FakeFishClient([_FakeUpstream(401, b'{"message": "Invalid Token", "status": 401}')])

    async def _run() -> None:
        out = await _fish_send(
            cast(httpx.AsyncClient, client),
            stream=False,
            method="POST",
            url="https://api.fish.audio/v1/tts",
        )
        assert out.status_code == 401

    asyncio.run(_run())
    assert client.sends == 1
    sleeps.assert_not_awaited()


def test_fish_send_retries_timeout_then_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps = AsyncMock()
    monkeypatch.setattr("fish_audio_suite_proxy.server.asyncio.sleep", sleeps)
    client = _FakeFishClient(
        [
            httpx.TimeoutException("timed out"),
            _FakeUpstream(200),
        ]
    )

    async def _run() -> None:
        out = await _fish_send(
            cast(httpx.AsyncClient, client),
            stream=False,
            method="POST",
            url="https://api.fish.audio/v1/tts",
        )
        assert isinstance(out, _FakeUpstream)
        assert out.status_code == 200

    asyncio.run(_run())
    assert client.sends == 2
    sleeps.assert_awaited_once()
