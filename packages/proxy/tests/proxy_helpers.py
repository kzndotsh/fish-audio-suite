from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from fish_audio_suite_proxy.server import app
from fish_audio_suite_proxy.upstream import RetryPolicy, fish_send


def not_response[T](value: T | JSONResponse) -> T:
    """Narrow a ``value | JSONResponse`` result to ``value``, failing if it was an error."""
    assert not isinstance(value, JSONResponse)
    return value


class FakeUpstream:
    def __init__(
        self,
        status_code: int,
        body: bytes = b"",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self._body = body
        self.headers = headers or {}

    async def aread(self) -> bytes:
        return self._body

    async def aclose(self) -> None:
        return None


class FakeFishClient:
    def __init__(self, outcomes: list[Exception | FakeUpstream]) -> None:
        self._outcomes = list(outcomes)
        self.sends = 0

    def build_request(
        self,
        method: str,
        url: str,
        *,
        content: Any = None,
        data: Any = None,
        files: Any = None,
        json: Any = None,
        headers: Any = None,
    ) -> dict[str, Any]:
        return {
            "method": method,
            "url": url,
            "content": content,
            "data": data,
            "files": files,
            "json": json,
            "headers": headers,
        }

    async def send(self, request: Any, *, stream: bool = False) -> FakeUpstream:
        del request, stream
        self.sends += 1
        item = self._outcomes.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def run_fish_send(
    monkeypatch: pytest.MonkeyPatch,
    outcomes: list[Exception | FakeUpstream],
    *,
    policy: RetryPolicy | None = None,
    is_disconnected: Any = None,
) -> tuple[Any, FakeFishClient, AsyncMock]:
    sleeps = AsyncMock()
    monkeypatch.setattr("fish_audio_suite_proxy.upstream.asyncio.sleep", sleeps)
    client = FakeFishClient(outcomes)

    async def run() -> Any:
        return await fish_send(
            client,
            stream=False,
            policy=policy,
            is_disconnected=is_disconnected,
            method="POST",
            url="https://api.fish.audio/v1/tts",
        )

    return asyncio.run(run()), client, sleeps


WAV_UPLOAD = {"file": ("a.wav", b"xx", "audio/wav")}


class AsrJson:
    def json(self) -> dict[str, Any]:
        return {
            "text": "hello there",
            "duration": 1.5,
            "language": "en",
            "segments": [
                {"text": "hello", "start": 0, "end": 0.6},
                {"text": "there", "start": 0.6, "end": 1.5},
            ],
        }


class AudioStream:
    async def aiter_bytes(self, _n: int = 4096):
        yield b"mp3"

    async def aclose(self) -> None:
        return None


def capture_upstream(monkeypatch: pytest.MonkeyPatch, result: Any) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    async def fake_send(*_args: object, **kwargs: object) -> Any:
        captured.update(kwargs)
        return result

    monkeypatch.setenv("FISH_API_KEY", "test-key")
    monkeypatch.setattr("fish_audio_suite_proxy.server.fish_send", fake_send)
    return captured


def post_speech(
    monkeypatch: pytest.MonkeyPatch,
    body: dict[str, Any],
) -> tuple[httpx.Response, dict[str, Any]]:
    captured = capture_upstream(monkeypatch, AudioStream())
    with TestClient(app) as client:
        return client.post("/v1/audio/speech", json=body), captured


def fresh_app_env(monkeypatch: pytest.MonkeyPatch, **env: str) -> None:
    for name in (
        "FISH_PROXY_API_KEYS",
        "FISH_PROXY_MAX_BODY_BYTES",
        "FISH_PROXY_MAX_INPUT_CHARS",
        "FISH_PROXY_LOG_TEXT",
        "FISH_PROXY_HOST",
    ):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
