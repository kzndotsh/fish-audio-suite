from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest

from fish_audio_suite_kit import FishHttpError
from fish_audio_suite_voice.cli import fish_asr


class _FakeAsrResponse:
    def __init__(
        self,
        status_code: int,
        *,
        text: str = "",
        payload: dict[str, Any] | None = None,
    ) -> None:
        self.status_code = status_code
        self.text = text
        self._payload = payload

    def json(self) -> dict[str, Any]:
        assert self._payload is not None
        return self._payload


class _FakeAsrClient:
    def __init__(self, outcomes: list[_FakeAsrResponse | Exception]) -> None:
        self._outcomes = list(outcomes)
        self.posts = 0

    async def __aenter__(self) -> _FakeAsrClient:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        return None

    async def post(self, *_args: object, **_kwargs: object) -> _FakeAsrResponse:
        self.posts += 1
        item = self._outcomes.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    mock = AsyncMock()
    monkeypatch.setattr("fish_audio_suite_voice.cli.asyncio.sleep", mock)
    return mock


def test_fish_asr_retries_429_then_ok(monkeypatch: pytest.MonkeyPatch, sleeps: AsyncMock) -> None:
    client = _FakeAsrClient(
        [
            _FakeAsrResponse(429, text='{"message": "slow down", "status": 429}'),
            _FakeAsrResponse(200, payload={"text": "hello there"}),
        ]
    )
    monkeypatch.setattr(
        "fish_audio_suite_voice.cli.httpx.AsyncClient",
        lambda **_kwargs: client,
    )
    text = asyncio.run(fish_asr(b"wav", "key", base="https://api.fish.audio"))
    assert text == "hello there"
    assert client.posts == 2
    sleeps.assert_awaited_once()


def test_fish_asr_does_not_retry_401(monkeypatch: pytest.MonkeyPatch, sleeps: AsyncMock) -> None:
    client = _FakeAsrClient(
        [_FakeAsrResponse(401, text='{"message": "Invalid Token", "status": 401}')]
    )
    monkeypatch.setattr(
        "fish_audio_suite_voice.cli.httpx.AsyncClient",
        lambda **_kwargs: client,
    )
    with pytest.raises(FishHttpError) as exc:
        asyncio.run(fish_asr(b"wav", "key", base="https://api.fish.audio"))
    assert exc.value.status == 401
    assert client.posts == 1
    sleeps.assert_not_awaited()


def test_fish_asr_retries_timeout_then_ok(
    monkeypatch: pytest.MonkeyPatch, sleeps: AsyncMock
) -> None:
    client = _FakeAsrClient(
        [
            httpx.TimeoutException("timed out"),
            _FakeAsrResponse(200, payload={"text": "hello there"}),
        ]
    )
    monkeypatch.setattr(
        "fish_audio_suite_voice.cli.httpx.AsyncClient",
        lambda **_kwargs: client,
    )
    text = asyncio.run(fish_asr(b"wav", "key", base="https://api.fish.audio"))
    assert text == "hello there"
    assert client.posts == 2
    sleeps.assert_awaited_once()
