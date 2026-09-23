from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest

from fish_audio_suite_kit import FishHttpError
from fish_audio_suite_voice.asr import fish_asr


class _FakeAsrResponse:
    def __init__(
        self,
        status_code: int,
        *,
        text: str = "",
        payload: Any = None,
    ) -> None:
        self.status_code = status_code
        self.text = text
        self.headers: dict[str, str] = {}
        self._payload = payload

    def json(self) -> Any:
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
    monkeypatch.setattr("fish_audio_suite_kit.http_errors.asyncio.sleep", mock)
    return mock


def _install_asr(
    monkeypatch: pytest.MonkeyPatch,
    outcomes: list[_FakeAsrResponse | Exception],
) -> _FakeAsrClient:
    client = _FakeAsrClient(outcomes)
    monkeypatch.setattr(
        "fish_audio_suite_voice.asr.httpx.AsyncClient",
        lambda **_kwargs: client,
    )
    return client


def _run_asr(
    monkeypatch: pytest.MonkeyPatch,
    outcomes: list[_FakeAsrResponse | Exception],
) -> tuple[str, _FakeAsrClient]:
    client = _install_asr(monkeypatch, outcomes)
    text = asyncio.run(fish_asr(b"wav", "key", base="https://api.fish.audio"))
    return text, client


def test_fish_asr_retries_429_then_ok(monkeypatch: pytest.MonkeyPatch, sleeps: AsyncMock) -> None:
    text, client = _run_asr(
        monkeypatch,
        [
            _FakeAsrResponse(429, text='{"message": "slow down", "status": 429}'),
            _FakeAsrResponse(200, payload={"text": "hello there"}),
        ],
    )
    assert text == "hello there"
    assert client.posts == 2
    sleeps.assert_awaited_once()


def test_fish_asr_does_not_retry_401(monkeypatch: pytest.MonkeyPatch, sleeps: AsyncMock) -> None:
    client = _install_asr(
        monkeypatch,
        [_FakeAsrResponse(401, text='{"message": "Invalid Token", "status": 401}')],
    )
    with pytest.raises(FishHttpError) as exc:
        asyncio.run(fish_asr(b"wav", "key", base="https://api.fish.audio"))
    assert exc.value.status == 401
    assert client.posts == 1
    sleeps.assert_not_awaited()


def test_fish_asr_non_string_text_is_502(
    monkeypatch: pytest.MonkeyPatch, sleeps: AsyncMock
) -> None:
    client = _install_asr(monkeypatch, [_FakeAsrResponse(200, payload={"text": ["hello"]})])
    with pytest.raises(FishHttpError) as exc:
        asyncio.run(fish_asr(b"wav", "key", base="https://api.fish.audio"))
    assert exc.value.status == 502
    assert exc.value.message == "Fish returned a non-object body"
    assert client.posts == 1
    sleeps.assert_not_awaited()


def test_fish_asr_non_object_body_is_502(
    monkeypatch: pytest.MonkeyPatch, sleeps: AsyncMock
) -> None:
    client = _install_asr(monkeypatch, [_FakeAsrResponse(200, payload=["hello"])])
    with pytest.raises(FishHttpError) as exc:
        asyncio.run(fish_asr(b"wav", "key", base="https://api.fish.audio"))
    assert exc.value.status == 502
    assert exc.value.message == "Fish returned a non-object body"
    assert client.posts == 1
    sleeps.assert_not_awaited()


def test_fish_asr_non_json_body_is_502(monkeypatch: pytest.MonkeyPatch, sleeps: AsyncMock) -> None:
    class _Broken(_FakeAsrResponse):
        def json(self) -> dict[str, Any]:
            raise json.JSONDecodeError("Expecting value", "", 0)

    client = _install_asr(monkeypatch, [_Broken(200)])
    with pytest.raises(FishHttpError) as exc:
        asyncio.run(fish_asr(b"wav", "key", base="https://api.fish.audio"))
    assert exc.value.status == 502
    assert exc.value.message == "Fish returned a non-JSON body"
    assert client.posts == 1
    sleeps.assert_not_awaited()

    class _NotUtf8(_FakeAsrResponse):
        def json(self) -> dict[str, Any]:
            raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")

    client = _install_asr(monkeypatch, [_NotUtf8(200)])
    with pytest.raises(FishHttpError) as exc:
        asyncio.run(fish_asr(b"wav", "key", base="https://api.fish.audio"))
    assert exc.value.status == 502
    assert exc.value.message == "Fish returned a non-JSON body"


def test_fish_asr_retries_timeout_then_ok(
    monkeypatch: pytest.MonkeyPatch, sleeps: AsyncMock
) -> None:
    text, client = _run_asr(
        monkeypatch,
        [
            httpx.TimeoutException("timed out"),
            _FakeAsrResponse(200, payload={"text": "hello there"}),
        ],
    )
    assert text == "hello there"
    assert client.posts == 2
    sleeps.assert_awaited_once()
