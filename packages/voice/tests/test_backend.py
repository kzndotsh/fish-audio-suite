from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from fish_audio_suite_voice.llm import ChatBackend, open_chat_backend
from fish_audio_suite_voice.transports import chat_completions_url
from fish_audio_suite_voice.tune import LlmTune

SSE = (
    'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n'
    'data: {"choices":[{"delta":{"content":" there"},"finish_reason":"stop"}]}\n\n'
    "data: [DONE]\n\n"
)


def _http_tune(**kw: Any) -> LlmTune:
    fields: dict[str, Any] = {
        "backend": "openai",
        "base": "https://llm.example.test/v1",
        "key": "sk-test",
        "model": "some/model",
    }
    fields.update(kw)
    return LlmTune(**fields)


def _install_transport(
    monkeypatch: pytest.MonkeyPatch, handler: Any, made: list[httpx.AsyncClient]
) -> None:
    real = httpx.AsyncClient

    def factory(**kwargs: Any) -> httpx.AsyncClient:
        client = real(transport=httpx.MockTransport(handler), **kwargs)
        made.append(client)
        return client

    monkeypatch.setattr("fish_audio_suite_voice.transports.httpx.AsyncClient", factory)


def test_openai_backend_streams_and_sends_no_attribution_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, text=SSE, headers={"content-type": "text/event-stream"})

    made: list[httpx.AsyncClient] = []
    _install_transport(monkeypatch, handler, made)

    async def run() -> list[str]:
        async with open_chat_backend(_http_tune(temperature=0.3)) as backend:
            backend_: ChatBackend = backend
            return [tok async for tok in backend_.stream([{"role": "user", "content": "hi"}])]

    assert "".join(asyncio.run(run())) == "Hello there"
    request = seen[0]
    assert str(request.url) == chat_completions_url("https://llm.example.test/v1")
    assert request.headers["authorization"] == "Bearer sk-test"
    assert "http-referer" not in request.headers
    assert "x-title" not in request.headers
    body = json.loads(request.content)
    assert body["temperature"] == 0.3
    assert body["max_tokens"] == 1200
    assert body["model"] == "some/model"


def test_openai_backend_reuses_one_pooled_client_and_closes_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=SSE)

    made: list[httpx.AsyncClient] = []
    _install_transport(monkeypatch, handler, made)

    async def run() -> None:
        async with open_chat_backend(_http_tune()) as backend:
            for _ in range(3):
                async for _tok in backend.stream([{"role": "user", "content": "hi"}]):
                    pass
            assert len(made) == 1
            assert not made[0].is_closed

    asyncio.run(run())
    assert len(made) == 1
    assert made[0].is_closed


def test_openai_backend_does_not_ask_for_nitro_routing(monkeypatch: pytest.MonkeyPatch) -> None:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, text=SSE)

    _install_transport(monkeypatch, handler, [])

    async def run() -> None:
        async with open_chat_backend(_http_tune(nitro=True)) as backend:
            async for _tok in backend.stream([{"role": "user", "content": "hi"}]):
                pass

    asyncio.run(run())
    assert bodies[0]["model"] == "some/model"
    assert "provider" not in bodies[0]


def test_openrouter_backend_checks_the_model_once_and_passes_the_session_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    sent: list[dict[str, Any]] = []

    class _Event:
        def __init__(self, text: str) -> None:
            self.choices = [
                type(
                    "C", (), {"delta": type("D", (), {"content": text})(), "finish_reason": None}
                )()
            ]

    class _Stream:
        async def __aenter__(self) -> _Stream:
            return self

        async def __aexit__(self, *_exc: object) -> None:
            return None

        async def __aiter__(self):
            yield _Event("Hi")

    class _Chat:
        async def send_async(self, **kwargs: Any) -> _Stream:
            sent.append(kwargs)
            return _Stream()

    class _Models:
        async def get_async(self, **kwargs: Any) -> object:
            calls.append(f"{kwargs['author']}/{kwargs['slug']}")
            return type("R", (), {"data": type("M", (), {"id": "org/model"})()})()

    class _Client:
        chat = _Chat()
        models = _Models()

        def __init__(self, **_kwargs: object) -> None:
            return None

        async def __aenter__(self) -> _Client:
            return self

        async def __aexit__(self, *_exc: object) -> None:
            return None

    monkeypatch.setattr("openrouter.OpenRouter", _Client)
    tune = LlmTune(backend="openrouter", key="sk", model="org/model")

    async def run() -> str:
        async with open_chat_backend(tune, session_id="sess-1") as backend:
            first = "".join([t async for t in backend.stream([{"role": "user", "content": "a"}])])
            await asyncio.sleep(0)
            async for _ in backend.stream([{"role": "user", "content": "b"}]):
                pass
            return first

    assert asyncio.run(run()) == "Hi"
    assert calls == ["org/model"]
    assert sent[0]["session_id"] == "sess-1"
    assert sent[1]["session_id"] == "sess-1"
