from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, ClassVar

import httpx
import pytest
from openrouter.errors import OpenRouterError

from fish_audio_suite_voice.llm import (
    _model_author_slug,
    _want_nitro,
    check_openrouter_model,
    llm_token_stream,
)

OR_BASE = "https://openrouter.ai/api/v1"


def _tokens(**extra: Any):
    return llm_token_stream(
        [{"role": "user", "content": "hi"}],
        base=OR_BASE,
        key="sk-test",
        model="org/model",
        **extra,
    )


def test_want_nitro_slash_model_on_openrouter() -> None:
    assert _want_nitro("org/model", OR_BASE)


def test_want_nitro_false_when_suffix_present() -> None:
    assert not _want_nitro("org/model:nitro", OR_BASE)


def test_want_nitro_false_when_env_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FISH_LLM_NITRO", "0")
    assert not _want_nitro("org/model", OR_BASE)


def test_want_nitro_false_for_non_openrouter_base() -> None:
    assert not _want_nitro("org/model", "https://api.example.com/v1")


def test_model_author_slug_splits_variant() -> None:
    assert _model_author_slug("openai/gpt-4") == ("openai", "gpt-4")
    assert _model_author_slug("openai/gpt-4:nitro") == ("openai", "gpt-4:nitro")
    assert _model_author_slug("gpt-4") is None


def _chunk(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=text), finish_reason=None)],
        usage=None,
        id="gen-1",
        model="org/model",
        error=None,
        provider=None,
    )


class _FakeStream:
    def __init__(self, chunks: list[object]) -> None:
        self._chunks = chunks

    async def __aenter__(self) -> _FakeStream:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def __aiter__(self) -> Any:
        for chunk in self._chunks:
            yield chunk


class _FakeChat:
    last_kw: dict[str, Any] | None = None

    def __init__(self, chunks: list[object]) -> None:
        self._chunks = chunks

    async def send_async(self, **kwargs: Any) -> Any:
        _FakeChat.last_kw = kwargs
        return _FakeStream(self._chunks)


class _FakeOpenRouter:
    last_init: dict[str, Any] | None = None
    chunks: ClassVar[list[object]] = []

    def __init__(self, **kwargs: Any) -> None:
        _FakeOpenRouter.last_init = kwargs
        self.chat = _FakeChat(list(self.chunks))

    async def __aenter__(self) -> _FakeOpenRouter:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None


@pytest.fixture
def fake_openrouter(monkeypatch: pytest.MonkeyPatch) -> type[_FakeOpenRouter]:
    import openrouter

    monkeypatch.setattr(openrouter, "OpenRouter", _FakeOpenRouter)
    _FakeOpenRouter.chunks = [_chunk("Hello"), _chunk(" world")]
    _FakeOpenRouter.last_init = None
    _FakeChat.last_kw = None
    return _FakeOpenRouter


def test_openrouter_stream_joins_tokens(fake_openrouter: type[_FakeOpenRouter]) -> None:
    async def run() -> list[str]:
        return [tok async for tok in _tokens()]

    pieces = asyncio.run(run())
    assert "".join(pieces) == "Hello world"
    assert fake_openrouter.last_init is not None
    assert fake_openrouter.last_init["http_referer"].endswith("fish-audio-suite")
    assert fake_openrouter.last_init["x_open_router_title"] == "fish-audio-suite-voice"
    assert _FakeChat.last_kw is not None
    assert _FakeChat.last_kw["stream"] is True
    assert _FakeChat.last_kw["model"] == "org/model:nitro"
    assert _FakeChat.last_kw["provider"] == {"sort": "throughput"}
    assert _FakeChat.last_kw["max_completion_tokens"] == 1200
    assert "max_tokens" not in _FakeChat.last_kw
    assert fake_openrouter.last_init["x_open_router_categories"] == "cli-agent"


def test_max_tokens_must_be_positive(
    fake_openrouter: type[_FakeOpenRouter],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        async for _tok in _tokens():
            pass

    monkeypatch.setenv("FISH_LLM_MAX_TOKENS", "0")
    asyncio.run(run())
    assert _FakeChat.last_kw is not None
    assert _FakeChat.last_kw["max_completion_tokens"] == 1200
    monkeypatch.setenv("FISH_LLM_MAX_TOKENS", "-5")
    asyncio.run(run())
    assert _FakeChat.last_kw["max_completion_tokens"] == 1200
    monkeypatch.setenv("FISH_LLM_MAX_TOKENS", "128")
    asyncio.run(run())
    assert _FakeChat.last_kw["max_completion_tokens"] == 128
    assert fake_openrouter.last_init is not None


def test_openrouter_message_content_when_delta_empty(
    fake_openrouter: type[_FakeOpenRouter],
) -> None:
    fake_openrouter.chunks = [
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content=None),
                    message=SimpleNamespace(content="from-message"),
                    finish_reason="stop",
                )
            ],
            usage=None,
            id="gen-1",
            model="org/model",
            error=None,
            provider=None,
        )
    ]

    async def run() -> list[str]:
        return [tok async for tok in _tokens()]

    assert asyncio.run(run()) == ["from-message"]


def test_openrouter_cancel_stops_after_chunk(
    fake_openrouter: type[_FakeOpenRouter],
) -> None:
    fake_openrouter.chunks = [_chunk("one"), _chunk("two")]

    async def run() -> list[str]:
        cancel = asyncio.Event()
        pieces: list[str] = []
        async for tok in _tokens(cancel=cancel):
            pieces.append(tok)
            cancel.set()
        return pieces

    assert asyncio.run(run()) == ["one"]


def test_passed_client_skips_openrouter_constructor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import openrouter

    inits: list[object] = []

    class _Boom:
        def __init__(self, **kwargs: object) -> None:
            inits.append(kwargs)

    monkeypatch.setattr(openrouter, "OpenRouter", _Boom)
    chat = _FakeChat([_chunk("ok")])
    passed = SimpleNamespace(chat=chat)

    async def run() -> list[str]:
        return [
            tok
            async for tok in _tokens(
                client=passed,
                session_id="sess-1",
                trace_id="trace-abc",
            )
        ]

    assert asyncio.run(run()) == ["ok"]
    assert inits == []
    assert _FakeChat.last_kw is not None
    assert _FakeChat.last_kw["session_id"] == "sess-1"
    assert _FakeChat.last_kw["trace"] == {
        "trace_id": "trace-abc",
        "trace_name": "fish-audio-suite-voice",
    }


def test_check_openrouter_model_calls_get_with_nitro() -> None:
    seen: dict[str, str] = {}

    async def get_async(*, author: str, slug: str, **_kwargs: Any) -> Any:
        seen["author"] = author
        seen["slug"] = slug
        return SimpleNamespace(
            data=SimpleNamespace(id="org/model", name="Model", context_length=8000)
        )

    client = SimpleNamespace(models=SimpleNamespace(get_async=get_async))
    asyncio.run(check_openrouter_model(client, "org/model", OR_BASE))
    assert seen == {"author": "org", "slug": "model:nitro"}


def test_check_openrouter_model_404(capsys: pytest.CaptureFixture[str]) -> None:
    async def get_async(**_kwargs: Any) -> Any:
        req = httpx.Request("GET", "https://openrouter.ai/api/v1/model/org/missing")
        raise OpenRouterError("missing", httpx.Response(404, request=req), body="nope")

    client = SimpleNamespace(models=SimpleNamespace(get_async=get_async))
    asyncio.run(check_openrouter_model(client, "org/missing", OR_BASE))
    assert "unknown model org/missing:nitro" in capsys.readouterr().err


def test_check_openrouter_model_skips_unsplit_slug() -> None:
    async def get_async(**_kwargs: Any) -> Any:
        raise AssertionError("must not call get")

    client = SimpleNamespace(models=SimpleNamespace(get_async=get_async))
    asyncio.run(check_openrouter_model(client, "norgslash", OR_BASE))
