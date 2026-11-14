from __future__ import annotations

import asyncio
import gc
import json
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from types import SimpleNamespace
from typing import Any, ClassVar, cast

import httpx
import openrouter
import pytest
from openrouter.errors import OpenRouterError

from fish_audio_suite_voice.llm.client import llm_token_stream
from fish_audio_suite_voice.llm.transports import (
    ChatChunk,
    ChatHttpError,
    ChatUnusableError,
    check_openrouter_model,
    describe_http_error,
    openrouter_client,
    routed_model,
)
from fish_audio_suite_voice.llm.tune import LlmTune, is_openrouter_host
from fish_audio_suite_voice.logutil.setup import configure_voice_logging
from fish_audio_suite_voice.session.events import EVENTS, Event, Notice

OR_BASE = "https://openrouter.ai/api/v1"


def _tune(**kw: Any) -> LlmTune:
    fields: dict[str, Any] = {
        "base": OR_BASE,
        "api_key": "sk-test",
        "model": "org/model",
        "nitro": True,
        "continuation": True,
    }
    fields.update(kw)
    return LlmTune(**fields)


def _openai_tune(**kw: Any) -> LlmTune:
    fields: dict[str, Any] = {
        "base": "https://api.example.com/v1",
        "api_key": "sk-test",
        "model": "org/model",
    }
    fields.update(kw)
    return LlmTune(**fields)


Handler = Callable[[httpx.Request], httpx.Response]


def _over_http(
    handler: Handler,
    *,
    messages: list[Any] | None = None,
    cancel_after_first: bool = False,
) -> list[str]:
    """Stream one reply from the OpenAI-compatible path through a fake HTTP server."""

    async def run() -> list[str]:
        cancel = asyncio.Event()
        pieces: list[str] = []
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            async for tok in llm_token_stream(
                messages or [{"role": "user", "content": "hi"}],
                tune=_openai_tune(),
                cancel=cancel,
                http=http,
            ):
                pieces.append(tok)
                if cancel_after_first:
                    cancel.set()
        return pieces

    return asyncio.run(asyncio.wait_for(run(), 5))


def _sse_reply(text: str, *, status: int = 200, headers: dict[str, str] | None = None) -> Handler:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, text=text, headers=headers)

    return handler


class _Stalls(httpx.AsyncByteStream):
    """A body that sends ``first`` and then either never ends or fails the next read."""

    def __init__(self, first: bytes, *, then_fail: bool) -> None:
        self._first = first
        self._then_fail = then_fail

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self._first
        if self._then_fail:
            await asyncio.sleep(0)
            raise httpx.ReadError("connection dropped")
        await asyncio.Event().wait()


def test_openrouter_client_drops_a_key_that_breaks_the_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    class _Client:
        def __init__(self, **kwargs: object) -> None:
            seen.update(kwargs)

        async def __aenter__(self) -> _Client:
            return self

        async def __aexit__(self, *_exc: object) -> None:
            return None

    monkeypatch.setattr("openrouter.OpenRouter", _Client)

    async def open_with(key: str) -> None:
        async with openrouter_client(_tune(api_key=key)):
            pass

    asyncio.run(open_with("sk-good"))
    assert seen["api_key"] == "sk-good"
    asyncio.run(open_with("sk-\nbad"))
    assert seen["api_key"] == ""


def test_surrogate_in_chat_messages_still_encodes() -> None:
    sent: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, text='data: {"choices":[{"delta":{"content":"ok"}}]}\n\n')

    messages = [{"role": "system", "content": "Be brief. \ud800 Answer plainly."}]
    assert _over_http(handler, messages=messages) == ["ok"]
    text = sent[0]["messages"][0]["content"]
    assert "\ud800" not in text
    assert "Be brief." in text
    assert "Answer plainly." in text


def test_sse_bom_does_not_drop_the_first_token() -> None:
    first = '\ufeffdata: {"choices":[{"delta":{"content":"Hello there"}}]}\n\n'
    assert _over_http(_sse_reply(first)) == ["Hello there"]
    # A BOM before [DONE] still ends the reply.
    done = (
        'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n'
        "\ufeffdata: [DONE]\n\n"
        'data: {"choices":[{"delta":{"content":" dropped"}}]}\n\n'
    )
    assert _over_http(_sse_reply(done)) == ["Hello"]


def test_a_provider_error_group_is_reported(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    async def events() -> Any:
        yield ChatChunk(text="Hello there")
        raise ExceptionGroup("llm", [RuntimeError("provider down")])

    monkeypatch.setattr("fish_audio_suite_voice.llm.client.chat_events", lambda _call: events())

    async def collect() -> list[str]:
        return [
            tok
            async for tok in llm_token_stream(
                [{"role": "user", "content": "hi"}],
                tune=LlmTune(
                    base="https://api.example.com/v1",
                    api_key="sk-test",
                    model="org/model",
                ),
            )
        ]

    assert asyncio.run(collect()) == ["Hello there"]
    assert "provider down" in capsys.readouterr().err


def test_a_cancel_group_does_not_warn(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    async def events() -> Any:
        yield ChatChunk(text="Hello there")
        raise BaseExceptionGroup("cancel", [asyncio.CancelledError()])

    monkeypatch.setattr("fish_audio_suite_voice.llm.client.chat_events", lambda _call: events())

    async def collect() -> list[str]:
        return [
            tok
            async for tok in llm_token_stream(
                [{"role": "user", "content": "hi"}],
                tune=LlmTune(
                    base="https://api.example.com/v1",
                    api_key="sk-test",
                    model="org/model",
                ),
            )
        ]

    assert asyncio.run(collect()) == ["Hello there"]
    assert capsys.readouterr().err == ""


def test_split_sse_data_lines_stay_one_event() -> None:
    lines = [
        'data: {"choices":[{"delta":{"content":',
        'data: "Hello"}}]}',
        "",
        'data: {"choices":[{"delta":{"content":" there"}}]}',
        "data: [DONE]",
        'data: {"choices":[{"delta":{"content":"dropped"}}]}',
    ]
    assert _over_http(_sse_reply("\n".join(lines) + "\n")) == ["Hello", " there"]


def _tokens(**extra: Any):
    tune = extra.pop("tune", None) or _tune()
    return llm_token_stream([{"role": "user", "content": "hi"}], tune=tune, **extra)


def test_nitro_routes_a_slash_model_on_openrouter() -> None:
    assert routed_model(_tune()) == "org/model:nitro"


def test_nitro_keeps_a_model_that_already_names_a_variant() -> None:
    assert routed_model(_tune(model="org/model:nitro")) == "org/model:nitro"
    assert routed_model(_tune(model="org/model:free")) == "org/model:free"


def test_nitro_is_off_unless_asked() -> None:
    assert routed_model(_tune(nitro=False)) == "org/model"
    assert not LlmTune().nitro


def test_nitro_is_never_added_off_openrouter() -> None:
    other = _tune(base="https://api.example.com/v1")
    assert routed_model(other) == "org/model"


def test_openrouter_host_checks_the_hostname_not_the_text() -> None:
    assert is_openrouter_host("https://openrouter.ai/api/v1")
    assert is_openrouter_host("https://eu.openrouter.ai/api/v1")
    assert not is_openrouter_host("https://api.example.com/v1?via=openrouter.ai")
    assert not is_openrouter_host("https://notopenrouter.ai/v1")
    assert not is_openrouter_host("http://localhost:11434/v1")


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
    calls: ClassVar[list[dict[str, Any]]] = []
    rounds: ClassVar[list[list[object]] | None] = None

    def __init__(self, chunks: list[object]) -> None:
        self._chunks = chunks

    async def send_async(self, **kwargs: Any) -> Any:
        _FakeChat.last_kw = kwargs
        _FakeChat.calls.append(kwargs)
        chunks = self._chunks
        if _FakeChat.rounds is not None:
            index = len(_FakeChat.calls) - 1
            if index < len(_FakeChat.rounds):
                chunks = _FakeChat.rounds[index]
        return _FakeStream(chunks)


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
    _FakeChat.calls = []
    _FakeChat.rounds = None
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
    assert _FakeChat.last_kw["provider"] == {"sort": "latency"}
    assert _FakeChat.last_kw["max_completion_tokens"] == 1200
    assert "reasoning" not in _FakeChat.last_kw
    assert "max_tokens" not in _FakeChat.last_kw
    assert fake_openrouter.last_init["x_open_router_categories"] == "cli-agent"


def test_request_settings_come_from_the_tune(
    fake_openrouter: type[_FakeOpenRouter],
) -> None:
    async def run(tune: LlmTune) -> None:
        async for _tok in _tokens(tune=tune):
            pass

    asyncio.run(run(_tune(max_tokens=128, temperature=0.2, timeout_s=30.0, provider_sort="price")))
    assert _FakeChat.last_kw is not None
    assert _FakeChat.last_kw["max_completion_tokens"] == 128
    assert _FakeChat.last_kw["temperature"] == 0.2
    assert _FakeChat.last_kw["timeout_ms"] == 30_000
    assert _FakeChat.last_kw["provider"] == {"sort": "price"}
    asyncio.run(run(_tune(nitro=False)))
    # The sort does not need :nitro.
    assert _FakeChat.last_kw["provider"] == {"sort": "latency"}
    assert _FakeChat.last_kw["model"] == "org/model"
    asyncio.run(run(_tune(provider_sort="")))
    assert "provider" not in _FakeChat.last_kw
    assert "reasoning" not in _FakeChat.last_kw
    asyncio.run(run(_tune(reasoning_effort="low")))
    assert _FakeChat.last_kw["reasoning"] == {"effort": "low"}
    assert fake_openrouter.last_init is not None


def test_openrouter_attribution_is_configurable(
    fake_openrouter: type[_FakeOpenRouter],
) -> None:
    async def run(tune: LlmTune) -> None:
        async for _tok in _tokens(tune=tune):
            pass

    asyncio.run(run(_tune(referer="https://example.test/app", title="my-app", categories="")))
    init = fake_openrouter.last_init
    assert init is not None
    assert init["http_referer"] == "https://example.test/app"
    assert init["x_open_router_title"] == "my-app"
    assert init["x_open_router_categories"] is None


def test_llm_env_numbers_must_be_usable(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("FISH_LLM_MAX_TOKENS", "0")
    assert LlmTune.from_env().max_tokens == 1200
    monkeypatch.setenv("FISH_LLM_MAX_TOKENS", "-5")
    assert LlmTune.from_env().max_tokens == 1200
    monkeypatch.setenv("FISH_LLM_MAX_TOKENS", "128")
    assert LlmTune.from_env().max_tokens == 128
    monkeypatch.setenv("FISH_LLM_TIMEOUT", "0")
    assert LlmTune.from_env().timeout_s == 120.0
    assert "FISH_LLM_MAX_TOKENS" in capsys.readouterr().err


def test_content_parts_keep_text_and_skip_reasoning(
    fake_openrouter: type[_FakeOpenRouter],
) -> None:
    fake_openrouter.chunks = [
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        content=[
                            {"type": "reasoning", "text": "secret"},
                            {"type": "text", "text": "Hello"},
                            " there",
                        ]
                    ),
                    finish_reason=None,
                )
            ],
            usage=None,
            id="gen-1",
            model="org/model",
            error=None,
            provider=None,
        ),
    ]

    async def run() -> list[str]:
        return [tok async for tok in _tokens()]

    assert asyncio.run(run()) == ["Hello there"]


def test_reasoning_field_is_not_spoken(
    fake_openrouter: type[_FakeOpenRouter],
) -> None:
    fake_openrouter.chunks = [
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        content=None,
                        reasoning="secret plan",
                        reasoning_details=[{"type": "reasoning.text", "text": "more secret"}],
                    ),
                    message=SimpleNamespace(content=None, reasoning="message secret"),
                    finish_reason=None,
                )
            ],
            usage=None,
            id="gen-1",
            model="org/model",
            error=None,
            provider=None,
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content="Hello", reasoning="still secret"),
                    finish_reason=None,
                )
            ],
            usage=None,
            id="gen-1",
            model="org/model",
            error=None,
            provider=None,
        ),
    ]

    async def run() -> list[str]:
        return [tok async for tok in _tokens()]

    assert asyncio.run(run()) == ["Hello"]


def test_one_content_object_is_spoken(fake_openrouter: type[_FakeOpenRouter]) -> None:
    fake_openrouter.chunks = [
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content={"type": "text", "text": "Hello"}),
                    finish_reason=None,
                )
            ],
            usage=None,
            id="gen-1",
            model="org/model",
            error=None,
            provider=None,
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content={"type": "reasoning", "text": "secret"}),
                    finish_reason=None,
                )
            ],
            usage=None,
            id="gen-1",
            model="org/model",
            error=None,
            provider=None,
        ),
    ]

    async def run() -> list[str]:
        return [tok async for tok in _tokens()]

    assert asyncio.run(run()) == ["Hello"]


def test_openrouter_message_content_when_delta_empty(
    fake_openrouter: type[_FakeOpenRouter],
) -> None:
    fake_openrouter.chunks = [
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content=None),
                    message=SimpleNamespace(content="from-message."),
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

    assert asyncio.run(run()) == ["from-message."]
    assert len(_FakeChat.calls) == 1


def test_refusal_is_spoken_when_content_is_empty(
    fake_openrouter: type[_FakeOpenRouter],
) -> None:
    fake_openrouter.chunks = [
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content=None, refusal="I can't do that."),
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

    assert asyncio.run(run()) == ["I can't do that."]
    fake_openrouter.chunks = [
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content=""),
                    message=SimpleNamespace(content=None, refusal="I can't do that."),
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
    assert asyncio.run(run()) == ["I can't do that."]


def test_blank_first_delta_keeps_the_message(
    fake_openrouter: type[_FakeOpenRouter],
) -> None:
    fake_openrouter.chunks = [
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content=""),
                    message=SimpleNamespace(content="from-message"),
                    finish_reason=None,
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


def test_empty_delta_does_not_repeat_the_full_message(
    fake_openrouter: type[_FakeOpenRouter],
) -> None:
    fake_openrouter.chunks = [
        _chunk("Hello."),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content=""),
                    message=SimpleNamespace(content="Hello."),
                    finish_reason="stop",
                )
            ],
            usage=None,
            id="gen-1",
            model="org/model",
            error=None,
            provider=None,
        ),
    ]

    async def run() -> list[str]:
        return [tok async for tok in _tokens()]

    assert asyncio.run(run()) == ["Hello."]
    assert len(_FakeChat.calls) == 1


def test_null_delta_after_tokens_does_not_repeat_the_message(
    fake_openrouter: type[_FakeOpenRouter],
) -> None:
    fake_openrouter.chunks = [
        _chunk("Hello."),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content=None),
                    message=SimpleNamespace(content="Hello."),
                    finish_reason="stop",
                )
            ],
            usage=None,
            id="gen-1",
            model="org/model",
            error=None,
            provider=None,
        ),
    ]

    async def run() -> list[str]:
        return [tok async for tok in _tokens()]

    assert asyncio.run(run()) == ["Hello."]
    assert len(_FakeChat.calls) == 1


def _stopped(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content=text),
                finish_reason="stop",
            )
        ],
        usage=None,
        id="gen-1",
        model="org/model",
        error=None,
        provider=None,
    )


def test_unfinished_stop_continues_once(fake_openrouter: type[_FakeOpenRouter]) -> None:
    _FakeChat.rounds = [
        [_stopped("Thank you for")],
        [_stopped(" listening.")],
    ]

    async def run() -> list[str]:
        return [tok async for tok in _tokens()]

    assert asyncio.run(run()) == ["Thank you for", " listening."]
    assert len(_FakeChat.calls) == 2
    assert _FakeChat.calls[1]["messages"][-1] == {
        "role": "assistant",
        "content": "Thank you for",
    }


def test_echoed_prefill_is_not_spoken_twice(fake_openrouter: type[_FakeOpenRouter]) -> None:
    _FakeChat.rounds = [
        [_stopped("Thank you for")],
        [_stopped("Thank you for listening.")],
    ]

    async def run() -> str:
        return "".join([tok async for tok in _tokens()])

    assert asyncio.run(run()) == "Thank you for listening."
    assert len(_FakeChat.calls) == 2


def test_a_new_document_is_not_appended(fake_openrouter: type[_FakeOpenRouter]) -> None:
    essay = "Révolution industrielle et transformations sociales. " * 4
    _FakeChat.rounds = [
        [_stopped("Alright, honey. You can")],
        [_stopped(essay)],
    ]

    async def run() -> str:
        return "".join([tok async for tok in _tokens()])

    assert asyncio.run(run()) == "Alright, honey. You can"


def test_continuation_stops_at_the_first_sentence(
    fake_openrouter: type[_FakeOpenRouter],
) -> None:
    _FakeChat.rounds = [
        [_stopped("Thank you for")],
        [_stopped(" listening. Then a second sentence that must not be spoken.")],
    ]

    async def run() -> str:
        return "".join([tok async for tok in _tokens()])

    assert asyncio.run(run()) == "Thank you for listening."


def test_cancel_returns_while_the_next_token_is_still_pending() -> None:
    # The server sends one token and then nothing. Cancel has to end the turn anyway.
    first = b'data: {"choices":[{"delta":{"content":"one"}}]}\n\n'

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=_Stalls(first, then_fail=False))

    assert _over_http(handler, cancel_after_first=True) == ["one"]


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
    asyncio.run(check_openrouter_model(client, _tune()))
    assert seen == {"author": "org", "slug": "model:nitro"}


def test_check_openrouter_model_reads_a_dict_reply() -> None:
    async def get_async(**_kwargs: Any) -> Any:
        return {"data": {"id": "org/other", "name": "Other", "context_length": 8000}}

    seen: list[Event] = []
    unsubscribe = EVENTS.subscribe(seen.append)
    client = SimpleNamespace(models=SimpleNamespace(get_async=get_async))
    try:
        asyncio.run(check_openrouter_model(client, _tune()))
    finally:
        unsubscribe()
    assert seen == [Notice("[llm model org/model:nitro → org/other]")]


def test_a_failed_read_after_the_consumer_stops_is_not_logged() -> None:
    first = b'data: {"choices":[{"delta":{"content":"one"}}]}\n\n'

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=_Stalls(first, then_fail=True))

    reported: list[dict[str, Any]] = []

    async def run() -> None:
        asyncio.get_running_loop().set_exception_handler(lambda _loop, ctx: reported.append(ctx))
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            stream = cast(
                AsyncGenerator[str, None],
                llm_token_stream(
                    [{"role": "user", "content": "hi"}], tune=_openai_tune(), http=http
                ),
            )
            assert await anext(stream) == "one"
            # The read already in flight fails while nobody is waiting for it.
            await asyncio.sleep(0.01)
            await stream.aclose()
            del stream
            gc.collect()
            for _ in range(5):
                await asyncio.sleep(0)

    asyncio.run(run())
    assert reported == []


def test_check_openrouter_model_skips_unsplit_slug() -> None:
    async def get_async(**_kwargs: Any) -> Any:
        raise AssertionError("must not call get")

    client = SimpleNamespace(models=SimpleNamespace(get_async=get_async))
    asyncio.run(check_openrouter_model(client, _tune(model="norgslash")))


def _events_once_429(wait: float | None, then: list[str]):
    calls = {"n": 0}

    def events(call: Any) -> Any:
        async def gen() -> Any:
            calls["n"] += 1
            if calls["n"] == 1:
                raise ChatHttpError(429, wait, quota_exhausted=False)
            for text in then:
                yield ChatChunk(text=text)

        return gen()

    return calls, events


def test_429_within_cap_retries_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls, events = _events_once_429(4.0, ["back"])
    waits: list[float] = []

    async def pause(seconds: float, _stopped: Callable[[], bool]) -> bool:
        waits.append(seconds)
        return False

    monkeypatch.setattr("fish_audio_suite_voice.llm.client.chat_events", events)
    monkeypatch.setattr("fish_audio_suite_voice.llm.client.sleep_unless", pause)

    seen: list[Event] = []
    unsubscribe = EVENTS.subscribe(seen.append)

    async def collect() -> list[str]:
        return [
            tok
            async for tok in llm_token_stream(
                [{"role": "user", "content": "hi"}],
                tune=LlmTune(
                    base="https://api.example.com/v1",
                    api_key="sk-test",
                    model="org/model",
                ),
            )
        ]

    try:
        assert asyncio.run(collect()) == ["back"]
    finally:
        unsubscribe()
    assert calls["n"] == 2
    assert waits == [4.0]
    assert seen == [Notice("[llm 429, retrying in 4s]")]


def test_429_over_cap_returns_to_listening(monkeypatch: pytest.MonkeyPatch) -> None:
    calls, events = _events_once_429(31.0, ["back"])
    monkeypatch.setattr("fish_audio_suite_voice.llm.client.chat_events", events)

    async def collect() -> list[str]:
        return [
            tok
            async for tok in llm_token_stream(
                [{"role": "user", "content": "hi"}],
                tune=LlmTune(
                    base="https://api.example.com/v1",
                    api_key="sk-test",
                    model="org/model",
                ),
            )
        ]

    assert asyncio.run(collect()) == []
    assert calls["n"] == 1


def test_429_without_retry_after_retries_once_after_a_short_pause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A shared provider pool (one OpenRouter host) sends a bare 429 that clears fast.
    calls, events = _events_once_429(None, ["back"])
    monkeypatch.setattr("fish_audio_suite_voice.llm.client.chat_events", events)
    waits: list[float] = []

    async def no_sleep(seconds: float, _stopped: Callable[[], bool]) -> bool:
        waits.append(seconds)
        return False

    monkeypatch.setattr("fish_audio_suite_voice.llm.client.sleep_unless", no_sleep)

    async def collect() -> list[str]:
        return [
            tok
            async for tok in llm_token_stream(
                [{"role": "user", "content": "hi"}],
                tune=LlmTune(
                    base="https://api.example.com/v1",
                    api_key="sk-test",
                    model="org/model",
                ),
            )
        ]

    assert asyncio.run(collect()) == ["back"]
    assert calls["n"] == 2
    assert waits == [1.0]


def test_a_quota_429_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def events(call: Any) -> Any:
        async def gen() -> Any:
            calls["n"] += 1
            raise ChatHttpError(429, None, quota_exhausted=True)
            yield ChatChunk()

        return gen()

    monkeypatch.setattr("fish_audio_suite_voice.llm.client.chat_events", events)

    async def collect() -> list[str]:
        return [
            tok
            async for tok in llm_token_stream(
                [{"role": "user", "content": "hi"}],
                tune=LlmTune(
                    base="https://api.example.com/v1",
                    api_key="sk-test",
                    model="org/model",
                ),
            )
        ]

    assert asyncio.run(collect()) == []
    assert calls["n"] == 1


def test_a_429_wait_cut_short_does_not_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    calls, events = _events_once_429(4.0, ["back"])

    async def pause(_seconds: float, _stopped: Callable[[], bool]) -> bool:
        return True  # the wait reports it was stopped

    monkeypatch.setattr("fish_audio_suite_voice.llm.client.chat_events", events)
    monkeypatch.setattr("fish_audio_suite_voice.llm.client.sleep_unless", pause)

    async def collect() -> list[str]:
        return [
            tok
            async for tok in llm_token_stream(
                [{"role": "user", "content": "hi"}], tune=_openai_tune(), cancel=asyncio.Event()
            )
        ]

    assert asyncio.run(collect()) == []
    assert calls["n"] == 1


def test_cancel_wakes_the_429_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    _calls, events = _events_once_429(4.0, ["back"])
    cancel = asyncio.Event()
    woken: list[bool] = []

    async def pause(_seconds: float, stopped: Callable[[], bool]) -> bool:
        # Ctrl+C lands during the wait: the wait's stop check has to see it.
        woken.append(stopped())
        cancel.set()
        woken.append(stopped())
        return stopped()

    monkeypatch.setattr("fish_audio_suite_voice.llm.client.chat_events", events)
    monkeypatch.setattr("fish_audio_suite_voice.llm.client.sleep_unless", pause)

    async def collect() -> list[str]:
        return [
            tok
            async for tok in llm_token_stream(
                [{"role": "user", "content": "hi"}], tune=_openai_tune(), cancel=cancel
            )
        ]

    assert asyncio.run(collect()) == []
    assert woken == [False, True]


def test_429_after_a_token_does_not_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def events(call: Any) -> Any:
        async def gen() -> Any:
            calls["n"] += 1
            yield ChatChunk(text="partial")
            raise ChatHttpError(429, 2.0, quota_exhausted=False)

        return gen()

    monkeypatch.setattr("fish_audio_suite_voice.llm.client.chat_events", events)

    async def collect() -> list[str]:
        return [
            tok
            async for tok in llm_token_stream(
                [{"role": "user", "content": "hi"}],
                tune=LlmTune(
                    base="https://api.example.com/v1",
                    api_key="sk-test",
                    model="org/model",
                ),
            )
        ]

    assert asyncio.run(collect()) == ["partial"]
    assert calls["n"] == 1


_OK_SSE = 'data: {"choices":[{"delta":{"content":"back"}}]}\n\n'


def _refused_then_ok(
    status: int, body: str, headers: dict[str, str] | None = None
) -> tuple[list[httpx.Request], Handler]:
    """A server that refuses the first request and answers the next."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if len(seen) == 1:
            return httpx.Response(status, text=body, headers=headers)
        return httpx.Response(200, text=_OK_SSE)

    return seen, handler


@pytest.fixture
def waits(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record each retry wait instead of sleeping."""
    asked: list[float] = []

    async def pause(seconds: float, _stopped: Callable[[], bool]) -> bool:
        asked.append(seconds)
        return False

    monkeypatch.setattr("fish_audio_suite_voice.llm.client.sleep_unless", pause)
    return asked


def test_a_429_waits_for_its_retry_after_header_before_the_body(waits: list[float]) -> None:
    body = '{"error":{"metadata":{"retry_after_seconds":2}}}'
    seen, handler = _refused_then_ok(429, body, {"Retry-After": "12"})
    assert _over_http(handler) == ["back"]
    assert len(seen) == 2
    assert waits == [12.0]


@pytest.mark.parametrize(
    ("value", "wait"),
    [
        (8, 8.0),
        ("1.5", 1.5),
        # An unusable wait in the body is ignored, so the bare-429 pause applies.
        (-3, 1.0),
        (10**400, 1.0),
        (True, 1.0),
        ("soon", 1.0),
    ],
)
def test_a_retry_wait_in_the_error_body_is_read_the_way_a_header_is(
    waits: list[float], value: object, wait: float
) -> None:
    body = json.dumps({"error": {"metadata": {"retry_after_seconds": value}}})
    seen, handler = _refused_then_ok(429, body)
    assert _over_http(handler) == ["back"]
    assert len(seen) == 2
    assert waits == [wait]


def test_openrouter_429_header_retries_once(monkeypatch: pytest.MonkeyPatch) -> None:
    req = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    err = OpenRouterError(
        "slow",
        httpx.Response(429, headers={"Retry-After": "3"}, request=req),
        body="{}",
    )
    calls = {"n": 0}

    class _Chat:
        async def send_async(self, **_kwargs: Any) -> Any:
            calls["n"] += 1
            if calls["n"] == 1:
                raise err
            return _FakeStream([_chunk("ok")])

    class _Client:
        def __init__(self, **_kwargs: Any) -> None:
            self.chat = _Chat()

        async def __aenter__(self) -> _Client:
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

    async def pause(seconds: float, _stopped: Callable[[], bool]) -> bool:
        assert seconds == 3.0
        return False

    monkeypatch.setattr(openrouter, "OpenRouter", _Client)
    monkeypatch.setattr("fish_audio_suite_voice.llm.client.sleep_unless", pause)

    async def collect() -> list[str]:
        return [tok async for tok in _tokens()]

    assert asyncio.run(collect()) == ["ok"]
    assert calls["n"] == 2


def test_a_retry_wait_on_another_status_is_not_honoured(waits: list[float]) -> None:
    seen, handler = _refused_then_ok(500, '{"retry_after_seconds":2}', {"Retry-After": "2"})
    assert _over_http(handler) == []
    assert len(seen) == 1
    assert waits == []


def test_log_briefs_show_the_provider_and_the_token_counts() -> None:
    from fish_audio_suite_voice.llm.transports import provider_brief, usage_brief

    assert provider_brief("available=1, selected=Venice") == "Venice"
    assert provider_brief("Venice") == "Venice"
    assert provider_brief(None) == ""
    usage = {"prompt_tokens": 203, "completion_tokens": 20, "cost": 5.86e-05}
    assert usage_brief(usage) == "in=203 out=20 cost=$0.00006"
    assert usage_brief(None) == "usage=?"
    # A cache hit shows how much of the prompt the provider read back, so a change
    # that breaks the cache can be seen.
    cached = {**usage, "prompt_tokens_details": {"cached_tokens": 180}}
    assert usage_brief(cached) == "in=203 cached=180 out=20 cost=$0.00006"
    no_hit = {**usage, "prompt_tokens_details": {"cached_tokens": 0}}
    assert usage_brief(no_hit) == "in=203 cached=0 out=20 cost=$0.00006"


def test_a_429_that_says_the_quota_is_spent_is_not_retried(waits: list[float]) -> None:
    quota = '{"error":{"code":"insufficient_quota","message":"free_limit_reached"}}'
    seen, handler = _refused_then_ok(429, quota)
    assert _over_http(handler) == []
    assert len(seen) == 1
    assert waits == []
    limited = '{"error":{"message":"temporarily rate-limited upstream"}}'
    seen, handler = _refused_then_ok(429, limited)
    assert _over_http(handler) == ["back"]
    assert len(seen) == 2


def test_a_wrapped_provider_429_becomes_one_readable_line() -> None:
    # The body OpenRouter returned when Parasail was rate limited.
    body = json.dumps(
        {
            "error": {
                "message": "Provider returned error",
                "code": 429,
                "metadata": {
                    "raw": "thedrummer/cydonia-24b-v4.1 is temporarily rate-limited upstream. Please retry shortly",
                    "provider_name": "Parasail",
                },
            },
            "user_id": "user_123",
        }
    )
    line = describe_http_error(429, "thedrummer/cydonia-24b-v4.1", body)
    assert line.startswith("HTTP 429 from Parasail, model=thedrummer/cydonia-24b-v4.1: ")
    assert "temporarily rate-limited upstream" in line
    assert "\n" not in line
    assert "user_123" not in line


def test_a_provider_refusal_shows_what_the_provider_said() -> None:
    # xAI answered 403 with its own JSON inside OpenRouter's.
    inner = json.dumps(
        {"code": "permission-denied", "error": "I'm sorry, I can't help with that request."}
    )
    body = json.dumps(
        {
            "error": {
                "message": "Provider returned error",
                "code": 403,
                "metadata": {"raw": inner, "provider_name": "xAI"},
            }
        }
    )
    assert (
        describe_http_error(403, "x-ai/grok-4.20", body)
        == "HTTP 403 from xAI, model=x-ai/grok-4.20: I'm sorry, I can't help with that request."
    )


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (
            '{"error":{"message":"No auth credentials found","code":401}}',
            "No auth credentials found",
        ),
        ('{"error":"insufficient_quota"}', "insufficient_quota"),
        ("<html>Bad gateway</html>", "<html>Bad gateway</html>"),
        ("", "no detail in the reply"),
        ("[1, 2]", "[1, 2]"),
    ],
)
def test_other_error_bodies_still_give_a_short_line(body: str, expected: str) -> None:
    line = describe_http_error(500, "m", body)
    assert line.endswith(expected)
    assert line.startswith("HTTP 500, model=m: ")


def test_a_huge_error_body_is_cut_to_one_short_line() -> None:
    line = describe_http_error(500, "m", "x" * 5000)
    assert len(line) < 300


def test_an_empty_raw_field_keeps_the_message_the_reply_did_have() -> None:
    body = json.dumps(
        {
            "error": {
                "message": "Provider returned error",
                "metadata": {"raw": "", "provider_name": "X"},
            }
        }
    )
    assert (
        describe_http_error(502, "m", body) == "HTTP 502 from X, model=m: Provider returned error"
    )


def test_a_deeply_nested_error_body_still_gives_one_line() -> None:
    line = describe_http_error(502, "some/model", "[" * 20_000)
    assert line.startswith("HTTP 502, model=some/model: [[[")
    assert "\n" not in line


@pytest.mark.parametrize(
    ("raw", "sort"),
    [
        (None, "latency"),
        ("Throughput", "throughput"),
        ("price", "price"),
        ("off", ""),
        ("fastest", "latency"),
    ],
)
def test_provider_sort_reads_the_env(
    monkeypatch: pytest.MonkeyPatch, raw: str | None, sort: str
) -> None:
    if raw is None:
        monkeypatch.delenv("FISH_LLM_PROVIDER_SORT", raising=False)
    else:
        monkeypatch.setenv("FISH_LLM_PROVIDER_SORT", raw)
    assert LlmTune.from_env().provider_sort == sort


def _refused_once(status: int) -> Any:
    def events(_call: Any) -> Any:
        async def gen() -> AsyncIterator[ChatChunk]:
            raise ChatHttpError(status, None, quota_exhausted=False)
            yield ChatChunk()

        return gen()

    return events


async def _drain(stream: AsyncIterator[str]) -> list[str]:
    return [tok async for tok in stream]


def _hi() -> AsyncIterator[str]:
    return llm_token_stream(
        [{"role": "user", "content": "hi"}],
        tune=LlmTune(base="https://api.example.com/v1", api_key="sk-test", model="org/gone"),
    )


@pytest.mark.parametrize(
    ("status", "says"),
    [
        (401, "the key was refused"),
        (402, "out of credits"),
        (404, "no provider serves model org/gone"),
    ],
)
def test_a_refusal_that_every_turn_would_get_stops_the_stream(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], status: int, says: str
) -> None:
    monkeypatch.setattr("fish_audio_suite_voice.llm.client.chat_events", _refused_once(status))
    with pytest.raises(ChatUnusableError, match=says):
        asyncio.run(_drain(_hi()))
    assert says in capsys.readouterr().err


@pytest.mark.parametrize("status", [400, 403, 500, 503])
def test_a_refusal_of_one_request_ends_only_that_reply(
    monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    monkeypatch.setattr("fish_audio_suite_voice.llm.client.chat_events", _refused_once(status))
    assert asyncio.run(_drain(_hi())) == []


def _lookup_client(*, endpoints: Any) -> Any:
    async def get_async(**_kwargs: Any) -> Any:
        return SimpleNamespace(data=SimpleNamespace(id="org/model", name="M", context_length=8))

    return SimpleNamespace(
        models=SimpleNamespace(get_async=get_async),
        endpoints=SimpleNamespace(list_async=endpoints),
    )


def test_a_model_with_no_provider_is_refused_before_the_session_starts(
    capsys: pytest.CaptureFixture[str],
) -> None:
    seen: dict[str, str] = {}

    async def none_serve(*, author: str, slug: str, **_kwargs: Any) -> Any:
        seen.update(author=author, slug=slug)
        return SimpleNamespace(data=SimpleNamespace(endpoints=[]))

    with pytest.raises(ChatUnusableError, match="no provider serves model org/model"):
        asyncio.run(check_openrouter_model(_lookup_client(endpoints=none_serve), _tune()))
    # Asked about the model itself: :nitro is a way of routing, not a model.
    assert seen == {"author": "org", "slug": "model"}
    assert "no provider serves model org/model" in capsys.readouterr().err


def test_a_model_with_a_provider_starts_and_a_failed_lookup_does_not_block() -> None:
    async def one_serves(**_kwargs: Any) -> Any:
        return SimpleNamespace(data=SimpleNamespace(endpoints=[SimpleNamespace(name="P")]))

    async def lookup_fails(**_kwargs: Any) -> Any:
        raise httpx.ConnectError("offline")

    asyncio.run(check_openrouter_model(_lookup_client(endpoints=one_serves), _tune()))
    asyncio.run(check_openrouter_model(_lookup_client(endpoints=lookup_fails), _tune()))


def test_an_unknown_model_is_refused_before_the_session_starts() -> None:
    async def get_async(**_kwargs: Any) -> Any:
        req = httpx.Request("GET", "https://openrouter.ai/api/v1/model/org/missing")
        raise OpenRouterError("missing", httpx.Response(404, request=req), body="nope")

    client = SimpleNamespace(models=SimpleNamespace(get_async=get_async))
    with pytest.raises(ChatUnusableError, match="unknown model org/missing"):
        asyncio.run(check_openrouter_model(client, _tune(model="org/missing")))


def test_a_service_tier_is_sent_only_when_one_is_chosen(
    fake_openrouter: type[_FakeOpenRouter],
) -> None:
    del fake_openrouter

    async def run(**extra: Any) -> None:
        async for _tok in _tokens(**extra):
            pass

    asyncio.run(run())
    assert _FakeChat.last_kw is not None
    assert "service_tier" not in _FakeChat.last_kw
    asyncio.run(run(tune=_tune(service_tier="ultrafast")))
    assert _FakeChat.last_kw["service_tier"] == "ultrafast"


def test_the_tier_that_served_the_reply_is_logged(
    fake_openrouter: type[_FakeOpenRouter], capsys: pytest.CaptureFixture[str]
) -> None:
    served = _chunk("Hi.")
    served.service_tier = "priority"  # ultrafast was asked for and fell back
    fake_openrouter.chunks = [served]
    configure_voice_logging(debug=1)

    async def run() -> None:
        async for _tok in _tokens(tune=_tune(service_tier="ultrafast")):
            pass

    try:
        asyncio.run(run())
        lines = capsys.readouterr().err.splitlines()
    finally:
        configure_voice_logging(debug=False)
    assert any("stream_end" in line and "tier=priority" in line for line in lines)
