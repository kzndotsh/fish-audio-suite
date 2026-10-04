from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any, ClassVar

import httpx
import openrouter
import pytest
from openrouter.errors import OpenRouterError

from fish_audio_suite_voice.llm import (
    _ChatStats,
    _consume_chat_events,
    _model_author_slug,
    _want_nitro,
    check_openrouter_model,
    llm_token_stream,
)
from fish_audio_suite_voice.transports import (
    ChatCall,
    _abort_http,
    _chat_body,
    _feed_sse,
    describe_http_error,
    openrouter_client,
)
from fish_audio_suite_voice.tune import LlmTune, is_openrouter_host

OR_BASE = "https://openrouter.ai/api/v1"


def _tune(**kw: Any) -> LlmTune:
    fields: dict[str, Any] = {
        "backend": "openrouter",
        "base": OR_BASE,
        "api_key": "sk-test",
        "model": "org/model",
        "nitro": True,
        "continuation": True,
    }
    fields.update(kw)
    return LlmTune(**fields)


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
    call = ChatCall(
        messages=[{"role": "system", "content": "Be brief. \ud800 Answer plainly."}],
        tune=_tune(max_tokens=100),
        route_model="model",
        client=None,
        http=None,
        session_id=None,
        trace_id=None,
        stats=_ChatStats(),
    )
    body = _chat_body(call, "max_tokens")
    request = httpx.Request(
        "POST",
        "https://example.com/v1/chat/completions",
        json=body,
    )
    request.read()
    text = body["messages"][0]["content"]
    assert "\ud800" not in text
    assert "Be brief." in text
    assert "Answer plainly." in text


def test_sse_bom_does_not_drop_the_first_token() -> None:
    buf, event, stop = _feed_sse(
        "",
        '\ufeffdata: {"choices":[{"delta":{"content":"Hello there"}}]}',
    )
    assert stop is False
    assert buf == ""
    assert event is not None
    assert event["choices"][0]["delta"]["content"] == "Hello there"
    _, done, stopped = _feed_sse("", "\ufeffdata: [DONE]")
    assert done is None
    assert stopped is True


def test_a_provider_error_group_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    notes: list[str] = []
    monkeypatch.setattr(
        "fish_audio_suite_voice.llm.warn", lambda message: notes.append(str(message))
    )

    async def events() -> Any:
        yield _chunk("Hello there")
        raise ExceptionGroup("llm", [RuntimeError("provider down")])

    monkeypatch.setattr("fish_audio_suite_voice.llm.chat_events", lambda _call: events())

    async def collect() -> list[str]:
        return [
            tok
            async for tok in llm_token_stream(
                [{"role": "user", "content": "hi"}],
                base="https://api.example.com/v1",
                key="sk-test",
                model="org/model",
            )
        ]

    assert asyncio.run(collect()) == ["Hello there"]
    assert any("provider down" in note for note in notes)


def test_a_cancel_group_does_not_warn(monkeypatch: pytest.MonkeyPatch) -> None:
    notes: list[str] = []
    monkeypatch.setattr(
        "fish_audio_suite_voice.llm.warn", lambda message: notes.append(str(message))
    )

    async def events() -> Any:
        yield _chunk("Hello there")
        raise BaseExceptionGroup("cancel", [asyncio.CancelledError()])

    monkeypatch.setattr("fish_audio_suite_voice.llm.chat_events", lambda _call: events())

    async def collect() -> list[str]:
        return [
            tok
            async for tok in llm_token_stream(
                [{"role": "user", "content": "hi"}],
                base="https://api.example.com/v1",
                key="sk-test",
                model="org/model",
            )
        ]

    assert asyncio.run(collect()) == ["Hello there"]
    assert notes == []


def test_split_sse_data_lines_stay_one_event() -> None:
    lines = [
        'data: {"choices":[{"delta":{"content":',
        'data: "Hello"}}]}',
        "",
        'data: {"choices":[{"delta":{"content":" there"}}]}',
        "data: [DONE]",
        'data: {"choices":[{"delta":{"content":"dropped"}}]}',
    ]
    events: list[dict[str, Any]] = []
    buf = ""
    for line in lines:
        buf, event, stop = _feed_sse(buf, line)
        if event is not None:
            events.append(event)
        if stop:
            break
    assert [event["choices"][0]["delta"]["content"] for event in events] == [
        "Hello",
        " there",
    ]


def _tokens(**extra: Any):
    tune = extra.pop("tune", None) or _tune()
    return llm_token_stream([{"role": "user", "content": "hi"}], tune=tune, **extra)


def test_want_nitro_slash_model_on_openrouter() -> None:
    assert _want_nitro("org/model", _tune())


def test_want_nitro_false_when_suffix_present() -> None:
    assert not _want_nitro("org/model:nitro", _tune())


def test_want_nitro_is_off_unless_asked() -> None:
    assert not _want_nitro("org/model", _tune(nitro=False))
    assert not LlmTune().nitro


def test_want_nitro_false_for_non_openrouter_backend() -> None:
    other = _tune(backend="openai", base="https://api.example.com/v1")
    assert not _want_nitro("org/model", other)


def test_openrouter_host_checks_the_hostname_not_the_text() -> None:
    assert is_openrouter_host("https://openrouter.ai/api/v1")
    assert is_openrouter_host("https://eu.openrouter.ai/api/v1")
    assert not is_openrouter_host("https://api.example.com/v1?via=openrouter.ai")
    assert not is_openrouter_host("https://notopenrouter.ai/v1")
    assert not is_openrouter_host("http://localhost:11434/v1")


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
    assert _FakeChat.last_kw["provider"] == {"sort": "throughput"}
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
    assert "provider" not in _FakeChat.last_kw
    assert _FakeChat.last_kw["model"] == "org/model"
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
    # The next chunk never arrives. Cancel has to end the turn anyway.
    cancel = asyncio.Event()

    async def stalled() -> AsyncIterator[Any]:
        yield _chunk("one")
        await asyncio.Event().wait()

    async def run() -> list[str]:
        pieces: list[str] = []
        async for piece in _consume_chat_events(stalled(), cancel, _ChatStats()):
            pieces.append(piece)
            cancel.set()
        return pieces

    assert asyncio.run(asyncio.wait_for(run(), 1)) == ["one"]


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


def test_check_openrouter_model_404(capsys: pytest.CaptureFixture[str]) -> None:
    async def get_async(**_kwargs: Any) -> Any:
        req = httpx.Request("GET", "https://openrouter.ai/api/v1/model/org/missing")
        raise OpenRouterError("missing", httpx.Response(404, request=req), body="nope")

    client = SimpleNamespace(models=SimpleNamespace(get_async=get_async))
    asyncio.run(check_openrouter_model(client, _tune(model="org/missing")))
    assert "unknown model org/missing:nitro" in capsys.readouterr().err


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
                call.stats.aborted = True
                call.stats.http_status = 429
                call.stats.retry_after_s = wait
            else:
                for text in then:
                    yield _chunk(text)

        return gen()

    return calls, events


def test_429_within_cap_retries_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls, events = _events_once_429(4.0, ["back"])
    waits: list[float] = []
    printed: list[str] = []

    async def pause(_cancel: asyncio.Event | None, seconds: float) -> bool:
        waits.append(seconds)
        return False

    monkeypatch.setattr("fish_audio_suite_voice.llm.chat_events", events)
    monkeypatch.setattr("fish_audio_suite_voice.llm._pause_for_retry", pause)

    def record_print(*args: object, **_kwargs: object) -> None:
        printed.append(" ".join(str(arg) for arg in args))

    monkeypatch.setattr("fish_audio_suite_voice.llm.console_print", record_print)

    async def collect() -> list[str]:
        return [
            tok
            async for tok in llm_token_stream(
                [{"role": "user", "content": "hi"}],
                base="https://api.example.com/v1",
                key="sk-test",
                model="org/model",
            )
        ]

    assert asyncio.run(collect()) == ["back"]
    assert calls["n"] == 2
    assert waits == [4.0]
    assert printed == ["  [llm 429, retrying in 4s]"]


def test_429_over_cap_returns_to_listening(monkeypatch: pytest.MonkeyPatch) -> None:
    calls, events = _events_once_429(31.0, ["back"])
    monkeypatch.setattr("fish_audio_suite_voice.llm.chat_events", events)

    async def collect() -> list[str]:
        return [
            tok
            async for tok in llm_token_stream(
                [{"role": "user", "content": "hi"}],
                base="https://api.example.com/v1",
                key="sk-test",
                model="org/model",
            )
        ]

    assert asyncio.run(collect()) == []
    assert calls["n"] == 1


def test_429_without_retry_after_retries_once_after_a_short_pause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A shared provider pool (one OpenRouter host) sends a bare 429 that clears fast.
    calls, events = _events_once_429(None, ["back"])
    monkeypatch.setattr("fish_audio_suite_voice.llm.chat_events", events)
    waits: list[float] = []

    async def no_sleep(_cancel: object, seconds: float) -> bool:
        waits.append(seconds)
        return False

    monkeypatch.setattr("fish_audio_suite_voice.llm._pause_for_retry", no_sleep)

    async def collect() -> list[str]:
        return [
            tok
            async for tok in llm_token_stream(
                [{"role": "user", "content": "hi"}],
                base="https://api.example.com/v1",
                key="sk-test",
                model="org/model",
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
            call.stats.aborted = True
            call.stats.http_status = 429
            call.stats.retry_after_s = None
            call.stats.quota_exhausted = True
            for _ in ():
                yield None

        return gen()

    monkeypatch.setattr("fish_audio_suite_voice.llm.chat_events", events)
    monkeypatch.setattr("fish_audio_suite_voice.llm.chat_events", events)

    async def collect() -> list[str]:
        return [
            tok
            async for tok in llm_token_stream(
                [{"role": "user", "content": "hi"}],
                base="https://api.example.com/v1",
                key="sk-test",
                model="org/model",
            )
        ]

    assert asyncio.run(collect()) == []
    assert calls["n"] == 1


def test_cancel_during_429_wait_does_not_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    calls, events = _events_once_429(4.0, ["back"])

    async def pause(cancel: asyncio.Event | None, _seconds: float) -> bool:
        if cancel is not None:
            cancel.set()
        return True

    monkeypatch.setattr("fish_audio_suite_voice.llm.chat_events", events)
    monkeypatch.setattr("fish_audio_suite_voice.llm._pause_for_retry", pause)

    async def collect() -> list[str]:
        return [
            tok
            async for tok in llm_token_stream(
                [{"role": "user", "content": "hi"}],
                base="https://api.example.com/v1",
                key="sk-test",
                model="org/model",
                cancel=asyncio.Event(),
            )
        ]

    assert asyncio.run(collect()) == []
    assert calls["n"] == 1


def test_429_after_a_token_does_not_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def events(call: Any) -> Any:
        async def gen() -> Any:
            calls["n"] += 1
            yield _chunk("partial")
            call.stats.aborted = True
            call.stats.http_status = 429
            call.stats.retry_after_s = 2.0

        return gen()

    monkeypatch.setattr("fish_audio_suite_voice.llm.chat_events", events)

    async def collect() -> list[str]:
        return [
            tok
            async for tok in llm_token_stream(
                [{"role": "user", "content": "hi"}],
                base="https://api.example.com/v1",
                key="sk-test",
                model="org/model",
            )
        ]

    assert asyncio.run(collect()) == ["partial"]
    assert calls["n"] == 1


def test_abort_http_reads_retry_after_header() -> None:
    stats = _ChatStats()
    req = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    headers = httpx.Response(429, headers={"Retry-After": "12"}, request=req).headers
    body = '{"error":{"metadata":{"retry_after_seconds":2}}}'
    _abort_http(stats, 429, "org/model", body, headers)
    assert stats.http_status == 429
    assert stats.aborted is True
    assert stats.retry_after_s == 12.0


def test_abort_http_reads_retry_after_from_body() -> None:
    stats = _ChatStats()
    body = '{"error":{"metadata":{"retry_after_seconds":8}}}'
    _abort_http(stats, 429, "org/model", body, None)
    assert stats.retry_after_s == 8.0


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

    async def pause(_cancel: asyncio.Event | None, seconds: float) -> bool:
        assert seconds == 3.0
        return False

    monkeypatch.setattr(openrouter, "OpenRouter", _Client)
    monkeypatch.setattr("fish_audio_suite_voice.llm._pause_for_retry", pause)

    async def collect() -> list[str]:
        return [tok async for tok in _tokens()]

    assert asyncio.run(collect()) == ["ok"]
    assert calls["n"] == 2


def test_abort_http_ignores_retry_after_on_other_status() -> None:
    stats = _ChatStats()
    _abort_http(stats, 500, "org/model", '{"retry_after_seconds":2}', None)
    assert stats.http_status == 500
    assert stats.retry_after_s is None


def test_log_briefs_show_the_provider_and_the_token_counts() -> None:
    from fish_audio_suite_voice.llm import _provider_brief, _usage_brief

    assert _provider_brief("available=1, selected=Venice") == "Venice"
    assert _provider_brief("Venice") == "Venice"
    assert _provider_brief(None) == ""
    usage = {"prompt_tokens": 203, "completion_tokens": 20, "cost": 5.86e-05}
    assert _usage_brief(usage) == "in=203 out=20 cost=$0.00006"
    assert _usage_brief(None) == "usage=?"


def test_a_retry_wait_in_the_error_body_is_read_the_way_a_header_is() -> None:
    from fish_audio_suite_voice.transports import _retry_after_seconds

    def body(value: object) -> str:
        return json.dumps({"error": {"metadata": {"retry_after_seconds": value}}})

    assert _retry_after_seconds(None, body(8)) == 8.0
    assert _retry_after_seconds(None, body("1.5")) == 1.5
    assert _retry_after_seconds(None, body(-3)) is None
    assert _retry_after_seconds(None, body(10**400)) is None
    assert _retry_after_seconds(None, body(True)) is None
    assert _retry_after_seconds(None, body("soon")) is None
    headers = httpx.Headers({"Retry-After": "12"})
    assert _retry_after_seconds(headers, body(8)) == 12.0


def test_abort_http_marks_a_quota_429() -> None:
    quota = _ChatStats()
    _abort_http(
        quota,
        429,
        "m",
        '{"error":{"code":"insufficient_quota","message":"free_limit_reached"}}',
        None,
    )
    assert quota.quota_exhausted
    limited = _ChatStats()
    _abort_http(
        limited, 429, "m", '{"error":{"message":"temporarily rate-limited upstream"}}', None
    )
    assert not limited.quota_exhausted


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
