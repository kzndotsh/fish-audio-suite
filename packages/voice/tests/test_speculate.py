"""A reply written ahead of time for a turn that is probably over: kept, or thrown away."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest
from session_fakes import FakeBackend, make_ctx

from fish_audio_suite_kit import ChatMessage
from fish_audio_suite_voice.duplex_state import DuplexContext
from fish_audio_suite_voice.reply import collect_reply
from fish_audio_suite_voice.speculate import Speculation, same_words

ASKED: list[ChatMessage] = [{"role": "user", "content": "hello there"}]


def test_transcripts_match_ignoring_case_and_spacing_but_not_words() -> None:
    assert same_words("Hello  there.", "hello there.")
    assert not same_words("hello there", "hello there you")
    assert not same_words("", "")


async def _read(spec: Speculation, cancel: asyncio.Event | None = None) -> list[str]:
    return [token async for token in spec.replay(cancel or asyncio.Event())]


def test_the_tokens_written_before_and_after_it_is_taken_come_out_in_order() -> None:
    async def run() -> list[str]:
        gate = asyncio.Event()

        async def tokens(*_a: object, **_k: object) -> AsyncIterator[str]:
            yield "one "
            await gate.wait()
            yield "two"

        spec = Speculation(FakeBackend(tokens), ASKED, "hello there")
        await asyncio.sleep(0)
        await asyncio.sleep(0)  # "one " is written while nobody is reading
        asyncio.get_running_loop().call_later(0.01, gate.set)
        return await _read(spec)

    assert asyncio.run(run()) == ["one ", "two"]


def test_throwing_it_away_stops_the_model() -> None:
    async def run() -> bool:
        stopped = asyncio.Event()

        async def tokens(
            _messages: list[ChatMessage], *, cancel: asyncio.Event | None = None, **_k: object
        ) -> AsyncIterator[str]:
            try:
                yield "start "
                assert cancel is not None
                await cancel.wait()
                yield "never"
            finally:
                stopped.set()  # however it ended, the model stopped being asked for more

        spec = Speculation(FakeBackend(tokens), ASKED, "hello there")
        await asyncio.sleep(0.01)
        spec.cancel()
        await asyncio.wait_for(stopped.wait(), 1)
        return True

    assert asyncio.run(run())


def test_an_error_while_it_was_being_written_reaches_the_reader() -> None:
    async def run() -> None:
        async def tokens(*_a: object, **_k: object) -> AsyncIterator[str]:
            yield "partial "
            raise RuntimeError("model fell over")

        await _read(Speculation(FakeBackend(tokens), ASKED, "hello there"))

    with pytest.raises(RuntimeError, match="model fell over"):
        asyncio.run(run())


def test_the_reader_can_stop_waiting_when_the_user_interrupts() -> None:
    async def run() -> list[str]:
        async def tokens(*_a: object, **_k: object) -> AsyncIterator[str]:
            yield "hi "
            await asyncio.sleep(10)
            yield "late"

        spec = Speculation(FakeBackend(tokens), ASKED, "hello there")
        cancel = asyncio.Event()
        asyncio.get_running_loop().call_later(0.05, cancel.set)
        return await asyncio.wait_for(_read(spec, cancel), 2)

    assert asyncio.run(run()) == ["hi "]


def _ctx_with(reply: str, asked: str) -> tuple[DuplexContext, list[str]]:
    calls: list[str] = []

    async def tokens(*_a: object, **_k: object) -> AsyncIterator[str]:
        calls.append("model asked")
        yield reply

    ctx = make_ctx(tokens)
    ctx.history.append({"role": "user", "content": asked})
    return ctx, calls


def test_a_confirmed_turn_uses_the_reply_that_was_already_being_written() -> None:
    async def run() -> tuple[str, list[str]]:
        ctx, calls = _ctx_with("Hi back.", "hello there")
        ctx.speculation = Speculation(ctx.backend, list(ctx.history), "Hello there.")
        await asyncio.sleep(0.01)
        reply, _ = await collect_reply(ctx, llm_cancel=asyncio.Event(), trace_id=None, started=0.0)
        assert ctx.speculation is None
        return reply, calls

    reply, calls = asyncio.run(run())
    assert reply == "Hi back."
    assert calls == ["model asked"]  # once: the one written ahead of time


def test_when_the_words_changed_the_reply_is_asked_for_again() -> None:
    async def run() -> tuple[str, list[str]]:
        ctx, calls = _ctx_with("Fresh reply.", "hello there you")
        ctx.speculation = Speculation(ctx.backend, list(ctx.history), "hello there")
        await asyncio.sleep(0.01)
        reply, _ = await collect_reply(ctx, llm_cancel=asyncio.Event(), trace_id=None, started=0.0)
        return reply, calls

    reply, calls = asyncio.run(run())
    assert reply == "Fresh reply."
    assert calls == ["model asked", "model asked"]  # the guess, then the real one
