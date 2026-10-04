from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import pytest

from fish_audio_suite_voice.cancel import is_own_cancel, reap
from fish_audio_suite_voice.llm import llm_token_stream


def test_is_own_cancel_needs_a_set_flag() -> None:
    async def check() -> tuple[bool, bool, bool]:
        return (
            is_own_cancel(None),
            is_own_cancel(asyncio.Event()),
            is_own_cancel(threading.Event()),
        )

    assert asyncio.run(check()) == (False, False, False)


def test_is_own_cancel_accepts_either_kind_of_event() -> None:
    async def check() -> tuple[bool, bool]:
        aio = asyncio.Event()
        aio.set()
        thread = threading.Event()
        thread.set()
        return is_own_cancel(aio), is_own_cancel(thread)

    assert asyncio.run(check()) == (True, True)


def test_is_own_cancel_is_false_while_the_task_has_a_pending_cancellation() -> None:
    async def run() -> None:
        flag = asyncio.Event()
        flag.set()
        seen: list[bool] = []

        async def child() -> None:
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                seen.append(is_own_cancel(flag))
                raise

        task = asyncio.create_task(child())
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert seen == [False]

    asyncio.run(run())


def test_reap_waits_for_a_cancelled_child() -> None:
    async def run() -> None:
        async def child() -> None:
            try:
                await asyncio.sleep(30)
            finally:
                await asyncio.sleep(0.02)

        task = asyncio.create_task(child())
        await asyncio.sleep(0)
        task.cancel()
        await reap(task)
        assert task.done()
        assert task.cancelled()

    asyncio.run(run())


def test_reap_reads_a_childs_own_error_without_raising() -> None:
    async def run() -> None:
        async def child() -> None:
            raise RuntimeError("boom")

        task = asyncio.create_task(child())
        await reap(task)
        assert isinstance(task.exception(), RuntimeError)

    asyncio.run(run())


def test_reap_does_not_swallow_a_cancel_of_the_caller() -> None:
    async def run() -> None:
        sleeper = asyncio.create_task(asyncio.sleep(30))
        waiter = asyncio.create_task(reap(sleeper))
        await asyncio.sleep(0.02)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        sleeper.cancel()
        await reap(sleeper)

    asyncio.run(run())


def test_reap_gives_up_after_the_wait_and_leaves_the_task_running() -> None:
    async def run() -> None:
        sleeper = asyncio.create_task(asyncio.sleep(30))
        await reap(sleeper, wait_s=0.02)
        assert not sleeper.done()
        sleeper.cancel()
        await reap(sleeper)

    asyncio.run(run())


def _chunk(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=text), finish_reason=None)],
        usage=None,
        id="gen-1",
        model="org/model",
        error=None,
        provider=None,
    )


def _events_then(exc: BaseException, cancel: asyncio.Event | None = None) -> Any:
    async def events() -> AsyncIterator[Any]:
        yield _chunk("Hello")
        if cancel is not None:
            cancel.set()
        raise exc

    return events()


def _stream(
    monkeypatch: pytest.MonkeyPatch,
    events: Any,
    cancel: asyncio.Event | None,
) -> AsyncIterator[str]:
    monkeypatch.setattr("fish_audio_suite_voice.llm.chat_events", lambda _call: events)
    return llm_token_stream(
        [{"role": "user", "content": "hi"}],
        base="https://api.example.com/v1",
        key="sk-test",
        model="org/model",
        cancel=cancel,
    )


def test_llm_stream_ends_quietly_when_our_own_cancel_flag_is_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cancel = asyncio.Event()
    stream = _stream(monkeypatch, _events_then(asyncio.CancelledError(), cancel), cancel)

    async def collect() -> list[str]:
        return [piece async for piece in stream]

    assert asyncio.run(collect()) == ["Hello"]


def test_llm_stream_lets_an_unrelated_cancellation_propagate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = _stream(monkeypatch, _events_then(asyncio.CancelledError()), asyncio.Event())

    async def collect() -> list[str]:
        return [piece async for piece in stream]

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(collect())


def test_llm_stream_lets_asyncio_timeout_fire(monkeypatch: pytest.MonkeyPatch) -> None:
    async def stalled() -> AsyncIterator[Any]:
        await asyncio.sleep(30)
        yield None

    stream = _stream(monkeypatch, stalled(), asyncio.Event())

    async def collect() -> None:
        async with asyncio.timeout(0.1):
            async for _piece in stream:
                pass

    with pytest.raises(TimeoutError):
        asyncio.run(collect())
