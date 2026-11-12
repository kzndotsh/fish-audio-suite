from __future__ import annotations

import asyncio
import signal
import threading
from typing import Any

import pytest

from fish_audio_suite_voice.signals import DuplexSession, TurnSignals


def test_install_sigint_uses_the_loop_handler_and_runs_before_first() -> None:
    session = DuplexSession()
    speaking = threading.Event()
    session.turn.bind(speaking, asyncio.Event())
    order: list[str] = []
    registered: dict[str, Any] = {}

    async def scenario() -> None:
        loop = asyncio.get_running_loop()
        original = loop.add_signal_handler

        def capture(sig: int, callback: Any, *args: Any) -> None:
            registered["sig"] = sig
            registered["callback"] = callback

        loop.add_signal_handler = capture
        try:
            session.install_sigint(loop, before=lambda: order.append("before"))
        finally:
            loop.add_signal_handler = original

    previous = signal.getsignal(signal.SIGINT)
    try:
        asyncio.run(scenario())
        assert registered["sig"] == signal.SIGINT
        registered["callback"]()
        assert order == ["before"]
        assert session.quit_requested.is_set()
        # Ctrl+C stops the reply being spoken, not only the mic.
        assert speaking.is_set()
        assert signal.getsignal(signal.SIGINT) == signal.SIG_DFL
    finally:
        signal.signal(signal.SIGINT, previous)


def test_install_sigint_falls_back_to_signal_signal_without_loop_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = DuplexSession()
    installed: dict[str, Any] = {}

    async def scenario() -> None:
        loop = asyncio.get_running_loop()

        def refuse(*_args: Any) -> None:
            raise NotImplementedError

        monkeypatch.setattr(loop, "add_signal_handler", refuse)
        monkeypatch.setattr(
            "fish_audio_suite_voice.signals.signal.signal",
            lambda sig, handler: installed.update(sig=sig, handler=handler),
        )
        session.install_sigint(loop)
        installed["handler"](signal.SIGINT, None)
        await asyncio.sleep(0)

    asyncio.run(scenario())
    assert installed["sig"] == signal.SIGINT
    assert session.quit_requested.is_set()


def test_fire_on_a_closed_loop_sets_the_flag_without_raising() -> None:
    loop = asyncio.new_event_loop()
    event = asyncio.Event()
    waiter_loop_closed = threading.Event()
    session = DuplexSession()
    session.turn.bind(waiter_loop_closed, event, loop)
    loop.close()
    session.turn.fire()
    assert waiter_loop_closed.is_set()
    assert event.is_set()


def test_fire_on_a_stopped_loop_waits_for_the_loop_to_set_the_event() -> None:
    loop = asyncio.new_event_loop()
    try:
        event = asyncio.Event()
        session = DuplexSession()
        session.turn.bind(threading.Event(), event, loop)
        session.turn.fire()
        # Not set from this thread: the call is queued for the loop.
        assert not event.is_set()
        loop.run_until_complete(asyncio.sleep(0))
        assert event.is_set()
    finally:
        loop.close()


def test_cleared_turn_does_not_cancel_the_old_events() -> None:
    turn = threading.Event()
    llm = asyncio.Event()
    signals_ = TurnSignals()
    signals_.fire()
    signals_.bind(turn, llm)
    signals_.clear()
    signals_.fire()
    assert not turn.is_set()
    assert not llm.is_set()


def test_request_quit_sets_mic_and_turn_cancel() -> None:
    session = DuplexSession()
    turn = threading.Event()
    llm = asyncio.Event()
    session.turn.bind(turn, llm)
    session.request_quit()
    assert session.quit_requested.is_set()
    assert turn.is_set()
    assert llm.is_set()


def test_a_turn_bound_after_quit_starts_cancelled() -> None:
    session = DuplexSession()
    session.request_quit()
    turn = threading.Event()
    llm = asyncio.Event()
    session.turn.bind(turn, llm)
    assert turn.is_set()
    assert llm.is_set()


def test_quit_wakes_every_waiter_that_asked() -> None:
    session = DuplexSession()
    woken: list[str] = []
    session.on_quit(lambda: woken.append("input"))
    session.on_quit(lambda: woken.append("screen"))
    session.request_quit()
    assert woken == ["input", "screen"]


def test_request_quit_from_a_thread_sets_the_llm_event_on_its_loop() -> None:
    async def scenario() -> bool:
        session = DuplexSession()
        llm = asyncio.Event()
        session.turn.bind(threading.Event(), llm)
        worker = threading.Thread(target=session.request_quit)
        worker.start()
        await asyncio.wait_for(llm.wait(), timeout=1)
        worker.join()
        return session.quit_requested.is_set()

    assert asyncio.run(scenario())


def test_sessions_do_not_share_quit_state() -> None:
    first = DuplexSession()
    second = DuplexSession()
    first.request_quit()
    assert first.quit_requested.is_set()
    assert not second.quit_requested.is_set()
