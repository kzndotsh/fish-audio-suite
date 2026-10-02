from __future__ import annotations

import asyncio
import signal
from typing import Any

import pytest

from fish_audio_suite_voice.signals import DuplexSession


def test_install_sigint_uses_the_loop_handler_and_runs_before_first() -> None:
    session = DuplexSession()
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
        assert session.stop.is_set()
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

        loop.add_signal_handler = refuse
        monkeypatch.setattr(
            "fish_audio_suite_voice.signals.signal.signal",
            lambda sig, handler: installed.update(sig=sig, handler=handler),
        )
        session.install_sigint(loop)
        installed["handler"](signal.SIGINT, None)
        await asyncio.sleep(0)

    asyncio.run(scenario())
    assert installed["sig"] == signal.SIGINT
    assert session.stop.is_set()
