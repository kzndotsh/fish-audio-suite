from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator

import pytest
from wire_helpers import RecordingSink, play_stream

from fish_audio_suite_voice.wire import TtsResult


def _play(stream: AsyncIterator[bytes], sink: RecordingSink | None = None) -> TtsResult:
    result, _socket = asyncio.run(play_stream(stream, sink or RecordingSink(), threading.Event()))
    return result


def test_a_keyboard_interrupt_in_the_stream_stops_the_process_instead_of_being_queued() -> None:
    async def stream() -> AsyncIterator[bytes]:
        yield b"\x01\x02"
        raise KeyboardInterrupt

    sink = RecordingSink()
    with pytest.raises(KeyboardInterrupt):
        _play(stream(), sink)
    # Queued, it would wait behind the audio the reader had read ahead, and that would play.
    assert sink.chunks == []


def test_a_system_exit_in_the_stream_is_not_queued_either() -> None:
    async def stream() -> AsyncIterator[bytes]:
        yield b"\x01\x02"
        raise SystemExit(3)

    with pytest.raises(SystemExit):
        _play(stream())


def test_an_ordinary_error_in_the_stream_still_reaches_the_turn() -> None:
    async def stream() -> AsyncIterator[bytes]:
        yield b"\x01\x02"
        raise RuntimeError("socket died")

    result = _play(stream())
    assert result.got_audio is True
    assert result.error_message == "socket died"


def test_an_exception_group_from_the_sdk_still_reaches_the_turn() -> None:
    async def stream() -> AsyncIterator[bytes]:
        yield b"\x01\x02"
        raise BaseExceptionGroup("tasks", [ValueError("a"), asyncio.CancelledError()])

    result = _play(stream())
    assert result.got_audio is True
    assert result.error_message == "a"  # the real error inside the group, not the cancel
