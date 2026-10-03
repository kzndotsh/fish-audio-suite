from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import AsyncIterator

import pytest
from fishaudio.types import TTSConfig

from fish_audio_suite_voice.wire import EventAcc, Heard, TurnRun, TurnSpec, _pump_ws_audio


class _Sink:
    output_latency_s = 0.0

    def __init__(self) -> None:
        self.chunks: list[bytes] = []

    def start(self) -> None:
        return None

    def write(self, chunk: bytes) -> None:
        self.chunks.append(chunk)

    def finish(self, *, kill: bool = False) -> None:
        del kill

    def bytes_played(self) -> int:
        return sum(len(c) for c in self.chunks)


def _turn() -> TurnRun:
    spec = TurnSpec(
        api_key="k",
        base_url="https://api.fish.audio",
        voice_id="v",
        model="s2.1-pro",
        audio_format="pcm",
        latency="balanced",
        speed=1.0,
        sample_rate=44100,
        partial_chars=40,
        trace_headers={},
        config=TTSConfig(),
    )
    return TurnRun(
        spec=spec,
        sink=_Sink(),
        cancel=threading.Event(),
        sent_text="",
        acc=EventAcc(),
        t0=time.perf_counter(),
        audio=Heard(),
    )


async def _no_close() -> None:
    return None


def _pump(stream: AsyncIterator[bytes]) -> None:
    asyncio.run(_pump_ws_audio(stream, _turn(), _no_close))


def test_a_keyboard_interrupt_in_the_stream_stops_the_process_instead_of_being_queued() -> None:
    async def stream() -> AsyncIterator[bytes]:
        yield b"\x01\x02"
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        _pump(stream())


def test_a_system_exit_in_the_stream_is_not_queued_either() -> None:
    async def stream() -> AsyncIterator[bytes]:
        yield b"\x01\x02"
        raise SystemExit(3)

    with pytest.raises(SystemExit):
        _pump(stream())


def test_an_ordinary_error_in_the_stream_still_reaches_the_pump() -> None:
    async def stream() -> AsyncIterator[bytes]:
        yield b"\x01\x02"
        raise RuntimeError("socket died")

    with pytest.raises(RuntimeError, match="socket died"):
        _pump(stream())


def test_an_exception_group_from_the_sdk_still_reaches_the_pump() -> None:
    async def stream() -> AsyncIterator[bytes]:
        yield b"\x01\x02"
        raise BaseExceptionGroup("tasks", [ValueError("a"), asyncio.CancelledError()])

    with pytest.raises(BaseExceptionGroup):
        _pump(stream())
