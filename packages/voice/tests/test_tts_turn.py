from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator
from dataclasses import replace
from typing import Any

import pytest
from fishaudio import AsyncFishAudio
from fishaudio.exceptions import RateLimitError, WebSocketError
from wire_helpers import (
    RecordingSink,
    ScriptedSocket,
    make_run,
)

from fish_audio_suite_kit import (
    FishAuthError,
    FishRateLimitError,
)
from fish_audio_suite_voice.speaker import FishSpeaker
from fish_audio_suite_voice.tts_turn import SdkSocket, run_turn
from fish_audio_suite_voice.wire import TurnSpec, text_events


def test_run_turn_closes_the_sink_when_start_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    socket = ScriptedSocket()

    class _Boom(RecordingSink):
        def __init__(self) -> None:
            super().__init__()
            self.opened = False
            self.finished = False

        def start(self) -> None:
            self.opened = True
            raise RuntimeError("dac")

        def finish(self, *, kill: bool = False) -> None:
            del kill
            self.finished = True

    run, _sink = make_run()
    sink = _Boom()

    async def no_events():
        if False:
            yield ""

    with pytest.raises(RuntimeError, match="dac"):
        asyncio.run(
            run_turn(
                run.spec, no_events(), sink, threading.Event(), sent_text="hello", socket=socket
            )
        )
    assert sink.opened is True
    assert sink.finished is True
    assert socket.streams == 0


def test_run_turn_closes_the_client_when_finish_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    socket = ScriptedSocket()

    class _FinishBoom(RecordingSink):
        def finish(self, *, kill: bool = False) -> None:
            del kill
            raise RuntimeError("disk")

    run, _sink = make_run()

    async def no_events():
        if False:
            yield ""

    with pytest.raises(RuntimeError, match="disk"):
        asyncio.run(
            run_turn(
                run.spec,
                no_events(),
                _FinishBoom(),
                threading.Event(),
                sent_text="hello",
                socket=socket,
            )
        )
    assert socket.closed == 1


def test_client_close_after_audio_does_not_forget_the_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = AsyncFishAudio.close

    class _OneSecond(SdkSocket):
        """The SDK socket, with one second of audio in place of its websocket."""

        def stream(self, events: AsyncIterator[Any], spec: TurnSpec) -> AsyncIterator[bytes]:
            super().stream(events, spec)  # opens the real client, whose socket is never read

            async def audio() -> AsyncIterator[bytes]:
                async for _event in events:
                    pass
                yield b"\x00\x00" * 16_000

            return audio()

    async def boom_close(self: AsyncFishAudio) -> None:
        await original(self)
        raise BaseExceptionGroup("close", [asyncio.CancelledError()])

    monkeypatch.setattr("fish_audio_suite_voice.tts_turn.SdkSocket", _OneSecond)
    monkeypatch.setattr("fish_audio_suite_voice.tts_turn.AsyncFishAudio.close", boom_close)
    tts = FishSpeaker(api_key="k", voice_id="v", sample_rate=16_000, audio_format="pcm")

    class Sink:
        output_latency_s = 0.0

        def __init__(self) -> None:
            self.n = 0

        def start(self) -> None:
            return None

        def write(self, chunk: bytes) -> None:
            self.n += len(chunk)

        def finish(self, *, kill: bool = False) -> None:
            del kill

        def bytes_played(self) -> int:
            return self.n

    result = tts.speak("Hello there friend.", Sink())
    assert "Hello" in result.spoken_so_far
    assert result.bytes_played == 32_000
    assert result.got_audio is True


def test_run_turn_stops_when_the_api_key_cannot_be_a_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened: list[object] = []
    monkeypatch.setattr(
        "fish_audio_suite_voice.tts_turn.AsyncFishAudio", lambda **kw: opened.append(kw)
    )
    run, sink = make_run()
    spec = replace(run.spec, api_key="sk-\nbad")

    async def no_events():
        if False:
            yield ""

    result = asyncio.run(run_turn(spec, no_events(), sink, threading.Event(), sent_text="hello"))
    assert opened == []
    assert result.error_status == 401
    assert isinstance(result.error, FishAuthError)
    assert result.error.status == 401
    assert not result.error.retryable
    assert result.got_audio is False


def test_run_turn_retries_before_audio_and_stops_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    slept: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    socket = ScriptedSocket([RateLimitError(429, "slow", None)], [b"abcd"])
    monkeypatch.setattr("fish_audio_suite_voice.pause.asyncio.sleep", fake_sleep)
    run, sink = make_run()

    async def no_events():
        if False:
            yield ""

    result = asyncio.run(
        run_turn(run.spec, no_events(), sink, threading.Event(), sent_text="hello", socket=socket)
    )
    assert socket.streams == 2
    # Jitter: half to all of the 2**0 base.
    assert 0.5 <= sum(slept) <= 1.0
    assert result.got_audio is True
    assert result.spoken_so_far == "hello"

    slept.clear()
    socket = ScriptedSocket([b"abcd", RateLimitError(429, "slow", None)])
    run, sink = make_run()
    failed = asyncio.run(
        run_turn(run.spec, no_events(), sink, threading.Event(), sent_text="hello", socket=socket)
    )
    assert socket.streams == 1
    assert slept == []
    assert failed.error_status == 429
    assert isinstance(failed.error, FishRateLimitError)
    assert failed.error.retryable
    assert failed.got_audio is True

    # About 0.4 s at 44.1 kHz int16, enough for the first word only.
    socket = ScriptedSocket([b"\x00" * 33_075, WebSocketError("dropped")])
    run, sink = make_run()
    dropped = asyncio.run(
        run_turn(
            run.spec,
            no_events(),
            sink,
            threading.Event(),
            sent_text="hello there friend",
            socket=socket,
        )
    )
    assert socket.streams == 1
    assert dropped.error_status is None
    assert dropped.error is None
    assert dropped.got_audio is True
    assert dropped.spoken_so_far == "hello"


def test_retry_backoff_does_not_replay_after_cancel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cancel = threading.Event()
    socket = ScriptedSocket([RateLimitError(429, "slow", None)])

    async def fake_sleep(_seconds: float) -> None:
        cancel.set()

    monkeypatch.setattr("fish_audio_suite_voice.pause.asyncio.sleep", fake_sleep)
    run, sink = make_run()

    async def no_events():
        if False:
            yield ""

    result = asyncio.run(
        run_turn(run.spec, no_events(), sink, cancel, sent_text="hello", socket=socket)
    )
    assert socket.streams == 1
    assert result.cancelled is True
    assert result.error_status is None
    assert result.got_audio is False
    assert result.spoken_so_far == ""


def test_a_keyboard_interrupt_in_a_turn_is_not_treated_as_a_fish_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run, sink = make_run()
    socket = ScriptedSocket([KeyboardInterrupt()])
    with pytest.raises(KeyboardInterrupt):
        asyncio.run(
            run_turn(
                run.spec,
                text_events("hi", run.cancel, 40),
                sink,
                run.cancel,
                sent_text="",
                socket=socket,
            )
        )
    assert socket.streams == 1


def test_run_isolated_does_not_swallow_a_keyboard_interrupt_but_still_closes_the_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fish_audio_suite_voice.tts_turn import run_isolated

    seen: list[asyncio.AbstractEventLoop] = []

    async def interrupted_shutdown(loop: asyncio.AbstractEventLoop) -> None:
        seen.append(loop)
        raise KeyboardInterrupt

    monkeypatch.setattr("fish_audio_suite_voice.tts_turn.quiet_shutdown", interrupted_shutdown)

    async def work() -> Any:
        return None

    with pytest.raises(KeyboardInterrupt):
        run_isolated(work())
    assert seen
    assert seen[0].is_closed()
