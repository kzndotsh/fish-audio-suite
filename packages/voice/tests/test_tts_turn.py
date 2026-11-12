from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator
from dataclasses import replace
from typing import Any

import pytest
from fishaudio import AsyncFishAudio
from fishaudio.exceptions import AuthenticationError, RateLimitError, WebSocketError
from wire_helpers import (
    RecordingSink,
    ScriptedSocket,
    make_run,
    play_stream,
)

from fish_audio_suite_kit import (
    CueStyle,
    FishAuthError,
    FishRateLimitError,
)
from fish_audio_suite_voice.cancel import is_cancel_noise
from fish_audio_suite_voice.speaker import FishSpeaker
from fish_audio_suite_voice.tts_text import WholeText, text_events
from fish_audio_suite_voice.tts_tune import TtsTune
from fish_audio_suite_voice.tts_turn import SdkSocket, run_turn
from fish_audio_suite_voice.wire import TtsResult, TurnSpec


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

    with pytest.raises(RuntimeError, match="dac"):
        asyncio.run(
            run_turn(
                run.spec, WholeText("hello", 40, CueStyle()), sink, threading.Event(), socket=socket
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

    with pytest.raises(RuntimeError, match="disk"):
        asyncio.run(
            run_turn(
                run.spec,
                WholeText("hello", 40, CueStyle()),
                _FinishBoom(),
                threading.Event(),
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
    tts = FishSpeaker(api_key="k", voice_id="v", tune=TtsTune(sample_rate=16_000))

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

    result = asyncio.run(
        run_turn(spec, WholeText("hello", 40, CueStyle()), sink, threading.Event())
    )
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

    result = asyncio.run(
        run_turn(
            run.spec, WholeText("hello", 40, CueStyle()), sink, threading.Event(), socket=socket
        )
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
        run_turn(
            run.spec, WholeText("hello", 40, CueStyle()), sink, threading.Event(), socket=socket
        )
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
            WholeText("hello there friend", 40, CueStyle()),
            sink,
            threading.Event(),
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

    result = asyncio.run(
        run_turn(run.spec, WholeText("hello", 40, CueStyle()), sink, cancel, socket=socket)
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
                text_events("hi", run.cancel, 40, style=CueStyle()),
                sink,
                run.cancel,
                socket=socket,
            )
        )
    assert socket.streams == 1


def test_run_isolated_does_not_swallow_a_keyboard_interrupt_but_still_closes_the_loop() -> None:
    from fish_audio_suite_voice.tts_turn import run_isolated

    loops: list[asyncio.AbstractEventLoop] = []

    async def stubborn() -> None:
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            raise KeyboardInterrupt from None  # Ctrl+C lands while the loop is torn down

    async def work() -> Any:
        loops.append(asyncio.get_running_loop())
        asyncio.get_running_loop().create_task(stubborn())
        await asyncio.sleep(0)
        return None

    with pytest.raises(KeyboardInterrupt):
        run_isolated(work())
    assert loops
    assert loops[0].is_closed()


def _fail_every_attempt(
    monkeypatch: pytest.MonkeyPatch, error: BaseException
) -> tuple[ScriptedSocket, Any]:
    async def no_wait(_seconds: float) -> None:
        return None

    monkeypatch.setattr("fish_audio_suite_voice.pause.asyncio.sleep", no_wait)
    socket = ScriptedSocket([error])
    run, sink = make_run()
    result = asyncio.run(
        run_turn(
            run.spec, WholeText("hello", 40, CueStyle()), sink, threading.Event(), socket=socket
        )
    )
    return socket, result


def test_a_rate_limit_is_retried_and_a_bad_key_is_not(monkeypatch: pytest.MonkeyPatch) -> None:
    socket, limited = _fail_every_attempt(monkeypatch, RateLimitError(429, "slow down", None))
    assert socket.streams > 1
    assert limited.error_status == 429
    assert limited.error_message == "slow down"
    socket, refused = _fail_every_attempt(
        monkeypatch, AuthenticationError(401, "Invalid Token", None)
    )
    assert socket.streams == 1
    assert refused.error_status == 401
    assert refused.error_message == "Invalid Token"
    assert isinstance(refused.error, FishAuthError)
    assert not refused.error.retryable


def test_a_cancel_scope_error_before_audio_is_teardown_noise_not_a_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    err = RuntimeError("Attempted to exit cancel scope in a different task than it was entered in")
    assert is_cancel_noise(err)
    socket, result = _fail_every_attempt(monkeypatch, err)
    assert socket.streams == 1  # not retried
    assert result.error is None
    assert result.error_status is None
    assert result.error_message is None
    assert result.got_audio is False


class _AudioSocket(SdkSocket):
    """The SDK socket with two chunks of audio in place of its websocket, and no client."""

    def stream(self, events: AsyncIterator[Any], spec: TurnSpec) -> AsyncIterator[bytes]:
        del spec

        async def audio() -> AsyncIterator[bytes]:
            async for _event in events:
                pass
            yield b"\x00\x00" * 800
            yield b"\x00\x00" * 800

        return audio()


class _FailingSink(RecordingSink):
    """Raises ``error`` from ``stage`` (start, write or finish), after ``cancel`` is set if given."""

    def __init__(self, stage: str, error: Exception, cancel: threading.Event | None = None) -> None:
        super().__init__()
        self.stage = stage
        self.error = error
        self.cancel = cancel
        self.finished = False

    def _maybe_fail(self, stage: str) -> None:
        if stage == self.stage:
            if self.cancel is not None:
                self.cancel.set()
            raise self.error

    def start(self) -> None:
        self._maybe_fail("start")

    def write(self, chunk: bytes) -> None:
        self._maybe_fail("write")
        super().write(chunk)

    def finish(self, *, kill: bool = False) -> None:
        self.finished = True
        self._maybe_fail("finish")


@pytest.mark.parametrize("stage", ["start", "write", "finish"])
@pytest.mark.parametrize("error", [RuntimeError("device gone"), OSError("disk full")])
@pytest.mark.parametrize("streamed", [False, True])
def test_an_error_the_sink_raises_reaches_the_caller(
    monkeypatch: pytest.MonkeyPatch, stage: str, error: Exception, streamed: bool
) -> None:
    # A RuntimeError from the sink used to come back as a silent turn, and a failed write
    # as a Fish error in the result. Both now raise, as the speak docstring says.
    monkeypatch.setattr("fish_audio_suite_voice.tts_turn.SdkSocket", _AudioSocket)
    tts = FishSpeaker(api_key="k", voice_id="v")
    sink = _FailingSink(stage, error)

    def speak() -> object:
        if streamed:
            return tts.speak_stream(["Hello there ", "friend."], sink)
        return tts.speak("Hello there friend.", sink)

    with pytest.raises(type(error), match=str(error)):
        speak()
    assert sink.finished is True


def test_a_sink_error_while_the_turn_is_cancelled_ends_it_quietly(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("fish_audio_suite_voice.tts_turn.SdkSocket", _AudioSocket)
    tts = FishSpeaker(api_key="k", voice_id="v")
    cancel = threading.Event()
    sink = _FailingSink("write", RuntimeError("stream closed"), cancel=cancel)
    result = tts.speak("Hello there friend.", sink, cancel=cancel)
    assert result.cancelled is True
    assert result.error is None
    assert result.error_message is None
    assert "stream closed" not in capsys.readouterr().err


async def _hanging_audio() -> AsyncIterator[bytes]:
    """One chunk, then a websocket that never sends again."""
    yield b"\x01\x02"
    await asyncio.sleep(30)
    yield b"\x03\x04"


def test_a_timeout_around_a_turn_is_not_swallowed() -> None:
    # The turn's cancel flag is never set, so the CancelledError the timeout throws in
    # is not the turn's own: it must reach asyncio.timeout, which turns it into TimeoutError.
    sink = RecordingSink()

    async def main() -> None:
        async with asyncio.timeout(0.2):
            await play_stream(_hanging_audio(), sink, threading.Event())

    with pytest.raises(TimeoutError):
        asyncio.run(main())
    assert sink.chunks == [b"\x01\x02"]


def test_an_outer_task_cancel_reaches_the_caller() -> None:
    played = threading.Event()

    class SignalOnWrite(RecordingSink):
        def write(self, chunk: bytes) -> None:
            super().write(chunk)
            played.set()

    sink = SignalOnWrite()

    async def main() -> None:
        task = asyncio.ensure_future(play_stream(_hanging_audio(), sink, threading.Event()))
        # The sink writes on a worker thread, so wait for it off the loop.
        assert await asyncio.to_thread(played.wait, 5.0)
        task.cancel()
        await task

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(main())
    assert sink.chunks == [b"\x01\x02"]


def test_the_turns_own_cancel_ends_a_hanging_stream_quietly() -> None:
    cancel = threading.Event()

    class CancelOnFirstChunk(RecordingSink):
        def write(self, chunk: bytes) -> None:
            super().write(chunk)
            cancel.set()

    sink = CancelOnFirstChunk()

    async def main() -> tuple[TtsResult, int]:
        result, socket = await asyncio.wait_for(play_stream(_hanging_audio(), sink, cancel), 5.0)
        return result, socket.closed

    result, closed = asyncio.run(main())
    assert result.cancelled is True
    assert result.error is None
    assert result.error_message is None
    assert sink.chunks == [b"\x01\x02"]
    assert closed >= 1
