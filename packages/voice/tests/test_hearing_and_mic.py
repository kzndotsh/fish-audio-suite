"""Listening: ASR outcomes, ``hear_line`` exits, mic framing and the console's closed-pipe guards."""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import replace
from typing import Any

import httpx
import pytest
from session_fakes import AsrReply, ScriptedAsr, ScriptedRecordings, barge_clip, make_ctx
from voice_fakes import ScriptedMic, install_vad

from fish_audio_suite_voice import mic, pcm
from fish_audio_suite_voice.barge import (
    BargeGate,
)
from fish_audio_suite_voice.console_sink import ConsoleSink, console_print
from fish_audio_suite_voice.duplex_state import DuplexContext
from fish_audio_suite_voice.events import EventBus, ExitCode, ReplyToken
from fish_audio_suite_voice.heard import HeardLine
from fish_audio_suite_voice.hearing import hear_line
from fish_audio_suite_voice.log_setup import configure_voice_logging
from fish_audio_suite_voice.mic import SessionMic
from fish_audio_suite_voice.pcm import FRAME_BYTES, take_frames
from fish_audio_suite_voice.portaudio import PortAudioMissingError


def _heard(
    monkeypatch: pytest.MonkeyPatch,
    reply: AsrReply | Callable[[], AsrReply],
    last_user: str = "",
    ctx: DuplexContext | None = None,
) -> str:
    """Hear one recorded clip that Fish ASR answers with ``reply``, and return its kind."""
    ScriptedRecordings(monkeypatch, b"wav")
    asr = ScriptedAsr(monkeypatch, reply)
    ctx = replace(ctx or make_ctx(), asr_http=asr.client())
    return asyncio.run(hear_line(ctx, last_user)).kind


@pytest.mark.parametrize(
    ("error", "kind"),
    [(401, "fatal"), (503, "again"), (httpx.ConnectError("no route"), "again")],
)
def test_an_asr_failure_is_fatal_only_for_a_bad_key(
    monkeypatch: pytest.MonkeyPatch, error: AsrReply, kind: str
) -> None:
    assert _heard(monkeypatch, error) == kind


@pytest.mark.parametrize("error", [503, httpx.ConnectError("cancelled")])
def test_a_failure_after_quit_is_just_a_bye(
    monkeypatch: pytest.MonkeyPatch, error: AsrReply
) -> None:
    ctx = make_ctx()

    def quit_then_fail() -> AsrReply:
        ctx.turn_input.session.quit_requested.set()  # Ctrl+C while the request was out
        return error

    assert _heard(monkeypatch, quit_then_fail, ctx=ctx) == "bye"


def test_a_transcript_that_says_goodbye_ends_the_session(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _heard(monkeypatch, "goodbye") == "bye"


def test_a_transcript_that_repeats_the_last_line_is_noise(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _heard(monkeypatch, "Tell me a story", "tell me a story") == "noise"


def test_hear_line_exits_when_the_mic_cannot_open(monkeypatch: pytest.MonkeyPatch) -> None:
    ScriptedRecordings(monkeypatch, PortAudioMissingError("missing"))
    heard = asyncio.run(hear_line(make_ctx(), ""))
    assert (heard.kind, heard.code) == ("fatal", ExitCode.FATAL)


@pytest.mark.parametrize(("clip", "quit_set", "kind"), [(b"", False, "noise"), (b"x", True, "bye")])
def test_hear_line_reports_a_dropped_clip_and_quit(
    monkeypatch: pytest.MonkeyPatch, clip: bytes, quit_set: bool, kind: str
) -> None:
    ctx = make_ctx()

    def record(_stop: object) -> bytes:
        if quit_set:
            ctx.turn_input.session.quit_requested.set()
        return clip

    ScriptedRecordings(monkeypatch, record)
    assert asyncio.run(hear_line(ctx, "")).kind == kind


def test_a_batch_turn_reports_the_silence_it_waited_out(monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = make_ctx(asr=ScriptedAsr(monkeypatch, "what time is it"))
    ScriptedRecordings(monkeypatch, b"wav")
    heard = asyncio.run(hear_line(ctx, ""))
    assert heard.end_wait_ms == ctx.config.listen.end_silence_frames * pcm.FRAME_MS


def test_a_barge_in_clip_is_never_a_stale_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = make_ctx(asr=ScriptedAsr(monkeypatch, "hello there friend"))
    clip = barge_clip(b"clip", since=42)
    ctx.turn_input.carry.keep(clip, "")
    recordings = ScriptedRecordings(monkeypatch, b"wav")
    heard = asyncio.run(hear_line(ctx, "hello there friend"))
    # The listen starts from the clip and reads on after it.
    assert recordings.calls[0]["carried"] == clip
    assert heard.kind == "line"  # the same words as the last line, but said fresh over the reply
    assert ctx.turn_input.carry.clip is None


class _FakeStream:
    """A ``RawInputStream`` that delivers scripted buffers to the callback when started."""

    opened = 0

    def __init__(self, buffers: list[bytes], **kwargs: Any) -> None:
        self._buffers = buffers
        self._callback = kwargs["callback"]
        self.closed = False
        _FakeStream.opened += 1

    def start(self) -> None:
        for buffer in self._buffers:
            self._callback(buffer, len(buffer), None, None)

    def stop(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True


class _Aec:
    def clean(self, frame: bytes) -> bytes:
        return frame.upper()


def _install(monkeypatch: pytest.MonkeyPatch, buffers: list[bytes]) -> None:
    class Sd:
        @staticmethod
        def RawInputStream(**kwargs: Any) -> _FakeStream:  # noqa: N802 - mirrors sounddevice
            return _FakeStream(buffers, **kwargs)

    _FakeStream.opened = 0
    monkeypatch.setattr(mic, "load_sounddevice", lambda: Sd)


def _take(frames: Iterator[bytes], stop: threading.Event, want: int) -> list[bytes]:
    got: list[bytes] = []
    for frame in frames:
        got.append(frame)
        if len(got) == want:
            stop.set()
    return got


def _frames_from(
    monkeypatch: pytest.MonkeyPatch, buffers: list[bytes], want: int, aec: _Aec | None = None
) -> list[bytes]:
    _install(monkeypatch, buffers)
    with SessionMic(aec=aec) as source:  # pyright: ignore[reportArgumentType]
        stop = threading.Event()
        return _take(source.frames(stop), stop, want)


def _numbered(count: int) -> list[bytes]:
    return [index.to_bytes(4, "little") * (FRAME_BYTES // 4) for index in range(count)]


def test_the_mic_regroups_ragged_buffers_into_whole_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    half = FRAME_BYTES // 2
    # 1.5 frames, then 1.5 more: three whole frames, none dropped or split.
    got = _frames_from(monkeypatch, [b"a" * (half * 3), b"b" * (half * 3)], want=3)
    assert [len(f) for f in got] == [FRAME_BYTES] * 3
    assert got[0] == b"a" * FRAME_BYTES
    assert got[1] == b"a" * half + b"b" * half
    assert got[2] == b"b" * FRAME_BYTES


def test_the_mic_cleans_each_frame_with_the_echo_canceller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    got = _frames_from(monkeypatch, [b"a" * FRAME_BYTES], want=1, aec=_Aec())
    assert got == [b"A" * FRAME_BYTES]


def test_the_mic_waits_out_an_empty_queue_until_stopped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch, [])
    stop = threading.Event()
    timer = threading.Timer(0.05, stop.set)
    timer.start()
    try:
        with SessionMic() as source:
            assert list(source.frames(stop)) == []
    finally:
        timer.cancel()


def test_a_reader_picks_up_exactly_where_the_last_one_stopped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A barge-in stops reading on the frame that tripped it, and the next listen reads
    # on from the frame after. Opening the device again lost what came in between.
    buffers = _numbered(30)
    _install(monkeypatch, buffers)
    with SessionMic() as source:
        stop = threading.Event()
        start = source.position
        first = _take(source.frames(stop, since=start), stop, 10)
        stop = threading.Event()
        rest = _take(source.frames(stop, since=start + len(first)), stop, 20)
    assert first + rest == buffers
    assert _FakeStream.opened == 1  # one device for the whole session


def test_a_reader_that_falls_behind_keeps_only_the_newest_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Ten seconds of frames arrive while no one reads. A backlog that size is
    # stale, so only about the last two seconds are kept.
    buffers = _numbered(333)
    _install(monkeypatch, buffers)
    with SessionMic() as source:
        stop = threading.Event()
        assert _take(source.frames(stop), stop, 1)  # opens the device
        # The device's own queue already dropped all but the newest blocks, so wait
        # for the reader to stop moving rather than for every frame.
        settled, deadline = -1, time.monotonic() + 5
        while source.position != settled and time.monotonic() < deadline:
            settled = source.position
            time.sleep(0.05)
        stop = threading.Event()
        timer = threading.Timer(0.3, stop.set)
        timer.start()
        try:
            got = list(source.frames(stop, since=0))
        finally:
            timer.cancel()
    assert 0 < len(got) <= 2_000 // pcm.FRAME_MS  # at most about two seconds
    assert got == buffers[-len(got) :]  # the newest frames, in order, with no gap


def test_a_failure_in_the_mic_thread_reaches_the_reader(monkeypatch: pytest.MonkeyPatch) -> None:
    class _BrokenAec:
        def clean(self, frame: bytes) -> bytes:
            raise OSError("device gone")

    _install(monkeypatch, _numbered(3))
    with (
        SessionMic(aec=_BrokenAec()) as source,  # pyright: ignore[reportArgumentType]
        pytest.raises(OSError, match="device gone"),
    ):
        list(source.frames(threading.Event()))


def test_the_barge_heartbeat_only_logs_at_trace_level(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class Quiet:
        def __init__(self, mode: int) -> None:
            del mode

        def is_speech(self, frame: bytes, rate: int) -> bool:
            del frame, rate
            return False

    install_vad(monkeypatch, Quiet)
    quiet = b"\x01\x00" * (pcm.FRAME_BYTES // 2)

    def heartbeats(level: int, frames: int) -> int:
        configure_voice_logging(debug=level)
        capsys.readouterr()
        BargeGate(mic=ScriptedMic([quiet] * frames)).watch(threading.Event())
        err = capsys.readouterr().err
        return sum("far=" in line and "aec=" in line for line in err.splitlines())

    try:
        assert heartbeats(1, pcm.LISTEN_HEARTBEAT_FRAMES) == 0  # events only: no heartbeat
        assert heartbeats(2, pcm.LISTEN_HEARTBEAT_FRAMES - 1) == 0  # not due yet
        assert heartbeats(2, pcm.LISTEN_HEARTBEAT_FRAMES) == 1
    finally:
        configure_voice_logging(debug=False)


class _BrokenPipe:
    def write(self, _text: str) -> int:
        raise BrokenPipeError

    def flush(self) -> None:
        raise BrokenPipeError


def test_a_closed_stdout_never_raises_from_the_console_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bus = EventBus()
    with ConsoleSink(bus) as display:
        bus.emit(ReplyToken("hello"))  # the reply line is open
        monkeypatch.setattr(sys, "stdout", _BrokenPipe())  # and then the reader goes away
        console_print("status")
        bus.emit(ReplyToken(" there"))
        display.end_reply_line()
        display.end_reply_line()


def test_a_buffer_shorter_than_a_frame_waits_and_a_zero_frame_size_yields_nothing() -> None:
    pending = bytearray()
    assert pcm.take_frames(pending, b"ab", 4) == []
    assert pcm.take_frames(pending, b"cdef", 4) == [b"abcd"]
    assert bytes(pending) == b"ef"
    assert pcm.take_frames(pending, b"gh", 0) == []


def test_hear_line_logs_instead_of_printing_listening_in_debug_mode(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("fish_audio_suite_voice.hearing.debug_enabled", lambda: True)
    ScriptedRecordings(monkeypatch)  # hears nothing
    assert asyncio.run(hear_line(make_ctx(), "")).kind == "noise"
    assert "listening" not in capsys.readouterr().out


def test_releasing_the_mic_closes_the_device_forgets_its_audio_and_a_read_reopens_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    buffers = _numbered(5)
    _install(monkeypatch, buffers)
    with SessionMic() as source:
        stop = threading.Event()
        first = _take(source.frames(stop), stop, 5)
        source.release()  # muted: nothing is kept or heard
        assert source.position == 5  # numbers keep counting
        stop = threading.Event()
        timer = threading.Timer(0.2, stop.set)
        timer.start()
        try:
            # A read from before the release finds nothing kept, and opens the device
            # again, which delivers its buffers anew.
            again = _take(source.frames(stop, since=0), stop, 5)
        finally:
            timer.cancel()
    assert first == buffers
    assert again == buffers
    assert _FakeStream.opened == 2


def test_recent_returns_what_arrived_since_a_position_without_waiting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    buffers = _numbered(10)
    _install(monkeypatch, buffers)
    with SessionMic() as source:
        stop = threading.Event()
        _take(source.frames(stop), stop, 10)
        assert source.recent(6) == buffers[6:]
        assert source.recent(10) == []
        assert source.recent(0) == buffers


def test_a_reader_that_is_already_stopped_does_not_open_the_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Mute sets the stop flag and then releases the mic. A recording that starts in
    # between must not open it again while muted.
    _install(monkeypatch, _numbered(3))
    stopped = threading.Event()
    stopped.set()
    with SessionMic() as source:
        assert list(source.frames(stopped)) == []
    assert _FakeStream.opened == 0


def test_a_stop_flag_ends_the_recording_and_quit_still_does_too(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The stop flag used to replace the quit flag, so a caller that forgot to fold quit into
    # it left Ctrl+C unable to stop the mic.
    recordings = ScriptedRecordings(monkeypatch)
    typed = threading.Event()
    ctx = make_ctx()
    asyncio.run(hear_line(ctx, "", stop=typed))
    stop = recordings.calls[0]["stop"]
    assert not stop.is_set()
    typed.set()
    assert stop.is_set()
    typed.clear()
    ctx.turn_input.session.quit_requested.set()
    assert stop.is_set()


def _heard_line(monkeypatch: pytest.MonkeyPatch, reply: AsrReply) -> HeardLine:
    """Record one clip and hear it with Fish ASR answering ``reply``."""
    ScriptedRecordings(monkeypatch, b"wav")
    return asyncio.run(hear_line(make_ctx(asr=ScriptedAsr(monkeypatch, reply)), ""))


def test_recognize_fatal_again_and_quit(monkeypatch: pytest.MonkeyPatch) -> None:
    fatal = _heard_line(monkeypatch, 401)
    assert fatal.kind == "fatal"
    assert fatal.code == 2
    assert _heard_line(monkeypatch, TimeoutError("late")).kind == "again"
    assert _heard_line(monkeypatch, "goodbye").kind == "bye"


def test_recognize_listens_again_after_a_network_error(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _heard_line(monkeypatch, httpx.ConnectError("down")).kind == "again"


@pytest.mark.parametrize("bug", [RuntimeError("bug"), ValueError("bad"), KeyError("k")])
def test_recognize_does_not_hide_a_programming_error(
    monkeypatch: pytest.MonkeyPatch, bug: Exception
) -> None:
    with pytest.raises(type(bug)):
        _heard_line(monkeypatch, bug)


@pytest.mark.parametrize(("barge_prefix", "kind"), [(b"", "line"), (b"clip", "noise")])
def test_hear_line_answers_a_yeah_unless_it_was_said_over_the_reply(
    monkeypatch: pytest.MonkeyPatch, barge_prefix: bytes, kind: str
) -> None:
    ctx = make_ctx(asr=ScriptedAsr(monkeypatch, "Yeah."))
    if barge_prefix:
        ctx.turn_input.carry.keep(barge_clip(barge_prefix), "")
    ScriptedRecordings(monkeypatch, b"RIFFwav")
    assert asyncio.run(hear_line(ctx, "")).kind == kind


def test_mic_chunks_join_when_short_and_split_when_long() -> None:
    pending = bytearray()
    half = b"\x01" * (FRAME_BYTES // 2)
    assert take_frames(pending, half, FRAME_BYTES) == []
    assert take_frames(pending, half, FRAME_BYTES) == [b"\x01" * FRAME_BYTES]
    assert pending == bytearray()
    long = b"\x02" * (FRAME_BYTES * 2 + 3)
    frames = take_frames(pending, long, FRAME_BYTES)
    assert frames == [b"\x02" * FRAME_BYTES, b"\x02" * FRAME_BYTES]
    rest = b"\x03" * (FRAME_BYTES - 3)
    done = take_frames(pending, rest, FRAME_BYTES)
    assert done == [b"\x02" * 3 + rest]
    assert pending == bytearray()
