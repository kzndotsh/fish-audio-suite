"""Listening: ASR outcomes, ``hear_line`` exits, mic framing and the console's closed-pipe guards."""

from __future__ import annotations

import asyncio
import sys
import threading
from collections.abc import Iterator
from types import TracebackType
from typing import Any

import httpx
import pytest
from test_duplex import _ctx  # pyright: ignore[reportPrivateUsage]

from fish_audio_suite_kit import FishHttpError
from fish_audio_suite_voice import barge, console
from fish_audio_suite_voice.barge import FRAME_BYTES, mic_frames
from fish_audio_suite_voice.console import console_print, end_reply_line, write_reply_token
from fish_audio_suite_voice.debug import configure_voice_logging
from fish_audio_suite_voice.duplex_state import EXIT_FATAL, DuplexContext
from fish_audio_suite_voice.hearing import hear_line, recognize
from fish_audio_suite_voice.playback import PortAudioMissingError


def _asr_returns(monkeypatch: pytest.MonkeyPatch, outcome: str | BaseException) -> None:
    async def asr(*_args: object, **_kwargs: object) -> str:
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr("fish_audio_suite_voice.hearing.fish_asr", asr)


def _recognize(ctx: DuplexContext, last_user: str = "") -> str:
    return asyncio.run(recognize(ctx, b"wav", last_user)).kind


@pytest.mark.parametrize(
    ("error", "kind"),
    [
        (FishHttpError.from_status(401, "bad key"), "fatal"),
        (FishHttpError.from_status(503, "down"), "again"),
        (httpx.ConnectError("no route"), "again"),
    ],
)
def test_an_asr_failure_is_fatal_only_for_a_bad_key(
    monkeypatch: pytest.MonkeyPatch, error: BaseException, kind: str
) -> None:
    _asr_returns(monkeypatch, error)
    assert _recognize(_ctx()) == kind


@pytest.mark.parametrize(
    "error", [FishHttpError.from_status(503, "down"), httpx.ConnectError("cancelled")]
)
def test_a_failure_after_quit_is_just_a_bye(
    monkeypatch: pytest.MonkeyPatch, error: BaseException
) -> None:
    ctx = _ctx()
    ctx.session.quit_requested.set()
    _asr_returns(monkeypatch, error)
    assert _recognize(ctx) == "bye"


def test_a_transcript_that_says_goodbye_ends_the_session(monkeypatch: pytest.MonkeyPatch) -> None:
    _asr_returns(monkeypatch, "goodbye")
    assert _recognize(_ctx()) == "bye"


def test_a_transcript_that_repeats_the_last_line_is_noise(monkeypatch: pytest.MonkeyPatch) -> None:
    _asr_returns(monkeypatch, "Tell me a story")
    assert _recognize(_ctx(), "tell me a story") == "noise"


def test_hear_line_exits_when_the_mic_cannot_open(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_audio(*_args: object, **_kwargs: object) -> bytes:
        raise PortAudioMissingError("missing")

    monkeypatch.setattr("fish_audio_suite_voice.hearing.record_utterance", no_audio)
    heard = asyncio.run(hear_line(_ctx(), ""))
    assert (heard.kind, heard.code) == ("fatal", EXIT_FATAL)


@pytest.mark.parametrize(("clip", "quit_set", "kind"), [(b"", False, "noise"), (b"x", True, "bye")])
def test_hear_line_reports_a_dropped_clip_and_quit(
    monkeypatch: pytest.MonkeyPatch, clip: bytes, quit_set: bool, kind: str
) -> None:
    ctx = _ctx()

    def record(*_args: object, **_kwargs: object) -> bytes:
        if quit_set:
            ctx.session.quit_requested.set()
        return clip

    monkeypatch.setattr("fish_audio_suite_voice.hearing.record_utterance", record)
    assert asyncio.run(hear_line(ctx, "")).kind == kind


def test_a_barge_in_clip_is_never_a_stale_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = _ctx()
    ctx.barge_prefix = b"clip"
    seen: dict[str, Any] = {}

    def record(*_args: object, **kwargs: object) -> bytes:
        seen["prefix"] = kwargs["prefix"]
        return b"wav"

    async def fake_recognize(*_args: object, **kwargs: object) -> Any:
        seen.update(kwargs)
        from fish_audio_suite_voice.hearing import HeardLine

        return HeardLine("line", text="hello there friend")

    monkeypatch.setattr("fish_audio_suite_voice.hearing.record_utterance", record)
    monkeypatch.setattr("fish_audio_suite_voice.hearing.recognize", fake_recognize)
    asyncio.run(hear_line(ctx, "hello there friend"))
    assert seen["prefix"] == b"clip"
    assert seen["stale"] is False
    assert seen["over_reply"] is True
    assert ctx.barge_prefix == b""


class _FakeStream:
    """A ``RawInputStream`` that delivers scripted buffers to the callback once opened."""

    def __init__(self, buffers: list[bytes], **kwargs: Any) -> None:
        self._buffers = buffers
        self._callback = kwargs["callback"]

    def __enter__(self) -> _FakeStream:
        for buffer in self._buffers:
            self._callback(buffer, len(buffer), None, None)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        return None


class _Aec:
    def clean(self, frame: bytes) -> bytes:
        return frame.upper()


def _frames_from(
    monkeypatch: pytest.MonkeyPatch, buffers: list[bytes], want: int, aec: _Aec | None = None
) -> list[bytes]:
    class Sd:
        @staticmethod
        def RawInputStream(**kwargs: Any) -> _FakeStream:  # noqa: N802 - mirrors sounddevice
            return _FakeStream(buffers, **kwargs)

    monkeypatch.setattr(barge, "load_sounddevice", lambda: Sd)
    stop = threading.Event()
    got: list[bytes] = []
    frames: Iterator[bytes] = mic_frames(None, stop, timeout=0.01, aec=aec)  # pyright: ignore[reportArgumentType]
    for frame in frames:
        got.append(frame)
        if len(got) == want:
            stop.set()
    return got


def test_mic_frames_regroups_ragged_buffers_into_whole_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    half = FRAME_BYTES // 2
    # 1.5 frames, then 1.5 more: three whole frames, none dropped or split.
    got = _frames_from(monkeypatch, [b"a" * (half * 3), b"b" * (half * 3)], want=3)
    assert [len(f) for f in got] == [FRAME_BYTES] * 3
    assert got[0] == b"a" * FRAME_BYTES
    assert got[1] == b"a" * half + b"b" * half
    assert got[2] == b"b" * FRAME_BYTES


def test_mic_frames_cleans_each_frame_with_the_echo_canceller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    got = _frames_from(monkeypatch, [b"a" * FRAME_BYTES], want=1, aec=_Aec())
    assert got == [b"A" * FRAME_BYTES]


def test_mic_frames_waits_out_an_empty_queue_until_stopped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Sd:
        @staticmethod
        def RawInputStream(**kwargs: Any) -> _FakeStream:  # noqa: N802 - mirrors sounddevice
            return _FakeStream([], **kwargs)

    monkeypatch.setattr(barge, "load_sounddevice", lambda: Sd)
    stop = threading.Event()
    timer = threading.Timer(0.05, stop.set)
    timer.start()
    try:
        assert list(mic_frames(None, stop, timeout=0.01)) == []
    finally:
        timer.cancel()


def test_the_barge_heartbeat_only_logs_at_trace_level(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lines: list[str] = []
    monkeypatch.setattr(barge, "debug", lambda message, **_kw: lines.append(message))
    configure_voice_logging(debug=1)
    try:
        barge._barge_heartbeat(idle_frames=20, rms=10.0, need=220.0, far=True, hit=0, aec_on=True)  # pyright: ignore[reportPrivateUsage]
        assert lines == []
        configure_voice_logging(debug=2)
        barge._barge_heartbeat(idle_frames=7, rms=10.0, need=220.0, far=True, hit=0, aec_on=True)  # pyright: ignore[reportPrivateUsage]
        assert lines == []
        barge._barge_heartbeat(idle_frames=20, rms=10.0, need=220.0, far=True, hit=0, aec_on=True)  # pyright: ignore[reportPrivateUsage]
        assert len(lines) == 1
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
    monkeypatch.setattr(sys, "stdout", _BrokenPipe())
    console_print("status")
    write_reply_token("hello")
    console._REPLY.open = True  # pyright: ignore[reportPrivateUsage]
    end_reply_line()
    assert console._REPLY.open is False  # pyright: ignore[reportPrivateUsage]


def test_a_buffer_shorter_than_a_frame_waits_and_a_zero_frame_size_yields_nothing() -> None:
    pending = bytearray()
    assert barge._take_full_frames(pending, b"ab", 4) == []  # pyright: ignore[reportPrivateUsage]
    assert barge._take_full_frames(pending, b"cdef", 4) == [b"abcd"]  # pyright: ignore[reportPrivateUsage]
    assert bytes(pending) == b"ef"
    assert barge._take_full_frames(pending, b"gh", 0) == []  # pyright: ignore[reportPrivateUsage]


def test_hear_line_logs_instead_of_printing_listening_in_debug_mode(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("fish_audio_suite_voice.hearing.debug_enabled", lambda: True)
    monkeypatch.setattr("fish_audio_suite_voice.hearing.record_utterance", lambda *_a, **_k: b"")
    assert asyncio.run(hear_line(_ctx(), "")).kind == "noise"
    assert "listening" not in capsys.readouterr().out
