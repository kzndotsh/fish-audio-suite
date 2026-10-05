"""A turn streamed to Deepgram Flux, driven by a fake connection and a fake mic thread."""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import AsyncIterator, Callable, Sequence
from typing import Any

import pytest

from fish_audio_suite_voice import streaming
from fish_audio_suite_voice.barge import FRAME_BYTES, StopFlag
from fish_audio_suite_voice.deepgram import (
    DeepgramError,
    FluxMessage,
    FluxStream,
    StreamError,
    TurnEnded,
    TurnStarted,
    TurnUpdate,
)
from fish_audio_suite_voice.events import EVENTS, Event, Interim
from fish_audio_suite_voice.streaming import StreamedTurn, stream_turn
from fish_audio_suite_voice.tune import ListenTune, SttTune

FRAME = b"\x01\x00" * (FRAME_BYTES // 2)
STT = SttTune(provider="deepgram", deepgram_key="secret")


class _FakeFlux(FluxStream):
    """A Flux connection that records what it is sent and answers from a script."""

    def __init__(
        self,
        on_audio: Callable[[int], Sequence[FluxMessage | None]] | None = None,
        on_force: list[FluxMessage] | None = None,
        *,
        open_errors: list[DeepgramError] | None = None,
        drops_while_idle: bool = False,
    ) -> None:
        super().__init__("wss://fake/v2/listen", "secret")
        self.sent: list[bytes] = []
        self.forced = 0
        self.closed = False
        self.opens = 0
        self._on_audio = on_audio
        self._on_force = on_force or []
        self._open_errors = open_errors or []
        self._drops = drops_while_idle
        self._connected = False
        self._inbox: asyncio.Queue[FluxMessage | None] = asyncio.Queue()  # None closes it

    @property
    def connected(self) -> bool:
        return self._connected

    async def open(self) -> None:
        self.opens += 1
        if self._open_errors:
            raise self._open_errors.pop(0)
        self._connected = True

    async def send_audio(self, pcm: bytes) -> None:
        self.sent.append(pcm)
        answers: Sequence[FluxMessage | None] = (
            self._on_audio(sum(map(len, self.sent))) if self._on_audio else ()
        )
        for message in answers:
            self._inbox.put_nowait(message)

    async def force_end_turn(self) -> None:
        self.forced += 1
        for message in self._on_force:
            self._inbox.put_nowait(message)

    async def messages(self) -> AsyncIterator[FluxMessage]:
        while (message := await self._inbox.get()) is not None:
            yield message

    async def close(self) -> None:
        self.closed = True
        self._inbox.put_nowait(None)


def _mic(
    frames: int,
    *,
    local_end: bool = False,
    voiced: bool = True,
    seen: dict[str, Any] | None = None,
) -> Callable[..., bool]:
    """A mic thread that hands over ``frames`` frames, then waits to be stopped."""

    def listen(
        _device: object,
        stop: StopFlag,
        sink: Callable[[bytes, bool], None],
        *,
        prefix: bytes,
        tune: ListenTune,
        aec: object,
    ) -> bool:
        if seen is not None:
            seen.update(prefix=prefix, tune=tune)
        for _ in range(frames):
            sink(FRAME, voiced)
        if local_end:
            return True
        while not stop.is_set():
            time.sleep(0.005)
        return False

    return listen


def _run(
    fake: _FakeFlux,
    listen_fn: Callable[..., bool],
    *,
    stt: SttTune = STT,
    stop: StopFlag | None = None,
    prefix: bytes = b"",
) -> streaming.StreamedTurn | str:
    quit_flag = threading.Event()

    async def go() -> streaming.StreamedTurn | str:
        return await stream_turn(
            stt=stt,
            listen=ListenTune(),
            device=None,
            aec=None,
            quit_requested=quit_flag,
            stop=stop or threading.Event(),
            prefix=prefix,
            make_stream=lambda _url, _key: fake,
            listen_fn=listen_fn,
        )

    return asyncio.run(asyncio.wait_for(go(), 10))


def _events() -> tuple[list[Event], Callable[[], None]]:
    seen: list[Event] = []
    return seen, EVENTS.subscribe(seen.append)


def test_a_turn_is_sent_as_it_is_spoken_and_ends_when_flux_says_so() -> None:
    ninety_ms = FRAME_BYTES * 3
    script = {
        ninety_ms: [TurnStarted("hel")],
        ninety_ms * 2: [TurnUpdate("hello there")],
        ninety_ms * 3: [TurnEnded("hello there", 0.8, "model")],
    }
    fake = _FakeFlux(lambda total: script.get(total, []))
    seen, stop = _events()
    try:
        result = _run(fake, _mic(9))
    finally:
        stop()
    assert isinstance(result, StreamedTurn)
    assert result.text == "hello there"
    assert result.asr_ms >= 0.0
    assert result.trace_id
    assert [len(chunk) for chunk in fake.sent] == [ninety_ms] * 3  # about 90 ms at a time
    assert fake.forced == 0  # Flux decided; no need to ask
    assert fake.closed
    interim = [event.text for event in seen if isinstance(event, Interim)]
    assert interim == ["hel", "hello there"]  # the words as they arrived, once each


def test_nothing_is_sent_while_waiting_for_speech() -> None:
    fake = _FakeFlux()
    stop = threading.Event()
    stop.set()  # a typed line, or quit, before anyone spoke
    result = _run(fake, _mic(0), stop=stop)
    assert result == "stopped"
    assert fake.sent == []  # only speech leaves the machine
    assert fake.closed


def test_the_pre_roll_and_a_barge_prefix_are_handed_to_the_mic_side_first() -> None:
    seen: dict[str, Any] = {}
    fake = _FakeFlux(
        lambda total: [TurnEnded("go on", 0.9, "model")] if total >= FRAME_BYTES * 3 else []
    )
    result = _run(fake, _mic(3, seen=seen), prefix=b"\x07\x00" * 480)
    assert isinstance(result, StreamedTurn)
    assert seen["prefix"] == b"\x07\x00" * 480
    assert isinstance(seen["tune"], ListenTune)


def test_when_the_silence_limit_comes_first_flux_is_asked_to_end_the_turn() -> None:
    fake = _FakeFlux(
        lambda total: [TurnUpdate("what time is it")] if total >= FRAME_BYTES * 3 else [],
        on_force=[TurnEnded("what time is it", 0.5, "manual")],
    )
    result = _run(fake, _mic(4, local_end=True))
    assert isinstance(result, StreamedTurn)
    assert result.text == "what time is it"
    assert fake.forced == 1
    assert sum(map(len, fake.sent)) == FRAME_BYTES * 4  # the leftover frame was flushed too


def test_when_flux_never_answers_a_forced_end_the_words_so_far_are_used(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(streaming, "_FORCE_WAIT_S", 0.05)
    fake = _FakeFlux(
        lambda total: [TurnUpdate("tell me a joke")] if total >= FRAME_BYTES * 3 else []
    )
    result = _run(fake, _mic(3, local_end=True))
    assert isinstance(result, StreamedTurn)
    assert result.text == "tell me a joke"
    assert fake.forced == 1


def test_a_turn_with_no_words_is_noise(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeFlux(
        lambda total: [TurnEnded("  ", 0.9, "model")] if total >= FRAME_BYTES * 3 else []
    )
    assert _run(fake, _mic(3)) == "noise"
    monkeypatch.setattr(streaming, "_FORCE_WAIT_S", 0.05)
    assert _run(_FakeFlux(), _mic(3, local_end=True)) == "noise"  # nothing came back at all


def test_a_missing_key_and_a_refused_connection_are_fatal_but_an_outage_falls_back() -> None:
    no_key = SttTune(provider="deepgram", deepgram_key="")
    assert _run(_FakeFlux(), _mic(1), stt=no_key) == "fatal"
    assert _run(_FakeFlux(open_errors=[DeepgramError("HTTP 401", fatal=True)]), _mic(1)) == "fatal"
    assert _run(_FakeFlux(open_errors=[DeepgramError("no route")]), _mic(1)) == "fallback"


def test_an_idle_connection_that_dropped_is_opened_again_when_speech_starts() -> None:
    fake = _FakeFlux(
        lambda total: [TurnEnded("hi", 0.9, "model")] if total >= FRAME_BYTES * 3 else []
    )

    def drop_after_first_open(listen: Callable[..., bool]) -> Callable[..., bool]:
        def wrapped(*args: Any, **kwargs: Any) -> bool:
            fake._connected = False
            return listen(*args, **kwargs)

        return wrapped

    result = _run(fake, drop_after_first_open(_mic(3)))
    assert isinstance(result, StreamedTurn)
    assert fake.opens == 2

    dead = _FakeFlux(open_errors=[])
    dead._open_errors = []

    class Fussy(_FakeFlux):
        async def open(self) -> None:
            await super().open()
            if self.opens > 1:
                raise DeepgramError("gone")

    broken = Fussy()

    def listen_then_drop(*args: Any, **kwargs: Any) -> bool:
        broken._connected = False
        return _mic(3)(*args, **kwargs)

    assert _run(broken, listen_then_drop) == "again"


def test_an_error_from_flux_or_a_dropped_connection_mid_turn_means_listen_again() -> None:
    erroring = _FakeFlux(
        lambda total: (
            [StreamError("INTERNAL_SERVER_ERROR", "oops")] if total >= FRAME_BYTES * 3 else []
        )
    )
    assert _run(erroring, _mic(3)) == "again"
    closing = _FakeFlux(lambda total: [None] if total >= FRAME_BYTES * 3 else [])  # Flux hangs up
    assert _run(closing, _mic(3)) == "again"


def test_the_mic_thread_is_stopped_once_the_turn_is_decided() -> None:
    stopped = threading.Event()
    fake = _FakeFlux(
        lambda total: [TurnEnded("done", 0.9, "model")] if total >= FRAME_BYTES * 3 else []
    )

    def listen(
        _device: object,
        stop: StopFlag,
        sink: Callable[[bytes, bool], None],
        **_kwargs: Any,
    ) -> bool:
        for _ in range(3):
            sink(FRAME, True)
        while not stop.is_set():
            time.sleep(0.005)
        stopped.set()
        return False

    assert isinstance(_run(fake, listen), StreamedTurn)
    assert stopped.is_set()  # it did not keep recording after the answer


def _loud(level: int = 1000) -> bytes:
    return level.to_bytes(2, "little", signed=True) * (FRAME_BYTES // 2)


class _EnergyVad:
    """A voice-activity detector that calls any frame with sound in it speech."""

    def is_speech(self, frame: bytes, _rate: int) -> bool:
        return any(frame)


def test_the_mic_gate_sends_nothing_until_speech_starts_and_then_the_pre_roll_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fish_audio_suite_voice import listen

    silence = b"\x00\x00" * (FRAME_BYTES // 2)
    quiet_then_speech = [silence] * 30 + [_loud()] * 12 + [silence] * 8
    levels: list[Event] = []
    stop_listening = EVENTS.subscribe(levels.append)

    def frames(*_args: Any, **_kwargs: Any) -> Any:
        yield from quiet_then_speech

    monkeypatch.setattr(listen, "mic_frames", frames)
    tune = ListenTune(
        start_speech_frames=2, pre_pad_frames=8, end_silence_frames=4, min_speech_rms=100.0
    )
    got: list[tuple[bytes, bool]] = []
    try:
        local_end = listen.stream_utterance(
            None,
            None,
            lambda frame, voiced: got.append((frame, voiced)),
            tune=tune,
            vad=_EnergyVad(),
        )
    finally:
        stop_listening()
    assert local_end  # the silence after the speech ended it
    assert got  # speech was found
    # Nothing from the 30 quiet frames before the pre-roll was sent: only speech leaves the machine.
    assert len(got) < len(quiet_then_speech)
    assert any(voiced for _, voiced in got)
    first_voiced = next(index for index, (_, voiced) in enumerate(got) if voiced)
    assert first_voiced < 8  # the pre-roll comes first, so the first words are not clipped
    assert any(event.__class__.__name__ == "MicLevel" for event in levels)  # the meter still moves


def test_the_mic_gate_returns_false_when_it_is_stopped_before_any_speech(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fish_audio_suite_voice import listen

    def frames(*_args: Any, **_kwargs: Any) -> Any:
        yield b"\x00\x00" * (FRAME_BYTES // 2)

    monkeypatch.setattr(listen, "mic_frames", frames)
    got: list[bytes] = []
    assert not listen.stream_utterance(
        None, None, lambda frame, _voiced: got.append(frame), vad=_EnergyVad()
    )
    assert got == []


def test_the_connection_goes_to_the_regional_endpoint_that_was_chosen() -> None:
    from dataclasses import replace

    urls: list[str] = []

    def make(url: str, _key: str) -> _FakeFlux:
        urls.append(url)
        return _FakeFlux(open_errors=[DeepgramError("stop here")])

    async def go(region: str) -> object:
        return await stream_turn(
            stt=replace(STT, deepgram_region=region),
            listen=ListenTune(),
            device=None,
            aec=None,
            quit_requested=threading.Event(),
            stop=threading.Event(),
            make_stream=make,
            listen_fn=_mic(0),
        )

    for region in ("global", "eu"):
        assert asyncio.run(go(region)) == "fallback"
    assert urls[0].startswith("wss://api.deepgram.com/v2/listen?")
    assert urls[1].startswith("wss://api.eu.deepgram.com/v2/listen?")
    assert all("mip_opt_out=true" in url for url in urls)  # whichever region, nothing is kept
