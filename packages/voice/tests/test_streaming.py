"""A turn streamed to Deepgram Flux, driven by a fake connection and a fake mic thread."""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import sys
import threading
import time
import wave
from collections.abc import AsyncIterator, Callable, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from voice_fakes import ScriptedMic

from fish_audio_suite_voice import streaming
from fish_audio_suite_voice.cancel import StopFlag
from fish_audio_suite_voice.deepgram import FluxStream
from fish_audio_suite_voice.events import EVENTS, Event, Interim
from fish_audio_suite_voice.mic import Mic
from fish_audio_suite_voice.pcm import FRAME_BYTES
from fish_audio_suite_voice.portaudio import PortAudioMissingError
from fish_audio_suite_voice.recognised import BargeIn, NoTurn, StreamFallback, Transcript
from fish_audio_suite_voice.streaming import stream_turn
from fish_audio_suite_voice.tune import (
    DEFAULT_BARGE_RMS,
    DeepgramTune,
    ListenTune,
    SttProvider,
    SttTune,
)
from fish_audio_suite_voice.turn_stream import (
    EagerTurnEnd,
    RecogniserSettings,
    StreamError,
    SttStreamError,
    TurnDropped,
    TurnEnded,
    TurnMessage,
    TurnResumed,
    TurnStarted,
    TurnUpdate,
)

CHUNK = streaming._SEND_BYTES  # 80 ms of audio: the size Deepgram recommends
FRAME = b"\x01\x00" * (FRAME_BYTES // 2)
STT = SttTune(provider="deepgram", deepgram=DeepgramTune(key="secret"))
# Most tests are about the stream, so one voiced frame is enough to open it.
ONE_VOICED = ListenTune(min_voiced_frames=1)


class _FakeFlux(FluxStream):
    """A Flux connection that records what it is sent and answers from a script."""

    def __init__(
        self,
        on_audio: Callable[[int], Sequence[TurnMessage | None]] | None = None,
        on_force: list[TurnMessage] | None = None,
        *,
        open_errors: list[SttStreamError] | None = None,
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
        self._inbox: asyncio.Queue[TurnMessage | None] = asyncio.Queue()  # None closes it

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
        answers: Sequence[TurnMessage | None] = (
            self._on_audio(sum(map(len, self.sent))) if self._on_audio else ()
        )
        for message in answers:
            self._inbox.put_nowait(message)

    async def force_end_turn(self) -> None:
        self.forced += 1
        for message in self._on_force:
            self._inbox.put_nowait(message)

    async def messages(self) -> AsyncIterator[TurnMessage]:
        while (message := await self._inbox.get()) is not None:
            yield message

    async def close(self) -> None:
        self.closed = True
        self._inbox.put_nowait(None)


def _mic(
    frames: int,
    *,
    local_end: bool = False,
    voiced: bool | Sequence[bool] = True,
    seen: dict[str, Any] | None = None,
) -> Callable[..., bool]:
    """A mic thread that hands over ``frames`` frames, then waits to be stopped.

    ``voiced`` is one flag for every frame, or one per frame.
    """

    def listen(
        _mic: Mic,
        stop: StopFlag,
        sink: Callable[[bytes, bool], None],
        *,
        carried: BargeIn | None,
        tune: ListenTune,
        since: int | None = None,
    ) -> bool:
        if seen is not None:
            seen.update(carried=carried, tune=tune)
        for index in range(frames):
            sink(FRAME, voiced if isinstance(voiced, bool) else voiced[index])
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
    fish_key: str = "",
    stop: StopFlag | None = None,
    carried: BargeIn | None = None,
    listen: ListenTune = ONE_VOICED,
    mic: ScriptedMic | None = None,
    **callbacks: Any,
) -> Transcript | NoTurn | StreamFallback:
    quit_flag = threading.Event()

    async def go() -> Transcript | NoTurn | StreamFallback:
        return await stream_turn(
            settings=RecogniserSettings(stt, fish_key=fish_key),
            listen=listen,
            mic=mic or ScriptedMic(),
            quit_requested=quit_flag,
            stop=stop or threading.Event(),
            carried=carried,
            make_stream=lambda _url, _key: fake,
            listen_fn=listen_fn,
            **callbacks,
        )

    return asyncio.run(asyncio.wait_for(go(), 10))


def _events() -> tuple[list[Event], Callable[[], None]]:
    seen: list[Event] = []
    return seen, EVENTS.subscribe(seen.append)


def test_a_turn_is_sent_as_it_is_spoken_and_ends_when_flux_says_so() -> None:
    eighty_ms = CHUNK
    script = {
        eighty_ms: [TurnStarted("hel")],
        eighty_ms * 2: [TurnUpdate("hello there")],
        eighty_ms * 3: [TurnEnded("hello there", 0.8, "model")],
    }
    fake = _FakeFlux(lambda total: script.get(total, []))
    seen, stop = _events()
    try:
        result = _run(fake, _mic(9))
    finally:
        stop()
    assert isinstance(result, Transcript)
    assert result.text == "hello there"
    assert result.asr_ms >= 0.0
    assert result.trace_id
    assert [len(chunk) for chunk in fake.sent] == [eighty_ms] * 3  # 80 ms at a time
    assert fake.forced == 0  # Flux decided; no need to ask
    assert fake.closed
    interim = [event.text for event in seen if isinstance(event, Interim)]
    assert interim == ["hel", "hello there"]  # the words as they arrived, once each


def test_nothing_is_sent_while_waiting_for_speech() -> None:
    fake = _FakeFlux()
    stop = threading.Event()
    stop.set()  # a typed line, or quit, before anyone spoke
    result = _run(fake, _mic(0), stop=stop)
    assert result == NoTurn("stopped")
    assert fake.sent == []  # only speech leaves the machine
    assert fake.closed


def test_the_pre_roll_and_a_barge_prefix_are_handed_to_the_mic_side_first() -> None:
    seen: dict[str, Any] = {}
    fake = _FakeFlux(lambda total: [TurnEnded("go on", 0.9, "model")] if total >= CHUNK else [])
    clip = BargeIn(b"\x07\x00" * 480, None, DEFAULT_BARGE_RMS)
    result = _run(fake, _mic(3, seen=seen), carried=clip)
    assert isinstance(result, Transcript)
    assert seen["carried"] == clip
    assert isinstance(seen["tune"], ListenTune)


def test_when_the_silence_limit_comes_first_flux_is_asked_to_end_the_turn() -> None:
    fake = _FakeFlux(
        lambda total: [TurnUpdate("what time is it")] if total >= CHUNK else [],
        on_force=[TurnEnded("what time is it", 0.5, "manual")],
    )
    result = _run(fake, _mic(4, local_end=True))
    assert isinstance(result, Transcript)
    assert result.text == "what time is it"
    assert fake.forced == 1
    assert sum(map(len, fake.sent)) == FRAME_BYTES * 4  # the leftover frame was flushed too


def test_when_flux_never_answers_a_forced_end_the_words_so_far_are_used(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(streaming, "_FORCE_WAIT_S", 0.05)
    fake = _FakeFlux(lambda total: [TurnUpdate("tell me a joke")] if total >= CHUNK else [])
    result = _run(fake, _mic(3, local_end=True))
    assert isinstance(result, Transcript)
    assert result.text == "tell me a joke"
    assert fake.forced == 1


def test_a_turn_with_no_words_is_noise(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeFlux(lambda total: [TurnEnded("  ", 0.9, "model")] if total >= CHUNK else [])
    assert _run(fake, _mic(3)) == NoTurn("noise")
    monkeypatch.setattr(streaming, "_FORCE_WAIT_S", 0.05)
    assert _run(_FakeFlux(), _mic(3, local_end=True)) == NoTurn("noise")  # nothing came back at all


def test_a_missing_key_and_a_refused_connection_are_fatal_but_an_outage_hands_over_the_speech() -> (
    None
):
    no_key = SttTune(provider="deepgram", deepgram=DeepgramTune(key=""))
    assert _run(_FakeFlux(), _mic(1), stt=no_key) == NoTurn("fatal")
    assert _run(_FakeFlux(open_errors=[SttStreamError("HTTP 401", fatal=True)]), _mic(1)) == NoTurn(
        "fatal"
    )
    outage = _run(_FakeFlux(open_errors=[SttStreamError("no route")]), _mic(3))
    # Deepgram could not be reached once speech had started, so the batch recogniser takes
    # over from the speech captured so far, and the first words are not lost.
    assert isinstance(outage, StreamFallback)
    assert outage.pcm.startswith(FRAME)
    assert len(outage.pcm) >= FRAME_BYTES


def test_an_outage_hands_over_the_frames_the_mic_was_still_passing_on() -> None:
    tail = b"\x02\x00" * (FRAME_BYTES // 2)

    def listen(
        _mic: Mic,
        stop: StopFlag,
        sink: Callable[[bytes, bool], None],
        *,
        carried: BargeIn | None,
        tune: ListenTune,
        since: int | None = None,
    ) -> bool:
        del carried, tune
        for _ in range(3):
            sink(FRAME, True)
        while not stop.is_set():
            time.sleep(0.005)
        # Frames the mic thread had already handed over when the turn let go of it.
        sink(tail, True)
        sink(tail, True)
        return False

    outage = _run(_FakeFlux(open_errors=[SttStreamError("no route")]), listen)
    assert isinstance(outage, StreamFallback)
    assert outage.pcm == FRAME * 3 + tail * 2


def test_a_turn_cancelled_while_the_mic_lets_go_still_closes_the_stream() -> None:
    released = threading.Event()

    def listen(
        _mic: Mic,
        stop: StopFlag,
        sink: Callable[[bytes, bool], None],
        *,
        carried: BargeIn | None,
        tune: ListenTune,
        since: int | None = None,
    ) -> bool:
        del carried, tune
        for _ in range(3):
            sink(FRAME, True)
        while not stop.is_set():
            time.sleep(0.005)
        released.wait(5)  # a mic thread slow to let go
        return False

    fake = _FakeFlux(open_errors=[SttStreamError("no route")])

    async def go() -> None:
        turn = asyncio.create_task(
            stream_turn(
                settings=RecogniserSettings(STT),
                listen=ONE_VOICED,
                mic=ScriptedMic(),
                quit_requested=threading.Event(),
                stop=threading.Event(),
                make_stream=lambda _url, _key: fake,
                listen_fn=listen,
            )
        )
        await asyncio.sleep(0.1)  # the outage has happened and the mic is being released
        turn.cancel()
        await asyncio.sleep(0.05)
        released.set()
        with contextlib.suppress(asyncio.CancelledError):
            await turn

    asyncio.run(asyncio.wait_for(go(), 10))
    assert fake.closed


def test_nothing_is_connected_until_speech_starts_and_then_it_is_connected_once() -> None:
    idle = _FakeFlux()
    stop = threading.Event()
    stop.set()
    assert _run(idle, _mic(0), stop=stop) == NoTurn("stopped")
    assert (
        idle.opens == 0
    )  # an idle connection would be closed by Deepgram, and sends nothing anyway

    spoken = _FakeFlux(lambda total: [TurnEnded("hi", 0.9, "model")] if total >= CHUNK else [])
    assert isinstance(_run(spoken, _mic(3)), Transcript)
    assert spoken.opens == 1


def test_a_forced_end_with_no_turn_to_end_is_noise_straight_away_not_after_the_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(streaming, "_FORCE_WAIT_S", 5.0)  # it must not wait this long
    # Flux took the sound for no speech, so it answers ForceEndTurn with a warning, not an
    # EndOfTurn, and its parser reads that warning as a dropped turn.
    fake = _FakeFlux(on_force=[TurnDropped()])
    started = time.monotonic()
    assert _run(fake, _mic(3, local_end=True)) == NoTurn("noise")
    assert time.monotonic() - started < 2.0
    assert fake.forced == 1


def test_an_error_from_flux_or_a_dropped_connection_mid_turn_means_listen_again() -> None:
    erroring = _FakeFlux(
        lambda total: [StreamError("INTERNAL_SERVER_ERROR", "oops")] if total >= CHUNK else []
    )
    assert _run(erroring, _mic(3)) == NoTurn("again")
    closing = _FakeFlux(lambda total: [None] if total >= CHUNK else [])  # Flux hangs up
    assert _run(closing, _mic(3)) == NoTurn("again")


def test_the_mic_thread_is_stopped_once_the_turn_is_decided() -> None:
    stopped = threading.Event()
    fake = _FakeFlux(lambda total: [TurnEnded("done", 0.9, "model")] if total >= CHUNK else [])

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

    assert isinstance(_run(fake, listen), Transcript)
    assert stopped.is_set()  # it did not keep recording after the answer


def _loud(level: int = 1000) -> bytes:
    return level.to_bytes(2, "little", signed=True) * (FRAME_BYTES // 2)


class _EnergyVad:
    """A voice-activity detector that calls any frame with sound in it speech."""

    def is_speech(self, frame: bytes, _rate: int) -> bool:
        return any(frame)


def _energy_engine(monkeypatch: pytest.MonkeyPatch) -> str:
    """Install ``_EnergyVad`` as a detector plugin and return its engine name."""
    plugin = ModuleType("energy_vad_plugin")
    vars(plugin)["make"] = lambda _strictness: _EnergyVad()
    monkeypatch.setitem(sys.modules, "energy_vad_plugin", plugin)
    return "energy_vad_plugin:make"


def test_the_mic_gate_sends_nothing_until_speech_starts_and_then_the_pre_roll_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fish_audio_suite_voice import listen

    # Each quiet frame has its own level, all under the speech floor, so the pre-roll
    # can be checked frame by frame.
    quiet = [
        level.to_bytes(2, "little", signed=True) * (FRAME_BYTES // 2) for level in range(1, 31)
    ]
    silence = b"\x00\x00" * (FRAME_BYTES // 2)
    # Under eight times the floor, so two voiced frames start the speech and the
    # pre-roll still holds quiet frames. A spike needs ten, which fill the pre-roll.
    speech = _loud(300)
    script = [*quiet, *[speech] * 12, *[silence] * 8]
    levels: list[Event] = []
    stop_listening = EVENTS.subscribe(levels.append)
    tune = ListenTune(
        start_speech_frames=2,
        pre_pad_frames=8,
        end_silence_frames=4,
        min_speech_rms=100.0,
        vad_engine=_energy_engine(monkeypatch),
    )
    got: list[bytes] = []
    try:
        local_end = listen.stream_utterance(
            ScriptedMic(script),
            None,
            lambda frame, _voiced: got.append(frame),
            tune=tune,
        )
    finally:
        stop_listening()
    assert local_end  # the silence after the speech ended it
    first_loud = got.index(speech)
    # The pre-roll comes first, so the first words are not clipped: the quiet frames
    # just before the speech, in order. Nothing older leaves the machine.
    assert first_loud == tune.pre_pad_frames - tune.start_speech_frames
    assert got[:first_loud] == quiet[-first_loud:]
    assert got[first_loud:].count(speech) == 12
    assert any(event.__class__.__name__ == "MicLevel" for event in levels)  # the meter still moves


def test_the_mic_gate_returns_false_when_it_is_stopped_before_any_speech(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fish_audio_suite_voice import listen

    silence = b"\x00\x00" * (FRAME_BYTES // 2)
    stop = threading.Event()

    def frames(flag: StopFlag | None) -> Any:
        # An endless quiet mic. Only the stop flag reaching it can end the listen.
        assert flag is stop
        for index in itertools.count():
            if index == 5:
                stop.set()
            if index > 200:
                raise AssertionError("the stop flag never ended the listen")
            yield silence

    got: list[bytes] = []
    assert not listen.stream_utterance(
        ScriptedMic(frames),
        stop,
        lambda frame, _voiced: got.append(frame),
        tune=ListenTune(vad_engine=_energy_engine(monkeypatch)),
    )
    assert got == []


def test_the_connection_goes_to_the_regional_endpoint_that_was_chosen() -> None:
    from dataclasses import replace

    urls: list[str] = []

    def make(url: str, _key: str) -> _FakeFlux:
        urls.append(url)
        return _FakeFlux(open_errors=[SttStreamError("stop here")])

    async def go(region: str) -> object:
        return await stream_turn(
            settings=RecogniserSettings(
                replace(STT, deepgram=replace(STT.deepgram, region=region))
            ),
            listen=ONE_VOICED,
            mic=ScriptedMic(),
            quit_requested=threading.Event(),
            stop=threading.Event(),
            make_stream=make,
            listen_fn=_mic(1),
        )

    for region in ("global", "eu"):
        assert isinstance(asyncio.run(go(region)), StreamFallback)
    assert urls[0].startswith("wss://api.deepgram.com/v2/listen?")
    assert urls[1].startswith("wss://api.eu.deepgram.com/v2/listen?")
    assert all("mip_opt_out=true" in url for url in urls)  # whichever region, nothing is kept


def test_a_knock_or_a_cough_is_dropped_before_anything_is_connected_or_sent() -> None:
    needs_voice = ListenTune(min_voiced_frames=4)
    for voiced in (False, [True, False, True, False, False]):  # no voice, or too little of it
        fake = _FakeFlux()
        assert _run(fake, _mic(5, local_end=True, voiced=voiced), listen=needs_voice) == NoTurn(
            "noise"
        )
        assert fake.opens == 0
        assert fake.sent == []


def test_stopping_while_the_first_frames_are_held_is_not_noise() -> None:
    stop = threading.Event()
    stop.set()
    fake = _FakeFlux()
    result = _run(fake, _mic(2, voiced=False), listen=ListenTune(min_voiced_frames=4), stop=stop)
    assert result == NoTurn("stopped")
    assert fake.opens == 0


def test_once_there_is_enough_voice_the_held_pre_roll_goes_first_and_in_order() -> None:
    fake = _FakeFlux(lambda total: [TurnEnded("hi", 0.9, "model")] if total >= CHUNK * 2 else [])
    pattern = [False, False, True, True, True, True]  # two quiet pre-roll frames, then speech
    result = _run(fake, _mic(6, voiced=pattern), listen=ListenTune(min_voiced_frames=3))
    assert isinstance(result, Transcript)
    assert fake.opens == 1
    sent = b"".join(fake.sent)
    assert len(sent) >= CHUNK * 2
    assert sent == (FRAME * 6)[: len(sent)]  # the pre-roll was not lost, and it is in order


def test_the_audio_sent_is_saved_as_a_wav_only_when_a_folder_is_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dataclasses import replace

    fake = _FakeFlux(lambda total: [TurnEnded("hi", 0.9, "model")] if total >= CHUNK else [])
    assert isinstance(_run(fake, _mic(3)), Transcript)
    assert not list(tmp_path.iterdir())  # nothing is written unless asked

    folder = tmp_path / "heard"
    fake = _FakeFlux(lambda total: [TurnEnded("hi", 0.9, "model")] if total >= CHUNK else [])
    assert isinstance(_run(fake, _mic(3), stt=replace(STT, save_dir=str(folder))), Transcript)
    (saved,) = folder.glob("stt-*.wav")
    with wave.open(str(saved)) as clip:
        assert clip.getframerate() == 16000
        assert clip.readframes(clip.getnframes()) == b"".join(fake.sent)

    blocked = tmp_path / "a-file"
    blocked.write_text("x")  # a folder cannot be made under a file: it warns and the turn goes on
    fake = _FakeFlux(lambda total: [TurnEnded("hi", 0.9, "model")] if total >= CHUNK else [])
    result = _run(fake, _mic(3), stt=replace(STT, save_dir=str(blocked / "sub")))
    assert isinstance(result, Transcript)


def test_an_early_end_of_turn_and_a_resumed_turn_are_passed_on_as_they_happen() -> None:
    told: list[str] = []
    script = {
        CHUNK: [EagerTurnEnd("so I was"), TurnResumed(), EagerTurnEnd("so I was saying")],
        CHUNK * 2: [TurnEnded("so I was saying", 0.9, "model")],
    }
    fake = _FakeFlux(lambda total: script.get(total, []))
    result = _run(
        fake,
        _mic(6),
        on_eager=lambda text: told.append(f"eager {text}"),
        on_resumed=lambda: told.append("resumed"),
    )
    assert isinstance(result, Transcript)
    assert told == ["eager so I was", "resumed", "eager so I was saying"]


def test_a_short_barge_in_clip_counts_as_speech_in_full_and_is_not_dropped_as_noise() -> None:
    # The interrupt gate already accepted 4 voiced frames, which is fewer than the 12 a fresh
    # utterance needs. The batch path credits them, and so must this one.
    prefix = FRAME * 5
    fake = _FakeFlux(lambda total: [TurnEnded("stop", 0.9, "model")] if total >= CHUNK else [])
    result = _run(
        fake,
        _mic(5, voiced=[True, True, True, True, False]),
        listen=ListenTune(min_voiced_frames=12),
        carried=BargeIn(prefix, None, DEFAULT_BARGE_RMS),
    )
    assert isinstance(result, Transcript)
    assert fake.opens == 1


def test_a_short_clip_with_no_barge_in_is_still_dropped_as_noise() -> None:
    fake = _FakeFlux()
    result = _run(
        fake,
        _mic(5, voiced=[True, True, True, True, False], local_end=True),
        listen=ListenTune(min_voiced_frames=12),
    )
    assert result == NoTurn("noise")
    assert fake.opens == 0


def test_a_turn_with_no_words_from_flux_is_given_up_on_early_as_noise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(streaming, "_NO_WORDS_S", 0.05)
    monkeypatch.setattr(streaming, "_FORCE_WAIT_S", 5.0)  # it must not come to this
    fake = _FakeFlux()  # taps: audio goes in, no words come back
    started = time.monotonic()
    assert _run(fake, _mic(3)) == NoTurn("noise")  # the mic is never ended: it gave up by itself
    assert time.monotonic() - started < 2.0
    assert fake.forced == 0
    assert fake.closed


def test_a_turn_whose_words_arrive_in_time_is_not_given_up_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(streaming, "_NO_WORDS_S", 0.3)
    script = {CHUNK: [TurnUpdate("hello")], CHUNK * 2: [TurnEnded("hello there", 0.9, "model")]}
    fake = _FakeFlux(lambda total: script.get(total, []))
    result = _run(fake, _mic(6))
    assert isinstance(result, Transcript)
    assert result.text == "hello there"


def test_fish_realtime_connects_to_the_fish_base_with_the_fish_key_and_the_eager_setting() -> None:
    from fish_audio_suite_voice.fish_realtime import FISH_REALTIME, FishRealtimeStream

    made: list[tuple[str, str]] = []

    def make(url: str, key: str) -> _FakeFlux:
        made.append((url, key))
        return _FakeFlux(open_errors=[SttStreamError("stop here")])

    async def go(fish_key: str = "", fish_base: str = "https://api.fish.audio") -> object:
        return await stream_turn(
            settings=RecogniserSettings(
                SttTune(provider="fish-realtime"), fish_key=fish_key, fish_base=fish_base
            ),
            listen=ONE_VOICED,
            mic=ScriptedMic(),
            quit_requested=threading.Event(),
            stop=threading.Event(),
            make_stream=make,
            listen_fn=_mic(1),
        )

    assert asyncio.run(go()) == NoTurn("fatal")  # no Fish key: nothing is connected
    assert made == []
    outcome = asyncio.run(go(fish_key="fish-secret", fish_base="https://fish.example"))
    assert isinstance(outcome, StreamFallback)  # an outage still hands the speech to Fish ASR
    assert made == [("wss://fish.example/v1/asr/realtime", "fish-secret")]

    for threshold, eager in ((0.0, False), (0.5, True)):
        settings = RecogniserSettings(
            SttTune(provider="fish-realtime", eager_eot_threshold=threshold),
            fish_key="k",
            fish_base="https://f",
        )
        stream = FISH_REALTIME.stream(
            FISH_REALTIME.url(settings), FISH_REALTIME.key(settings), settings
        )
        assert isinstance(stream, FishRealtimeStream)
        assert stream._config["turn_detection"] == {"mode": "auto", "eager": eager}


def test_a_turn_the_recogniser_drops_is_noise_even_after_words_arrived() -> None:
    fake = _FakeFlux(
        on_audio=lambda sent: [TurnUpdate("mm-hmm"), TurnDropped()] if sent == CHUNK else []
    )
    events, stop_events = _events()
    try:
        assert _run(fake, _mic(6)) == NoTurn("noise")
    finally:
        stop_events()
    assert Interim("mm-hmm") in events
    assert fake.closed


def test_fish_realtime_keeps_the_local_silence_limit_behind_fishs_own_end_of_turn() -> None:
    def heard_tune(provider: SttProvider, frames: int) -> ListenTune:
        seen: dict[str, Any] = {}
        fake = _FakeFlux(on_audio=lambda sent: [TurnEnded("hi", 1.0, "model")] if sent else [])
        _run(
            fake,
            _mic(6, seen=seen),
            stt=SttTune(provider=provider, deepgram=DeepgramTune(key="k")),
            listen=ListenTune(min_voiced_frames=1, end_silence_frames=frames),
            fish_key="k",
        )
        return seen["tune"]

    # 25 frames is 750 ms, and Fish needs about 1.3 s, so the limit is raised to 3 s.
    assert heard_tune("fish-realtime", 25).end_silence_frames == 100
    assert heard_tune("fish-realtime", 120).end_silence_frames == 120  # a longer one is kept
    assert heard_tune("deepgram", 25).end_silence_frames == 25  # Flux answers well inside it


def test_an_outage_hands_over_where_the_mic_was_read_up_to_so_nothing_is_lost() -> None:
    # The batch listen used to read from "now", so frames that arrived between the stream
    # letting go of the mic and the batch listen starting were never heard.
    def listen(
        mic: Mic,
        stop: StopFlag,
        sink: Callable[[bytes, bool], None],
        *,
        carried: BargeIn | None,
        tune: ListenTune,
        since: int | None = None,
    ) -> bool:
        del carried, tune
        for frame in mic.frames(stop, since=since):
            sink(frame, True)
        return False

    outage = _run(
        _FakeFlux(open_errors=[SttStreamError("no route")]), listen, mic=ScriptedMic([FRAME] * 3)
    )
    assert isinstance(outage, StreamFallback)
    assert outage.since == len(outage.frames) == 3  # read from position 0, three frames
    assert all(speech for _frame, speech in outage.frames)  # each frame keeps its verdict


def test_a_mic_that_cannot_open_is_fatal_as_it_is_on_the_batch_path() -> None:
    def listen(*_args: object, **_kwargs: object) -> bool:
        raise PortAudioMissingError("no PortAudio")

    fake = _FakeFlux()
    assert _run(fake, listen) == NoTurn("fatal")
    assert fake.opens == 0
