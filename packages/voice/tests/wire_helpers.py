"""Fakes and builders shared by the wire, turn, cancel, tts-text and playback tests."""

import asyncio
import threading
import time
from collections.abc import AsyncIterator, Callable, Iterable, Sequence
from typing import Any

import pytest
from fishaudio.types import Prosody, TTSConfig

from fish_audio_suite_kit import CueStyle
from fish_audio_suite_voice.tts.playback import PlaybackSink, SounddeviceSink
from fish_audio_suite_voice.tts.speaker import FishSpeaker
from fish_audio_suite_voice.tts.text import WholeText, delta_events, text_events
from fish_audio_suite_voice.tts.tune import TtsTune
from fish_audio_suite_voice.tts.turn import run_turn
from fish_audio_suite_voice.tts.websocket import (
    AudioArrival,
    FlushEvent,
    SentText,
    TextEvent,
    TtsResult,
    TurnRun,
    TurnSpec,
)


class ScriptedSocket:
    """A ``FishSocket`` that plays one script per attempt: audio chunks, then maybe an error.

    Each attempt reads every text event first, as Fish does before it speaks. The last
    script repeats when there are more attempts than scripts.
    """

    def __init__(self, *attempts: Sequence[bytes | BaseException]) -> None:
        self._attempts = attempts or ((),)
        self.streams = 0
        self.closed = 0

    def stream(self, events: AsyncIterator[Any], spec: TurnSpec) -> AsyncIterator[bytes]:
        del spec
        plan = self._attempts[min(self.streams, len(self._attempts) - 1)]
        self.streams += 1

        async def audio() -> AsyncIterator[bytes]:
            async for _event in events:
                pass
            for step in plan:
                if isinstance(step, BaseException):
                    raise step
                yield step

        return audio()

    async def close(self) -> None:
        self.closed += 1


class RecordingSink:
    output_latency_s = 0.0

    def __init__(self) -> None:
        self.chunks: list[bytes] = []

    def start(self) -> None:
        return None

    def write(self, chunk: bytes) -> None:
        self.chunks.append(chunk)

    def finish(self, *, kill: bool = False) -> None:
        return None

    def bytes_played(self) -> int:
        return sum(len(chunk) for chunk in self.chunks)


def make_run() -> tuple[TurnRun, RecordingSink]:
    sink = RecordingSink()
    spec = TurnSpec(
        api_key="k",
        base_url="https://api.fish.audio",
        voice_id="v",
        model="s2.1-pro",
        trace_headers={},
        config=TTSConfig(
            format="pcm", latency="balanced", sample_rate=44100, prosody=Prosody(speed=1.0)
        ),
    )
    run = TurnRun(
        spec=spec,
        sink=sink,
        cancel=threading.Event(),
        sent_text="",
        acc=SentText(),
        t0=time.perf_counter(),
        audio=AudioArrival(),
    )
    return run, sink


def collect_events(
    prepared: str, cancel: threading.Event, partial: int
) -> list[TextEvent | FlushEvent]:
    async def collect() -> list[TextEvent | FlushEvent]:
        return [event async for event in text_events(prepared, cancel, partial, style=CueStyle())]

    return asyncio.run(collect())


def stream_events(tts: FishSpeaker, deltas: Iterable[str], cancel: threading.Event):
    return delta_events(
        deltas, cancel, partial_chars=tts.tune.partial_chars, style=CueStyle(lead=True)
    )


def stream_text(deltas: list[str], partial_chars: int) -> str:
    tts = FishSpeaker(api_key="k", voice_id="v", tune=TtsTune(partial_chars=partial_chars))

    async def collect() -> str:
        events = [event async for event in stream_events(tts, deltas, threading.Event())]
        return "".join(event.text for event in events if isinstance(event, TextEvent))

    return asyncio.run(collect())


def event_kinds(events: list[object]) -> list[str]:
    return [type(event).__name__ for event in events]


class FakeOutputStream:
    """A sounddevice ``RawOutputStream`` that keeps what is written to it.

    ``on_write`` runs on each write and its result is the underflow flag PortAudio
    returns (True when the device ran dry before this write).
    """

    def __init__(
        self, on_write: Callable[[bytes], bool] | None = None, *, latency: float = 0.0
    ) -> None:
        self.on_write = on_write
        self.latency = latency
        self.written: list[bytes] = []
        self.opened_with: dict[str, object] = {}
        self.stopped = False
        self.aborted = False
        self.closed = False

    def start(self) -> None:
        return None

    def write(self, chunk: bytes) -> bool:
        self.written.append(bytes(chunk))
        return self.on_write(chunk) if self.on_write is not None else False

    def stop(self) -> None:
        self.stopped = True

    def abort(self) -> None:
        self.aborted = True

    def close(self) -> None:
        self.closed = True


def install_output_device(monkeypatch: pytest.MonkeyPatch, stream: FakeOutputStream) -> None:
    """Make every sounddevice sink that starts open ``stream`` instead of PortAudio."""

    class Sd:
        @staticmethod
        def RawOutputStream(**kwargs: object) -> FakeOutputStream:  # noqa: N802 - sounddevice's name
            stream.opened_with = kwargs
            return stream

    monkeypatch.setattr("fish_audio_suite_voice.tts.playback.load_sounddevice", lambda: Sd)


def open_sounddevice_sink(
    monkeypatch: pytest.MonkeyPatch, stream: FakeOutputStream, **kwargs: Any
) -> SounddeviceSink:
    """Start a ``SounddeviceSink`` on ``stream``."""
    install_output_device(monkeypatch, stream)
    sink = SounddeviceSink(**kwargs)
    sink.start()
    return sink


class StreamSocket:
    """A ``FishSocket`` whose one stream is ``audio``, whatever text it is sent."""

    def __init__(self, audio: AsyncIterator[bytes]) -> None:
        self._audio = audio
        self.closed = 0

    def stream(self, events: AsyncIterator[Any], spec: TurnSpec) -> AsyncIterator[bytes]:
        del events, spec
        return self._audio

    async def close(self) -> None:
        self.closed += 1


async def play_stream(
    audio: AsyncIterator[bytes], sink: RecordingSink, cancel: threading.Event
) -> tuple[TtsResult, StreamSocket]:
    """Run one turn whose websocket yields ``audio``, on the running loop."""
    run, _sink = make_run()
    socket = StreamSocket(audio)
    result = await run_turn(
        run.spec, WholeText("hello there friend", 40, CueStyle()), sink, cancel, socket=socket
    )
    return result, socket


def install_fish_audio(monkeypatch: pytest.MonkeyPatch, chunks: Sequence[bytes]) -> list[TurnSpec]:
    """Make each Fish websocket a ``FishSpeaker`` opens read the text, then send ``chunks``.

    Returns the list each websocket's ``TurnSpec`` is added to as it opens.
    """
    seen: list[TurnSpec] = []

    class Socket:
        def __init__(self, headers: dict[str, str]) -> None:
            del headers

        def stream(self, events: AsyncIterator[Any], spec: TurnSpec) -> AsyncIterator[bytes]:
            seen.append(spec)

            async def audio() -> AsyncIterator[bytes]:
                async for _event in events:
                    pass
                for chunk in chunks:
                    yield chunk

            return audio()

        async def close(self) -> None:
            return None

    monkeypatch.setattr("fish_audio_suite_voice.tts.turn.SdkSocket", Socket)
    return seen


def spec_sent(
    monkeypatch: pytest.MonkeyPatch,
    tts: FishSpeaker,
    sink: PlaybackSink | None = None,
    *,
    trace_headers: dict[str, str] | None = None,
) -> TurnSpec:
    """Speak one line with ``tts`` and return the ``TurnSpec`` the websocket was opened with."""
    seen = install_fish_audio(monkeypatch, ())
    tts.speak("Hello.", sink or RecordingSink(), trace_headers=trace_headers)
    return seen[0]
