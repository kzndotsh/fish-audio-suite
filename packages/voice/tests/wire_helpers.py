"""Fakes and builders shared by the wire, turn, cancel and stream-scrub tests."""

import asyncio
import threading
import time
from collections.abc import Iterable

from fishaudio.types import TTSConfig

from fish_audio_suite_voice.speaker import FishSpeaker
from fish_audio_suite_voice.stream_scrub import delta_events
from fish_audio_suite_voice.wire import (
    AudioArrival,
    FlushEvent,
    SentText,
    TextEvent,
    TurnRun,
    TurnSpec,
    text_events,
)


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
        audio_format="pcm",
        latency="balanced",
        speed=1.0,
        sample_rate=44100,
        partial_chars=40,
        trace_headers={},
        config=TTSConfig(),
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
        return [event async for event in text_events(prepared, cancel, partial)]

    return asyncio.run(collect())


def stream_events(tts: FishSpeaker, deltas: Iterable[str], cancel: threading.Event):
    return delta_events(deltas, cancel, partial_chars=tts.partial_chars, mood_lead=True)


def stream_text(deltas: list[str], partial_chars: int) -> str:
    tts = FishSpeaker(api_key="k", voice_id="v", partial_chars=partial_chars)

    async def collect() -> str:
        events = [event async for event in stream_events(tts, deltas, threading.Event())]
        return "".join(event.text for event in events if isinstance(event, TextEvent))

    return asyncio.run(collect())


def event_kinds(events: list[object]) -> list[str]:
    return [type(event).__name__ for event in events]
