"""One Fish TTS websocket per turn, isolated from the caller's event loop."""

from __future__ import annotations

import asyncio
import sys
import threading
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass, field
from typing import Any, cast

from fishaudio import TextEvent
from fishaudio.types import AudioFormat, LatencyMode, Prosody, TTSConfig

from fish_audio_suite_kit import (
    SuiteDefaults,
    known_mp3_bitrate,
    normalize_cues,
    scrub_tts,
    skip_empty_delta,
    split_tts_piece,
    strip_base,
)
from fish_audio_suite_voice.playback import PlaybackSink
from fish_audio_suite_voice.session import (
    IsolatedResult,
    TurnSpec,
    as_async,
    is_cancel_noise,
    run_isolated,
    run_turn,
    text_events,
)
from fish_audio_suite_voice.wire import flush_if_sent

_STOCK = SuiteDefaults()


def _spoken(text: str) -> str:
    return normalize_cues(scrub_tts(text))


def _quiet_result(cancelled: bool) -> IsolatedResult:
    return IsolatedResult(
        spoken_so_far="",
        bytes_played=0,
        got_audio=False,
        cancelled=cancelled,
        ttfa_ms=None,
        llm_ttfs_ms=None,
    )


@dataclass(kw_only=True, repr=False, eq=False)
class IsolatedFishTts:
    """Fish `stream_websocket` on a fresh loop. Do not merge the LLM socket onto this WS."""

    api_key: str
    voice_id: str
    model: str = _STOCK.tts_model
    latency: str = _STOCK.latency
    speed: float = _STOCK.speed
    audio_format: str = "pcm"
    sample_rate: int = _STOCK.sample_rate
    temperature: float = _STOCK.temperature
    top_p: float = _STOCK.top_p
    repetition_penalty: float = _STOCK.repetition_penalty
    chunk_length: int = _STOCK.chunk_length
    min_chunk_length: int = _STOCK.min_chunk_length
    volume: float = _STOCK.volume
    partial_chars: int = _STOCK.tts_partial_chars
    base_url: str = _STOCK.fish_base
    trace_headers: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.base_url = strip_base(self.base_url)
        self.trace_headers = dict(self.trace_headers)

    def speak_isolated(
        self,
        text: str,
        sink: PlaybackSink,
        cancel: threading.Event | None = None,
    ) -> IsolatedResult:
        """Run Fish WS on a private thread and loop. Safe from asyncio.run / to_thread."""
        cancel = cancel or threading.Event()
        result: IsolatedResult | None = None

        def worker() -> None:
            nonlocal result
            try:
                result = run_isolated(self.speak(text, sink, cancel))
            except (asyncio.CancelledError, BaseExceptionGroup, RuntimeError, GeneratorExit) as e:
                if not is_cancel_noise(e):
                    print(f"[tts] {e}", file=sys.stderr)
                result = _quiet_result(cancel.is_set())

        thread = threading.Thread(target=worker, name="fish-tts", daemon=True)
        thread.start()
        thread.join()
        if result is None:
            return _quiet_result(cancel.is_set())
        return result

    async def speak(
        self,
        text: str,
        sink: PlaybackSink,
        cancel: threading.Event,
    ) -> IsolatedResult:
        prepared = _spoken(text)
        return await run_turn(
            self._spec(),
            text_events(prepared, cancel, self.partial_chars),
            sink,
            cancel,
            sent_text=prepared,
        )

    async def speak_deltas(
        self,
        deltas: Iterable[str] | AsyncIterator[str],
        sink: PlaybackSink,
        cancel: threading.Event,
    ) -> IsolatedResult:
        return await run_turn(
            self._spec(),
            self._delta_events(deltas, cancel),
            sink,
            cancel,
            sent_text="",
        )

    async def _delta_events(
        self,
        deltas: Iterable[str] | AsyncIterator[str],
        cancel: threading.Event,
    ) -> AsyncIterator[Any]:
        buf = ""
        sent = 0

        def counted(piece: str) -> TextEvent | None:
            nonlocal sent
            event = self._text_event(piece)
            if event is None:
                return None
            sent += 1
            return event

        async for tok in as_async(deltas):
            if cancel.is_set():
                break
            if skip_empty_delta(tok):
                continue
            buf += tok
            while True:
                split = split_tts_piece(buf, self.partial_chars, flush_rest=False)
                if split is None:
                    break
                piece, buf = split
                event = counted(piece)
                if event is not None:
                    yield event
        tail = split_tts_piece(buf, self.partial_chars, flush_rest=True)
        if tail is not None and not cancel.is_set():
            event = counted(tail[0])
            if event is not None:
                yield event
        flush = flush_if_sent(sent, cancel)
        if flush is not None:
            yield flush

    def _text_event(self, piece: str) -> TextEvent | None:
        if skip_empty_delta(piece):
            return None
        return TextEvent(text=_spoken(piece))

    def _spec(self) -> TurnSpec:
        return TurnSpec(
            api_key=self.api_key,
            base_url=self.base_url,
            voice_id=self.voice_id,
            model=self.model,
            audio_format=self.audio_format,
            latency=self.latency,
            speed=self.speed,
            sample_rate=self.sample_rate,
            partial_chars=self.partial_chars,
            trace_headers=self.trace_headers,
            config=self._tts_config(),
        )

    def _tts_config(self) -> TTSConfig:
        return TTSConfig(
            format=cast(AudioFormat, self.audio_format),
            mp3_bitrate=known_mp3_bitrate(_STOCK.mp3_bitrate),
            sample_rate=self.sample_rate,
            latency=cast(LatencyMode, self.latency),
            normalize=_STOCK.normalize,
            chunk_length=self.chunk_length,
            min_chunk_length=self.min_chunk_length,
            temperature=self.temperature,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            max_new_tokens=_STOCK.max_new_tokens,
            condition_on_previous_chunks=_STOCK.condition_on_previous_chunks,
            early_stop_threshold=_STOCK.early_stop_threshold,
            prosody=Prosody(speed=self.speed, volume=self.volume),
        )


__all__ = ["IsolatedFishTts", "IsolatedResult", "is_cancel_noise"]
