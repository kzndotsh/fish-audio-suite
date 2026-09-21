"""One Fish TTS websocket per turn, isolated from the caller's event loop."""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass, field
from typing import Any

from fishaudio import AsyncFishAudio, FlushEvent, TextEvent
from fishaudio.types import Prosody, TTSConfig

from fish_audio_suite_kit import next_tts_cut, normalize_cues, scrub_tts, skip_empty_delta
from fish_audio_suite_kit.defaults import (
    DEFAULT_CHUNK_LENGTH,
    DEFAULT_LATENCY,
    DEFAULT_MIN_CHUNK_LENGTH,
    DEFAULT_REPETITION_PENALTY,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_SPEED,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_TTS_MODEL,
    DEFAULT_TTS_PARTIAL_CHARS,
)
from fish_audio_suite_voice.playback import PlaybackSink


@dataclass(frozen=True)
class IsolatedResult:
    spoken_so_far: str
    bytes_played: int
    got_audio: bool
    cancelled: bool
    ttfa_ms: float | None
    llm_ttfs_ms: float | None


class IsolatedFishTts:
    """Fish `stream_websocket` on a fresh loop. Do not merge the LLM socket onto this WS."""

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str,
        model: str = DEFAULT_TTS_MODEL,
        latency: str = DEFAULT_LATENCY,
        speed: float = DEFAULT_SPEED,
        audio_format: str = "pcm",
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
        chunk_length: int = DEFAULT_CHUNK_LENGTH,
        min_chunk_length: int = DEFAULT_MIN_CHUNK_LENGTH,
        volume: float = 0.0,
        partial_chars: int = DEFAULT_TTS_PARTIAL_CHARS,
    ) -> None:
        self.api_key = api_key
        self.voice_id = voice_id
        self.model = model
        self.latency = latency
        self.speed = speed
        self.audio_format = audio_format
        self.sample_rate = sample_rate
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.chunk_length = chunk_length
        self.min_chunk_length = min_chunk_length
        self.volume = volume
        self.partial_chars = partial_chars

    def speak_isolated(
        self,
        text: str,
        sink: PlaybackSink,
        cancel: threading.Event | None = None,
    ) -> IsolatedResult:
        cancel = cancel or threading.Event()

        async def _run() -> IsolatedResult:
            return await self.speak(text, sink, cancel)

        try:
            return asyncio.run(_run())
        except (asyncio.CancelledError, BaseExceptionGroup, RuntimeError, GeneratorExit) as e:
            msg = str(e).lower()
            if not any(x in msg for x in ("cancel scope", "athrow", "generator didn't stop")):
                if type(e) not in (BaseExceptionGroup, GeneratorExit, asyncio.CancelledError):
                    print(f"[tts] {e}", file=sys.stderr)
            return IsolatedResult("", 0, False, cancel.is_set(), None, None)

    async def speak(
        self,
        text: str,
        sink: PlaybackSink,
        cancel: threading.Event,
    ) -> IsolatedResult:
        prepared = normalize_cues(scrub_tts(text, dialogue_only=False))

        async def events() -> AsyncIterator[Any]:
            buf = prepared
            sent = 0
            while buf:
                if cancel.is_set():
                    break
                cut = next_tts_cut(buf, partial_chars=self.partial_chars)
                if cut < 0:
                    piece = buf
                    buf = ""
                else:
                    piece = buf[:cut]
                    buf = buf[cut:]
                if skip_empty_delta(piece):
                    continue
                yield TextEvent(text=piece)
                sent += 1
            if sent and not cancel.is_set():
                yield FlushEvent()

        return await self._stream(events(), sink, cancel, sent_text=prepared)

    async def speak_deltas(
        self,
        deltas: Iterable[str] | AsyncIterator[str],
        sink: PlaybackSink,
        cancel: threading.Event,
    ) -> IsolatedResult:
        async def events() -> AsyncIterator[Any]:
            buf = ""
            sent = 0
            async for tok in _as_async(deltas):
                if cancel.is_set():
                    break
                if skip_empty_delta(tok):
                    continue
                buf += tok
                while True:
                    cut = next_tts_cut(buf, partial_chars=self.partial_chars)
                    if cut < 0:
                        break
                    piece = buf[:cut]
                    buf = buf[cut:]
                    if skip_empty_delta(piece):
                        continue
                    yield TextEvent(text=normalize_cues(scrub_tts(piece)))
                    sent += 1
            if buf.strip() and not cancel.is_set():
                yield TextEvent(text=normalize_cues(scrub_tts(buf)))
                sent += 1
            if sent and not cancel.is_set():
                yield FlushEvent()

        return await self._stream(events(), sink, cancel, sent_text="")

    async def _stream(
        self,
        events: AsyncIterator[Any],
        sink: PlaybackSink,
        cancel: threading.Event,
        *,
        sent_text: str,
    ) -> IsolatedResult:
        client = AsyncFishAudio(api_key=self.api_key)
        got_audio = False
        ttfa_ms: float | None = None
        acc = _EventAcc()
        t0 = time.perf_counter()
        sink.start()
        try:
            cfg = TTSConfig(
                format=self.audio_format,  # type: ignore[arg-type]
                mp3_bitrate=128,
                sample_rate=self.sample_rate,
                latency=self.latency,  # type: ignore[arg-type]
                normalize=True,
                chunk_length=self.chunk_length,
                min_chunk_length=self.min_chunk_length,
                temperature=self.temperature,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                max_new_tokens=1024,
                condition_on_previous_chunks=True,
                prosody=Prosody(speed=self.speed, volume=self.volume),
            )
            logged_events = _tee_text_events(events, t0, acc)

            stream = client.tts.stream_websocket(
                logged_events,
                reference_id=self.voice_id,
                format=self.audio_format,  # type: ignore[arg-type]
                latency=self.latency,  # type: ignore[arg-type]
                speed=self.speed,
                config=cfg,
                model=self.model,  # type: ignore[arg-type]
            )
            async for chunk in stream:
                if cancel.is_set():
                    break
                if chunk:
                    if not got_audio:
                        got_audio = True
                        ttfa_ms = (time.perf_counter() - t0) * 1000
                        print("  [tts first audio ttfa]", flush=True)
                    sink.write(chunk)
        except (asyncio.CancelledError, GeneratorExit):
            pass
        except BaseExceptionGroup:
            pass
        except RuntimeError as e:
            if "cancel scope" not in str(e).lower() and "athrow" not in str(e).lower():
                if not cancel.is_set():
                    print(f"[tts] {e}", file=sys.stderr)
        except Exception as e:
            msg = str(e).lower()
            if "athrow" in msg or "cancel scope" in msg or "generator didn't stop" in msg:
                pass
            elif not cancel.is_set():
                print(f"[tts] {e}", file=sys.stderr)
        finally:
            sink.finish(kill=cancel.is_set())
            try:
                close = getattr(client, "aclose", None) or client.close
                res = close()
                if hasattr(res, "__await__"):
                    await res
            except Exception:
                pass

        full = sent_text or "".join(acc.flushed)
        spoken = _spoken_prefix(
            full,
            bytes_played=sink.bytes_played(),
            sample_rate=self.sample_rate,
            audio_format=self.audio_format,
            got_audio=got_audio,
            cancelled=cancel.is_set(),
        )
        return IsolatedResult(
            spoken_so_far=spoken,
            bytes_played=sink.bytes_played(),
            got_audio=got_audio,
            cancelled=cancel.is_set(),
            ttfa_ms=ttfa_ms,
            llm_ttfs_ms=acc.ttfs_ms,
        )


@dataclass
class _EventAcc:
    flushed: list[str] = field(default_factory=list)
    ttfs_ms: float | None = None


def _tee_text_events(
    events: AsyncIterator[Any],
    t0: float,
    acc: _EventAcc,
) -> AsyncIterator[Any]:
    async def gen() -> AsyncIterator[Any]:
        async for ev in events:
            if isinstance(ev, TextEvent) and getattr(ev, "text", None):
                acc.flushed.append(ev.text)
                if acc.ttfs_ms is None:
                    acc.ttfs_ms = (time.perf_counter() - t0) * 1000
            yield ev

    return gen()


def _spoken_prefix(
    sent_text: str,
    *,
    bytes_played: int,
    sample_rate: int,
    audio_format: str,
    got_audio: bool,
    cancelled: bool,
) -> str:
    if not sent_text.strip():
        return ""
    if not got_audio:
        return ""
    if not cancelled:
        return sent_text.strip()
    if bytes_played <= 0:
        return ""
    if audio_format == "pcm":
        secs = bytes_played / max(sample_rate * 2, 1)
        n = max(1, int(secs * 16))
        cut = sent_text[:n]
        if " " in cut:
            cut = cut.rsplit(" ", 1)[0]
        return cut.strip()
    return sent_text.strip()


async def _as_async(deltas: Iterable[str] | AsyncIterator[str]) -> AsyncIterator[str]:
    if hasattr(deltas, "__anext__") or hasattr(deltas, "__aiter__"):
        async for item in deltas:  # type: ignore[union-attr]
            yield item
        return
    for item in deltas:  # type: ignore[union-attr]
        yield item
