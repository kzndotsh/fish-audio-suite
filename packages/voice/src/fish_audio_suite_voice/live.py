"""One Fish TTS websocket per turn, isolated from the caller's event loop."""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from collections.abc import AsyncIterable, AsyncIterator, Iterable
from dataclasses import dataclass, field
from typing import Any, cast

import httpx
from fishaudio import AsyncFishAudio, FlushEvent, TextEvent
from fishaudio.exceptions import APIError, ValidationError, WebSocketError
from fishaudio.types import AudioFormat, LatencyMode, Model, Prosody, TTSConfig

from fish_audio_suite_kit import (
    FISH_RETRY_ATTEMPTS,
    SuiteDefaults,
    fish_backoff_seconds,
    make_traceparent,
    next_tts_cut,
    normalize_cues,
    scrub_tts,
    should_retry_fish_status,
    skip_empty_delta,
    w3c_trace_headers,
)
from fish_audio_suite_voice.playback import PlaybackSink

_STOCK = SuiteDefaults()


@dataclass(frozen=True)
class IsolatedResult:
    spoken_so_far: str
    bytes_played: int
    got_audio: bool
    cancelled: bool
    ttfa_ms: float | None
    llm_ttfs_ms: float | None
    error_status: int | None = None
    error_message: str | None = None


class IsolatedFishTts:
    """Fish `stream_websocket` on a fresh loop. Do not merge the LLM socket onto this WS."""

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str,
        model: str = _STOCK.tts_model,
        latency: str = _STOCK.latency,
        speed: float = _STOCK.speed,
        audio_format: str = "pcm",
        sample_rate: int = _STOCK.sample_rate,
        temperature: float = _STOCK.temperature,
        top_p: float = _STOCK.top_p,
        repetition_penalty: float = _STOCK.repetition_penalty,
        chunk_length: int = _STOCK.chunk_length,
        min_chunk_length: int = _STOCK.min_chunk_length,
        volume: float = 0.0,
        partial_chars: int = _STOCK.tts_partial_chars,
        base_url: str = _STOCK.fish_base,
        trace_headers: dict[str, str] | None = None,
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
        self.base_url = base_url.rstrip("/")
        self.trace_headers = dict(trace_headers or {})

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
            if not any(
                x in msg for x in ("cancel scope", "athrow", "generator didn't stop")
            ) and type(e) not in (BaseExceptionGroup, GeneratorExit, asyncio.CancelledError):
                print(f"[tts] {e}", file=sys.stderr)
            return IsolatedResult("", 0, False, cancel.is_set(), None, None)

    async def speak(
        self,
        text: str,
        sink: PlaybackSink,
        cancel: threading.Event,
    ) -> IsolatedResult:
        prepared = normalize_cues(scrub_tts(text))
        return await self._stream(
            _text_events(prepared, cancel, self.partial_chars),
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
        extra = w3c_trace_headers(self.trace_headers)
        if not extra:
            extra = {"traceparent": make_traceparent()}
        got_audio = False
        ttfa_ms: float | None = None
        err_status: int | None = None
        err_message: str | None = None
        acc = _EventAcc()
        t0 = time.perf_counter()
        sink.start()
        client: AsyncFishAudio | None = None

        async def close_client() -> None:
            nonlocal client
            if client is None:
                return
            try:
                close = getattr(client, "aclose", None) or client.close
                res = close()
                if hasattr(res, "__await__"):
                    await res
            except Exception:
                pass
            client = None

        try:
            for attempt in range(FISH_RETRY_ATTEMPTS):
                http = httpx.AsyncClient(
                    base_url=self.base_url,
                    headers=extra,
                    timeout=httpx.Timeout(240.0),
                    http2=False,
                )
                client = AsyncFishAudio(
                    api_key=self.api_key,
                    base_url=self.base_url,
                    httpx_client=http,
                )
                try:
                    cfg = TTSConfig(
                        format=cast(AudioFormat, self.audio_format),
                        mp3_bitrate=128,
                        sample_rate=self.sample_rate,
                        latency=cast(LatencyMode, self.latency),
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
                    acc.flushed.clear()
                    acc.ttfs_ms = None
                    replay = (
                        _text_events(sent_text, cancel, self.partial_chars) if sent_text else events
                    )
                    logged_events = _tee_text_events(replay, t0, acc)
                    stream = client.tts.stream_websocket(
                        logged_events,
                        reference_id=self.voice_id,
                        format=cast(AudioFormat, self.audio_format),
                        latency=cast(LatencyMode, self.latency),
                        speed=self.speed,
                        config=cfg,
                        model=cast(Model, self.model),
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
                    break
                except (asyncio.CancelledError, GeneratorExit):
                    break
                except BaseException as exc:
                    if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                        raise
                    root = _root_exc(exc)
                    retry, status, message = _classify_fish_exc(root)
                    if _is_cancel_noise(root) or cancel.is_set():
                        await close_client()
                        break
                    last = attempt + 1 >= FISH_RETRY_ATTEMPTS
                    can_replay = bool(sent_text) and not got_audio
                    if got_audio or not retry or last or not can_replay:
                        err_status = status
                        err_message = message or str(root)
                        print(
                            f"[tts] {err_status} {err_message}"
                            if err_status is not None
                            else f"[tts] {err_message}",
                            file=sys.stderr,
                        )
                        await close_client()
                        break
                    print(
                        f"[tts] retry status={status} attempt={attempt + 1}/{FISH_RETRY_ATTEMPTS}",
                        file=sys.stderr,
                    )
                    await close_client()
                    await asyncio.sleep(fish_backoff_seconds(attempt))
        finally:
            sink.finish(kill=cancel.is_set())
            await close_client()

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
            error_status=err_status,
            error_message=err_message,
        )


@dataclass
class _EventAcc:
    flushed: list[str] = field(default_factory=list)
    ttfs_ms: float | None = None


async def _text_events(
    prepared: str,
    cancel: threading.Event,
    partial_chars: int,
) -> AsyncIterator[Any]:
    buf = prepared
    sent = 0
    while buf:
        if cancel.is_set():
            break
        cut = next_tts_cut(buf, partial_chars=partial_chars)
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


async def _as_async(deltas: Iterable[str] | AsyncIterable[str]) -> AsyncIterator[str]:
    if isinstance(deltas, AsyncIterable):
        async for item in deltas:
            yield item
        return
    for item in deltas:
        yield item


def _root_exc(exc: BaseException) -> BaseException:
    cur = exc
    while isinstance(cur, BaseExceptionGroup) and cur.exceptions:
        cur = cur.exceptions[0]
    return cur


def _is_cancel_noise(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "athrow" in msg or "cancel scope" in msg or "generator didn't stop" in msg


def _classify_fish_exc(exc: BaseException) -> tuple[bool, int | None, str]:
    if _is_cancel_noise(exc):
        return False, None, ""
    if isinstance(exc, APIError):
        return should_retry_fish_status(exc.status), exc.status, exc.message
    if isinstance(exc, ValidationError):
        return False, 400, str(exc)
    if isinstance(exc, WebSocketError):
        return True, None, str(exc)
    if isinstance(exc, httpx.TimeoutException):
        return True, 504, "Fish request timed out"
    if isinstance(exc, httpx.RequestError):
        return True, 502, str(exc) or "Fish upstream unreachable"
    return False, None, str(exc)
