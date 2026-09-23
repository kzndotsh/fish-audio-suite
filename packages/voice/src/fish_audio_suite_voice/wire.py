"""Fish websocket audio: text events, the pump, and how much was spoken."""

from __future__ import annotations

import asyncio
import contextlib
import threading
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, cast

import httpx
from fishaudio import AsyncFishAudio, FlushEvent, TextEvent
from fishaudio.exceptions import APIError, ValidationError, WebSocketError
from fishaudio.types import AudioFormat, LatencyMode, Model, TTSConfig

from fish_audio_suite_kit import (
    FISH_RETRY_ATTEMPTS,
    elapsed_ms,
    fish_attempt_exhausted,
    fish_request_error,
    should_retry_fish_status,
    skip_empty_delta,
    split_tts_piece,
)
from fish_audio_suite_voice.aec import SAMPLE_BYTES
from fish_audio_suite_voice.debug import debug, warn
from fish_audio_suite_voice.playback import PlaybackSink

ANEXT_POLL_S = 0.25


@dataclass(frozen=True)
class IsolatedResult:
    """What one TTS turn actually played.

    Attributes
    ----------
    spoken_so_far : str
        The prefix that reached the sink, not the full unplayed reply.
        Barge-in history must use this.
    bytes_played : int
        Bytes the sink accepted.
    got_audio : bool
        True after the first Fish audio chunk.
    cancelled : bool
        True when barge-in or Ctrl+C stopped the turn.
    error_status : int or None
        Fish HTTP status when the turn failed. 401, 402, and 403 end duplex.
    """

    spoken_so_far: str
    bytes_played: int
    got_audio: bool
    cancelled: bool
    ttfa_ms: float | None
    llm_ttfs_ms: float | None
    error_status: int | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class TurnSpec:
    """Inputs for one Fish websocket. Built by ``IsolatedFishTts``, not by apps.

    Notes
    -----
    ``trace_headers`` are copied onto the httpx client that upgrades the
    socket. ``partial_chars`` is the cut size from ``SuiteDefaults``.
    """

    api_key: str
    base_url: str
    voice_id: str
    model: str
    audio_format: str
    latency: str
    speed: float
    sample_rate: int
    partial_chars: int
    trace_headers: dict[str, str]
    config: TTSConfig


@dataclass
class Heard:
    """Audio that actually arrived on this Fish websocket."""

    got_audio: bool = False
    ttfa_ms: float | None = None


@dataclass
class EventAcc:
    """Text events and first-sentence timing collected during one turn."""

    flushed: list[str] = field(default_factory=list)
    ttfs_ms: float | None = None


@dataclass
class TurnRun:
    """Mutable state for one Fish websocket turn."""

    spec: TurnSpec
    sink: PlaybackSink
    cancel: threading.Event
    sent_text: str
    acc: EventAcc
    t0: float
    audio: Heard
    err_status: int | None = None
    err_message: str | None = None


async def text_events(
    prepared: str,
    cancel: threading.Event,
    partial_chars: int,
) -> AsyncIterator[Any]:
    """Yield Fish text events for a whole reply, then one flush.

    Parameters
    ----------
    prepared : str
        Already scrubbed text.
    cancel : threading.Event
        Stops before the next piece. A cancel after zero pieces yields no flush.
    partial_chars : int
        Passed to ``split_tts_piece``. ``flush_rest`` is True, so the tail
        is not left buffered.

    Yields
    ------
    TextEvent or FlushEvent
        Empty pieces are skipped. ``FlushEvent`` is yielded only after at
        least one ``TextEvent``. A bare flush on an empty turn is invalid.
    """
    buf = prepared
    sent = 0
    while buf:
        if cancel.is_set():
            break
        split = split_tts_piece(buf, partial_chars, flush_rest=True)
        if split is None:
            break
        piece, buf = split
        if skip_empty_delta(piece):
            continue
        yield TextEvent(text=piece)
        sent += 1
    flush = flush_if_sent(sent, cancel)
    if flush is not None:
        yield flush


def flush_if_sent(sent: int, cancel: threading.Event) -> FlushEvent | None:
    """Return a flush only when text was sent and the turn was not cancelled.

    Parameters
    ----------
    sent : int
        How many ``TextEvent`` values were yielded.
    cancel : threading.Event
        When set, return None so Fish is not asked to flush a dead turn.

    Returns
    -------
    FlushEvent or None
        None when ``sent`` is 0. An empty turn plus a bare flush is invalid.
    """
    if sent and not cancel.is_set():
        return FlushEvent()
    return None


async def as_async(deltas: Iterable[str] | AsyncIterable[str]) -> AsyncIterator[str]:
    """Yield text deltas from either an async or a plain iterable."""
    if isinstance(deltas, AsyncIterable):
        async for item in deltas:
            yield item
        return
    for item in deltas:
        yield item


_CANCEL_NOISE = (
    "athrow",
    "cancel scope",
    "generator didn't stop",
    "different task than it was entered",
)


def is_cancel_noise(exc: BaseException) -> bool:
    """Return whether ``exc`` is a barge-in or Ctrl+C tear-down, not a Fish error.

    Parameters
    ----------
    exc : BaseException
        An error from the websocket task or a task group.

    Returns
    -------
    bool
        True for ``CancelledError``, ``GeneratorExit``, and a few anyio or
        asyncio messages that show up when the client is closed mid-stream.
    """
    if isinstance(exc, (asyncio.CancelledError, GeneratorExit)):
        return True
    msg = str(exc).lower()
    return any(phrase in msg for phrase in _CANCEL_NOISE)


async def send_turn(
    client: AsyncFishAudio,
    events: AsyncIterator[Any],
    run: TurnRun,
    *,
    close_client: Callable[[], Awaitable[None]],
) -> None:
    """Open one Fish realtime websocket and write audio to the sink.

    Parameters
    ----------
    client : AsyncFishAudio
        Client whose base URL and trace headers are already set.
    events : AsyncIterator
        Text events. Ignored when ``run.sent_text`` is set, because the turn
        is replayed from that string.
    run : TurnRun
        Mutable turn state. First-audio time and flushed text land here.
    close_client : Callable
        Awaited on barge-in or Ctrl+C. This ends the socket. Do not also
        ``aclose`` the stream iterator.

    Notes
    -----
    ``TTSConfig`` has no ``features``. Quality-guard stays on the proxy HTTP path.
    """
    spec = run.spec
    run.acc.flushed.clear()
    run.acc.ttfs_ms = None
    replay = text_events(run.sent_text, run.cancel, spec.partial_chars) if run.sent_text else events
    stream = client.tts.stream_websocket(
        _tee_text_events(replay, run.t0, run.acc),
        reference_id=spec.voice_id,
        format=cast(AudioFormat, spec.audio_format),
        latency=cast(LatencyMode, spec.latency),
        speed=spec.speed,
        config=spec.config,
        model=cast(Model, spec.model),
    )
    await _pump_ws_audio(stream, run, close_client)


def _note_first_audio(audio: Heard, chunk: bytes, t0: float) -> None:
    if audio.got_audio:
        return
    audio.got_audio = True
    audio.ttfa_ms = elapsed_ms(t0)
    print("  [tts first audio ttfa]", flush=True)
    debug(
        "tts.first_audio ttfa_ms={:.0f} chunk={}",
        audio.ttfa_ms,
        len(chunk),
    )


async def _pump_ws_audio(
    stream: AsyncIterator[Any],
    run: TurnRun,
    close_client: Callable[[], Awaitable[None]],
) -> None:
    it = aiter(stream)
    pending: asyncio.Task[Any] = asyncio.create_task(_anext_chunk(it))
    try:
        while True:
            if run.cancel.is_set():
                debug("tts.cancel before/during stream")
                await close_client()
                return
            if not await _wait_task(pending, ANEXT_POLL_S):
                continue
            try:
                chunk = pending.result()
            except StopAsyncIteration:
                return
            pending = asyncio.create_task(_anext_chunk(it))
            if not chunk:
                continue
            _note_first_audio(run.audio, chunk, run.t0)
            if run.cancel.is_set():
                await close_client()
                return
            run.sink.write(chunk)
    finally:
        if not pending.done():
            pending.cancel()
            with contextlib.suppress(BaseException):
                await pending


def _remember_event(ev: Any, acc: EventAcc, t0: float) -> None:
    if isinstance(ev, TextEvent) and ev.text:
        text = ev.text
        acc.flushed.append(text)
        if acc.ttfs_ms is None:
            acc.ttfs_ms = elapsed_ms(t0)
            debug(
                "tts.text_event chars={} ttfs_ms={:.0f}",
                len(text),
                acc.ttfs_ms,
            )
        return
    if isinstance(ev, FlushEvent):
        debug("tts.flush")


async def _tee_text_events(
    events: AsyncIterator[Any],
    t0: float,
    acc: EventAcc,
) -> AsyncIterator[Any]:
    async for ev in events:
        _remember_event(ev, acc, t0)
        yield ev


def isolated_result(run: TurnRun) -> IsolatedResult:
    """Build the result returned after an isolated speak finishes."""
    spec = run.spec
    audio = run.audio
    if not audio.got_audio and not run.cancel.is_set() and run.err_status is None:
        warn(f"[tts] no audio voice={spec.voice_id} model={spec.model}")
    played = run.sink.bytes_played()
    full = run.sent_text or "".join(run.acc.flushed)
    spoken = _spoken_prefix(
        full,
        bytes_played=played,
        sample_rate=spec.sample_rate,
        audio_format=spec.audio_format,
        got_audio=audio.got_audio,
        cancelled=run.cancel.is_set(),
    )
    return IsolatedResult(
        spoken_so_far=spoken,
        bytes_played=played,
        got_audio=audio.got_audio,
        cancelled=run.cancel.is_set(),
        ttfa_ms=audio.ttfa_ms,
        llm_ttfs_ms=run.acc.ttfs_ms,
        error_status=run.err_status,
        error_message=run.err_message,
    )


_CHARS_PER_S = 16


def _pcm_chars(bytes_played: int, sample_rate: int) -> int:
    secs = bytes_played / max(sample_rate * SAMPLE_BYTES, 1)
    return max(1, int(secs * _CHARS_PER_S))


def _word_prefix(text: str, n: int) -> str:
    cut = text[:n]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut.strip()


def _spoken_prefix(
    sent_text: str,
    *,
    bytes_played: int,
    sample_rate: int,
    audio_format: str,
    got_audio: bool,
    cancelled: bool,
) -> str:
    if not sent_text.strip() or not got_audio:
        return ""
    if cancelled and bytes_played <= 0:
        return ""
    if cancelled and audio_format == "pcm":
        return _word_prefix(sent_text, _pcm_chars(bytes_played, sample_rate))
    return sent_text.strip()


def _root_exc(exc: BaseException) -> BaseException:
    cur = exc
    while isinstance(cur, BaseExceptionGroup) and cur.exceptions:
        cur = cur.exceptions[0]
    return cur


async def _anext_chunk(it: AsyncIterator[Any]) -> Any:
    return await anext(it)


async def _wait_task(task: asyncio.Task[Any], wait_s: float) -> bool:
    """Return whether the task finished.

    A timeout must not cancel the task. ``asyncio.wait_for`` would kill the Fish websocket.
    """
    done, _ = await asyncio.wait({task}, timeout=wait_s)
    return bool(done)


async def quiet_shutdown(loop: asyncio.AbstractEventLoop) -> None:
    """Cancel leftover tasks on a private loop before it is closed.

    Parameters
    ----------
    loop : asyncio.AbstractEventLoop
        The loop ``run_isolated`` created. The current task is left running
        so this coroutine can finish the gather.
    """
    current = asyncio.current_task()
    pending = [task for task in asyncio.all_tasks(loop) if task is not current]
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)


@dataclass(frozen=True)
class _TtsFailure:
    retry: bool
    err_status: int | None = None
    err_message: str | None = None


def turn_failure(
    exc: BaseException,
    *,
    attempt: int,
    sent_text: str,
    got_audio: bool,
    cancel: threading.Event,
) -> _TtsFailure:
    """Decide whether a Fish exception can be retried.

    Parameters
    ----------
    exc : BaseException
        Error from one attempt. Exception groups are unwrapped to the root.
    attempt : int
        Zero-based attempt. The fifth is final.
    sent_text : str
        Text available to replay. Empty blocks a retry.
    got_audio : bool
        True after the first audio byte. Retries stop there so the listener
        does not hear the sentence twice.
    cancel : threading.Event
        When set, the failure is a barge-in, not a retry.

    Returns
    -------
    _TtsFailure
        ``retry`` is True only for 429 or 5xx before audio, with text to replay,
        and attempts left.
    """
    root = _root_exc(exc)
    retry, status, message = _classify_fish_exc(root)
    if is_cancel_noise(root) or cancel.is_set():
        return _TtsFailure(retry=False)
    last = fish_attempt_exhausted(attempt)
    can_replay = bool(sent_text) and not got_audio
    if retry and not last and can_replay:
        warn(f"[tts] retry status={status} attempt={attempt + 1}/{FISH_RETRY_ATTEMPTS}")
        return _TtsFailure(retry=True)
    err_message = message or str(root)
    warn(f"[tts] {status} {err_message}" if status is not None else f"[tts] {err_message}")
    return _TtsFailure(retry=False, err_status=status, err_message=err_message)


def _classify_fish_exc(exc: BaseException) -> tuple[bool, int | None, str]:
    if is_cancel_noise(exc):
        return False, None, ""
    if isinstance(exc, APIError):
        return should_retry_fish_status(exc.status), exc.status, exc.message
    if isinstance(exc, ValidationError):
        return False, 400, str(exc)
    if isinstance(exc, WebSocketError):
        return True, None, str(exc)
    if isinstance(exc, httpx.RequestError):
        status, message = fish_request_error(exc, httpx.TimeoutException)
        return True, status, message
    return False, None, str(exc)
