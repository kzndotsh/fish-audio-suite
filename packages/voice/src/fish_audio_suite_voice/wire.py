"""Fish websocket audio: text events, the pump, and how much was spoken."""

from __future__ import annotations

import asyncio
import contextlib
import threading
from collections.abc import (
    AsyncIterable,
    AsyncIterator,
    Awaitable,
    Callable,
    Iterable,
    Mapping,
)
from dataclasses import dataclass, field
from typing import Any, Final, cast

import httpx
from fishaudio import AsyncFishAudio, FlushEvent, TextEvent
from fishaudio.exceptions import APIError, ValidationError, WebSocketError
from fishaudio.types import LatencyMode, Model, TTSConfig

from fish_audio_suite_kit import (
    FISH_RETRY_ATTEMPTS,
    AudioFormat,
    FishHttpError,
    describe_request_error,
    elapsed_ms,
    fish_attempt_exhausted,
    is_empty_delta,
    should_retry_fish_status,
    split_tts_piece,
)
from fish_audio_suite_voice.cancel import is_cancel_noise, reap
from fish_audio_suite_voice.debug import debug, warn, with_detail
from fish_audio_suite_voice.declick import EdgeFade
from fish_audio_suite_voice.playback import PlaybackSink
from fish_audio_suite_voice.spoken import spoken_prefix

__all__ = [
    "AudioArrival",
    "SentText",
    "TtsFailure",
    "TtsResult",
    "TurnRun",
    "TurnSpec",
    "as_async",
    "flush_if_sent",
    "reap",
    "send_turn",
    "text_events",
    "tts_result",
    "turn_failure",
]

# How often the cancel watcher checks that its turn is still running. Cancel
# itself wakes the pump at once; this only bounds how long the watcher thread
# outlives a turn that ended without one.
_CANCEL_WATCH_S: Final = 1.0


@dataclass(frozen=True, slots=True)
class TtsResult:
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
    tts_first_audio_ms : float or None
        Milliseconds from the start of the TTS turn to the first audio chunk from
        Fish (Fish's time-to-first-audio). None when no audio arrived.
    tts_first_text_ms : float or None
        Milliseconds from the start of the TTS turn to the first text event sent
        to Fish. None when no text was sent.
    error_status : int or None
        Fish HTTP status when the turn failed. A socket drop has no status.
    error_message : str or None
        Text of the failure, set for a socket drop too.
    error : FishHttpError or None
        The failure as a kit error, set when there is a status. Test it with
        ``isinstance``: ``FishAuthError`` (401, 402, 403) ends duplex.
    """

    spoken_so_far: str
    bytes_played: int
    got_audio: bool
    cancelled: bool
    tts_first_audio_ms: float | None
    tts_first_text_ms: float | None
    error_status: int | None = None
    error_message: str | None = None
    error: FishHttpError | None = None

    def __post_init__(self) -> None:
        """Derive ``error`` from ``error_status`` when only the status was given."""
        if self.error is None and self.error_status is not None:
            error = FishHttpError.from_status(self.error_status, self.error_message or "")
            object.__setattr__(self, "error", error)


@dataclass(frozen=True, slots=True)
class TurnSpec:
    """Inputs for one Fish websocket. Built by ``FishSpeaker``, not by apps.

    Notes
    -----
    ``trace_headers`` are copied onto the httpx client that upgrades the
    socket. ``partial_chars`` is the cut size from ``SuiteDefaults``.
    """

    api_key: str = field(repr=False)
    base_url: str
    voice_id: str
    model: str
    audio_format: AudioFormat
    latency: LatencyMode
    speed: float
    sample_rate: int
    partial_chars: int
    trace_headers: Mapping[str, str]
    config: TTSConfig
    fade_ms: float = 0.0


@dataclass(slots=True)
class AudioArrival:
    """Audio that actually arrived on this Fish websocket."""

    got_audio: bool = False
    tts_first_audio_ms: float | None = None


@dataclass(slots=True)
class SentText:
    """Every text piece sent to Fish during one turn, and when the first one went out."""

    pieces: list[str] = field(default_factory=list)
    tts_first_text_ms: float | None = None


@dataclass(slots=True)
class TurnRun:
    """Mutable state for one Fish websocket turn."""

    spec: TurnSpec
    sink: PlaybackSink
    cancel: threading.Event
    sent_text: str
    acc: SentText
    t0: float
    audio: AudioArrival
    error_status: int | None = None
    error_message: str | None = None
    on_first_audio: Callable[[], None] | None = None


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
        if is_empty_delta(piece):
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
    run.acc.pieces.clear()
    run.acc.tts_first_text_ms = None
    replay = text_events(run.sent_text, run.cancel, spec.partial_chars) if run.sent_text else events
    stream = client.tts.stream_websocket(
        _tee_text_events(replay, run.t0, run.acc),
        reference_id=spec.voice_id,
        format=spec.audio_format,
        latency=spec.latency,
        speed=spec.speed,
        config=spec.config,
        model=cast(Model, spec.model),
    )
    await _pump_ws_audio(stream, run, close_client)


def _note_first_audio(
    audio: AudioArrival,
    chunk: bytes,
    t0: float,
    on_first_audio: Callable[[], None] | None = None,
) -> None:
    if audio.got_audio:
        return
    audio.got_audio = True
    audio.tts_first_audio_ms = elapsed_ms(t0)
    debug(
        "tts.first_audio {:.0f}ms after tts.start ({} bytes)",
        audio.tts_first_audio_ms,
        len(chunk),
    )
    if on_first_audio is not None:
        on_first_audio()


_STREAM_END: Final = object()
# A cancelled reader unwinds in milliseconds. A longer wait means a stuck socket,
# which the loop shutdown cleans up.
_REAP_TIMEOUT_S: Final = 2.0


async def _read_stream(stream: AsyncIterator[Any], queue: asyncio.Queue[Any]) -> None:
    # One task owns the whole iterator. The SDK holds an anyio task group
    # inside the generator, and exiting it from a task other than the one
    # that entered it raises "cancel scope in a different task".
    try:
        async for chunk in stream:
            await queue.put(chunk)
    except asyncio.CancelledError:
        raise
    except (Exception, BaseExceptionGroup) as exc:  # noqa: BLE001 - handed to the pump, which re-raises it
        # KeyboardInterrupt and SystemExit are deliberately not caught: queued,
        # they would wait behind the audio instead of stopping the process.
        await queue.put(exc)
    else:
        await queue.put(_STREAM_END)


async def _pump_ws_audio(
    stream: AsyncIterator[Any],
    run: TurnRun,
    close_client: Callable[[], Awaitable[None]],
) -> None:
    # One chunk of lookahead. A reader that runs ahead of playback keeps the
    # rest of a long reply in memory.
    queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=1)
    reader = asyncio.create_task(_read_stream(aiter(stream), queue))
    # Fish can start or stop a sentence on a loud sample next to silence, which clicks.
    fade = (
        EdgeFade(run.spec.sample_rate, run.spec.fade_ms)
        if run.spec.audio_format == "pcm" and run.spec.fade_ms > 0
        else None
    )
    cancelled, stop_watching = _watch_cancel(run.cancel)
    cancel_wait = asyncio.ensure_future(cancelled.wait())
    get: asyncio.Future[Any] | None = None
    try:
        while True:
            if run.cancel.is_set():
                debug("tts.cancel before/during stream")
                await close_client()
                return
            if get is None:
                get = asyncio.ensure_future(queue.get())
            done, _ = await asyncio.wait(
                {get, cancel_wait, reader}, return_when=asyncio.FIRST_COMPLETED
            )
            if get not in done:
                # The reader ends without a marker only when the stream itself
                # was cancelled. Raise that instead of waiting forever.
                if reader in done and queue.empty():
                    reader.result()
                    return
                # Cancel was set: the loop top closes the client.
                continue
            item = get.result()
            get = None
            if item is _STREAM_END:
                tail = fade.finish() if fade is not None else b""
                if tail:
                    await _play(run.sink, tail)
                return
            if isinstance(item, BaseException):
                raise item
            if not item:
                continue
            _note_first_audio(run.audio, item, run.t0, run.on_first_audio)
            if run.cancel.is_set():
                await close_client()
                return
            chunk = item if fade is None else fade.process(item)
            if chunk:
                await _play(run.sink, chunk)
    finally:
        stop_watching.set()
        for waiter in (get, cancel_wait):
            if waiter is not None and not waiter.done():
                waiter.cancel()
        if not reader.done():
            reader.cancel()
            await reap(reader, wait_s=_REAP_TIMEOUT_S)


def _watch_cancel(cancel: threading.Event) -> tuple[asyncio.Event, threading.Event]:
    """Mirror a thread-side cancel flag into an event this loop can await.

    A barge-in or Ctrl+C sets ``cancel`` on another thread. Polling it from the
    loop delayed the stop by up to the poll interval, so a thread waits on it
    and wakes the loop as soon as it is set. Set the returned stop event when
    the turn ends, and the thread exits within ``_CANCEL_WATCH_S``.
    """
    loop = asyncio.get_running_loop()
    cancelled = asyncio.Event()
    stop = threading.Event()

    def watch() -> None:
        while not stop.is_set():
            if cancel.wait(_CANCEL_WATCH_S):
                # The loop may already be closed if the turn ended meanwhile.
                with contextlib.suppress(RuntimeError):
                    loop.call_soon_threadsafe(cancelled.set)
                return

    threading.Thread(target=watch, name="fish-tts-cancel", daemon=True).start()
    return cancelled, stop


async def _play(sink: PlaybackSink, chunk: bytes) -> None:
    # A device write blocks for as long as the chunk plays. On this loop that
    # would also stall the text sender and the websocket reader, so the next
    # chunk could not arrive before this one ran out.
    write = asyncio.ensure_future(asyncio.to_thread(sink.write, chunk))
    try:
        await asyncio.shield(write)
    except asyncio.CancelledError:
        # Let the slice in flight finish before the caller closes the sink.
        await reap(write, wait_s=_REAP_TIMEOUT_S)
        raise


def _remember_event(ev: Any, acc: SentText, t0: float) -> None:
    if isinstance(ev, TextEvent) and ev.text:
        text = ev.text
        acc.pieces.append(text)
        if acc.tts_first_text_ms is None:
            acc.tts_first_text_ms = elapsed_ms(t0)
            debug(
                "tts.say {!r} ({} chars, first sentence at +{:.0f}ms)",
                text,
                len(text),
                acc.tts_first_text_ms,
            )
            return
        debug("tts.say {!r} ({} chars)", text, len(text))
        return
    if isinstance(ev, FlushEvent):
        debug("tts.flush")


async def _tee_text_events(
    events: AsyncIterator[Any],
    t0: float,
    acc: SentText,
) -> AsyncIterator[Any]:
    async for ev in events:
        _remember_event(ev, acc, t0)
        yield ev


def tts_result(run: TurnRun) -> TtsResult:
    """Build the result returned after a speak turn finishes."""
    spec = run.spec
    audio = run.audio
    if (
        not audio.got_audio
        and not run.cancel.is_set()
        and run.error_status is None
        and not run.error_message
    ):
        warn(f"[tts] no audio voice={spec.voice_id} model={spec.model}")
    played = run.sink.bytes_played()
    full = run.sent_text or "".join(run.acc.pieces)
    spoken = spoken_prefix(
        full,
        bytes_played=played,
        sample_rate=spec.sample_rate,
        audio_format=spec.audio_format,
        got_audio=audio.got_audio,
        cancelled=run.cancel.is_set(),
        # A socket drop has no HTTP status. The message is still a failed turn.
        failed=run.error_status is not None or bool(run.error_message),
        speed=spec.speed,
        # A sink written against the older protocol has no latency attribute,
        # and a turn that already played must not fail on it.
        output_latency_s=float(getattr(run.sink, "output_latency_s", 0.0) or 0.0),
    )
    return TtsResult(
        spoken_so_far=spoken,
        bytes_played=played,
        got_audio=audio.got_audio,
        cancelled=run.cancel.is_set(),
        tts_first_audio_ms=audio.tts_first_audio_ms,
        tts_first_text_ms=run.acc.tts_first_text_ms,
        error_status=run.error_status,
        error_message=run.error_message,
        error=(
            None
            if run.error_status is None
            else FishHttpError.from_status(run.error_status, run.error_message or "")
        ),
    )


def _root_exc(exc: BaseException) -> BaseException:
    # The first entry is often the cancel from a sibling task. Stopping there
    # hid a 503, so a turn with no audio was not retried.
    if isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        for item in exc.exceptions:
            found = _root_exc(item)
            if not is_cancel_noise(found):
                return found
        return _root_exc(exc.exceptions[0])
    return exc


@dataclass(frozen=True, slots=True)
class TtsFailure:
    """How one failed Fish attempt should be handled.

    Attributes
    ----------
    retry : bool
        Whether the turn should be tried again.
    error_status : int or None
        Fish HTTP status, when the failure had one.
    error_message : str or None
        Text to report. None when the failure was a barge-in.
    """

    retry: bool
    error_status: int | None = None
    error_message: str | None = None


def turn_failure(
    exc: BaseException,
    *,
    attempt: int,
    sent_text: str,
    got_audio: bool,
    cancel: threading.Event,
) -> TtsFailure:
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
    TtsFailure
        ``retry`` is True only for 429 or 5xx before audio, with text to replay,
        and attempts left.
    """
    root = _root_exc(exc)
    retry, status, message = _classify_fish_exc(root)
    if cancel.is_set() or (is_cancel_noise(root) and not got_audio):
        return TtsFailure(retry=False)
    # The socket died after audio started, and the turn was not cancelled.
    # Recording the unplayed tail makes the next turn assume it was heard.
    if is_cancel_noise(root):
        error_message = str(root)
        warn(f"[tts] {error_message}")
        return TtsFailure(retry=False, error_message=error_message)
    last = fish_attempt_exhausted(attempt)
    can_replay = bool(sent_text) and not got_audio
    if retry and not last and can_replay:
        warn(f"[tts] retry status={status} attempt={attempt + 1}/{FISH_RETRY_ATTEMPTS}")
        return TtsFailure(retry=True)
    error_message = message or str(root)
    warn(f"[tts] {status} {error_message}" if status is not None else f"[tts] {error_message}")
    return TtsFailure(retry=False, error_status=status, error_message=error_message)


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
        status, message = describe_request_error(exc, httpx.TimeoutException)
        return True, status, with_detail(message, exc)
    return False, None, str(exc)
