"""Fish websocket audio: text events, the pump, and how much was spoken."""

from __future__ import annotations

import asyncio
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
from fish_audio_suite_voice.debug import debug, warn, with_detail
from fish_audio_suite_voice.playback import PlaybackSink
from fish_audio_suite_voice.spoken import spoken_prefix

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

    api_key: str = field(repr=False)
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


# Last resort when a teardown error is not an exception type we know. Matching
# on message text breaks when an SDK rewords it, so a hit is logged.
_CANCEL_NOISE = (
    "athrow",
    "cancel scope",
    "generator didn't stop",
    "different task than it was entered",
)


def is_cancel_noise(exc: BaseException, *, cancelled: bool = False) -> bool:
    """Return whether ``exc`` is a barge-in or Ctrl+C tear-down, not a Fish error.

    Parameters
    ----------
    exc : BaseException
        An error from the websocket task or a task group.
    cancelled : bool, optional
        True when the caller's cancel flag is already set. Any ordinary error
        raised while the turn is being torn down is then expected noise.

    Returns
    -------
    bool
        True for ``CancelledError`` and ``GeneratorExit``, and for any error
        raised after cancel was requested. Without a cancel flag, a known
        anyio or asyncio message still matches, and the match is logged at
        debug so a reworded message shows up.
    """
    if isinstance(exc, (asyncio.CancelledError, GeneratorExit)):
        return True
    if cancelled and isinstance(exc, (Exception, BaseExceptionGroup)):
        return True
    msg = str(exc).lower()
    if any(phrase in msg for phrase in _CANCEL_NOISE):
        debug("cancel.noise matched by message: {}", msg)
        return True
    return False


def own_cancel(flag: asyncio.Event | threading.Event | None) -> bool:
    """Return whether a ``CancelledError`` here comes from our own cancel flag.

    Parameters
    ----------
    flag : asyncio.Event or threading.Event or None
        The cancel flag the caller was given. ``None`` means the caller has no flag.

    Returns
    -------
    bool
        True when the flag is set and the running task has no pending
        cancellation request of its own. Only then may the caller turn a
        ``CancelledError`` into a normal early stop. Anything else, such as
        ``asyncio.timeout``, a task group or an outer ``task.cancel()``, must
        propagate.
    """
    if flag is None or not flag.is_set():
        return False
    task = asyncio.current_task()
    return task is None or task.cancelling() == 0


async def reap(task: asyncio.Task[Any], *, wait_s: float | None = None) -> None:
    """Wait for a task that was just cancelled and drop its outcome.

    Parameters
    ----------
    task : asyncio.Task
        A child task that ``task.cancel()`` was already called on.
    wait_s : float or None, optional
        Seconds to wait. None waits until the task finishes. A task that is
        still running after that is left for the loop shutdown.

    Notes
    -----
    Unlike ``suppress(BaseException)`` around ``await task``, this never
    swallows ``KeyboardInterrupt``, ``SystemExit`` or a cancellation of the
    caller. The child's own exception is read so asyncio does not report it
    as never retrieved.
    """
    done, _pending = await asyncio.wait({task}, timeout=wait_s)
    if task in done and not task.cancelled():
        task.exception()


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


def _note_first_audio(
    audio: Heard,
    chunk: bytes,
    t0: float,
    on_first_audio: Callable[[], None] | None = None,
) -> None:
    if audio.got_audio:
        return
    audio.got_audio = True
    audio.ttfa_ms = elapsed_ms(t0)
    debug(
        "tts.first_audio {:.0f}ms after tts.start ({} bytes)",
        audio.ttfa_ms,
        len(chunk),
    )
    if on_first_audio is not None:
        on_first_audio()


_STREAM_END = object()
# A cancelled reader unwinds in milliseconds. A longer wait means a stuck socket,
# which the loop shutdown cleans up.
_REAP_TIMEOUT_S = 2.0


async def _read_stream(stream: AsyncIterator[Any], queue: asyncio.Queue[Any]) -> None:
    # One task owns the whole iterator. The SDK holds an anyio task group
    # inside the generator, and exiting it from a task other than the one
    # that entered it raises "cancel scope in a different task".
    try:
        async for chunk in stream:
            await queue.put(chunk)
    except asyncio.CancelledError:
        raise
    except BaseException as exc:
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
    try:
        while True:
            if run.cancel.is_set():
                debug("tts.cancel before/during stream")
                await close_client()
                return
            try:
                # A timeout only re-checks cancel. It never touches the reader.
                item = await asyncio.wait_for(queue.get(), ANEXT_POLL_S)
            except TimeoutError:
                # The reader ends without a marker only when the stream itself
                # was cancelled. Raise that instead of polling forever.
                if reader.done() and queue.empty():
                    reader.result()
                    return
                continue
            if item is _STREAM_END:
                return
            if isinstance(item, BaseException):
                raise item
            if not item:
                continue
            _note_first_audio(run.audio, item, run.t0, run.on_first_audio)
            if run.cancel.is_set():
                await close_client()
                return
            run.sink.write(item)
    finally:
        if not reader.done():
            reader.cancel()
            await reap(reader, wait_s=_REAP_TIMEOUT_S)


def _remember_event(ev: Any, acc: EventAcc, t0: float) -> None:
    if isinstance(ev, TextEvent) and ev.text:
        text = ev.text
        acc.flushed.append(text)
        if acc.ttfs_ms is None:
            acc.ttfs_ms = elapsed_ms(t0)
            debug(
                "tts.say {!r} ({} chars, first sentence at +{:.0f}ms)",
                text,
                len(text),
                acc.ttfs_ms,
            )
            return
        debug("tts.say {!r} ({} chars)", text, len(text))
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
    if (
        not audio.got_audio
        and not run.cancel.is_set()
        and run.err_status is None
        and not run.err_message
    ):
        warn(f"[tts] no audio voice={spec.voice_id} model={spec.model}")
    played = run.sink.bytes_played()
    full = run.sent_text or "".join(run.acc.flushed)
    spoken = spoken_prefix(
        full,
        bytes_played=played,
        sample_rate=spec.sample_rate,
        audio_format=spec.audio_format,
        got_audio=audio.got_audio,
        cancelled=run.cancel.is_set(),
        # A socket drop has no HTTP status. The message is still a failed turn.
        failed=run.err_status is not None or bool(run.err_message),
        speed=spec.speed,
        output_latency_s=float(getattr(run.sink, "output_latency_s", 0.0) or 0.0),
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
    if cancel.is_set() or (is_cancel_noise(root) and not got_audio):
        return _TtsFailure(retry=False)
    # The socket died after audio started, and the turn was not cancelled.
    # Recording the unplayed tail makes the next turn assume it was heard.
    if is_cancel_noise(root):
        err_message = str(root)
        warn(f"[tts] {err_message}")
        return _TtsFailure(retry=False, err_message=err_message)
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
        return True, status, with_detail(message, exc)
    return False, None, str(exc)
