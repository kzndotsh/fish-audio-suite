"""Fish websocket audio: the pump that sends a turn's text events, and how much was spoken."""

from __future__ import annotations

import asyncio
import contextlib
import threading
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Final, Protocol

from fishaudio import FlushEvent, TextEvent
from fishaudio.types import LatencyMode, TTSConfig

from fish_audio_suite_kit import (
    AudioFormat,
    FishAuthError,
    FishHttpError,
    SuiteDefaults,
    elapsed_ms,
)
from fish_audio_suite_voice.audio_chain import AudioChain
from fish_audio_suite_voice.cancel import reap
from fish_audio_suite_voice.debug import debug, warn
from fish_audio_suite_voice.playback import PlaybackSink, sink_output_latency
from fish_audio_suite_voice.spoken import spoken_prefix

__all__ = [
    "AudioArrival",
    "FishSocket",
    "SentText",
    "SinkError",
    "TtsResult",
    "TurnRun",
    "TurnSpec",
    "send_turn",
    "tts_result",
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
        The failure as a kit error, set when there is a status. ``fatal`` says whether
        it is a ``FishAuthError`` (401, 402, 403), which ends duplex.
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
            error = FishHttpError(self.error_status, self.error_message or "")
            object.__setattr__(self, "error", error)

    @property
    def failed(self) -> bool:
        """Whether Fish failed the turn: an HTTP status, or a message for a socket drop."""
        return self.error_status is not None or bool(self.error_message)

    @property
    def fatal(self) -> bool:
        """Whether the failure is the key's (401, 402 or 403), which no later turn can pass."""
        return isinstance(self.error, FishAuthError)


_STOCK_RATE: Final = SuiteDefaults().sample_rate


@dataclass(frozen=True, slots=True)
class TurnSpec:
    """Inputs for one Fish websocket. Built by ``FishSpeaker``, not by apps.

    Attributes
    ----------
    api_key : str
        Fish key the socket opens with.
    base_url : str
        Fish origin.
    voice_id : str
        Fish voice, sent in the start event.
    model : str
        Fish TTS model, sent as a header.
    trace_headers : Mapping of str to str
        Copied onto the httpx client that upgrades the socket.
    config : TTSConfig
        Everything Fish is sent about the audio. The format, rate, latency and speed are
        read from here, so the socket and the result agree with what Fish was asked for.
    fade_ms : float
        Edge fade applied before the sink. PCM only.
    loudness_db : float or None
        Level the reply is brought to before the sink, or None. PCM only.
    """

    api_key: str = field(repr=False)
    base_url: str
    voice_id: str
    model: str
    trace_headers: Mapping[str, str]
    config: TTSConfig
    fade_ms: float = 0.0
    loudness_db: float | None = None

    @property
    def audio_format(self) -> AudioFormat:
        """The format Fish sends."""
        return self.config.format

    @property
    def latency(self) -> LatencyMode:
        """The latency mode Fish is asked for."""
        return self.config.latency

    @property
    def sample_rate(self) -> int:
        """The PCM rate Fish sends. The SDK's default when the config leaves it unset."""
        rate = self.config.sample_rate
        return _STOCK_RATE if rate is None else rate

    @property
    def speed(self) -> float:
        """The speaking rate Fish is asked for, 1.0 when the config sets no prosody."""
        prosody = self.config.prosody
        return 1.0 if prosody is None else prosody.speed


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


class SinkError(Exception):
    """The sink raised while a chunk was being written.

    Raised out of ``send_turn`` so the turn loop can tell a sink that failed from a Fish
    failure, and re-raise the sink's own error instead of reporting it as Fish's.

    Parameters
    ----------
    error : Exception
        What the sink raised.
    """

    def __init__(self, error: Exception) -> None:
        super().__init__(str(error))
        self.error = error


class FishSocket(Protocol):
    """Where one Fish turn attempt sends its text and reads its audio.

    Notes
    -----
    ``stream`` opens a realtime websocket for one attempt. ``close`` ends it, and
    may be called again after it has closed. Closing is the only way to stop the
    socket early: the audio iterator must not be ``aclose``'d.
    """

    def stream(self, events: AsyncIterator[Any], spec: TurnSpec) -> AsyncIterator[bytes]:
        """Send ``events`` with the voice and format in ``spec``, yielding audio.

        Parameters
        ----------
        events : AsyncIterator
            Text and flush events for this attempt.
        spec : TurnSpec
            Voice, model, format and the key the socket opens with.

        Returns
        -------
        AsyncIterator
            Audio chunks as they arrive.
        """
        ...

    async def close(self) -> None:
        """End the socket. Safe to call more than once."""
        ...


async def send_turn(socket: FishSocket, events: AsyncIterator[Any], run: TurnRun) -> None:
    """Run one attempt over ``socket`` and write its audio to the sink.

    Parameters
    ----------
    socket : FishSocket
        Opens the websocket. Closed on barge-in or Ctrl+C, which ends it.
    events : AsyncIterator
        This attempt's text events. A retry passes a fresh stream.
    run : TurnRun
        Mutable turn state. First-audio time and flushed text land here.
    """
    run.acc.pieces.clear()
    run.acc.tts_first_text_ms = None
    stream = socket.stream(_tee_text_events(events, run.t0, run.acc), run.spec)
    await _pump_ws_audio(stream, run, socket.close)


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
    spec = run.spec
    chain = AudioChain(
        pcm=spec.audio_format == "pcm",
        sample_rate=spec.sample_rate,
        fade_ms=spec.fade_ms,
        loudness_db=spec.loudness_db,
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
                tail = chain.finish()
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
            chunk = chain.process(item)
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
    loop would delay the stop by up to the poll interval, so a thread waits on
    it and wakes the loop as soon as it is set. Set the returned stop event when
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
    except Exception as exc:
        raise SinkError(exc) from exc


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
        output_latency_s=sink_output_latency(run.sink),
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
    )
