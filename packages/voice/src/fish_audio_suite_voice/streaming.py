"""Hear one turn by streaming it to Deepgram Flux while the user is still speaking.

The mic is gated as it always is: nothing leaves the machine until the voice detector hears
speech start. From then on every frame is sent as it is captured (with the pre-roll first),
and Flux says when the turn is over. The silence timer of the batch path stays as a limit:
if Flux has not ended the turn by then, it is asked to.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, Literal

from fish_audio_suite_kit import elapsed_ms, make_traceparent, trace_id_of
from fish_audio_suite_voice.aec import EchoCanceller
from fish_audio_suite_voice.barge import FRAME_BYTES, SAMPLE_RATE, StopFlag
from fish_audio_suite_voice.debug import debug, warn
from fish_audio_suite_voice.deepgram import (
    DeepgramError,
    FluxStream,
    StreamError,
    StreamWarning,
    TurnEnded,
    TurnStarted,
    TurnUpdate,
    flux_base,
    flux_url,
)
from fish_audio_suite_voice.events import EVENTS, Interim
from fish_audio_suite_voice.listen import stream_utterance
from fish_audio_suite_voice.tune import ListenTune, SttTune

__all__ = [
    "StreamedTurn",
    "stream_turn",
]

# Flux works best with about 80 ms of audio in each message. Three of our 30 ms frames is 90 ms.
_SEND_BYTES: Final = FRAME_BYTES * 3
# How long to wait for Flux to answer when it was asked to end the turn.
_FORCE_WAIT_S: Final = 2.5

type Stopped = Literal["stopped", "noise", "again", "fallback", "fatal"]


@dataclass(frozen=True, slots=True)
class StreamedTurn:
    """What one streamed turn produced.

    Attributes
    ----------
    text : str
        The whole turn, as Flux recognised it. Never empty.
    asr_ms : float
        From the last moment of speech until the text was final.
    started : float
        ``time.perf_counter()`` at the last moment of speech, which the reply's timings count from.
    trace_id : str or None
        The trace id for this turn, shared with its reply.
    """

    text: str
    asr_ms: float
    started: float
    trace_id: str | None


class _AnyStop:
    """A stop flag that is set when any of several are."""

    def __init__(self, *flags: StopFlag) -> None:
        self._flags = flags

    def is_set(self) -> bool:
        return any(flag.is_set() for flag in self._flags)


class _Turn:
    """What the reader has learnt about the turn so far. Only the event loop touches it."""

    def __init__(self) -> None:
        self.text = ""
        self.ended = False
        self.error = ""
        self.last_voice = 0.0

    def hear(self, text: str) -> None:
        if text and text != self.text:
            self.text = text
            EVENTS.emit(Interim(text))


async def _read(stream: FluxStream, turn: _Turn) -> None:
    async for message in stream.messages():
        match message:
            case TurnStarted(text=text) | TurnUpdate(text=text):
                turn.hear(text)
            case TurnEnded(text=text):
                turn.hear(text)
                turn.ended = True
                return
            case StreamError(code=code, description=description):
                turn.error = f"{code}: {description}"
                return
            case StreamWarning(code=code, description=description):
                debug("deepgram.warning {}: {}", code, description)


async def _send(
    stream: FluxStream,
    frames: asyncio.Queue[tuple[bytes, bool, float] | None],
    first: tuple[bytes, bool, float],
    turn: _Turn,
) -> None:
    pending = bytearray()
    item: tuple[bytes, bool, float] | None = first
    while item is not None:
        frame, voiced, at = item
        if voiced:
            turn.last_voice = at
        pending += frame
        if len(pending) >= _SEND_BYTES:
            await stream.send_audio(bytes(pending))
            pending.clear()
        item = await frames.get()
    if pending:
        await stream.send_audio(bytes(pending))


class _Capture:
    """The mic thread and the queue it feeds, for one turn."""

    def __init__(
        self,
        listen_fn: Callable[..., bool],
        *,
        device: str | int | None,
        quit_requested: StopFlag,
        stop: StopFlag,
        prefix: bytes,
        listen: ListenTune,
        aec: EchoCanceller | None,
    ) -> None:
        loop = asyncio.get_running_loop()
        self.frames: asyncio.Queue[tuple[bytes, bool, float] | None] = asyncio.Queue()
        self._decided = threading.Event()  # set to stop the mic thread once the turn is decided

        def sink(frame: bytes, voiced: bool) -> None:
            loop.call_soon_threadsafe(self.frames.put_nowait, (frame, voiced, time.perf_counter()))

        self.pump = loop.run_in_executor(
            None,
            functools.partial(
                listen_fn,
                device,
                _AnyStop(quit_requested, stop, self._decided),
                sink,
                prefix=prefix,
                tune=listen,
                aec=aec,
            ),
        )
        self.pump.add_done_callback(lambda _done: self.frames.put_nowait(None))

    def decided(self) -> None:
        """Stop the mic thread: the turn has its answer."""
        self._decided.set()


async def _connect(stream: FluxStream) -> Stopped | None:
    """Open the connection, or say why it cannot be used."""
    try:
        await stream.open()
    except DeepgramError as exc:
        warn(f"[deepgram] {exc}")
        return "fatal" if exc.fatal else "fallback"
    return None


async def _reopen(stream: FluxStream) -> Stopped | None:
    """Open the connection again if it dropped while we waited for speech."""
    if stream.connected:
        return None
    try:
        await stream.open()
    except DeepgramError as exc:
        warn(f"[deepgram] {exc}")
        return "fatal" if exc.fatal else "again"
    return None


async def _stream_until_decided(
    stream: FluxStream,
    capture: _Capture,
    first: tuple[bytes, bool, float],
    turn: _Turn,
    tasks: list[asyncio.Task[None]],
) -> bool:
    """Send the speech and read the answer until the turn is decided.

    Returns
    -------
    bool
        True when the silence limit ended the mic side, so Flux was asked to end the turn.
    """
    reader = asyncio.create_task(_read(stream, turn))
    sender = asyncio.create_task(_send(stream, capture.frames, first, turn))
    tasks += [reader, sender]
    await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    if reader.done():
        return False
    # The mic side finished first. Either the silence limit was reached, or it was stopped.
    # At the limit, ask Flux to end the turn and give it a moment.
    await sender
    local_end = await capture.pump
    if local_end:
        await stream.force_end_turn()
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(asyncio.shield(reader), _FORCE_WAIT_S)
    return local_end


def _outcome(turn: _Turn, *, local_end: bool, reader_done: bool) -> StreamedTurn | Stopped:
    """Turn what happened into the result."""
    if turn.error:
        warn(f"[deepgram] {turn.error}")
        return "again"
    if reader_done and not turn.ended:
        warn("[deepgram] the connection closed before the turn ended")
        return "again"
    if not turn.ended and not local_end:
        return "stopped"
    text = turn.text.strip()
    if not text:
        debug("deepgram.empty turn")
        return "noise"
    started = turn.last_voice or time.perf_counter()
    return StreamedTurn(
        text=text,
        asr_ms=elapsed_ms(started),
        started=started,
        trace_id=trace_id_of(make_traceparent()),
    )


async def stream_turn(
    *,
    stt: SttTune,
    listen: ListenTune,
    device: str | int | None,
    aec: EchoCanceller | None,
    quit_requested: StopFlag,
    stop: StopFlag,
    prefix: bytes = b"",
    make_stream: Callable[[str, str], FluxStream] = FluxStream,
    listen_fn: Callable[..., bool] = stream_utterance,
) -> StreamedTurn | Stopped:
    """Open the mic, stream one turn to Flux, and return what it heard.

    Parameters
    ----------
    stt : SttTune
        The Deepgram key, model and end-of-turn setting.
    listen : ListenTune
        The voice-activity settings that decide when speech starts.
    device : str or int or None
        PortAudio input.
    aec : EchoCanceller or None
        Cleans each frame against the far-end tap.
    quit_requested : StopFlag
        Set when the session is ending.
    stop : StopFlag
        Set to end the turn early, such as when a line is typed.
    prefix : bytes, optional
        PCM kept from the barge-in that interrupted the previous reply. Sent first.
    make_stream : Callable, optional
        Builds the connection from ``(url, key)``. A fake in tests.
    listen_fn : Callable, optional
        Gates the mic and passes frames on, as ``stream_utterance`` does. A fake in tests.

    Returns
    -------
    StreamedTurn or str
        The turn, or why there is none: ``"stopped"`` (quit or ``stop`` was set),
        ``"noise"`` (nothing recognisable was said), ``"again"`` (the connection failed
        mid-turn, so listen again), ``"fallback"`` (Deepgram cannot be reached, so use the
        batch recogniser for this turn) or ``"fatal"`` (the key is missing, wrong or unpaid).
    """
    if not stt.deepgram_key:
        warn("[deepgram] DEEPGRAM_API_KEY is not set")
        return "fatal"
    url = flux_url(
        stt.deepgram_model,
        sample_rate=SAMPLE_RATE,
        eot_threshold=stt.eot_threshold,
        base=flux_base(stt.deepgram_region),
    )
    stream = make_stream(url, stt.deepgram_key)
    # Connect before the mic opens, so the speech that follows has nowhere to wait.
    if (failed := await _connect(stream)) is not None:
        return failed
    capture = _Capture(
        listen_fn,
        device=device,
        quit_requested=quit_requested,
        stop=stop,
        prefix=prefix,
        listen=listen,
        aec=aec,
    )
    turn = _Turn()
    tasks: list[asyncio.Task[None]] = []
    try:
        # Nothing is sent until speech has started: the first frame is the pre-roll.
        first = await capture.frames.get()
        if first is None:
            return "stopped"
        if (failed := await _reopen(stream)) is not None:
            return failed
        local_end = await _stream_until_decided(stream, capture, first, turn, tasks)
        capture.decided()
        reader_done = tasks[0].done()
        return _outcome(turn, local_end=local_end, reader_done=reader_done)
    finally:
        capture.decided()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        with contextlib.suppress(Exception):
            await capture.pump
        await stream.close()
