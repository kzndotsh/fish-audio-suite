"""Hear one turn by streaming it to Deepgram Flux while the user is still speaking.

The mic is gated as it always is: nothing leaves the machine until the voice detector hears
speech start, and the connection is not even opened until then (Deepgram closes a connection
that sits without audio). A bump or a cough starts the gate too, so the first frames are held
until there is as much voiced audio as the batch path asks for (``min_voiced_frames``). Then
the connection opens, the held audio goes first, and every frame after it is sent as it is
captured, and Flux says when the turn is over. The silence timer of the batch path stays as a
limit: if Flux has not ended the turn by then, it is asked to.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

from fish_audio_suite_kit import elapsed_ms, make_traceparent, trace_id_of
from fish_audio_suite_voice.aec import EchoCanceller
from fish_audio_suite_voice.barge import SAMPLE_RATE, StopFlag
from fish_audio_suite_voice.debug import debug, trace, warn
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
from fish_audio_suite_voice.playback import write_mono_wav
from fish_audio_suite_voice.tune import ListenTune, SttTune

__all__ = [
    "StreamFallback",
    "StreamedTurn",
    "stream_turn",
]

# Deepgram strongly recommends about 80 ms of audio in each message for Flux. That is not a whole
# number of our 30 ms frames, so the audio is re-cut into exact pieces of it as it is sent.
_SEND_MS: Final = 80
_SEND_BYTES: Final = SAMPLE_RATE * 2 * _SEND_MS // 1000
# How long to wait for Flux to answer when it was asked to end the turn.
_FORCE_WAIT_S: Final = 2.5

type Stopped = Literal["stopped", "noise", "again", "fatal"]
# Flux says this when asked to end a turn that never started: there is nothing to end.
_NO_TURN_CODE: Final = "FORCE_END_TURN_NO_ACTIVE_TURN"


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


@dataclass(frozen=True, slots=True)
class StreamFallback:
    """Deepgram could not be reached once speech had started, so use Fish ASR for this turn.

    Attributes
    ----------
    prefix : bytes
        The speech captured so far, the pre-roll included. The batch recogniser carries on from
        it, so the first words are not lost.
    """

    prefix: bytes


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
        self.sent = 0  # bytes of audio sent to Flux
        self.audio = bytearray()  # the same audio, kept only when it is to be saved
        self.keep = False

    def hear(self, text: str) -> None:
        if text and text != self.text:
            self.text = text
            trace("stt.turn {}", text)
            EVENTS.emit(Interim(text))


async def _read(stream: FluxStream, turn: _Turn) -> None:
    async for message in stream.messages():
        match message:
            case TurnStarted(text=text) | TurnUpdate(text=text):
                turn.hear(text)
            case TurnEnded(text=text, confidence=confidence, trigger=trigger):
                debug("stt.end trigger={} confidence={:.2f}", trigger, confidence)
                turn.hear(text)
                turn.ended = True
                return
            case StreamError(code=code, description=description):
                turn.error = f"{code}: {description}"
                return
            case StreamWarning(code=code, description=description):
                debug("stt.warning {}: {}", code, description)
                if code == _NO_TURN_CODE:  # nothing was said that Flux took for speech
                    turn.ended = True
                    return


async def _send(
    stream: FluxStream,
    frames: asyncio.Queue[tuple[bytes, bool, float] | None],
    held: list[tuple[bytes, bool, float]],
    turn: _Turn,
) -> None:
    pending = bytearray()
    backlog = iter(held)
    item: tuple[bytes, bool, float] | None = next(backlog, None) or await frames.get()
    while item is not None:
        frame, voiced, at = item
        if voiced:
            turn.last_voice = at
        pending += frame
        while len(pending) >= _SEND_BYTES:
            piece = bytes(pending[:_SEND_BYTES])
            del pending[:_SEND_BYTES]
            await stream.send_audio(piece)
            turn.sent += len(piece)
            if turn.keep:
                turn.audio += piece
        item = next(backlog, None) or await frames.get()
    if pending:
        await stream.send_audio(bytes(pending))
        turn.sent += len(pending)
        if turn.keep:
            turn.audio += pending


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
        self.held: list[tuple[bytes, bool, float]] = []  # frames taken while waiting for speech
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

    def captured(self) -> bytes:
        """Return every frame captured so far that has not been taken from the queue."""
        self._decided.set()  # a fallback listens afresh, so this thread has to let go of the mic
        pieces = [frame for frame, _voiced, _at in self.held]
        while not self.frames.empty():
            item = self.frames.get_nowait()
            if item is not None:
                pieces.append(item[0])
        return b"".join(pieces)


def _save(folder: str, pcm: bytes | bytearray) -> None:
    """Write what was sent to Flux as a WAV, so it can be listened to. Never raises."""
    path = (
        Path(folder).expanduser()
        / f"stt-{time.strftime('%Y%m%d-%H%M%S')}-{int(time.time() * 1000) % 1000:03d}.wav"
    )
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_mono_wav(path, bytes(pcm), SAMPLE_RATE)
    except OSError as exc:
        warn(f"[stt] could not save the audio to {path}: {exc}")
        return
    debug("stt.saved {} ({:.1f} s)", path, len(pcm) / (SAMPLE_RATE * 2))


async def _hold_until_voiced(capture: _Capture, need: int) -> Stopped | None:
    """Keep the frames until there is enough voiced audio to be speech and not a bump.

    Returns
    -------
    str or None
        None when there is. ``"stopped"`` when the mic was stopped first, ``"noise"`` when the
        gate ended the turn first: it was a cough or a knock, and nothing was connected or sent.
    """
    voiced = 0
    while voiced < need:
        item = await capture.frames.get()
        if item is None:
            local_end = await capture.pump
            if local_end:
                debug("stt.noise {} voiced frames of {} needed, nothing sent", voiced, need)
            return "noise" if local_end else "stopped"
        capture.held.append(item)
        voiced += item[1]
    return None


async def _connect(
    stream: FluxStream, capture: _Capture
) -> StreamFallback | Literal["fatal"] | None:
    """Open the connection now that speech has started.

    Returns
    -------
    StreamFallback or str or None
        None when it is open. ``"fatal"`` when the key is refused. Otherwise a fallback
        carrying the speech captured while it was trying.
    """
    try:
        await stream.open()
    except DeepgramError as exc:
        warn(f"[stt] {exc}")
        if exc.fatal:
            return "fatal"
        return StreamFallback(capture.captured())
    debug("stt.open connected, sending speech")
    return None


async def _stream_until_decided(
    stream: FluxStream,
    capture: _Capture,
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
    sender = asyncio.create_task(_send(stream, capture.frames, capture.held, turn))
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


def _outcome(
    turn: _Turn, *, local_end: bool, reader_done: bool
) -> StreamedTurn | StreamFallback | Stopped:
    """Turn what happened into the result."""
    if turn.error:
        warn(f"[stt] {turn.error}")
        return "again"
    if reader_done and not turn.ended:
        warn("[stt] the connection closed before the turn ended")
        return "again"
    if not turn.ended and not local_end:
        return "stopped"
    text = turn.text.strip()
    if not text:
        debug(
            "stt.empty Flux was sent {} kB ({:.1f} s of audio) and found no speech",
            turn.sent // 1000,
            turn.sent / (SAMPLE_RATE * 2),
        )
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
) -> StreamedTurn | StreamFallback | Stopped:
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
    StreamedTurn or StreamFallback or str
        The turn, or why there is none: ``"stopped"`` (quit or ``stop`` was set), ``"noise"``
        (nothing recognisable was said), ``"again"`` (the connection failed mid-turn, so listen
        again), ``"fatal"`` (the key is missing, wrong or unpaid) or a ``StreamFallback``
        (Deepgram could not be reached, so the batch recogniser takes over this turn).
    """
    if not stt.deepgram_key:
        warn("[stt] DEEPGRAM_API_KEY is not set")
        return "fatal"
    url = flux_url(
        stt.deepgram_model,
        sample_rate=SAMPLE_RATE,
        eot_threshold=stt.eot_threshold,
        base=flux_base(stt.deepgram_region),
    )
    stream = make_stream(url, stt.deepgram_key)
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
    turn.keep = bool(stt.save_dir)
    tasks: list[asyncio.Task[None]] = []
    try:
        # Nothing is connected or sent until the gate has opened and there is enough voice.
        if (quiet := await _hold_until_voiced(capture, listen.min_voiced_frames)) is not None:
            return quiet
        if (failed := await _connect(stream, capture)) is not None:
            return failed
        local_end = await _stream_until_decided(stream, capture, turn, tasks)
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
        if turn.audio:
            _save(stt.save_dir, turn.audio)
