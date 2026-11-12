"""Hear one turn by streaming it to a recogniser while the user is still speaking.

The recogniser is the one ``SttTune.provider`` names in ``recognisers.RECOGNISERS``. Each is a
``TurnStream``, and this module names none of them.

The mic is gated as it always is: nothing leaves the machine until the voice detector hears
speech start, and the connection is not even opened until then (a recogniser may close a
connection that sits without audio). A bump or a cough starts the gate too, so the first
frames are held until there is as much voiced audio as the batch path asks for
(``min_voiced_frames``). Then the connection opens, the held audio goes first, and every frame
after it is sent as it is captured, and the recogniser says when the turn is over. The silence
timer of the batch path stays as a limit: if the recogniser has not ended the turn by then, it
is asked to.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final, Literal

from fish_audio_suite_kit import MS_PER_S, elapsed_ms, make_traceparent, trace_id_of
from fish_audio_suite_voice.cancel import StopFlag, any_of
from fish_audio_suite_voice.debug import debug, trace, warn
from fish_audio_suite_voice.events import EVENTS, Interim
from fish_audio_suite_voice.listen import stream_utterance
from fish_audio_suite_voice.mic import Mic
from fish_audio_suite_voice.pcm import BYTES_PER_S, FRAME_BYTES, SAMPLE_RATE, write_mono_wav
from fish_audio_suite_voice.recognisers import recogniser
from fish_audio_suite_voice.tune import ListenTune
from fish_audio_suite_voice.turn_stream import (
    EagerTurnEnd,
    RecogniserSettings,
    StreamError,
    StreamWarning,
    SttStreamError,
    TurnDropped,
    TurnEnded,
    TurnResumed,
    TurnStarted,
    TurnStream,
    TurnUpdate,
)

__all__ = [
    "StreamFallback",
    "StreamedTurn",
    "stream_turn",
]

# Audio goes to the recogniser in pieces of about 80 ms, which Deepgram strongly recommends for
# Flux. That is not a whole number of our 30 ms frames, so the audio is re-cut into exact pieces
# of it as it is sent.
_SEND_MS: Final = 80
_SEND_BYTES: Final = BYTES_PER_S * _SEND_MS // MS_PER_S
# How long to wait for the recogniser to answer when it was asked to end the turn.
_FORCE_WAIT_S: Final = 2.5
# Real speech has its first word back well inside this. A knock or a run of taps has none, so
# the turn is given up on instead of being streamed until the silence timer ends.
_NO_WORDS_S: Final = 1.2

type Stopped = Literal["stopped", "noise", "again", "fatal"]


@dataclass(frozen=True, slots=True)
class StreamedTurn:
    """What one streamed turn produced.

    Attributes
    ----------
    text : str
        The whole turn, as the recogniser heard it. Never empty.
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
    """The recogniser could not be reached once speech had started, so use Fish ASR for this turn.

    Attributes
    ----------
    prefix : bytes
        The speech captured so far, the pre-roll included. The batch recogniser carries on from
        it, so the first words are not lost.
    """

    prefix: bytes


class _Turn:
    """What the reader has learnt about the turn so far. Only the event loop touches it."""

    def __init__(self, who: str) -> None:
        self.text = ""
        self.ended = False
        self.error = ""
        self.last_voice = 0.0
        self.sent = 0  # bytes of audio sent to the recogniser
        self.gave_up = False  # no words came back in time, so it was not speech
        self.who = who  # the recogniser's name, for the log
        self.audio = bytearray()  # the same audio, kept only when it is to be saved
        self.keep = False
        self.on_eager: Callable[[str], None] | None = None
        self.on_resumed: Callable[[], None] | None = None

    def hear(self, text: str) -> None:
        if text and text != self.text:
            self.text = text
            trace("stt.turn {}", text)
            EVENTS.emit(Interim(text))


async def _read(stream: TurnStream, turn: _Turn) -> None:
    async for message in stream.messages():
        match message:
            case TurnStarted(text=text) | TurnUpdate(text=text):
                turn.hear(text)
            case TurnEnded(text=text, confidence=confidence, trigger=trigger):
                debug("stt.end trigger={} confidence={:.2f}", trigger, confidence)
                turn.hear(text)
                turn.ended = True
                return
            case EagerTurnEnd(text=text):
                debug("stt.eager {!r}", text)
                if turn.on_eager is not None and text.strip():
                    turn.on_eager(text)
            case TurnDropped():
                debug("stt.dropped not the user's turn")
                turn.text = ""
                turn.ended = True
                return
            case TurnResumed():
                debug("stt.resumed the speaker carried on")
                if turn.on_resumed is not None:
                    turn.on_resumed()
            case StreamError(code=code, description=description):
                turn.error = f"{code}: {description}"
                return
            case StreamWarning(code=code, description=description):
                debug("stt.warning {}: {}", code, description)


async def _send(
    stream: TurnStream,
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
        mic: Mic,
        quit_requested: StopFlag,
        stop: StopFlag,
        prefix: bytes,
        listen: ListenTune,
        since: int | None,
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
                mic,
                any_of(quit_requested, stop, self._decided),
                sink,
                prefix=prefix,
                tune=listen,
                since=since,
            ),
        )
        self.pump.add_done_callback(lambda _done: self.frames.put_nowait(None))

    def decided(self) -> None:
        """Stop the mic thread: the turn has its answer."""
        self._decided.set()

    async def captured(self) -> bytes:
        """Stop the mic thread and return every frame it captured that was not taken yet."""
        self._decided.set()
        # Frames the thread handed over are still callbacks on the loop until it has
        # stopped. The thread's own failure is not this turn's to report. The shield
        # keeps a cancelled turn from cancelling the pump, which cleanup still awaits.
        with contextlib.suppress(Exception):
            await asyncio.shield(self.pump)
        pieces = [frame for frame, _voiced, _at in self.held]
        while not self.frames.empty():
            item = self.frames.get_nowait()
            if item is not None:
                pieces.append(item[0])
        return b"".join(pieces)


def _save(folder: str, pcm: bytes | bytearray) -> None:
    """Write what was sent to the recogniser as a WAV, so it can be listened to. Never raises."""
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
    debug("stt.saved {} ({:.1f} s)", path, len(pcm) / BYTES_PER_S)


async def _hold_until_voiced(capture: _Capture, need: int, prefix_frames: int) -> Stopped | None:
    """Keep the frames until there is enough voiced audio to be speech and not a bump.

    Parameters
    ----------
    capture : _Capture
        The mic side.
    need : int
        Voiced frames wanted.
    prefix_frames : int
        How many of the first frames are a barge-in clip. The interrupt gate already judged
        that clip to be speech, so the voice in it counts in full, as it does on the batch
        path: a short interruption must not be dropped for being shorter than ``need``.

    Returns
    -------
    str or None
        None when there is. ``"stopped"`` when the mic was stopped first, ``"noise"`` when the
        gate ended the turn first: it was a cough or a knock, and nothing was connected or sent.
    """
    voiced = 0
    taken = 0
    while voiced < need:
        item = await capture.frames.get()
        if item is None:
            local_end = await capture.pump
            if local_end:
                debug("stt.noise {} voiced frames of {} needed, nothing sent", voiced, need)
            return "noise" if local_end else "stopped"
        capture.held.append(item)
        voiced += item[1]
        taken += 1
        if taken == prefix_frames and 0 < voiced < need:
            need = voiced  # the barge-in clip was speech: that much voice is enough
    return None


async def _connect(
    stream: TurnStream, capture: _Capture
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
    except SttStreamError as exc:
        warn(f"[stt] {exc}")
        if exc.fatal:
            return "fatal"
        return StreamFallback(await capture.captured())
    debug("stt.open connected, sending speech")
    return None


async def _stream_until_decided(
    stream: TurnStream,
    capture: _Capture,
    turn: _Turn,
    tasks: list[asyncio.Task[None]],
) -> bool:
    """Send the speech and read the answer until the turn is decided.

    Returns
    -------
    bool
        True when the silence limit ended the mic side, so the recogniser was asked to end
        the turn.
    """
    reader = asyncio.create_task(_read(stream, turn))
    sender = asyncio.create_task(_send(stream, capture.frames, capture.held, turn))
    tasks += [reader, sender]
    done, _pending = await asyncio.wait(
        tasks, timeout=_NO_WORDS_S, return_when=asyncio.FIRST_COMPLETED
    )
    if not done and not turn.text:
        debug("stt.gave_up no words from {} after {:.1f} s: not speech", turn.who, _NO_WORDS_S)
        turn.gave_up = True
        return False
    if not done:  # words are coming in: carry on until the turn is over
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    if reader.done():
        return False
    # The mic side finished first. Either the silence limit was reached, or it was stopped.
    # At the limit, ask the recogniser to end the turn and give it a moment.
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
    if turn.gave_up:
        return "noise"
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
            "stt.empty {} was sent {} kB ({:.1f} s of audio) and found no speech",
            turn.who,
            turn.sent // 1000,
            turn.sent / BYTES_PER_S,
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
    settings: RecogniserSettings,
    listen: ListenTune,
    mic: Mic,
    quit_requested: StopFlag,
    stop: StopFlag,
    prefix: bytes = b"",
    since: int | None = None,
    make_stream: Callable[[str, str], TurnStream] | None = None,
    listen_fn: Callable[..., bool] = stream_utterance,
    on_eager: Callable[[str], None] | None = None,
    on_resumed: Callable[[], None] | None = None,
) -> StreamedTurn | StreamFallback | Stopped:
    """Read one turn from the mic, stream it to the recogniser, and return what it heard.

    Parameters
    ----------
    settings : RecogniserSettings
        Which recogniser, and what it is opened with.
    listen : ListenTune
        The voice-activity settings that decide when speech starts.
    mic : Mic
        Where the frames come from, already cleaned by the session's audio processor.
    quit_requested : StopFlag
        Set when the session is ending.
    stop : StopFlag
        Set to end the turn early, such as when a line is typed.
    prefix : bytes, optional
        PCM kept from the barge-in that interrupted the previous reply. Sent first.
    since : int or None, optional
        The mic position just after ``prefix``, read on from there. None reads from now.
    make_stream : Callable, optional
        Builds the connection from ``(url, key)``. A fake in tests.
    listen_fn : Callable, optional
        Gates the mic and passes frames on, as ``stream_utterance`` does. A fake in tests.
    on_eager : Callable, optional
        Called with the words so far when the recogniser thinks the turn is probably over.
        Needs ``SttTune.eager_eot_threshold`` to be set.
    on_resumed : Callable, optional
        Called when the speaker carried on after that.

    Returns
    -------
    StreamedTurn or StreamFallback or str
        The turn, or why there is none: ``"stopped"`` (quit or ``stop`` was set), ``"noise"``
        (nothing recognisable was said), ``"again"`` (the connection failed mid-turn, so listen
        again), ``"fatal"`` (the key is missing, wrong or unpaid) or a ``StreamFallback``
        (the recogniser could not be reached, so the batch one takes over this turn).
    """
    stt = settings.stt
    spec = recogniser(stt.provider)
    if spec is None:
        warn(f"[stt] FISH_VOICE_STT={stt.provider} does not stream")
        return "fatal"
    key = spec.key(settings)
    if not key:
        warn(f"[stt] {spec.key_name} is not set")
        return "fatal"
    url = spec.url(settings)
    stream = make_stream(url, key) if make_stream is not None else spec.stream(url, key, settings)
    if listen.end_silence_frames < spec.min_silence_frames:
        # The silence limit stays a backstop, behind the recogniser's own end of turn.
        listen = replace(listen, end_silence_frames=spec.min_silence_frames)
    capture = _Capture(
        listen_fn,
        mic=mic,
        quit_requested=quit_requested,
        stop=stop,
        prefix=prefix,
        listen=listen,
        since=since,
    )
    turn = _Turn(spec.label)
    turn.keep = bool(stt.save_dir)
    turn.on_eager = on_eager
    turn.on_resumed = on_resumed
    tasks: list[asyncio.Task[None]] = []
    try:
        # Nothing is connected or sent until the gate has opened and there is enough voice.
        if (
            quiet := await _hold_until_voiced(
                capture, listen.min_voiced_frames, len(prefix) // FRAME_BYTES
            )
        ) is not None:
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
