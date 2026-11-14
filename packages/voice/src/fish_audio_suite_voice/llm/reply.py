"""Answer a heard line: the model writes it, then Fish speaks it, whole or streamed."""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import replace
from typing import Final

from fish_audio_suite_kit import (
    elapsed_ms,
    ensure_lead_cue,
    is_tts_junk,
    make_traceparent,
)
from fish_audio_suite_voice.asr.outcome import HeardLine
from fish_audio_suite_voice.capture.barge_in import BargeGate
from fish_audio_suite_voice.capture.portaudio import PortAudioMissingError
from fish_audio_suite_voice.llm.speculate import Speculation
from fish_audio_suite_voice.llm.transports import ChatUnusableError
from fish_audio_suite_voice.logutil.debug import debug, warn
from fish_audio_suite_voice.session.cancel import is_cancel_noise, is_own_cancel
from fish_audio_suite_voice.session.events import (
    EVENTS,
    ExitCode,
    ReplyEnd,
    ReplyToken,
    Speaking,
    notice,
)
from fish_audio_suite_voice.session.latency import LatencySnapshot
from fish_audio_suite_voice.session.state import Chat, DuplexContext
from fish_audio_suite_voice.tts.playback import PlaybackSink, make_sink
from fish_audio_suite_voice.tts.played_text import Unheard, heard_written, unheard
from fish_audio_suite_voice.tts.websocket import TtsResult

__all__ = [
    "collect_reply",
    "drop_speculation",
    "speak_reply",
    "start_speculation",
    "stream_turn",
]

# The barge watch notices its stop flag within one mic poll (0.2 s). The join
# waits that out, so the watch has stopped reading before the next listen starts.
_BARGE_JOIN_S: Final = 1.0


class _TokenPipe:
    """Hand LLM tokens to the TTS thread's event loop.

    The two sides run on different loops and threads. ``push`` appends under a
    lock and wakes the reader's loop with ``call_soon_threadsafe``, so a token
    is read as soon as it arrives and nothing polls while the model is quiet.
    Tokens pushed before the reader starts wait in the buffer.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: list[str | None] = []
        self._loop: asyncio.AbstractEventLoop | None = None
        self._wake: asyncio.Event | None = None

    def push(self, token: str) -> None:
        """Add one token."""
        self._put(token)

    def close(self) -> None:
        """Mark the end of the reply."""
        self._put(None)

    def _put(self, item: str | None) -> None:
        with self._lock:
            self._pending.append(item)
            loop, wake = self._loop, self._wake
        if loop is not None and wake is not None:
            # The reader's loop is gone once the turn has ended.
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(wake.set)

    def __aiter__(self) -> AsyncIterator[str]:
        return self._drain()

    async def _drain(self) -> AsyncIterator[str]:
        wake = asyncio.Event()
        with self._lock:
            self._loop = asyncio.get_running_loop()
            self._wake = wake
        while True:
            with self._lock:
                items, self._pending = self._pending, []
                if not items:
                    # Cleared under the lock, so a push after this sets it again.
                    wake.clear()
            if not items:
                await wake.wait()
                continue
            for item in items:
                if item is None:
                    return
                yield item


def start_speculation(chat: Chat, text: str) -> None:
    """Start writing the reply for ``text``, which the recogniser thinks is probably the whole turn.

    Parameters
    ----------
    chat : Chat
        The conversation. Its history is not touched: it only gains the line once the
        turn is confirmed.
    text : str
        The words so far.

    Notes
    -----
    The request is built the way the reply's own is, from a copy of the conversation with
    the line added. A streaming recogniser hears no tone, so there is no tone hint, and the
    turn's trace id does not exist yet. The reply is tagged with the user's whole last
    message, which joins ``text`` to an earlier line that got no reply, because that is
    what the confirmed turn is checked against.
    """
    drop_speculation(chat)
    asked = chat.history.copy()
    asked.add_user_line(text)
    chat.speculation = Speculation(chat.backend, chat.view.messages(asked), asked.last_user())


def drop_speculation(chat: Chat) -> None:
    """Throw away a reply that was being written ahead of time, if there is one.

    Parameters
    ----------
    chat : Chat
        The conversation.
    """
    if chat.speculation is not None:
        chat.speculation.cancel()
        chat.speculation = None
        debug("llm.speculate dropped")


def _tokens_for(
    chat: Chat,
    llm_cancel: asyncio.Event,
    trace_id: str | None,
    tone: tuple[str, ...] = (),
) -> AsyncIterator[str]:
    """Return the reply's tokens: the one already being written if it answers this line."""
    ahead, chat.speculation = chat.speculation, None
    if ahead is not None and ahead.matches(chat.history.last_user()):
        debug("llm.speculate hit, the reply was already being written")
        return ahead.replay(llm_cancel)
    if ahead is not None:
        debug("llm.speculate miss: the words changed, asking again")
        ahead.cancel()
    return chat.backend.stream(chat.messages(tone), cancel=llm_cancel, trace_id=trace_id)


async def collect_reply(
    chat: Chat,
    *,
    llm_cancel: asyncio.Event,
    trace_id: str | None,
    started: float,
    on_token: Callable[[str], None] | None = None,
    tone: tuple[str, ...] = (),
) -> tuple[str, float | None]:
    """Stream the model's reply into one string, timing its first token.

    A ``ChatUnusableError`` from the backend propagates to the caller.
    """
    parts: list[str] = []
    ttft_ms: float | None = None
    tokens = _tokens_for(chat, llm_cancel, trace_id, tone)
    try:
        async for tok in tokens:
            if ttft_ms is None:
                ttft_ms = elapsed_ms(started)
                debug("llm.first_token {:.0f}ms", ttft_ms)
            parts.append(tok)
            EVENTS.emit(ReplyToken(tok))
            if on_token is not None:
                on_token(tok)
    except asyncio.CancelledError:
        # Our own cancel flag ends the reply early. Any other cancellation
        # (asyncio.timeout, a task group, an outer cancel) must propagate.
        if not is_own_cancel(llm_cancel):
            raise
    except (BaseExceptionGroup, RuntimeError) as e:
        if not is_cancel_noise(e, cancelled=llm_cancel.is_set()):
            warn(f"[llm] {e}")
    reply = "".join(parts).strip()
    EVENTS.emit(ReplyEnd(reply))
    return reply, ttft_ms


def _first_audio(into: list[float]) -> None:
    into.append(time.perf_counter())
    EVENTS.emit(Speaking())


def _after_speech(
    ctx: DuplexContext,
    snapshot: LatencySnapshot,
    result: TtsResult,
    *,
    written: str,
    started: float,
    first_audio_at: float | None = None,
) -> tuple[LatencySnapshot, ExitCode | None]:
    """Record what was heard of ``written`` and fold the TTS result into the timings."""
    c = ctx.config
    notes = ["cancelled" if result.cancelled else "", "no audio" if not result.got_audio else ""]
    if result.failed:
        notes.append(f"error={result.error_status} {result.error_message or ''}".strip())
    debug(
        "tts.done spoken={} chars, played {} kB {}",
        len(result.spoken_so_far),
        result.bytes_played // 1000,
        " ".join(note for note in notes if note),
    )
    if result.got_audio or not result.cancelled:
        # Fish was sent cues the model never wrote: the mood carried into each sentence and
        # the default lead. Kept, they would teach the model to open every reply the same way.
        ctx.chat.history.add_reply(
            heard_written(written, result.spoken_so_far),
            interrupted=result.cancelled or result.failed,
        )
    if result.fatal:
        return snapshot, ExitCode.FATAL
    if not result.got_audio and result.cancelled:
        notice("  [tts cancelled before audio]")
    elif not result.got_audio and not result.failed:
        notice(f"  [tts silent] voice={c.fish_voice_id} model={c.tts_model}")
    return (
        replace(
            snapshot,
            tts_first_text_ms=result.tts_first_text_ms,
            tts_first_audio_ms=result.tts_first_audio_ms,
            # From the start of ASR, so it covers the whole wait after you stop talking.
            first_audio_ms=None if first_audio_at is None else (first_audio_at - started) * 1000,
            voice_to_voice_ms=elapsed_ms(started),
        ),
        None,
    )


def _reply_sink(ctx: DuplexContext, cancel: threading.Event) -> PlaybackSink:
    """Open the session's playback sink for one reply. It says which format Fish sends."""
    c = ctx.config
    return make_sink(
        c.playback,
        path=None,
        sample_rate=c.tts.sample_rate,
        device=ctx.device,
        cancel=cancel,
        aec=ctx.turn_input.mic.aec,
        latency=ctx.sink_latency,
    )


def _trace(trace_id: str | None) -> dict[str, str]:
    """Return the trace header a reply's Fish turn shares with the turn it answers."""
    return {"traceparent": make_traceparent(trace_id=trace_id)}


def _led(written: str) -> str:
    """Return a shaped reply as Fish is sent it whole: led by a cue."""
    return ensure_lead_cue(written, default="calm")


def _keep_for_resume(
    ctx: DuplexContext, sent: str, written: str, result: TtsResult, barge: BargeGate
) -> None:
    # A barge-in keeps the clip that tripped it. If that clip turns out to be
    # noise, the rest of the reply is spoken again.
    clip = barge.clip()
    if result.cancelled and clip is not None:
        ctx.turn_input.carry.keep(
            clip, unheard(sent, written, result.spoken_so_far) if result.got_audio else None
        )


async def speak_reply(
    ctx: DuplexContext,
    reply: str | Unheard,
    cancel: threading.Event,
    snapshot: LatencySnapshot,
    *,
    started: float,
    trace_id: str | None,
) -> tuple[LatencySnapshot, ExitCode | None]:
    """Speak a finished reply through one TTS turn on a private loop.

    ``reply`` is the model's text, or the rest of a reply a false barge-in cut off, which
    joins the reply it came from in the conversation.
    """
    c = ctx.config
    if isinstance(reply, Unheard):
        written, prepared = reply.written, _led(c.cue_style.shape(reply.sent))
    else:
        written = c.cue_style.shape(reply)
        prepared = _led(written)
    if is_tts_junk(prepared, drop_narration=c.drop_narration):
        notice("  (skip junk TTS)")
        return snapshot, None
    barge = BargeGate(
        mic=ctx.turn_input.mic, tune=c.barge, listen=c.listen, muted=ctx.turn_input.muted
    )
    thread = barge.start_after_bleed(cancel)
    sink = _reply_sink(ctx, cancel)
    try:
        try:
            first_audio: list[float] = []
            result = await asyncio.to_thread(
                ctx.tts.speak_shaped,
                prepared,
                sink,
                cancel=cancel,
                on_first_audio=lambda: _first_audio(first_audio),
                trace_headers=_trace(trace_id),
            )
        except PortAudioMissingError as exc:
            warn(str(exc))
            return snapshot, ExitCode.FATAL
        _keep_for_resume(ctx, prepared, written, result, barge)
        return _after_speech(
            ctx,
            snapshot,
            result,
            written=written,
            started=started,
            first_audio_at=first_audio[0] if first_audio else None,
        )
    finally:
        # The barge watch reads the mic until this is set. The next listen
        # starts as soon as this function returns.
        cancel.set()
        thread.join(timeout=_BARGE_JOIN_S)


def _end_llm_when_tts_stops(task: asyncio.Future[TtsResult], llm_cancel: asyncio.Event) -> None:
    # A barge-in or a dead sink ends the reply, so the model should stop too.
    # A Fish failure before any audio leaves the model running. The finished
    # reply is spoken afterwards on the whole-string path.
    if (
        task.cancelled()
        or task.exception() is not None
        or task.result().cancelled
        or task.result().fatal
    ):
        llm_cancel.set()


async def stream_turn(
    ctx: DuplexContext,
    heard: HeardLine,
    cancel: threading.Event,
    llm_cancel: asyncio.Event,
) -> tuple[LatencySnapshot, ExitCode | None]:
    """Speak the reply while the model is still writing it.

    Parameters
    ----------
    ctx : DuplexContext
        Session state.
    heard : HeardLine
        The accepted user line.
    cancel : threading.Event
        Stops the TTS turn.
    llm_cancel : asyncio.Event
        Stops the model.

    Returns
    -------
    tuple of LatencySnapshot and int or None
        Timings and an exit code when the turn hit a fatal error.

    Notes
    -----
    Tokens go to the TTS thread as they arrive. The barge-in gate arms at the
    first audio chunk, not at the start, so it does not listen to a silent
    speaker. A Fish failure before any audio falls back to speaking the
    finished reply. A ``ChatUnusableError`` lets the Fish turn end, then
    returns ``ExitCode.FATAL``.
    """
    c = ctx.config
    pipe = _TokenPipe()
    barge = BargeGate(
        mic=ctx.turn_input.mic, tune=c.barge, listen=c.listen, muted=ctx.turn_input.muted
    )
    barge_threads: list[threading.Thread] = []
    first_audio: list[float] = []

    def on_first_audio() -> None:
        _first_audio(first_audio)
        barge_threads.append(barge.start_after_bleed(cancel))

    sink = _reply_sink(ctx, cancel)
    snapshot = LatencySnapshot(
        asr_ms=heard.asr_ms, trace_id=heard.trace_id, end_wait_ms=heard.end_wait_ms
    )
    # The streamed reply goes to Fish as the model writes it, so unlike ``speak_reply`` it
    # gets no default [calm] lead cue and no junk check: neither can be decided before the
    # whole reply is known. A cue the model writes still leads it.
    tts_task = asyncio.create_task(
        asyncio.to_thread(
            ctx.tts.speak_stream,
            pipe,
            sink,
            cancel=cancel,
            on_first_audio=on_first_audio,
            trace_headers=_trace(heard.trace_id),
        )
    )
    tts_task.add_done_callback(lambda task: _end_llm_when_tts_stops(task, llm_cancel))
    reply = ""
    ttft_ms: float | None = None
    unusable = False
    try:
        try:
            reply, ttft_ms = await collect_reply(
                ctx.chat,
                llm_cancel=llm_cancel,
                trace_id=heard.trace_id,
                started=time.perf_counter(),
                on_token=pipe.push,
                tone=heard.tone,
            )
        except ChatUnusableError:
            # The Fish turn is already open, so it ends the usual way first.
            unusable = True
        finally:
            pipe.close()
        snapshot = replace(snapshot, llm_first_token_ms=ttft_ms)
        if not reply.strip():
            # Stop the turn before Fish is asked to flush nothing.
            cancel.set()
        try:
            result = await tts_task
        except PortAudioMissingError as exc:
            warn(str(exc))
            return snapshot, ExitCode.FATAL
        # Not the pieces the stream sent: the resumed rest and what the conversation records
        # are cut from the reply shaped whole, as it would have been spoken without streaming.
        written = c.cue_style.shape(reply)
        _keep_for_resume(ctx, _led(written), written, result, barge)
    finally:
        cancel.set()
        for thread in barge_threads:
            thread.join(timeout=_BARGE_JOIN_S)
    if not reply.strip():
        # Nothing to say. The non-streaming path never opens a Fish turn for an
        # empty reply, so skip the "silent" report and the history entry too.
        debug("tts.stream empty reply, nothing to speak")
        return snapshot, ExitCode.FATAL if unusable else None
    failed_early = (
        not result.got_audio and not result.cancelled and result.failed and not result.fatal
    )
    if failed_early and reply and not ctx.turn_input.session.quit_requested.is_set():
        debug("tts.stream failed before audio, speaking the finished reply")
        retry_cancel = threading.Event()
        # A quit after this point sets the new flag too: ``TurnSignals`` keeps it.
        ctx.turn_input.session.turn.bind(retry_cancel, llm_cancel)
        return await speak_reply(
            ctx, reply, retry_cancel, snapshot, started=heard.started, trace_id=heard.trace_id
        )
    return _after_speech(
        ctx,
        snapshot,
        result,
        written=written,
        started=heard.started,
        first_audio_at=first_audio[0] if first_audio else None,
    )
