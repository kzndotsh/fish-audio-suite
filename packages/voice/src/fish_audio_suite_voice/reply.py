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
    FishAuthError,
    LatencySnapshot,
    elapsed_ms,
    ensure_lead_cue,
    is_same_utterance,
    is_tts_junk,
    make_traceparent,
    normalize_cues,
    scrub_tts,
)
from fish_audio_suite_voice.barge import BargeGate
from fish_audio_suite_voice.cancel import is_cancel_noise, is_own_cancel
from fish_audio_suite_voice.console import end_reply_line, write_reply_token
from fish_audio_suite_voice.debug import conversation, debug, debug_enabled, warn
from fish_audio_suite_voice.duplex_state import EXIT_FATAL, DuplexContext
from fish_audio_suite_voice.events import EVENTS, ReplyEnd, ReplyToken, notice
from fish_audio_suite_voice.hearing import HeardLine
from fish_audio_suite_voice.playback import PortAudioMissingError, make_sink
from fish_audio_suite_voice.spoken import unspoken_text
from fish_audio_suite_voice.wire import TtsResult

__all__ = [
    "after_speech",
    "collect_reply",
    "speak_reply",
    "stream_turn",
    "turn_summary",
]

# watch() polls the mic for 0.2 s. The join has to cover that poll so the
# stream is closed before the next listen opens the device.
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


async def collect_reply(
    ctx: DuplexContext,
    *,
    llm_cancel: asyncio.Event,
    trace_id: str | None,
    started: float,
    on_token: Callable[[str], None] | None = None,
) -> tuple[str, float | None]:
    """Stream the model's reply into one string, timing its first token."""
    parts: list[str] = []
    ttft_ms: float | None = None
    # Debug lines print while the reply streams. Buffering the reply keeps it
    # on one line instead of split around them.
    live = not debug_enabled()
    try:
        async for tok in ctx.backend.stream(ctx.history, cancel=llm_cancel, trace_id=trace_id):
            if ttft_ms is None:
                ttft_ms = elapsed_ms(started)
                debug("llm.first_token {:.0f}ms", ttft_ms)
                if live:
                    write_reply_token("llm \u25b8 ")
            parts.append(tok)
            if live:
                write_reply_token(tok)
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
    finally:
        end_reply_line()
    reply = "".join(parts).strip()
    if reply and not live:
        conversation("llm", reply)
    EVENTS.emit(ReplyEnd(reply))
    return reply, ttft_ms


def after_speech(
    ctx: DuplexContext,
    snapshot: LatencySnapshot,
    result: TtsResult,
    *,
    started: float,
    first_audio_at: float | None = None,
    resumes: bool = False,
) -> tuple[LatencySnapshot, int | None]:
    """Record what was heard in history and fold the TTS result into the timings.

    With ``resumes``, the spoken text continues the last assistant message
    instead of starting a new one.
    """
    c = ctx.config
    history = ctx.history
    notes = ["cancelled" if result.cancelled else "", "no audio" if not result.got_audio else ""]
    if result.error_status is not None or result.error_message:
        notes.append(f"error={result.error_status} {result.error_message or ''}".strip())
    debug(
        "tts.done spoken={} chars, played {} kB {}",
        len(result.spoken_so_far),
        result.bytes_played // 1000,
        " ".join(note for note in notes if note),
    )
    if isinstance(result.error, FishAuthError):
        return snapshot, EXIT_FATAL
    if not result.got_audio and result.cancelled:
        notice("  [tts cancelled before audio]")
    elif not result.got_audio and result.error_status is None and not result.error_message:
        notice(f"  [tts silent] voice={c.fish_voice_id} model={c.tts_model}")
    spoken = result.spoken_so_far
    # The last history row is the user line this reply answers. Saving an
    # exact copy teaches the next turn to repeat them again.
    last_user = history[-1]["content"] if history and history[-1]["role"] == "user" else ""
    if (
        spoken
        and (result.got_audio or not result.cancelled)
        and not is_same_utterance(spoken, last_user)
    ):
        if resumes and history and history[-1]["role"] == "assistant":
            history[-1]["content"] = f"{history[-1]['content']} {spoken}"
        else:
            history.append({"role": "assistant", "content": spoken})
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


def _keep_for_resume(ctx: DuplexContext, sent: str, result: TtsResult, captured: bytes) -> None:
    # A barge-in keeps the clip that tripped it. If that clip turns out to be
    # noise, the rest of the reply is spoken again.
    if result.cancelled and captured:
        ctx.barge_prefix = captured
        ctx.resume_text = unspoken_text(sent, result.spoken_so_far) if result.got_audio else ""


async def speak_reply(
    ctx: DuplexContext,
    reply: str,
    cancel: threading.Event,
    snapshot: LatencySnapshot,
    *,
    started: float,
    trace_id: str | None,
    resumes: bool = False,
) -> tuple[LatencySnapshot, int | None]:
    """Speak a finished reply through one TTS turn on a private loop.

    With ``resumes``, ``reply`` is the rest of a reply a false barge-in cut
    off, and it joins the last assistant message in history.
    """
    c = ctx.config
    scrubbed = ensure_lead_cue(normalize_cues(scrub_tts(reply), lead=c.mood_lead))
    if is_tts_junk(scrubbed, drop_narration=c.drop_narration):
        notice("  (skip junk TTS)")
        return snapshot, None
    barge = BargeGate(device=ctx.device, tune=c.barge, aec=ctx.session.aec)
    thread = barge.start_after_bleed(cancel)
    sink = make_sink(
        c.playback,
        path=None,
        sample_rate=c.sample_rate,
        device=ctx.device,
        cancel=cancel,
        aec=ctx.session.aec,
    )
    ctx.tts.trace_headers = {"traceparent": make_traceparent(trace_id=trace_id)}
    try:
        try:
            first_audio: list[float] = []
            result = await asyncio.to_thread(
                ctx.tts.speak,
                scrubbed,
                sink,
                cancel=cancel,
                on_first_audio=lambda: first_audio.append(time.perf_counter()),
            )
        except PortAudioMissingError as exc:
            warn(str(exc))
            return snapshot, EXIT_FATAL
        _keep_for_resume(ctx, scrubbed, result, barge.captured)
        return after_speech(
            ctx,
            snapshot,
            result,
            started=started,
            first_audio_at=first_audio[0] if first_audio else None,
            resumes=resumes,
        )
    finally:
        # watch() holds the mic until this event is set. The next listen
        # opens the same device as soon as this function returns.
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
        or isinstance(task.result().error, FishAuthError)
    ):
        llm_cancel.set()


async def stream_turn(
    ctx: DuplexContext,
    heard: HeardLine,
    cancel: threading.Event,
    llm_cancel: asyncio.Event,
) -> tuple[LatencySnapshot, int | None]:
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
    finished reply.
    """
    c = ctx.config
    pipe = _TokenPipe()
    barge = BargeGate(device=ctx.device, tune=c.barge, aec=ctx.session.aec)
    barge_threads: list[threading.Thread] = []
    first_audio: list[float] = []

    def on_first_audio() -> None:
        first_audio.append(time.perf_counter())
        barge_threads.append(barge.start_after_bleed(cancel))

    sink = make_sink(
        c.playback,
        path=None,
        sample_rate=c.sample_rate,
        device=ctx.device,
        cancel=cancel,
        aec=ctx.session.aec,
    )
    ctx.tts.trace_headers = {"traceparent": make_traceparent(trace_id=heard.trace_id)}
    snapshot = LatencySnapshot(asr_ms=heard.asr_ms, trace_id=heard.trace_id)
    tts_task = asyncio.create_task(
        asyncio.to_thread(
            ctx.tts.speak_stream,
            pipe,
            sink,
            cancel=cancel,
            on_first_audio=on_first_audio,
        )
    )
    tts_task.add_done_callback(lambda task: _end_llm_when_tts_stops(task, llm_cancel))
    reply = ""
    try:
        try:
            reply, ttft_ms = await collect_reply(
                ctx,
                llm_cancel=llm_cancel,
                trace_id=heard.trace_id,
                started=time.perf_counter(),
                on_token=pipe.push,
            )
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
            return snapshot, EXIT_FATAL
        _keep_for_resume(
            ctx,
            ensure_lead_cue(normalize_cues(scrub_tts(reply), lead=c.mood_lead)),
            result,
            barge.captured,
        )
    finally:
        cancel.set()
        for thread in barge_threads:
            thread.join(timeout=_BARGE_JOIN_S)
    if not reply.strip():
        # Nothing to say. The non-streaming path never opens a Fish turn for an
        # empty reply, so skip the "silent" report and the history entry too.
        debug("tts.stream empty reply, nothing to speak")
        return snapshot, None
    failed_early = (
        not result.got_audio
        and not result.cancelled
        and (result.error_status is not None or bool(result.error_message))
        and not isinstance(result.error, FishAuthError)
    )
    if failed_early and reply and not ctx.session.quit_requested.is_set():
        debug("tts.stream failed before audio, speaking the finished reply")
        retry_cancel = threading.Event()
        ctx.session.turn.bind(retry_cancel, llm_cancel)
        if ctx.session.quit_requested.is_set():
            # Quit landed between the check above and the rebind, so the old
            # cancel flag no longer reaches this turn. Stop is sticky.
            retry_cancel.set()
            return snapshot, None
        return await speak_reply(
            ctx, reply, retry_cancel, snapshot, started=heard.started, trace_id=heard.trace_id
        )
    return after_speech(
        ctx,
        snapshot,
        result,
        started=heard.started,
        first_audio_at=first_audio[0] if first_audio else None,
    )


def _seconds(ms: float) -> str:
    return f"{ms / 1000:.2f}s"


def turn_summary(snapshot: LatencySnapshot) -> str:
    """One human line for a finished turn. The first number is the wait that matters."""
    fields = [
        ("first audio", snapshot.first_audio_ms),
        ("asr", snapshot.asr_ms),
        ("llm first token", snapshot.llm_first_token_ms),
        ("tts first audio", snapshot.tts_first_audio_ms),
        ("total", snapshot.voice_to_voice_ms),
    ]
    shown = [f"{name} {_seconds(value)}" for name, value in fields if value is not None]
    return "  \u21b3 " + " \u00b7 ".join(shown) if shown else ""
