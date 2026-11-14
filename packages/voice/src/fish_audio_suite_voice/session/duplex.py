"""The duplex session loop: listen, speech to text, LLM, then TTS, until quit."""

from __future__ import annotations

import asyncio
import threading
import time
from contextlib import AsyncExitStack

from fish_audio_suite_voice.asr.outcome import HeardLine
from fish_audio_suite_voice.asr.pipeline import hear_line
from fish_audio_suite_voice.asr.transcribe import asr_client
from fish_audio_suite_voice.capture.listen import cooldown_report
from fish_audio_suite_voice.capture.mic import SessionMic
from fish_audio_suite_voice.capture.vad import open_vad
from fish_audio_suite_voice.config.settings import VoiceCliConfig
from fish_audio_suite_voice.llm.client import ChatBackend
from fish_audio_suite_voice.llm.reply import collect_reply, speak_reply, stream_turn
from fish_audio_suite_voice.llm.transports import ChatUnusableError
from fish_audio_suite_voice.logutil.debug import debug, debug_enabled
from fish_audio_suite_voice.session.cancel import StopFlag, never
from fish_audio_suite_voice.session.events import EVENTS, Bye, ExitCode, TurnEnded
from fish_audio_suite_voice.session.inputs import TurnSource
from fish_audio_suite_voice.session.latency import LatencySnapshot
from fish_audio_suite_voice.session.signals import DuplexSession
from fish_audio_suite_voice.session.state import DuplexContext, session_context
from fish_audio_suite_voice.tts.speaker import FishSpeaker

__all__ = [
    "duplex_turns",
]


async def _answer_line(ctx: DuplexContext, heard: HeardLine) -> ExitCode | None:
    cancel = threading.Event()
    llm_cancel = asyncio.Event()
    ctx.turn_input.session.turn.bind(cancel, llm_cancel)
    if ctx.config.stream_tts:
        snapshot, fatal = await stream_turn(ctx, heard, cancel, llm_cancel)
        if fatal is not None:
            return fatal
    else:
        try:
            reply, ttft_ms = await collect_reply(
                ctx.chat,
                llm_cancel=llm_cancel,
                trace_id=heard.trace_id,
                started=time.perf_counter(),
                tone=heard.tone,
            )
        except ChatUnusableError:
            return ExitCode.FATAL
        snapshot = LatencySnapshot(
            asr_ms=heard.asr_ms,
            llm_first_token_ms=ttft_ms,
            trace_id=heard.trace_id,
            end_wait_ms=heard.end_wait_ms,
        )
        if ctx.turn_input.session.quit_requested.is_set():
            return ExitCode.OK
        if reply:
            snapshot, fatal = await speak_reply(
                ctx,
                reply,
                cancel,
                snapshot,
                started=heard.started,
                trace_id=heard.trace_id,
            )
            if fatal is not None:
                return fatal
    EVENTS.emit(TurnEnded(snapshot))
    return await _settle(ctx)


async def _settle(ctx: DuplexContext) -> ExitCode | None:
    turn = ctx.turn_input
    quit_requested = turn.session.quit_requested
    turn.session.turn.fire()
    turn.session.turn.clear()
    reply_ended = turn.mic.position
    if quit_requested.is_set() or (
        not turn.carry.barged
        and await asyncio.to_thread(quit_requested.wait, ctx.config.barge.cooldown_s)
    ):
        return ExitCode.OK
    if not turn.carry.barged and debug_enabled():
        _log_cooldown(ctx, reply_ended)
    return None


def _log_cooldown(ctx: DuplexContext, since: int) -> None:
    # The session mic keeps hearing through the cooldown. This reports whether the
    # tail of the reply came near starting a turn, which is what the cooldown guards,
    # to show whether FISH_VOICE_COOLDOWN could be shorter.
    tune = ctx.config.listen
    try:
        vad = open_vad(tune.vad_engine, tune.vad_aggressiveness)
    except Exception:  # noqa: BLE001 - a debug line must never end the session
        return
    debug("listen.cooldown {}", cooldown_report(ctx.turn_input.mic.recent(since), tune, vad))


async def _resume_reply(ctx: DuplexContext) -> ExitCode | None:
    """Speak again what a false barge-in cut off."""
    text = ctx.turn_input.carry.take_resume()
    debug("barge.resume {} chars", len(text))
    cancel = threading.Event()
    ctx.turn_input.session.turn.bind(cancel, asyncio.Event())
    _, fatal = await speak_reply(
        ctx,
        text,
        cancel,
        LatencySnapshot(),
        started=time.perf_counter(),
        trace_id=None,
        resumes=True,
    )
    if fatal is not None:
        return fatal
    return await _settle(ctx)


async def duplex_turns(
    c: VoiceCliConfig,
    tts: FishSpeaker,
    device: str | int | None,
    backend: ChatBackend,
    session: DuplexSession | None = None,
    *,
    source: TurnSource | None = None,
    full_screen: bool = False,
) -> ExitCode:
    """Run turns until quit: mic, speech to text, LLM, then one TTS turn per reply.

    Parameters
    ----------
    c : VoiceCliConfig
        Env-backed duplex settings.
    tts : FishSpeaker
        Live TTS client. Each reply uses ``speak``.
    device : str or int or None
        Mic and speaker device, or None for the host default.
    backend : ChatBackend
        Open chat backend, usually from ``open_chat_backend``.
    session : DuplexSession or None, optional
        Cancel flags. A new one is created when omitted. Pass your own to wire
        ``request_quit`` to a signal handler.
    source : TurnSource or None, optional
        Where each turn comes from, and the mute switch. The mic when omitted.
    full_screen : bool, optional
        Whether the session runs in the full-screen app, which decides an ``auto``
        ``output_latency`` (see ``config.settings.sink_latency``). False by default.

    Returns
    -------
    ExitCode
        ``OK`` on a normal bye. ``FATAL`` on a fatal error: Fish answers 401, 402 or
        403, Deepgram refuses the key, the chat model is unusable, or PortAudio is
        missing.

    Notes
    -----
    It prints nothing: every step is an event on ``EVENTS``, and a display, such as
    ``cli.console_sink.ConsoleSink``, subscribes to show it. One sampled trace id is shared
    by Fish ASR and the TTS websocket for that turn. One mic stream, with the echo
    canceller that cleans it, stays open for the whole session. Barge-in keeps the
    audio that tripped the gate and skips the post-reply cooldown.
    ``session.request_quit`` cancels the in-flight reply and ends the loop. A ``Bye``
    event carrying the exit code is sent once however the session ends, with
    ``ExitCode.CRASH`` when the session raises.
    """
    session = session or DuplexSession()
    code = ExitCode.CRASH  # replaced below unless the session raises
    try:
        code = await _session(
            c, tts, device, backend, session=session, source=source, full_screen=full_screen
        )
        return code
    finally:
        EVENTS.emit(Bye(code))


async def _session(
    c: VoiceCliConfig,
    tts: FishSpeaker,
    device: str | int | None,
    backend: ChatBackend,
    *,
    session: DuplexSession,
    source: TurnSource | None,
    full_screen: bool,
) -> ExitCode:
    async with AsyncExitStack() as stack:
        asr_http = await stack.enter_async_context(asr_client())
        ctx = session_context(
            c,
            session=session,
            open_mic=lambda aec: stack.enter_context(SessionMic(device, aec)),
            backend=backend,
            tts=tts,
            asr_http=asr_http,
            device=device,
            muted=(lambda: source.muted) if source is not None else never,
            full_screen=full_screen,
        )
        turn, history = ctx.turn_input, ctx.chat.history
        last_user = ""
        while True:
            heard = await _next_turn(ctx, source, last_user)
            if heard.kind == "bye":
                return ExitCode.OK
            if heard.kind == "fatal":
                return heard.code
            if heard.kind == "again":
                continue
            if heard.kind == "noise":
                # The interrupt was not speech, so the cut-off reply goes on.
                if turn.carry.resume:
                    code = await _resume_reply(ctx)
                    if code is not None:
                        return code
                continue
            turn.carry.take_resume()  # real speech, so the cut reply stays cut
            last_user = heard.text
            history.remember_user(heard.text)
            code = await _answer_line(ctx, heard)
            if code is not None:
                return code


async def _next_turn(ctx: DuplexContext, source: TurnSource | None, last_user: str) -> HeardLine:
    if source is None:
        return await hear_line(ctx, last_user)

    async def hear(stop: StopFlag) -> HeardLine:
        return await hear_line(ctx, last_user, stop=stop)

    return await source.next_turn(ctx.turn_input, hear)
