"""One duplex turn: listen, Fish ASR, LLM, then TTS on a private loop."""

from __future__ import annotations

import asyncio
import threading
import time
from contextlib import AsyncExitStack

from fish_audio_suite_kit import LatencySnapshot
from fish_audio_suite_voice.aec import EchoCanceller
from fish_audio_suite_voice.asr import asr_client
from fish_audio_suite_voice.config import VoiceCliConfig
from fish_audio_suite_voice.console_sink import ConsoleSink
from fish_audio_suite_voice.debug import debug
from fish_audio_suite_voice.duplex_state import EXIT_FATAL, EXIT_OK, DuplexContext
from fish_audio_suite_voice.events import EVENTS, Bye, TurnEnded
from fish_audio_suite_voice.hearing import HeardLine, hear_line
from fish_audio_suite_voice.history import opening_history, remember_user
from fish_audio_suite_voice.inputs import TurnSource
from fish_audio_suite_voice.llm import ChatBackend
from fish_audio_suite_voice.reply import collect_reply, speak_reply, stream_turn
from fish_audio_suite_voice.signals import DuplexSession
from fish_audio_suite_voice.speaker import FishSpeaker

__all__ = [
    "EXIT_FATAL",
    "EXIT_OK",
    "bye",
    "duplex_turns",
]


def bye() -> int:
    """Return the exit code of a normal quit.

    Returns
    -------
    int
        ``EXIT_OK`` (0). Fatal Fish and PortAudio failures use 2 instead.
    """
    return EXIT_OK


async def _answer_line(ctx: DuplexContext, heard: HeardLine) -> int | None:
    cancel = threading.Event()
    llm_cancel = asyncio.Event()
    ctx.session.turn.bind(cancel, llm_cancel)
    if ctx.config.stream_tts:
        snapshot, fatal = await stream_turn(ctx, heard, cancel, llm_cancel)
        if fatal is not None:
            return fatal
    else:
        reply, ttft_ms = await collect_reply(
            ctx,
            llm_cancel=llm_cancel,
            trace_id=heard.trace_id,
            started=time.perf_counter(),
        )
        snapshot = LatencySnapshot(
            asr_ms=heard.asr_ms, llm_first_token_ms=ttft_ms, trace_id=heard.trace_id
        )
        if ctx.session.quit_requested.is_set():
            return bye()
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


async def _settle(ctx: DuplexContext) -> int | None:
    quit_requested = ctx.session.quit_requested
    ctx.session.turn.fire()
    ctx.session.turn.clear()
    barged = bool(ctx.barge_prefix)
    if quit_requested.is_set() or (
        not barged and await asyncio.to_thread(quit_requested.wait, ctx.config.barge.cooldown_s)
    ):
        return bye()
    return None


async def _resume_reply(ctx: DuplexContext) -> int | None:
    """Speak again what a false barge-in cut off."""
    text, ctx.resume_text = ctx.resume_text, ""
    debug("barge.resume {} chars", len(text))
    cancel = threading.Event()
    ctx.session.turn.bind(cancel, asyncio.Event())
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
    console: bool = True,
) -> int:
    """Mic, Fish ASR, LLM, then one TTS turn on a private loop, until quit.

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
        Cancel flags and the echo canceller. A new one is created when omitted.
        Pass your own to wire ``request_quit`` to a signal handler.
    source : TurnSource or None, optional
        Where each turn comes from. The mic when omitted.
    console : bool, optional
        Print the session to the terminal. A display that draws its own screen
        passes False and follows ``EVENTS`` instead.

    Returns
    -------
    int
        ``0`` on a normal bye. ``2`` when Fish returns 401, 402, or 403, or
        PortAudio is missing.

    Notes
    -----
    One sampled trace id is shared by ASR and the TTS websocket for that turn.
    Barge-in keeps the audio that tripped the gate and skips the post-speak
    cooldown. ``session.request_quit`` cancels the in-flight reply and ends
    the loop. A ``Bye`` event carrying the exit code is sent once however the
    session ends, including a fatal error or a crash.
    """
    session = session or DuplexSession(aec=EchoCanceller(c.aec))
    printer = ConsoleSink(EVENTS) if console else None
    code = EXIT_FATAL  # a crash ends the session like a fatal error
    try:
        code = await _session(c, tts, device, backend, session=session, source=source)
        return code
    finally:
        EVENTS.emit(Bye(code))
        if printer is not None:
            printer.close()


async def _session(
    c: VoiceCliConfig,
    tts: FishSpeaker,
    device: str | int | None,
    backend: ChatBackend,
    *,
    session: DuplexSession,
    source: TurnSource | None,
) -> int:
    async with AsyncExitStack() as stack:
        asr_http = await stack.enter_async_context(asr_client())
        history, pinned = opening_history(c.system_prompt, seed=c.pin_seed)
        ctx = DuplexContext(
            config=c,
            tts=tts,
            device=device,
            backend=backend,
            session=session,
            asr_http=asr_http,
            history=history,
            pinned=pinned,
        )
        last_user = ""
        while True:
            heard = await (
                hear_line(ctx, last_user) if source is None else source.next_turn(ctx, last_user)
            )
            if heard.kind == "bye":
                return bye()
            if heard.kind == "fatal":
                return heard.code
            if heard.kind == "again":
                continue
            if heard.kind == "noise":
                # The interrupt was not speech, so the cut-off reply goes on.
                if ctx.resume_text:
                    code = await _resume_reply(ctx)
                    if code is not None:
                        return code
                continue
            ctx.resume_text = ""
            last_user = heard.text
            remember_user(ctx.history, heard.text, c.history_turns, ctx.pinned)
            code = await _answer_line(ctx, heard)
            if code is not None:
                return code
