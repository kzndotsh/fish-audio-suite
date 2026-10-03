"""One duplex turn: listen, Fish ASR, LLM, then isolated TTS."""

from __future__ import annotations

import asyncio
import threading
import time
from contextlib import AsyncExitStack

from fish_audio_suite_kit import LatencySnapshot
from fish_audio_suite_voice.aec import EchoCanceller
from fish_audio_suite_voice.asr import asr_client
from fish_audio_suite_voice.config import VoiceCliConfig
from fish_audio_suite_voice.debug import console_print, debug, env_debug, trace
from fish_audio_suite_voice.duplex_state import EXIT_FATAL, EXIT_OK, DuplexState
from fish_audio_suite_voice.hearing import HeardLine, hear_line
from fish_audio_suite_voice.history import opening_history, remember_user
from fish_audio_suite_voice.live import IsolatedFishTts
from fish_audio_suite_voice.llm import ChatBackend
from fish_audio_suite_voice.reply import collect_reply, speak_reply, stream_turn, turn_summary
from fish_audio_suite_voice.signals import DuplexSession

__all__ = [
    "EXIT_FATAL",
    "EXIT_OK",
    "bye",
    "duplex_turns",
]


def bye() -> int:
    """Print the quit line and return success.

    Returns
    -------
    int
        ``EXIT_OK`` (0). Fatal Fish and PortAudio failures use 2 instead.
    """
    console_print("\nbye")
    return EXIT_OK


async def _answer_line(loop: DuplexState, heard: HeardLine) -> int | None:
    cancel = threading.Event()
    llm_cancel = asyncio.Event()
    loop.session.turn.bind(cancel, llm_cancel)
    if loop.config.stream_tts:
        snapshot, fatal = await stream_turn(loop, heard, cancel, llm_cancel)
        if fatal is not None:
            return fatal
    else:
        reply, ttft_ms = await collect_reply(
            loop,
            llm_cancel=llm_cancel,
            trace_id=heard.trace_id,
            started=time.perf_counter(),
        )
        snapshot = LatencySnapshot(asr_ms=heard.asr_ms, llm_ttft=ttft_ms, trace_id=heard.trace_id)
        if loop.session.stop.is_set():
            return bye()
        if reply:
            snapshot, fatal = await speak_reply(
                loop,
                reply,
                cancel,
                snapshot,
                started=heard.started,
                trace_id=heard.trace_id,
            )
            if fatal is not None:
                return fatal
    summary = turn_summary(snapshot)
    if env_debug():
        debug("turn.summary {}", summary.strip().removeprefix("\u21b3 "))
        trace("turn.timing {}", snapshot.log_line())
    else:
        console_print(summary, flush=True)
    stop = loop.session.stop
    loop.session.turn.fire()
    loop.session.turn.clear()
    barged = bool(loop.barge_prefix)
    if stop.is_set() or (
        not barged and await asyncio.to_thread(stop.wait, loop.config.barge.cooldown_s)
    ):
        return bye()
    return None


async def duplex_turns(
    c: VoiceCliConfig,
    tts: IsolatedFishTts,
    device: str | int | None,
    backend: ChatBackend,
    session: DuplexSession | None = None,
) -> int:
    """Mic, Fish ASR, LLM, then one isolated TTS turn, until quit.

    Parameters
    ----------
    c : VoiceCliConfig
        Env-backed duplex settings.
    tts : IsolatedFishTts
        Live TTS client. Each reply uses ``speak_isolated``.
    device : str or int or None
        Mic and speaker device, or None for the host default.
    backend : ChatBackend
        Open chat backend, usually from ``open_chat_backend``.
    session : DuplexSession or None, optional
        Cancel flags and the echo canceller. A new one is created when omitted.
        Pass your own to wire ``request_quit`` to a signal handler.

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
    the loop.
    """
    session = session or DuplexSession(aec=EchoCanceller(c.aec))
    async with AsyncExitStack() as stack:
        asr_http = await stack.enter_async_context(asr_client())
        history, pinned = opening_history(c.system_prompt)
        loop = DuplexState(
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
            heard = await hear_line(loop, last_user)
            if heard.kind == "bye":
                return bye()
            if heard.kind == "fatal":
                return heard.code
            if heard.kind == "again":
                continue
            last_user = heard.text
            remember_user(loop.history, heard.text, c.history_turns, loop.pinned)
            code = await _answer_line(loop, heard)
            if code is not None:
                return code
