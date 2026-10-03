"""Listen for one utterance and turn it into a line of text through Fish ASR."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Literal

import httpx

from fish_audio_suite_kit import (
    FishAuthError,
    FishHttpError,
    elapsed_ms,
    is_asr_hallucination,
    is_backchannel,
    is_quit_utterance,
    is_same_utterance,
    make_traceparent,
    trace_id_of,
)
from fish_audio_suite_voice.asr import fish_asr
from fish_audio_suite_voice.debug import (
    clear_turn,
    console_print,
    conversation,
    debug,
    debug_enabled,
    mark_turn,
    trace,
    warn,
)
from fish_audio_suite_voice.duplex_state import EXIT_FATAL, DuplexContext
from fish_audio_suite_voice.listen import record_utterance
from fish_audio_suite_voice.playback import PortAudioMissingError

__all__ = [
    "HeardLine",
    "classify_transcript",
    "hear_line",
    "recognize",
]


def _skip_asr(reason: str, text: str) -> Literal["skip"]:
    debug("asr.skip {} ({} chars)", reason, len(text.strip()))
    return "skip"


def classify_transcript(
    text: str,
    last_user: str,
    *,
    stale: bool = True,
) -> Literal["skip", "quit", "ok"]:
    """Decide whether recognized text is skipped, a quit word, or a line to answer."""
    if is_backchannel(text):
        return _skip_asr("backchannel", text)
    if is_asr_hallucination(text):
        return _skip_asr("hallucination", text)
    if is_quit_utterance(text):
        return "quit"
    # Only a clip that ended almost as the mic opened can be a stale copy. A
    # later repeat is the user saying it again on purpose.
    if stale and is_same_utterance(text, last_user):
        return "skip"
    return "ok"


@dataclass(frozen=True, slots=True)
class HeardLine:
    """What one listen step produced, and what the loop should do next."""

    kind: Literal["bye", "again", "fatal", "line"]
    text: str = ""
    asr_ms: float = 0.0
    started: float = 0.0
    trace_id: str | None = None
    code: int = 0


async def recognize(
    ctx: DuplexContext,
    wav: bytes,
    last_user: str,
    *,
    stale: bool = True,
) -> HeardLine:
    """Send one utterance to Fish ASR and classify the text it returns."""
    c = ctx.config
    started = time.perf_counter()
    asr_parent = make_traceparent()
    try:
        text = await fish_asr(
            wav,
            c.fish_api_key,
            base=c.fish_base,
            language=c.fish_asr_language,
            model=c.asr_model,
            extra_headers={"traceparent": asr_parent},
            cancel=ctx.session.quit_requested,
            client=ctx.asr_http,
        )
    except FishHttpError as e:
        if ctx.session.quit_requested.is_set():
            return HeardLine("bye")
        warn(f"[asr] {e.status} {e.message}")
        if isinstance(e, FishAuthError):
            return HeardLine("fatal", code=EXIT_FATAL)
        return HeardLine("again")
    except (httpx.HTTPError, OSError) as e:
        # A network failure is worth another try on the next utterance. Any
        # other error is a bug and propagates instead of looping silently.
        if ctx.session.quit_requested.is_set():
            return HeardLine("bye")
        warn(f"[asr] {e}")
        return HeardLine("again")
    if ctx.session.quit_requested.is_set():
        return HeardLine("bye")
    asr_ms = elapsed_ms(started)
    decision = classify_transcript(text, last_user, stale=stale)
    if decision == "quit":
        return HeardLine("bye")
    if decision == "skip":
        return HeardLine("again")
    conversation("you", text)
    return HeardLine(
        "line",
        text=text,
        asr_ms=asr_ms,
        started=started,
        trace_id=trace_id_of(asr_parent),
    )


async def hear_line(ctx: DuplexContext, last_user: str) -> HeardLine:
    """Open the mic, record one utterance, and return what was heard."""
    if debug_enabled():
        debug("listen.waiting for you")
    else:
        console_print("listening…")
    clear_turn()
    trace("listen.waiting device={}", ctx.device)
    try:
        prefix = ctx.barge_prefix
        ctx.barge_prefix = b""
        opened = time.monotonic()
        wav = await asyncio.to_thread(
            record_utterance,
            ctx.device,
            ctx.session.quit_requested,
            prefix=prefix,
            tune=ctx.config.listen,
            aec=ctx.session.aec,
        )
    except PortAudioMissingError as e:
        warn(str(e))
        return HeardLine("fatal", code=EXIT_FATAL)
    if ctx.session.quit_requested.is_set():
        return HeardLine("bye")
    if not wav:
        debug("listen.dropped (too short or none)")
        return HeardLine("again")
    mark_turn()
    # A barge-in clip carries fresh speech, so it is never a stale copy.
    window = ctx.config.repeat_window_s
    stale = not prefix and time.monotonic() - opened < window
    heard = await recognize(ctx, wav, last_user, stale=stale)
    # Quit during the Fish request used to come back as a normal line, so
    # the LLM still answered after Ctrl+C.
    if ctx.session.quit_requested.is_set():
        return HeardLine("bye")
    return heard
