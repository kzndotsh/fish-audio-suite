"""Listen for one utterance and turn it into a line of text through Fish ASR."""

from __future__ import annotations

import asyncio
import functools
import time
from dataclasses import dataclass
from typing import Final, Literal

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
from fish_audio_suite_voice.barge import StopFlag
from fish_audio_suite_voice.debug import (
    clear_turn,
    debug,
    debug_enabled,
    mark_turn,
    trace,
    warn,
)
from fish_audio_suite_voice.duplex_state import EXIT_FATAL, DuplexContext
from fish_audio_suite_voice.events import EVENTS, Heard, Listening
from fish_audio_suite_voice.history import remember_user
from fish_audio_suite_voice.listen import record_utterance
from fish_audio_suite_voice.playback import PortAudioMissingError
from fish_audio_suite_voice.speculate import Speculation
from fish_audio_suite_voice.streaming import StreamedTurn, StreamFallback, stream_turn

__all__ = [
    "HeardLine",
    "accept_transcript",
    "classify_transcript",
    "drop_speculation",
    "hear_line",
    "recognize",
]


def _skip_asr(reason: str, text: str) -> Literal["skip"]:
    debug("asr.skip {} ({} chars)", reason, len(text.strip()))
    return "skip"


# Hesitation sounds, never an answer. "yeah", "yep", "uh-huh" and "嗯" are
# backchannels too, but after the assistant asks something they answer it.
_FILLERS: Final = frozenset(
    {"ah", "hmm", "huh", "mm", "uh", "um", "えっと", "呃", "唔", "啊", "어", "음"}
)


def classify_transcript(
    text: str,
    last_user: str,
    *,
    stale: bool = True,
    over_reply: bool = True,
) -> Literal["skip", "quit", "ok"]:
    """Decide whether recognized text is skipped, a quit word, or a line to answer.

    Parameters
    ----------
    text : str
        What Fish ASR heard.
    last_user : str
        The previous user line, to drop a stale copy of it.
    stale : bool, optional
        Whether the clip ended so soon after the mic opened that it may be a
        copy of the previous line.
    over_reply : bool, optional
        Whether the clip was said over the assistant's reply (a barge-in). There
        a lone "yeah" or "mm-hmm" means "go on" and is skipped. On a turn of its
        own it answers the assistant, so only hesitation sounds are skipped.

    Returns
    -------
    str
        ``"skip"``, ``"quit"`` or ``"ok"``.
    """
    if is_backchannel(text, phrases=None if over_reply else _FILLERS):
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

    kind: Literal["bye", "again", "noise", "fatal", "line"]
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
    over_reply: bool = True,
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
    return accept_transcript(
        text,
        last_user,
        asr_ms=elapsed_ms(started),
        started=started,
        trace_id=trace_id_of(asr_parent),
        stale=stale,
        over_reply=over_reply,
    )


def accept_transcript(
    text: str,
    last_user: str,
    *,
    asr_ms: float,
    started: float,
    trace_id: str | None,
    stale: bool = True,
    over_reply: bool = True,
) -> HeardLine:
    """Decide what a recognised transcript is, whichever recogniser made it.

    Parameters
    ----------
    text : str
        What was recognised.
    last_user : str
        The previous user line, to drop a stale copy of it.
    asr_ms : float
        How long recognition took after the speech ended.
    started : float
        ``time.perf_counter()`` when the speech ended, which the reply's timings count from.
    trace_id : str or None
        The trace id shared with the reply's TTS turn.
    stale : bool, optional
        See ``classify_transcript``.
    over_reply : bool, optional
        See ``classify_transcript``.

    Returns
    -------
    HeardLine
        ``bye`` for a quit word, ``noise`` for something to skip, otherwise the ``line``, which
        is also announced with a ``Heard`` event.
    """
    decision = classify_transcript(text, last_user, stale=stale, over_reply=over_reply)
    if decision == "quit":
        return HeardLine("bye")
    if decision == "skip":
        return HeardLine("noise")
    EVENTS.emit(Heard(text, asr_ms))
    return HeardLine("line", text=text, asr_ms=asr_ms, started=started, trace_id=trace_id)


async def hear_line(
    ctx: DuplexContext, last_user: str, *, stop: StopFlag | None = None
) -> HeardLine:
    """Open the mic, record one utterance, and return what was heard.

    Parameters
    ----------
    ctx : DuplexContext
        The session.
    last_user : str
        The previous line, used to drop an echo of it.
    stop : StopFlag or None, optional
        Ends the recording when set, in place of the session's quit flag. Pass one
        that is set on quit too, or Ctrl+C will not stop the mic.

    Returns
    -------
    HeardLine
        What the loop should do next.
    """
    if debug_enabled():
        debug("listen.waiting for you")
    EVENTS.emit(Listening())
    clear_turn()
    trace("listen.waiting device={}", ctx.device)
    if ctx.config.stt.provider == "deepgram" and (
        streamed := await _hear_streaming(ctx, last_user, stop)
    ):
        if streamed.kind != "line":
            drop_speculation(ctx)  # no turn to answer, so a reply started for one is no use
        return streamed
    drop_speculation(ctx)
    try:
        prefix = ctx.barge_prefix
        ctx.barge_prefix = b""
        opened = time.monotonic()
        wav = await asyncio.to_thread(
            record_utterance,
            ctx.device,
            ctx.session.quit_requested if stop is None else stop,
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
        return HeardLine("noise")
    mark_turn()
    # A barge-in clip carries fresh speech, so it is never a stale copy.
    window = ctx.config.repeat_window_s
    stale = not prefix and time.monotonic() - opened < window
    heard = await recognize(ctx, wav, last_user, stale=stale, over_reply=bool(prefix))
    # Quit during the Fish request used to come back as a normal line, so
    # the LLM still answered after Ctrl+C.
    if ctx.session.quit_requested.is_set():
        return HeardLine("bye")
    return heard


def drop_speculation(ctx: DuplexContext) -> None:
    """Throw away a reply that was being written ahead of time, if there is one."""
    if ctx.speculation is not None:
        ctx.speculation.cancel()
        ctx.speculation = None
        debug("llm.speculate dropped")


def _speculate(ctx: DuplexContext, text: str) -> None:
    """Start writing the reply for ``text``, which Flux thinks is probably the whole turn."""
    drop_speculation(ctx)
    asked = list(ctx.history)  # the real history only gains the line once the turn is confirmed
    remember_user(asked, text, ctx.config.history_turns, ctx.pinned)
    ctx.speculation = Speculation(ctx.backend, asked, text)


async def _hear_streaming(
    ctx: DuplexContext, last_user: str, stop: StopFlag | None
) -> HeardLine | None:
    """Hear one turn through Deepgram, or return None to use the batch recogniser instead.

    Parameters
    ----------
    ctx : DuplexContext
        The session.
    last_user : str
        The previous line, used to drop an echo of it.
    stop : StopFlag or None
        Ends the turn early, in place of the session's quit flag.

    Returns
    -------
    HeardLine or None
        What the loop should do next, or None when Deepgram cannot be reached, so this
        turn is heard the usual way.
    """
    quit_requested = ctx.session.quit_requested
    prefix = ctx.barge_prefix
    opened = time.monotonic()
    eager = ctx.config.stt.eager_eot_threshold > 0
    outcome = await stream_turn(
        stt=ctx.config.stt,
        listen=ctx.config.listen,
        device=ctx.device,
        aec=ctx.session.aec,
        quit_requested=quit_requested,
        stop=quit_requested if stop is None else stop,
        prefix=prefix,
        on_eager=functools.partial(_speculate, ctx) if eager else None,
        on_resumed=functools.partial(drop_speculation, ctx) if eager else None,
    )
    if isinstance(outcome, StreamFallback):
        # Deepgram failed once speech had started. The batch recogniser carries on from the
        # speech captured so far, the way it carries on from a barge-in.
        ctx.barge_prefix = outcome.prefix
        return None
    ctx.barge_prefix = b""  # the stream had it
    if quit_requested.is_set():
        return HeardLine("bye")
    if isinstance(outcome, StreamedTurn):
        mark_turn()
        stale = not prefix and time.monotonic() - opened < ctx.config.repeat_window_s
        return accept_transcript(
            outcome.text,
            last_user,
            asr_ms=outcome.asr_ms,
            started=outcome.started,
            trace_id=outcome.trace_id,
            stale=stale,
            over_reply=bool(prefix),
        )
    match outcome:
        case "fatal":
            return HeardLine("fatal", code=EXIT_FATAL)
        case "again":
            return HeardLine("again")
        case _:  # "stopped" or "noise": a typed line or mute cut it short, or nothing was said
            return HeardLine("noise")
