"""Listen for one turn, through Fish ASR or a streaming recogniser, and say what was heard.

This module only runs the listen: which recogniser hears the turn, the carried clip it starts
from, and the end-of-turn check's loop over clips. Each recogniser returns a
``asr.result.Recognised`` and ``asr.outcome.heard_line`` judges it, whichever path it took.
"""

from __future__ import annotations

import asyncio
import functools
import time
from dataclasses import replace

from fish_audio_suite_kit import is_asr_hallucination
from fish_audio_suite_voice.asr.backends import recogniser
from fish_audio_suite_voice.asr.outcome import Circumstances, HeardLine, heard_line
from fish_audio_suite_voice.asr.result import (
    BargeIn,
    CarriedClip,
    NoTurn,
    Recognised,
    StreamFallback,
    Transcript,
)
from fish_audio_suite_voice.asr.streaming import stream_turn
from fish_audio_suite_voice.asr.transcribe import transcribe_clip
from fish_audio_suite_voice.asr.turn_check import (
    OpenTurn,
    Reopened,
    continuation_tune,
    first_clip_tune,
    wait_frames,
)
from fish_audio_suite_voice.asr.turn_stream import RecogniserSettings
from fish_audio_suite_voice.capture.listen import mic_failure, record_utterance
from fish_audio_suite_voice.capture.pcm import FRAME_MS
from fish_audio_suite_voice.capture.portaudio import PortAudioMissingError
from fish_audio_suite_voice.config.tune import ListenTune
from fish_audio_suite_voice.llm.reply import drop_speculation, start_speculation
from fish_audio_suite_voice.logutil.debug import (
    clear_turn,
    debug,
    debug_enabled,
    mark_turn,
    trace,
)
from fish_audio_suite_voice.session.cancel import StopFlag, any_of
from fish_audio_suite_voice.session.events import EVENTS, Listening
from fish_audio_suite_voice.session.state import DuplexContext

__all__ = ["hear_line"]


async def hear_line(
    ctx: DuplexContext, last_user: str, *, stop: StopFlag | None = None
) -> HeardLine:
    """Listen for one turn and return what was heard.

    Parameters
    ----------
    ctx : DuplexContext
        The session.
    last_user : str
        The previous user line, to drop a stale copy of it.
    stop : StopFlag or None, optional
        Also ends the recording when set, such as a typed line or mute. The session's quit
        flag always does.

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
    quit_requested = ctx.turn_input.session.quit_requested
    stopper = quit_requested if stop is None else any_of(quit_requested, stop)
    barge = ctx.turn_input.carry.take_clip()
    carried: CarriedClip | None = barge
    opened = time.monotonic()
    outcome: Transcript | NoTurn | None = None
    ended = opened
    if recogniser(ctx.config.stt.provider) is not None:
        streamed = await _hear_streaming(ctx, barge, stopper)
        ended = time.monotonic()
        if isinstance(streamed, StreamFallback):
            # The streaming recogniser failed once speech had started. The batch recogniser
            # carries on from the speech captured so far, as one listen opened at ``opened``.
            carried = streamed
        else:
            outcome = streamed
            if isinstance(outcome, Transcript) and not quit_requested.is_set():
                mark_turn()
    if outcome is None:
        drop_speculation(ctx.chat)
        outcome, ended = await _hear_batch(ctx, carried, stopper)
    line = heard_line(
        outcome,
        Circumstances(
            last_user=last_user,
            carried=carried,
            opened=opened,
            ended=ended,
            repeat_window_s=ctx.config.repeat_window_s,
        ),
        quit_requested=quit_requested,
    )
    if line.kind != "line":
        drop_speculation(ctx.chat)  # no turn to answer, so a reply started for one is no use
    return line


async def _hear_streaming(ctx: DuplexContext, barge: BargeIn | None, stop: StopFlag) -> Recognised:
    """Hear one turn through the streaming recogniser, from the barge-in clip if there is one."""
    eager = ctx.config.stt.eager
    return await stream_turn(
        settings=RecogniserSettings(
            ctx.config.stt, fish_key=ctx.config.fish_api_key, fish_base=ctx.config.fish_base
        ),
        listen=ctx.config.listen,
        mic=ctx.turn_input.mic,
        quit_requested=ctx.turn_input.session.quit_requested,
        stop=stop,
        carried=barge,
        on_eager=functools.partial(start_speculation, ctx.chat) if eager else None,
        on_resumed=functools.partial(drop_speculation, ctx.chat) if eager else None,
    )


async def _hear_batch(
    ctx: DuplexContext, carried: CarriedClip | None, stop: StopFlag
) -> tuple[Transcript | NoTurn, float]:
    """Record a clip and hear it with Fish ASR, through the end-of-turn check when it is on.

    Returns
    -------
    tuple of (Transcript or NoTurn) and float
        What was heard, and ``time.monotonic()`` when the clip was recorded.
    """
    first_tune = first_clip_tune(ctx.config.listen, ctx.config.turn)
    try:
        wav = await asyncio.to_thread(
            record_utterance, ctx.turn_input.mic, stop, carried=carried, tune=first_tune
        )
    except PortAudioMissingError as e:
        return mic_failure(e), time.monotonic()
    ended = time.monotonic()
    if ctx.turn_input.session.quit_requested.is_set():
        return NoTurn("stopped"), ended
    if not wav:
        debug("listen.dropped (too short or none)")
        return NoTurn("noise"), ended
    mark_turn()
    if ctx.config.turn.enabled:
        return await _hear_checked(ctx, wav, stop, first_tune), ended
    heard = await transcribe_clip(
        wav, ctx.config, cancel=ctx.turn_input.session.quit_requested, client=ctx.asr_http
    )
    if isinstance(heard, Transcript):
        heard = replace(heard, end_wait_ms=_silence_ms(first_tune))
    return heard, ended


def _silence_ms(tune: ListenTune) -> float:
    """Return the silence a clip recorded with ``tune`` ended on, in milliseconds."""
    return float(tune.end_silence_frames * FRAME_MS)


async def _hear_checked(
    ctx: DuplexContext, wav: bytes, stop: StopFlag, first_tune: ListenTune
) -> Transcript | NoTurn:
    """Transcribe a clip that ended on a short silence, and carry on if the turn is not over.

    While each clip is transcribed the mic is already listening for a continuation. When
    the words so far look finished that recording is dropped. Otherwise it is awaited, its
    transcript is joined on, and the combined line is judged again, up to the turn's cap.
    The first clip's words are kept whatever happens to a continuation.
    """
    # A continuation ends on this tune's silence, so a joined line reports that wait.
    more_tune = continuation_tune(ctx.config.listen, ctx.config.turn)
    last: Transcript | None = None
    open_turn = OpenTurn()
    clip: bytes | None = wav
    while clip is not None:
        reopened = _reopen(ctx, stop, open_turn, more_tune)
        try:
            got = await transcribe_clip(
                clip, ctx.config, cancel=ctx.turn_input.session.quit_requested, client=ctx.asr_http
            )
            if isinstance(got, NoTurn):
                if got.reason != "again" or last is None:
                    return got
                break  # the words so far were heard, so a failed continuation only ends the turn
            if last is not None and is_asr_hallucination(got.text):
                break  # noise caught as a continuation
            last = got
            open_turn = open_turn.add(got.text, got.tone)
            if reopened is None or open_turn.judgement().finished:
                break
            debug(
                "turn.extend {} clip={} {!r}",
                open_turn.judgement().reason,
                open_turn.clips,
                open_turn.text[-60:],
            )
            try:
                clip = await reopened.clip()
            except PortAudioMissingError as e:
                return mic_failure(e)
        finally:
            if reopened is not None:
                await reopened.drop()
    if last is None or stop.is_set():
        return NoTurn("stopped")
    return replace(
        last,
        text=open_turn.text,
        tone=open_turn.tone,
        clips=open_turn.clips,
        end_wait_ms=_silence_ms(first_tune if open_turn.clips == 1 else more_tune),
    )


def _reopen(
    ctx: DuplexContext, stop: StopFlag, open_turn: OpenTurn, tune: ListenTune
) -> Reopened | None:
    """Start listening for a continuation, unless the turn has used them all or the mic is muted."""
    turn = ctx.config.turn
    if not open_turn.room_for_more(turn) or ctx.turn_input.muted() or stop.is_set():
        return None
    wait = wait_frames(turn)

    def record(flag: StopFlag) -> bytes | None:
        return record_utterance(ctx.turn_input.mic, flag, tune=tune, give_up_frames=wait)

    return Reopened(record, stop)
