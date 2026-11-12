"""What a heard turn becomes: the line the loop answers, or what it does instead.

Every recogniser's outcome goes through ``heard_line``, so a streamed turn, a Fish ASR clip and
a turn the end-of-turn check joined from several clips are judged by the same rules: a quit
ends the session, a failure maps to listening again or stopping, and the words are skipped
when they are a hallucination, a filler, a listener's "go on" said over the reply, or a stale
copy of the last line.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from fish_audio_suite_kit import (
    is_asr_hallucination,
    is_backchannel,
    is_filler,
    is_quit_utterance,
    is_same_utterance,
)
from fish_audio_suite_voice.cancel import StopFlag
from fish_audio_suite_voice.debug import debug
from fish_audio_suite_voice.events import EVENTS, ExitCode, Heard
from fish_audio_suite_voice.recognised import CarriedClip, NoTurn, Transcript, said_over_reply

__all__ = [
    "Circumstances",
    "HeardLine",
    "classify_transcript",
    "heard_line",
]


@dataclass(frozen=True, slots=True)
class HeardLine:
    """What one listen step produced, and what the loop should do next."""

    kind: Literal["bye", "again", "noise", "fatal", "line"]
    text: str = ""
    asr_ms: float = 0.0
    started: float = 0.0
    trace_id: str | None = None
    code: ExitCode = ExitCode.OK
    # The silence waited out after the last word, when the listener ended the turn.
    end_wait_ms: float | None = None
    # How Fish heard the voice (``laughter``, ``tired``). Only set with ``FISH_VOICE_HEAR_TONE``,
    # and only the model's request for this turn sees it.
    tone: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Circumstances:
    """What a turn was heard after, which decides how its words are judged.

    Attributes
    ----------
    last_user : str
        The previous user line, to drop a stale copy of it.
    carried : BargeIn or StreamFallback or None
        The speech the listen started from. A barge-in clip, or a fallback from a turn that
        began with one, was said over the reply.
    opened : float
        ``time.monotonic()`` when listening for the turn began.
    ended : float
        ``time.monotonic()`` when its speech ended: the clip was recorded, or the streamed
        turn was decided.
    repeat_window_s : float
        ``FISH_VOICE_REPEAT_WINDOW``: a line that ended this soon after listening began may be
        a stale copy of the last one.
    """

    last_user: str
    carried: CarriedClip | None
    opened: float
    ended: float
    repeat_window_s: float

    def stale(self, transcript: Transcript) -> bool:
        """Say whether ``transcript`` may be a stale copy of the last line.

        Parameters
        ----------
        transcript : Transcript
            The words heard.

        Returns
        -------
        bool
            True only for a single clip that ended almost as listening began, and not said
            over the reply: speech that interrupted a reply is fresh, and a later repeat or a
            line joined from several clips is the user saying it again on purpose.
        """
        return (
            not said_over_reply(self.carried)
            and transcript.clips == 1
            and self.ended - self.opened < self.repeat_window_s
        )


def _skip_asr(reason: str, text: str) -> Literal["skip"]:
    debug("asr.skip {} ({} chars)", reason, len(text.strip()))
    return "skip"


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
        What was recognised.
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
    if is_backchannel(text) if over_reply else is_filler(text):
        return _skip_asr("backchannel", text)
    if is_asr_hallucination(text):
        return _skip_asr("hallucination", text)
    if is_quit_utterance(text):
        return "quit"
    if stale and is_same_utterance(text, last_user):
        return "skip"
    return "ok"


def heard_line(
    outcome: Transcript | NoTurn, at: Circumstances, *, quit_requested: StopFlag
) -> HeardLine:
    """Turn what a recogniser heard into what the loop does next.

    Parameters
    ----------
    outcome : Transcript or NoTurn
        The words, or why there are none.
    at : Circumstances
        What the turn was heard after.
    quit_requested : StopFlag
        The session's quit flag. Once it is set nothing is answered, or the model would
        reply after Ctrl+C.

    Returns
    -------
    HeardLine
        ``bye`` on quit or a quit word, ``fatal`` (with ``ExitCode.FATAL``), ``again``, ``noise``
        for nothing to answer, otherwise the ``line``, which is also announced with a
        ``Heard`` event.
    """
    if quit_requested.is_set():
        return HeardLine("bye")
    match outcome:
        case NoTurn(reason="fatal"):
            return HeardLine("fatal", code=ExitCode.FATAL)
        case NoTurn(reason="again"):
            return HeardLine("again")
        case NoTurn():
            return HeardLine("noise")
        case Transcript():
            pass
    decision = classify_transcript(
        outcome.text,
        at.last_user,
        stale=at.stale(outcome),
        over_reply=said_over_reply(at.carried),
    )
    if decision == "quit":
        return HeardLine("bye")
    if decision == "skip":
        return HeardLine("noise")
    EVENTS.emit(Heard(outcome.text, outcome.asr_ms))
    return HeardLine(
        "line",
        text=outcome.text,
        asr_ms=outcome.asr_ms,
        started=outcome.started,
        trace_id=outcome.trace_id,
        tone=outcome.tone,
        end_wait_ms=outcome.end_wait_ms,
    )
