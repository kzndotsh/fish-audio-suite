"""A snapshot of a session, folded from its events, for a display to show."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Final

from fish_audio_suite_kit import LatencySnapshot
from fish_audio_suite_voice.events import (
    Bye,
    Event,
    Heard,
    MicLevel,
    Notice,
    ReplyEnd,
    ReplyToken,
    SessionState,
    TurnEnded,
    next_state,
)

__all__ = [
    "SessionView",
    "level_fraction",
    "reduce_view",
]

_FULL_SCALE: Final = 32768.0  # the peak of 16-bit audio
_QUIETEST_DB: Final = -60.0  # at or below this, a meter shows empty


@dataclass(frozen=True, slots=True)
class SessionView:
    """Everything a display shows about a session, as one immutable value.

    Attributes
    ----------
    state : SessionState
        What the session is doing.
    heard : str
        The last line that will be, or was, answered. Empty before the first.
    reply : str
        The model's reply so far, growing token by token, then the whole reply.
    last_turn : LatencySnapshot or None
        The timings of the last finished turn.
    turns : int
        How many turns have finished.
    mic_rms : float
        The latest mic level.
    mic_need : float
        The level that counts as speech.
    last_notice : str
        The latest status line, such as a retry.
    exit_code : int or None
        Set when the session has ended. None while it runs.
    """

    state: SessionState = SessionState.IDLE
    heard: str = ""
    reply: str = ""
    last_turn: LatencySnapshot | None = None
    turns: int = 0
    mic_rms: float = 0.0
    mic_need: float = 0.0
    last_notice: str = ""
    exit_code: int | None = None

    @property
    def mic_fraction(self) -> float:
        """How full the mic meter is, from 0 to 1."""
        return level_fraction(self.mic_rms)

    @property
    def mic_need_fraction(self) -> float:
        """Where the speech threshold sits on the meter, from 0 to 1."""
        return level_fraction(self.mic_need)


def level_fraction(rms: float) -> float:
    """Map a mic RMS level to how full a meter should be, from 0 to 1.

    Parameters
    ----------
    rms : float
        An RMS level of 16-bit audio, such as ``MicLevel.rms``.

    Returns
    -------
    float
        0 at 60 dB below full scale or quieter, 1 at full scale, in between on a
        decibel scale, which is how loudness is heard. A raw linear bar would look
        empty for ordinary speech, which sits at a few hundred out of 32768.

    Examples
    --------
    >>> level_fraction(0.0), level_fraction(32768.0)
    (0.0, 1.0)
    >>> round(level_fraction(1000.0), 2)
    0.5
    """
    if rms <= 0:
        return 0.0
    decibels = 20 * math.log10(rms / _FULL_SCALE)
    return min(1.0, max(0.0, (decibels - _QUIETEST_DB) / -_QUIETEST_DB))


def reduce_view(view: SessionView, event: Event) -> SessionView:
    """Return the view after ``event``, leaving ``view`` as it was.

    Parameters
    ----------
    view : SessionView
        The view before.
    event : Event
        What just happened.

    Returns
    -------
    SessionView
        A new view, or ``view`` itself when the event changes nothing. A display
        assigns the result to one reactive attribute, which refreshes only when the
        value changed.

    Notes
    -----
    Call it from one thread, in event order, such as the thread that drains an
    ``EventQueue``. ``StateChanged`` and ``LogLine`` events do not change the view:
    the state is worked out from the other events here, and log lines belong in a log.
    """
    moved = replace(view, state=next_state(view.state, event))
    match event:
        case Heard(text=text):
            return replace(moved, heard=text, reply="")
        case ReplyToken(text=text):
            return replace(moved, reply=view.reply + text)
        case ReplyEnd(text=text):
            return replace(moved, reply=text)
        case MicLevel(rms=rms, need=need):
            return replace(moved, mic_rms=rms, mic_need=need)
        case Notice(text=text):
            return replace(moved, last_notice=text)
        case TurnEnded(snapshot=snapshot):
            return replace(moved, last_turn=snapshot, turns=view.turns + 1)
        case Bye(code=code):
            return replace(moved, exit_code=code)
        case _:
            return moved
