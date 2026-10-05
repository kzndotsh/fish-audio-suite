"""A snapshot of a session, folded from its events, for a display to show."""

from __future__ import annotations

import math
from collections.abc import Sequence
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
    "TurnTimings",
    "level_fraction",
    "reduce_view",
    "split_cells",
    "turn_timings",
    "wave_dots",
]

_FULL_SCALE: Final = 32768.0  # the peak of 16-bit audio
_QUIETEST_DB: Final = -60.0  # at or below this, a meter shows empty
_BRAILLE: Final = 0x2800  # the first Braille pattern, which has no dots
# The bit that raises each dot of a Braille cell, by dot row (top to bottom) and column.
_DOT_BITS: Final = ((0x01, 0x08), (0x02, 0x10), (0x04, 0x20), (0x40, 0x80))


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
    0.49
    """
    if rms <= 0:
        return 0.0
    decibels = 20 * math.log10(rms / _FULL_SCALE)
    return min(1.0, max(0.0, (decibels - _QUIETEST_DB) / -_QUIETEST_DB))


def wave_dots(levels: Sequence[float], rows: int) -> list[str]:
    """Draw a mirrored, dot-matrix waveform: one bar per level, two bars to a character.

    Parameters
    ----------
    levels : Sequence[float]
        How loud each bar is, from 0 to 1, as ``level_fraction`` gives it, left to right.
        Even silence draws one dot above and one below the middle, so a quiet mic still
        shows a thin dotted line.
    rows : int
        How many lines tall the drawing is. Each line has four dot rows, so the middle
        splits ``rows * 4`` dot rows into two equal halves. At least 1.

    Returns
    -------
    list[str]
        ``rows`` strings of Braille characters, each ``ceil(len(levels) / 2)`` long, top to
        bottom. Braille draws with dots in the foreground colour only, so a cell needs no
        background and no reversed colours.

    Examples
    --------
    >>> wave_dots([0.0, 0.0], 1)
    ['\u2836']
    >>> wave_dots([1.0, 1.0], 1)
    ['\u28ff']
    """
    rows = max(1, rows)
    half = rows * 2  # dot rows above, and below, the middle
    cells = (len(levels) + 1) // 2
    grid = [[0] * cells for _ in range(rows)]
    for x, level in enumerate(levels):
        reach = max(1, round(min(1.0, max(0.0, level)) * half))
        column = x % 2
        for step in range(reach):
            for dot_row in (half - 1 - step, half + step):
                grid[dot_row // 4][x // 2] |= _DOT_BITS[dot_row % 4][column]
    return ["".join(chr(_BRAILLE + bits) for bits in row) for row in grid]


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


@dataclass(frozen=True, slots=True)
class TurnTimings:
    """Where the wait before a reply was heard went, for a display to draw.

    Attributes
    ----------
    wait_ms : float
        From the end of the utterance until the first audio played.
    total_ms : float or None
        From the end of the utterance until the reply finished playing, when known.
    stages : tuple[tuple[str, float], ...]
        What the wait was spent on, in order, as ``(name, milliseconds)``. They add up to
        ``wait_ms``. Stages that took no time are left out.
    """

    wait_ms: float
    total_ms: float | None
    stages: tuple[tuple[str, float], ...]


def turn_timings(snapshot: LatencySnapshot) -> TurnTimings | None:
    """Break a turn's wait into recognising, thinking and speaking.

    Parameters
    ----------
    snapshot : LatencySnapshot
        The finished turn.

    Returns
    -------
    TurnTimings or None
        ``None`` when the turn has no wait to break down (no ``first_audio_ms``).

    Notes
    -----
    ``tts`` is what is left of the wait after ``asr`` and ``llm``: the time from the model's
    first token to audio playing. It is not ``tts_first_audio_ms``, which is measured from
    the start of the TTS turn and so includes the wait for the model.

    Examples
    --------
    >>> t = turn_timings(LatencySnapshot(asr_ms=300, llm_first_token_ms=700, first_audio_ms=1500))
    >>> t.stages
    (('asr', 300), ('llm', 700), ('tts', 500))
    >>> turn_timings(LatencySnapshot(asr_ms=300)) is None
    True
    """
    wait = snapshot.first_audio_ms
    if wait is None:
        return None
    known = [("asr", snapshot.asr_ms), ("llm", snapshot.llm_first_token_ms)]
    stages = [(name, ms) for name, ms in known if ms is not None and ms > 0]
    rest = wait - sum(ms for _, ms in stages)
    if rest > 0:
        stages.append(("tts", rest))
    return TurnTimings(wait, snapshot.voice_to_voice_ms, tuple(stages))


def split_cells(parts: Sequence[float], width: int) -> list[int]:
    """Share ``width`` cells between ``parts`` in proportion, so a bar adds up exactly.

    Parameters
    ----------
    parts : Sequence[float]
        The sizes to draw. Negative sizes count as zero.
    width : int
        How many cells in all.

    Returns
    -------
    list[int]
        One count per part, adding up to ``width`` (all zeros when there is nothing to
        draw). A part that is not zero gets at least one cell when ``width`` allows it, so
        a short stage does not vanish.

    Examples
    --------
    >>> split_cells([300, 700, 500], 15)
    [3, 7, 5]
    >>> split_cells([1, 1000], 10)
    [1, 9]
    """
    sizes = [max(0.0, part) for part in parts]
    total = sum(sizes)
    if total <= 0 or width <= 0:
        return [0] * len(sizes)
    exact = [size / total * width for size in sizes]
    cells = [int(value) for value in exact]
    by_remainder = sorted(range(len(sizes)), key=lambda i: exact[i] - cells[i], reverse=True)
    for i in by_remainder[: width - sum(cells)]:
        cells[i] += 1
    for i, size in enumerate(sizes):
        if size > 0 and cells[i] == 0:
            donor = max(range(len(cells)), key=lambda j: cells[j])
            if cells[donor] > 1:
                cells[donor] -= 1
                cells[i] = 1
    return cells
