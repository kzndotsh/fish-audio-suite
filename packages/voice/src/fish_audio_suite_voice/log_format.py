"""How a log line looks: its tag, its time columns and its colours.

The terminal, the log file and the plain display's conversation lines all lay a line out
the same way, ``HH:MM:SS.mmm  +0.42s  tag    text``, from the pieces here.
"""

from __future__ import annotations

import os
import re
import sys
import time
from datetime import datetime
from typing import Final, TextIO

__all__ = [
    "TAG_WIDTH",
    "clock_stamp",
    "compose",
    "dim",
    "log_tag",
    "paint",
    "start_turn_clock",
    "stop_turn_clock",
    "turn_offset",
]

TAG_WIDTH: Final = 7
_TAGGED: Final = re.compile(r"^([a-z]+)\.([a-z_]+)\b ?(.*)$", re.DOTALL)
_BRACKETED: Final = re.compile(r"^\[([A-Za-z-]+)\]\s*(.*)$", re.DOTALL)
_TAG_COLORS: Final = {
    "you": "96",
    "listen": "36",
    "asr": "33",
    "stt": "33",
    "llm": "35",
    "turn": "37",
    "tts": "32",
    "barge": "31",
    "aec": "34",
    "warn": "93",
}


def log_tag(message: str, level: str) -> tuple[str, str]:
    """Split a log message into the tag a display shows and the rest.

    Parameters
    ----------
    message : str
        The logged message, such as ``tts.start voice=x`` or ``[llm] stopping``.
    level : str
        The log level name, such as ``WARNING``.

    Returns
    -------
    tuple of str
        The tag (``tts``, ``warn``, ``log``) and the text without it. A warning
        without a tag of its own is tagged ``warn``.
    """
    tagged = _TAGGED.match(message)
    if tagged:
        tag, name, rest = tagged.groups()
        body = f"{name} {rest}".rstrip()
    elif bracketed := _BRACKETED.match(message):
        tag, body = bracketed.group(1).lower(), bracketed.group(2)
    elif message.startswith("HTTP Request:"):
        tag, body = "http", message.removeprefix("HTTP Request:").strip()
    else:
        tag, body = "log", message
    if level == "WARNING" and tag == "log":
        tag = "warn"
    return tag, body


def clock_stamp(when: datetime | None = None) -> str:
    """Return the wall-clock column of a line, ``HH:MM:SS.mmm``.

    Parameters
    ----------
    when : datetime or None, optional
        The time of the line, such as a log record's. Now when omitted.

    Returns
    -------
    str
        The local time to the millisecond.
    """
    moment = when if when is not None else datetime.now().astimezone()
    return moment.strftime("%H:%M:%S.") + f"{moment.microsecond // 1000:03d}"


class _TurnClock:
    t0: float | None = None
    count: int = 0


_TURN: Final = _TurnClock()


def start_turn_clock() -> int:
    """Start the per-turn clock, so later lines show the seconds since now.

    Returns
    -------
    int
        The number of this turn, counting from 1.
    """
    _TURN.t0 = time.perf_counter()
    _TURN.count += 1
    return _TURN.count


def stop_turn_clock() -> None:
    """Stop the per-turn clock, so the lines that follow carry no stale offset."""
    _TURN.t0 = None


def turn_offset() -> str:
    """Return the turn column of a line: ``+0.42s`` since the turn began, or blanks.

    Returns
    -------
    str
        Eight characters wide.
    """
    t0 = _TURN.t0
    if t0 is None:
        return " " * 8
    return f"{time.perf_counter() - t0:+.2f}s".rjust(8)


def _color_on(stream: TextIO | None) -> bool:
    target = stream if stream is not None else sys.stderr
    return bool(target.isatty()) and "NO_COLOR" not in os.environ


def dim(text: str, stream: TextIO | None = None) -> str:
    """Return ``text`` dimmed when ``stream`` (stderr by default) is a colour terminal.

    Parameters
    ----------
    text : str
        What to show.
    stream : TextIO or None, optional
        Where it goes. ``NO_COLOR`` or a pipe keeps it plain.

    Returns
    -------
    str
        The text, wrapped in the dim escape or not.
    """
    return f"\x1b[2m{text}\x1b[0m" if _color_on(stream) else text


def paint(code: str, text: str, stream: TextIO | None = None) -> str:
    """Return ``text`` in an ANSI colour when ``stream`` (stderr by default) is a colour terminal.

    Parameters
    ----------
    code : str
        The SGR code, such as ``93`` or ``1;96``.
    text : str
        What to show.
    stream : TextIO or None, optional
        Where it goes. ``NO_COLOR`` or a pipe keeps it plain.

    Returns
    -------
    str
        The text, coloured or not.
    """
    return f"\x1b[{code}m{text}\x1b[0m" if _color_on(stream) else text


def compose(stamp: str, tag: str, label: str, body: str, stream: TextIO | None = None) -> str:
    """Lay out one terminal line: the time columns, the coloured label, then the body.

    Parameters
    ----------
    stamp : str
        The wall-clock column, from ``clock_stamp``.
    tag : str
        Picks the label's colour.
    label : str
        What the tag column shows, padded to ``TAG_WIDTH``.
    body : str
        The rest of the line.
    stream : TextIO or None, optional
        Where it goes, which decides whether it is coloured.

    Returns
    -------
    str
        The line, without a newline.
    """
    tag_text = paint(_TAG_COLORS.get(tag, "90"), label.ljust(TAG_WIDTH), stream)
    return f"{dim(stamp, stream)} {dim(turn_offset(), stream)}  {tag_text}{body}"
