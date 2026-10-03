"""OpenAI-shaped caption files from timed ASR cues. No network."""

from __future__ import annotations

import math
import re
from typing import NamedTuple

from fish_audio_suite_kit.defaults import MS_PER_S

__all__ = [
    "CaptionCue",
    "format_as_srt",
    "format_as_vtt",
]


class CaptionCue(NamedTuple):
    """One timed ASR phrase for SubRip or WebVTT.

    Attributes
    ----------
    start : float
        Start in seconds. Non-finite values render as ``00:00:00``.
    end : float
        End in seconds. A value before ``start`` is raised to ``start``.
    text : str
        Spoken text. A blank cue is left out of the file.
    """

    start: float
    end: float
    text: str


_MINUTE_MS = 60 * MS_PER_S
_HOUR_MS = 60 * _MINUTE_MS
_CUE_BLANK_RE = re.compile(r"\n(?:[ \t]*\n)+")


def _ms(seconds: float) -> int:
    """Whole milliseconds for a clock. Non-finite and negative values are 0."""
    if not math.isfinite(seconds):
        return 0
    try:
        return round(max(0.0, seconds) * MS_PER_S)
    except OverflowError:
        return 0


def _clock(ms: int, decimal: str) -> str:
    hours, ms = divmod(ms, _HOUR_MS)
    minutes, ms = divmod(ms, _MINUTE_MS)
    secs, ms = divmod(ms, MS_PER_S)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{decimal}{ms:03d}"


def _cue_body(text: str, *, escape: bool) -> str:
    flat = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    flat = _CUE_BLANK_RE.sub("\n", flat)
    # "-->" inside the transcript is a second timing line. Players then
    # drop or retimes the words after it.
    flat = flat.replace("-->", "->")
    # "&" and "<" start a WebVTT escape or tag. The words after them drop.
    # SubRip shows those entities as the letters amp and lt.
    if not escape:
        return flat
    return flat.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _span(cue: CaptionCue, decimal: str, *, escape: bool) -> tuple[str, str, str] | None:
    body = _cue_body(cue.text, escape=escape)
    if not body:
        return None
    start_ms = _ms(cue.start)
    end_ms = _ms(cue.end) if math.isfinite(cue.end) else start_ms
    # WebVTT drops a cue whose end is not later than its start, so an inverted
    # or zero-length segment gets one millisecond. The comparison is on integers:
    # formatted clocks do not sort as text once the hours reach three digits.
    end_ms = max(end_ms, start_ms + 1)
    return _clock(start_ms, decimal), _clock(end_ms, decimal), body


def _spans(cues: list[CaptionCue], decimal: str, *, escape: bool) -> list[tuple[str, str, str]]:
    spans: list[tuple[str, str, str]] = []
    for cue in cues:
        span = _span(cue, decimal, escape=escape)
        if span is not None:
            spans.append(span)
    return spans


def _range(start: str, end: str) -> str:
    return f"{start} --> {end}"


def format_as_srt(cues: list[CaptionCue]) -> str:
    """Render timed cues as SubRip.

    Parameters
    ----------
    cues : list of CaptionCue
        Timed phrases. Blank text is skipped. Clocks use a comma before milliseconds.

    Returns
    -------
    str
        A trailing-newline SRT document, or ``""`` when nothing is speakable.

    Examples
    --------
    >>> cues = [CaptionCue(0.0, 1.5, "Hello"), CaptionCue(1.5, 3.25, "world")]
    >>> format_as_srt(cues).splitlines()[:3]
    ['1', '00:00:00,000 --> 00:00:01,500', 'Hello']
    """
    spans = _spans(cues, ",", escape=False)
    if not spans:
        return ""
    blocks = [
        f"{index}\n{_range(start, end)}\n{body}"
        for index, (start, end, body) in enumerate(spans, start=1)
    ]
    return "\n\n".join(blocks) + "\n"


def format_as_vtt(cues: list[CaptionCue]) -> str:
    """Render timed cues as WebVTT.

    Parameters
    ----------
    cues : list of CaptionCue
        Timed phrases. Blank text is skipped. Clocks use a period before milliseconds.

    Returns
    -------
    str
        A file that always starts with ``WEBVTT``, even when ``cues`` is empty.
    """
    lines = ["WEBVTT", ""]
    for start, end, body in _spans(cues, ".", escape=True):
        lines.append(_range(start, end))
        lines.append(body)
        lines.append("")
    return "\n".join(lines)
