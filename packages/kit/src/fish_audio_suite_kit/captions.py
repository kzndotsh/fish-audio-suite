"""OpenAI-shaped caption files from timed ASR cues. No network."""

from __future__ import annotations

import math
import re
from typing import NamedTuple

from fish_audio_suite_kit.defaults import MS_PER_S


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


def _clock(seconds: float, decimal: str) -> str:
    if not math.isfinite(seconds):
        seconds = 0.0
    try:
        ms = round(max(0.0, seconds) * MS_PER_S)
    except OverflowError:
        ms = 0
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
    start_s = cue.start if math.isfinite(cue.start) else 0.0
    end_s = cue.end if math.isfinite(cue.end) else start_s
    end_s = max(end_s, start_s)
    start = _clock(start_s, decimal)
    end = _clock(end_s, decimal)
    # WebVTT drops a cue whose end is not later than its start. An inverted
    # segment, a zero-length one, and a clock that overflowed all rendered
    # as the same timestamp twice, so the words never appeared.
    if end <= start:
        end = _clock(start_s + 0.001, decimal)
    return start, end, body


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
