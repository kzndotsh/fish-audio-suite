"""OpenAI-shaped caption files from timed ASR cues. No network."""

from __future__ import annotations

from typing import NamedTuple


class CaptionCue(NamedTuple):
    start: float
    end: float
    text: str


def _clock(seconds: float, decimal: str) -> str:
    ms = round(max(0.0, seconds) * 1000.0)
    hours, ms = divmod(ms, 3_600_000)
    minutes, ms = divmod(ms, 60_000)
    secs, ms = divmod(ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{decimal}{ms:03d}"


def format_as_srt(cues: list[CaptionCue]) -> str:
    """SubRip. Empty input is an empty string."""
    if not cues:
        return ""
    blocks: list[str] = []
    numbered = 0
    for cue in cues:
        body = cue.text.strip()
        if not body:
            continue
        numbered += 1
        start = _clock(cue.start, ",")
        end = _clock(max(cue.end, cue.start), ",")
        blocks.append(f"{numbered}\n{start} --> {end}\n{body}")
    if not blocks:
        return ""
    return "\n\n".join(blocks) + "\n"


def format_as_vtt(cues: list[CaptionCue]) -> str:
    """WebVTT. Empty input is a header-only file."""
    lines = ["WEBVTT", ""]
    if not cues:
        return "WEBVTT\n"
    for cue in cues:
        body = cue.text.strip()
        if not body:
            continue
        start = _clock(cue.start, ".")
        end = _clock(max(cue.end, cue.start), ".")
        lines.append(f"{start} --> {end}")
        lines.append(body)
        lines.append("")
    return "\n".join(lines)
