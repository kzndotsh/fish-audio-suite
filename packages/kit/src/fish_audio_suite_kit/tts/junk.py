"""The TTS junk gate: text with too little speech to send to Fish, or to answer.

The gate drops only noise: a cue with no words, an unmatched ``[``, or text under the
speech floor (fewer than two letters, unless it is a short answer such as ``no``,
``ok`` or ``hi``). A single letter is judged junk.
"""

from __future__ import annotations

import re
from typing import Final

from fish_audio_suite_kit.text._speech_floor import too_little_speech
from fish_audio_suite_kit.tts.cues import cue_pattern
from fish_audio_suite_kit.tts.cuts import sentence_cuts
from fish_audio_suite_kit.tts.dialogue import QUOTE_RE

__all__ = [
    "is_tts_junk",
]


# Unquoted stage direction: a pronoun or article plus a body verb. This is a
# roleplay heuristic and is off unless a caller asks for it (see is_tts_junk).
_NARRATION_RE = re.compile(r"^(She|He|They|Her|His|The|A|An)\b.*", re.IGNORECASE)


_NARRATION_VERB_RE = re.compile(
    r"\b(shifts|leans|smiles|laughs|settles|tilts|watches|murmurs|"
    r"reaches|moves|looks|dropping|rustle|stretches)\b",
    re.IGNORECASE,
)


def _stage_pieces(text: str) -> list[str]:
    # Each line is split on its own, so a direction never runs across a line break.
    pieces: list[str] = []
    for line in text.split("\n"):
        start = 0
        for end in sentence_cuts(line):
            pieces.append(line[start:end])
            start = end
        pieces.append(line[start:])
    return [piece.strip() for piece in pieces if piece.strip()]


def _one_stage_direction(piece: str) -> bool:
    return bool(_NARRATION_RE.match(piece)) and bool(_NARRATION_VERB_RE.search(piece))


def _looks_like_narration(text: str) -> bool:
    s = text.strip()
    if not s or QUOTE_RE.search(s):
        return False
    # Every sentence must be a stage direction. One spoken sentence after a
    # direction ("She smiles. It's open today.") keeps the whole line.
    pieces = _stage_pieces(s)
    return bool(pieces) and all(_one_stage_direction(piece) for piece in pieces)


# A cue with an empty body counts as a cue too, so "[ ]" alone is no speech. This gate
# reads cues up to 80 characters, where the rules in ``cues`` read 200, so a longer
# bracket is judged as words.
_JUNK_CUE_CHARS: Final = 80
_EMPTY_CUE_RE: Final = re.compile(cue_pattern(_JUNK_CUE_CHARS, min_chars=0))


_SPACE_RE = re.compile(r"\s+")


def is_tts_junk(
    text: str,
    *,
    drop_narration: bool = False,
) -> bool:
    """Return whether scrubbed text should not be sent to Fish TTS.

    Parameters
    ----------
    text : str
        Text after ``scrub_tts``.
    drop_narration : bool, optional
        Also treat an unquoted stage direction ("She smiles softly.") as junk.
        Default False. This is a roleplay heuristic that matches ordinary
        English ("The weather looks great"), so enable it only next to
        ``extract_quoted_speech``.

    Returns
    -------
    bool
        True for cue-only text, an unmatched ``[``, or text with too little
        speech: fewer than two letters, unless it is a short answer such as ``no``. ``Été`` and a Russian sentence are speech,
        and three digits are a number. A lead cue is not speech: ``[clear] hi``
        is judged on ``hi``.
    """
    s = (text or "").strip()
    if s.count("[") > s.count("]"):
        return True
    # A cue name is not speech, and a leading cue must not hide a narration check.
    spoken = _SPACE_RE.sub(" ", _EMPTY_CUE_RE.sub(" ", s)).strip()
    bare = _SPACE_RE.sub(" ", QUOTE_RE.sub(" ", spoken)).strip()
    if not bare:
        return True
    if too_little_speech(spoken):
        return True
    return drop_narration and _looks_like_narration(spoken)
