"""The TTS junk gate: text with too little speech to send to Fish, or to answer."""

from __future__ import annotations

import re
from collections.abc import Set
from typing import Final

from fish_audio_suite_kit._charsets import cjk_latin_counts
from fish_audio_suite_kit.cuts import next_tts_cut
from fish_audio_suite_kit.dialogue import QUOTE_RE

__all__ = [
    "DEFAULT_MIN_LETTERS",
    "DEFAULT_SHORT_WORDS",
    "is_tts_junk",
    "too_thin",
]


# Unquoted stage direction: a pronoun or article plus a body verb. This is a
# roleplay heuristic and is off unless a caller asks for it (see is_tts_junk).
_NARRATION_RE = re.compile(r"^(She|He|They|Her|His|The|A|An)\b.*", re.IGNORECASE)


_NARRATION_VERB_RE = re.compile(
    r"\b(shifts|leans|smiles|laughs|settles|tilts|watches|murmurs|"
    r"reaches|moves|looks|dropping|rustle|stretches)\b",
    re.IGNORECASE,
)


# Default floor for "enough speech" and the words kept below it. A one-letter
# reply is noise. "no", "ok", and "hi" are answers.
DEFAULT_MIN_LETTERS: Final = 2


DEFAULT_SHORT_WORDS: Final = frozenset({"no", "ok", "hi", "yo", "go", "yes", "yep", "nope"})


def _short_word(text: str, short_words: Set[str]) -> bool:
    folded = "".join(ch for ch in text.lower() if ch.isalnum())
    return folded in short_words


def too_thin(
    text: str,
    *,
    min_letters: int = DEFAULT_MIN_LETTERS,
    short_words: Set[str] | None = None,
) -> bool:
    """Return whether text has too little speech to send or answer.

    Parameters
    ----------
    text : str
        Scrubbed text.
    min_letters : int, optional
        Fewest letters that count as speech. Default 2.
    short_words : AbstractSet of str or None, optional
        Lowercase words kept even when under ``min_letters`` (or when a caller
        raises the floor). Default ``DEFAULT_SHORT_WORDS``.

    Returns
    -------
    bool
        True for empty text, a lone CJK character, or fewer than
        ``min_letters`` letters. Two CJK characters, or three digits, are
        enough: ``100`` is an answer.
    """
    stripped = text.strip()
    if not stripped:
        return True
    cjk, letters = cjk_latin_counts(stripped)
    if cjk >= 2:
        return False
    if sum(ch.isdigit() for ch in stripped) >= 3:
        return False
    if _short_word(stripped, DEFAULT_SHORT_WORDS if short_words is None else short_words):
        return False
    return letters < min_letters


def _stage_pieces(text: str) -> list[str]:
    pieces: list[str] = []
    rest = text
    while rest:
        cut = next_tts_cut(rest, partial_chars=len(rest) + 1)
        newline = rest.find("\n")
        if newline >= 0 and (cut < 0 or newline + 1 < cut):
            cut = newline + 1
        if cut <= 0:
            pieces.append(rest)
            break
        pieces.append(rest[:cut])
        rest = rest[cut:]
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


# A cue with an empty body counts as a cue too, so "[ ]" alone is no speech.
_EMPTY_CUE_RE = re.compile(r"\[[^\]\n]{0,80}\]")


_SPACE_RE = re.compile(r"\s+")


def is_tts_junk(
    text: str,
    *,
    drop_narration: bool = False,
    min_letters: int = DEFAULT_MIN_LETTERS,
    short_words: Set[str] | None = None,
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
    min_letters : int, optional
        Fewest letters that count as speech. Default 2.
    short_words : AbstractSet of str or None, optional
        Words kept below the floor. Default ``DEFAULT_SHORT_WORDS``.

    Returns
    -------
    bool
        True for cue-only text, an unmatched ``[``, or text with too little
        speech (see ``too_thin``). ``Été`` and a Russian sentence are speech,
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
    if too_thin(spoken, min_letters=min_letters, short_words=short_words):
        return True
    return drop_narration and _looks_like_narration(spoken)
