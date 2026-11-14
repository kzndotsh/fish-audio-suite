"""The floor under which text is too little speech to send to Fish, or to answer.

The floor drops only noise. Fewer than two letters is noise, and a short-word
allowlist (``no``, ``ok``, ``hi``) keeps real one-word answers.
"""

from __future__ import annotations

from typing import Final

from fish_audio_suite_kit.text._charsets import cjk_latin_counts

__all__ = [
    "DEFAULT_SHORT_WORDS",
    "too_little_speech",
]

# A one-letter reply is noise. "no", "ok" and "hi" are answers.
_MIN_LETTERS: Final = 2
# Two CJK characters are a word; three digits are a number ("100" is an answer).
_MIN_CJK: Final = 2
_MIN_DIGITS: Final = 3
DEFAULT_SHORT_WORDS: Final = frozenset({"no", "ok", "hi", "yo", "go", "yes", "yep", "nope"})


def too_little_speech(text: str, *, any_cjk: bool = False) -> bool:
    """Return whether text has too little speech to send or answer.

    Parameters
    ----------
    text : str
        Scrubbed text.
    any_cjk : bool, optional
        How a single CJK character counts. False (the TTS and dialogue gates): it is
        not enough on its own, two are, and a lone one is judged with the letters
        around it. True (the ASR gate): any CJK character with a letter beside it is
        speech, and a lone one with no letter is noise whatever digits follow it.

    Returns
    -------
    bool
        True for empty text, fewer than two letters, or too little CJK. Two CJK
        characters, three digits, or a word from ``DEFAULT_SHORT_WORDS`` are enough.
        Accented and Cyrillic letters count as letters.

    Examples
    --------
    >>> too_little_speech("a"), too_little_speech("ok"), too_little_speech("100")
    (True, False, False)
    >>> too_little_speech("a好"), too_little_speech("a好", any_cjk=True)
    (True, False)
    """
    stripped = text.strip()
    if not stripped:
        return True
    cjk, letters = cjk_latin_counts(stripped)
    if any_cjk and cjk:
        return cjk == 1 and letters == 0
    if cjk >= _MIN_CJK:
        return False
    if sum(ch.isdigit() for ch in stripped) >= _MIN_DIGITS:
        return False
    if "".join(ch for ch in stripped.lower() if ch.isalnum()) in DEFAULT_SHORT_WORDS:
        return False
    return letters < _MIN_LETTERS
