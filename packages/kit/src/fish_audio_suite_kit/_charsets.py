"""Character classes and small text primitives shared by the kit modules.

Every sentence stop, closer, CJK range, and tag-name list lives here once, so a
filter and its streaming hold cannot disagree about what ends a sentence.
"""

from __future__ import annotations

import re

__all__ = [
    "ANGLE_TOKEN_RE",
    "ASCII_STOPS",
    "BREAKS_RE",
    "CJK_RANGES",
    "IDEOGRAPHIC_STOPS",
    "SENTENCE_CLOSERS",
    "SENTENCE_CLOSER_CHARS",
    "SENTENCE_STOPS",
    "SPACED_STOPS",
    "THOUGHT_CLOSE_RE",
    "THOUGHT_OPEN_RE",
    "THOUGHT_WORDS",
    "cjk_latin_counts",
    "is_cjk",
    "plain_breaks",
    "utf8_text",
]

# Ends a sentence when followed by a space or the end of the buffer.
ASCII_STOPS = ".!?…"
# Ideographic stops end a sentence with no space needed.
IDEOGRAPHIC_STOPS = "。！？"
# Arabic, Devanagari, and Urdu stops count only after a space.
SPACED_STOPS = "؟।۔"
SENTENCE_STOPS = frozenset(ASCII_STOPS + IDEOGRAPHIC_STOPS + SPACED_STOPS)
# Characters that stay on the sentence after its stop: quotes and a closing paren.
SENTENCE_CLOSER_CHARS = "\"'”’)」』»›〉》"
SENTENCE_CLOSERS = frozenset(SENTENCE_CLOSER_CHARS)

# Hiragana/katakana, CJK ext A, unified ideographs, compatibility, Hangul.
CJK_RANGES = (
    range(0x3040, 0x3100),
    range(0x3400, 0x4DC0),
    range(0x4E00, 0xA000),
    range(0xF900, 0xFB00),
    range(0xAC00, 0xD7B0),
)

# Any <|...|> token. TTS keeps these (phonemes). ASR replaces them with a space.
ANGLE_TOKEN_RE = re.compile(r"<\|[^|>]{0,200}\|>")
# Three or more blank lines become one paragraph break.
BREAKS_RE = re.compile(r"\n{3,}")

# Chain-of-thought tag names, longest first. Every thought-tag regex is built
# from this tuple. Stream holds also prefix-match it for a split "<thi".
THOUGHT_WORDS = ("reasoning", "thoughts", "thinking", "thought", "think")
_THOUGHT_NAMES = "|".join(THOUGHT_WORDS)
THOUGHT_OPEN_RE = re.compile(rf"<\s*(?:{_THOUGHT_NAMES})\s*>", re.IGNORECASE)
THOUGHT_CLOSE_RE = re.compile(rf"<\s*/\s*(?:{_THOUGHT_NAMES})\s*>", re.IGNORECASE)


def is_cjk(ch: str) -> bool:
    """Return whether a character is Han, kana, or Hangul."""
    code = ord(ch)
    return any(code in span for span in CJK_RANGES)


def cjk_latin_counts(text: str) -> tuple[int, int]:
    """Count CJK characters and other letters.

    Parameters
    ----------
    text : str
        Any string.

    Returns
    -------
    tuple of int and int
        ``(cjk, letters)``. Accented and Cyrillic letters count as letters.
    """
    cjk = letters = 0
    for ch in text:
        if is_cjk(ch):
            cjk += 1
        elif ch.isalpha():
            letters += 1
    return cjk, letters


def plain_breaks(text: str) -> str:
    """Normalize line breaks and drop invisible characters.

    Parameters
    ----------
    text : str
        Raw model or ASR text.

    Returns
    -------
    str
        CRLF, CR, and the Unicode line and paragraph separators become a
        newline. A zero-width space, word joiner, BOM, and soft hyphen are
        removed because they are not sounds.
    """
    return (
        text.replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\u2028", "\n")
        .replace("\u2029", "\n")
        .replace("\u200b", "")
        .replace("\ufeff", "")
        .replace("\u2060", "")
        .replace("\u00ad", "")
    )


def utf8_text(text: str) -> str:
    """Replace characters that cannot be encoded as UTF-8.

    Parameters
    ----------
    text : str
        Any string, including one built from a surrogate.

    Returns
    -------
    str
        A UTF-8 round trip with ``errors="replace"``.
    """
    return text.encode("utf-8", "replace").decode("utf-8")
