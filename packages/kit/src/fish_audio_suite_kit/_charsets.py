"""Character classes and small text primitives shared by the kit modules.

Every sentence stop, closer, emphasis mark, CJK range, and tag-name list lives here
once, so a filter and its streaming hold cannot disagree about what ends a sentence.
Where one use deliberately reads a different set, that set is named here too, built
from the shared one, with the reason beside it.
"""

from __future__ import annotations

import re

__all__ = [
    "ANGLE_TOKEN_RE",
    "ASCII_STOPS",
    "BREAKS_RE",
    "CJK_RANGES",
    "CLAUSE_MARKS",
    "EMPHASIS_MARKS",
    "IDEOGRAPHIC_STOPS",
    "LATIN_CLOSER_CHARS",
    "PUNCT_BEFORE_CLOSER",
    "SENTENCE_CLOSERS",
    "SENTENCE_CLOSER_CHARS",
    "SENTENCE_STOPS",
    "SPACED_STOPS",
    "THOUGHT_CLOSE_RE",
    "THOUGHT_OPEN_RE",
    "THOUGHT_WORDS",
    "TRAILING_EMPHASIS_MARKS",
    "TURN_CLOSER_CHARS",
    "TURN_TERMINALS",
    "cjk_latin_counts",
    "cut_at_control",
    "is_cjk",
    "is_header_safe",
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
# Commas, semicolons and colons, ASCII and fullwidth: a pause, not a sentence end.
CLAUSE_MARKS = ",;:，；："
# Closing quotes and a closing paren.
LATIN_CLOSER_CHARS = "\"'”’)"
_CJK_QUOTE_CLOSERS = "」』"
_ANGLE_CLOSERS = "»›〉》"
# Characters that stay on the sentence after its stop: closing quotes, guillemets,
# CJK corner and angle brackets, and a closing paren.
SENTENCE_CLOSER_CHARS = LATIN_CLOSER_CHARS + _CJK_QUOTE_CLOSERS + _ANGLE_CLOSERS
SENTENCE_CLOSERS = frozenset(SENTENCE_CLOSER_CHARS)
# What the turn judge strips after the last word before reading its punctuation. It also
# strips a closing square or curly bracket and the fullwidth paren, and keeps the
# guillemets and angle brackets, which an ASR transcript does not end on.
TURN_CLOSER_CHARS = LATIN_CLOSER_CHARS + _CJK_QUOTE_CLOSERS + "]}）"
# The stops that finish a turn. A trailing ellipsis is left out: it trails off.
TURN_TERMINALS = ASCII_STOPS.replace("…", "") + IDEOGRAPHIC_STOPS
# Markdown emphasis marks: bold, italic, code and strike.
EMPHASIS_MARKS = "*_`~"
# The marks a streamed chunk holds when it ends on them, in case they open emphasis. A
# backtick is left out: the fence holds own it.
TRAILING_EMPHASIS_MARKS = EMPHASIS_MARKS.replace("`", "")
# The punctuation a closing emphasis run may follow right after a word ("**done,**"):
# stops and clause marks, without the ellipsis, the ideographic full stop and question
# mark, and the fullwidth semicolon.
PUNCT_BEFORE_CLOSER = frozenset(ASCII_STOPS + IDEOGRAPHIC_STOPS + CLAUSE_MARKS) - frozenset(
    "…。？；"
)

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
THOUGHT_WORDS = (
    "scratchpad",
    "reasoning",
    "planning",
    "analysis",
    "thoughts",
    "thinking",
    "internal",
    "thought",
    "policy",
    "think",
    "plan",
)
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


def cut_at_control(text: str) -> str:
    r"""Cut text at its first control character.

    Parameters
    ----------
    text : str
        A value bound for one line: a header, a URL, a log field.

    Returns
    -------
    str
        ``text`` up to the first character below U+0020 (a newline, a carriage return,
        a tab or any other C0 control), or all of it when it has none. Nothing is
        stripped.

    Notes
    -----
    A newline in a header value starts another header, and one in a base URL makes
    ``httpx`` reject it, so what follows it is never part of the value.

    Examples
    --------
    >>> cut_at_control("https://api.fish.audio\nX-Evil: 1")
    'https://api.fish.audio'
    """
    cut = next((index for index, ch in enumerate(text) if ord(ch) < 32), None)
    return text if cut is None else text[:cut]


def is_header_safe(text: str, *, token: bool = False) -> bool:
    r"""Return whether text can be sent as an HTTP header value unchanged.

    Parameters
    ----------
    text : str
        The value.
    token : bool, optional
        Also require one non-empty word: no space anywhere. Default False, which
        accepts spaces and the empty string.

    Returns
    -------
    bool
        True when every character is printable ASCII (U+0020 to U+007E, without the
        space when ``token``). A control character would split the header and a
        non-ASCII one cannot be encoded into it.

    Examples
    --------
    >>> is_header_safe("Bearer abc"), is_header_safe("caf\u00e9")
    (True, False)
    >>> is_header_safe("s2-pro", token=True), is_header_safe("s2 pro", token=True)
    (True, False)
    """
    low = 33 if token else 32
    if token and not text:
        return False
    return all(low <= ord(ch) < 127 for ch in text)
