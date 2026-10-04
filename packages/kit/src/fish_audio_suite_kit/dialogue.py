"""Pull the quoted dialogue out of roleplay text."""

from __future__ import annotations

import re

from fish_audio_suite_kit._charsets import cjk_latin_counts

__all__ = [
    "QUOTE_RE",
    "extract_quoted_speech",
]

# A quote whose only content is a [cue] in this window is not speech.
_CUE_LOOKAHEAD = 24
# Optional [cue] glued to a quote, so dialogue extraction does not drop the cue.
_LEAD_CUE = r"\[[^\]\n]{1,80}\]\s*"
_LEAD_CUE_RE = re.compile(rf"({_LEAD_CUE})")

# Straight, curly, guillemet, and corner quotes. French uses «…» and German
# »…«, so either order of angle quotes is a quote. A digit before the mark is
# inches (5"), not dialogue.
_ANGLE_QUOTES = "«»‹›〈〉《》"
_QUOTE_OPEN = '"“「『' + _ANGLE_QUOTES
_QUOTE_CLOSE = '"”」』' + _ANGLE_QUOTES
QUOTE_RE = re.compile(f"[{_QUOTE_OPEN}{_QUOTE_CLOSE}]")
# Each opener and the marks that may close it. A straight or curly quote also
# closes with the other (models mix them), and the angle quotes close in either
# direction. A quote of another style inside is part of the speech, so nested
# dialogue does not split into fragments.
_QUOTE_PAIRS = (
    ('"', '"”'),
    ("“", '”"'),
    ("「", "」"),
    ("『", "』"),
    ("«", "»«"),
    ("»", "«»"),
    ("‹", "›‹"),
    ("›", "‹›"),
    ("〈", "〉"),
    ("《", "》"),
)
# A closed quote, with an optional cue in front. It may wrap across lines. The
# body may be empty so that narration around "" is not read as speech.
_DIALOGUE_RE = re.compile(
    rf"(?:{_LEAD_CUE})?(?<!\d)(?:"
    + "|".join(
        f"{re.escape(op)}([^{re.escape(cl)}]*){'[' + re.escape(cl) + ']'}"
        for op, cl in _QUOTE_PAIRS
    )
    + ")"
)
# A quote opened and not yet closed. Streaming replies arrive this way. The
# search starts after the last closed quote.
_OPEN_DIALOGUE_RE = re.compile(
    rf"(?:{_LEAD_CUE})?"
    rf"(?<!\d)[{_QUOTE_OPEN}](.+)$",
    re.DOTALL,
)
# Closers stripped from an unclosed quote before the speakable check.
_CLOSE_QUOTES = '"”」』' + _ANGLE_QUOTES


def _enough_speech(text: str) -> bool:
    # A quoted one-letter line and "42" are too thin. Two CJK characters are a
    # sentence, though a quoted Chinese line has no Latin letters at all.
    cjk, letters = cjk_latin_counts(text)
    if letters >= 2 or cjk >= 2:
        return True
    return sum(ch.isdigit() for ch in text) >= 3


def _speakable_quote(inner: str) -> bool:
    if not inner or (inner.startswith("[") and inner.endswith("]")):
        return False
    return _enough_speech(inner)


def _quote_body(match: re.Match[str]) -> str:
    # One group per quote style, and only the one that matched is set.
    for group in match.groups():
        if group is not None:
            return group.strip()
    return ""


def _closed_quotes(text: str) -> tuple[list[str], int]:
    """Return the speakable closed quotes and where the last closed pair ends."""
    parts: list[str] = []
    end = 0
    for m in _DIALOGUE_RE.finditer(text):
        end = m.end()
        if not _speakable_quote(_quote_body(m)):
            continue
        chunk = m.group(0).strip()
        if chunk:
            parts.append(chunk)
    return parts, end


def _trailing_open_quote(text: str, start: int, *, anchored: bool = False) -> str:
    """Return the quote still open after ``start``, with its lead cue, or ``""``.

    ``anchored`` requires the quote (after any lead cue) to begin at ``start``.
    """
    m = _OPEN_DIALOGUE_RE.match(text, start) if anchored else _OPEN_DIALOGUE_RE.search(text, start)
    if m is None:
        return ""
    inner = (m.group(1) or "").strip().rstrip(_CLOSE_QUOTES).strip()
    cjk, letters = cjk_latin_counts(inner)
    cue_only = inner.startswith("[") and "]" in inner[:_CUE_LOOKAHEAD] and letters < 3 and cjk < 2
    if not inner or not _enough_speech(inner) or cue_only:
        return ""
    cm = _LEAD_CUE_RE.match(m.group(0))
    cue = cm.group(1) if cm else ""
    return f'{cue}"{inner}"'


def extract_quoted_speech(text: str) -> str:
    """Keep quoted dialogue (optional leading ``[cue]``); drop stage notes.

    Parameters
    ----------
    text : str
        A reply that mixes quoted speech with narration.

    Returns
    -------
    str
        The quoted lines joined by spaces. An unclosed trailing quote counts,
        also after narration or after a quote that did close. Text with no
        quotes is returned unchanged. A quote that is not speech (a lone cue,
        one letter, an empty pair) returns empty rather than the narration.
    """
    if not text:
        return ""
    parts, closed_end = _closed_quotes(text)
    opened = _trailing_open_quote(text, closed_end)
    if not parts and not opened:
        # A reply that opens with a quote holding only a cue name is passed
        # through as a quote, as it always was.
        opened = _trailing_open_quote(text.strip(), 0, anchored=True)
    if opened:
        parts.append(opened)
    if parts:
        return " ".join(parts)
    # A closed pair that is not speech must not fall through to the stage
    # direction. A lone inch mark is not a pair, so that sentence stays.
    if _DIALOGUE_RE.search(text):
        return ""
    return text
