"""Quoted dialogue, optional narration filtering, and the TTS junk gate."""

from __future__ import annotations

import re

from fish_audio_suite_kit._charsets import cjk_latin_counts
from fish_audio_suite_kit.cuts import next_tts_cut

# A quote whose only content is a [cue] in this window is not speech.
_CUE_LOOKAHEAD = 24
# Optional [cue] glued to a quote, so dialogue extraction does not drop the cue.
_LEAD_CUE = r"\[[^\]\n]{1,80}\]\s*"
# A cue with an empty body counts as a cue too, so "[ ]" alone is no speech.
_EMPTY_CUE_RE = re.compile(r"\[[^\]\n]{0,80}\]")
_SPACE_RE = re.compile(r"\s+")
_LEAD_CUE_RE = re.compile(rf"({_LEAD_CUE})")

# Straight, curly, guillemet, and corner quotes. French uses «…» and German
# »…«, so either order of angle quotes is a quote. A digit before the mark is
# inches (5"), not dialogue.
_ANGLE_QUOTES = "«»‹›〈〉《》"
_QUOTE_OPEN = '"“「『' + _ANGLE_QUOTES
_QUOTE_CLOSE = '"”」』' + _ANGLE_QUOTES
_QUOTE_RE = re.compile(f"[{_QUOTE_OPEN}{_QUOTE_CLOSE}]")
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
DEFAULT_MIN_LETTERS = 2
DEFAULT_SHORT_WORDS = frozenset({"no", "ok", "hi", "yo", "go", "yes", "yep", "nope"})


def _short_word(text: str, short_words: frozenset[str]) -> bool:
    folded = "".join(ch for ch in text.lower() if ch.isalnum())
    return folded in short_words


def too_thin(
    text: str,
    *,
    min_letters: int = DEFAULT_MIN_LETTERS,
    short_words: frozenset[str] | None = None,
) -> bool:
    """Return whether text has too little speech to send or answer.

    Parameters
    ----------
    text : str
        Scrubbed text.
    min_letters : int, optional
        Fewest letters that count as speech. Default 2.
    short_words : frozenset of str or None, optional
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
    if not s or _QUOTE_RE.search(s):
        return False
    # Every sentence must be a stage direction. One spoken sentence after a
    # direction ("She smiles. It's open today.") keeps the whole line.
    pieces = _stage_pieces(s)
    return bool(pieces) and all(_one_stage_direction(piece) for piece in pieces)


def is_tts_junk(
    text: str,
    *,
    drop_narration: bool = False,
    min_letters: int = DEFAULT_MIN_LETTERS,
    short_words: frozenset[str] | None = None,
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
    short_words : frozenset of str or None, optional
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
    bare = _SPACE_RE.sub(" ", _QUOTE_RE.sub(" ", spoken)).strip()
    if not bare:
        return True
    if too_thin(spoken, min_letters=min_letters, short_words=short_words):
        return True
    return drop_narration and _looks_like_narration(spoken)
