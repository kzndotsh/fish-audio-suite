"""Sentence cuts for one Fish TTS send."""

from __future__ import annotations

import re

from fish_audio_suite_kit._charsets import (
    ASCII_STOPS,
    IDEOGRAPHIC_STOPS,
    SENTENCE_CLOSER_CHARS,
    SENTENCE_CLOSERS,
    SPACED_STOPS,
)
from fish_audio_suite_kit.defaults import SuiteDefaults

__all__ = [
    "ends_sentence",
    "next_tts_cut",
    "split_tts_piece",
]

# Flush a long clause when no sentence end has arrived. Default is 40.
_PARTIAL_CHARS = SuiteDefaults().tts_partial_chars
# A partial cut at the last space is used only when that space is at least this
# far in. Earlier than that, a hard cut at the window beats a tiny first piece.
_MIN_WORD_CUT = 12

# Sentence end. Trailing closers and one following space stay on the sentence.
# Ideographic stops need no space, and "Done.Excited" glued is still a new
# sentence. Arabic, Devanagari, and Urdu stops require the space. A decimal or
# an abbreviation is not a stop.
_SENT_END = re.compile(
    rf"(?:[{re.escape(ASCII_STOPS)}]+[{re.escape(SENTENCE_CLOSER_CHARS)}]*\s?"
    rf"|[{SPACED_STOPS}]+[{re.escape(SENTENCE_CLOSER_CHARS)}]*\s"
    rf"|[{IDEOGRAPHIC_STOPS}]+[{re.escape(SENTENCE_CLOSER_CHARS)}]*\s?)"
)


def _trailing_word(text: str) -> tuple[int, str] | None:
    """Find the token just before a period: an abbreviation, an initial or a number.

    Parameters
    ----------
    text : str
        Text that ends where the period starts.

    Returns
    -------
    tuple of int and str, or None
        The start index and the last run of digits, or of ASCII letters, once
        trailing whitespace is ignored. None when the text does not end in one.
    """
    end = len(text)
    while end and text[end - 1].isspace():
        end -= 1
    if not end:
        return None
    last = text[end - 1]
    if last.isdecimal():
        start = end
        while start and text[start - 1].isdecimal():
            start -= 1
    elif "A" <= last <= "Z" or "a" <= last <= "z":
        start = end
        while start and ("A" <= text[start - 1] <= "Z" or "a" <= text[start - 1] <= "z"):
            start -= 1
    else:
        return None
    return start, text[start:end]


# Titles and Latin-script abbreviations whose period is not a sentence end. A
# one-letter initial is handled separately. Words that are also common sentence
# enders ("no", "co", "est") are not listed here: they are abbreviations only in
# the contexts below.
_ABBREVIATIONS = frozenset(
    {
        "dr",
        "mr",
        "mrs",
        "ms",
        "prof",
        "sr",
        "jr",
        "vs",
        "etc",
        "st",
        "ave",
        "inc",
        "ltd",
        "vol",
        "fig",
        "approx",
    }
)


# "No. 5" is a number and "Acme Co. Ltd." is a company, but "No. I will not" is
# a sentence. These words count as abbreviations only when the text after the
# period says so. With nothing after it yet, the period ends the sentence.
_COMPANY_SUFFIXES = ("ltd", "inc", "corp", "llc", "plc", "&")
# How far before a stop the abbreviation check reads.
_LOOKBACK = 64


def _context_abbreviation(word: str, rest: str) -> bool:
    if word == "no":
        return rest[:1].isdigit()
    if word == "co":
        return rest.lower().startswith(_COMPANY_SUFFIXES)
    return False


def _skip_abbreviation(buf: str, end_start: int) -> bool:
    """Return whether the period belongs to ``Dr.``, a one-letter initial, or ``1.``.

    A sentence cut must not flush ``Dr.`` as its own TTS request. ``No.`` before
    a digit and ``Co.`` before a company suffix count too.
    """
    # Only the word before the stop and the start of its line matter, so look back a
    # short way. Copying and searching the whole buffer for every stop is quadratic
    # in a long reply of short sentences.
    window_start = max(0, end_start - _LOOKBACK)
    before = buf[window_start:end_start]
    found = _trailing_word(before)
    if found is None:
        return False
    word_start, w = found
    # A word that fills the window began before it. It is too long to be a title.
    if window_start and word_start == 0 and not w.isdigit():
        return False
    if w.isdigit():
        # "3.14" is one number. A leading "1." is a list marker, so it is
        # not its own sentence. "page 12." and "10:30." do end the sentence,
        # or the next mood is spoken as a word.
        index = end_start
        while index < len(buf) and buf[index] in ".!?…\"'”’)":
            index += 1
        if index < len(buf) and buf[index].isdigit():
            return True
        prefix = before[:word_start]
        newline = prefix.rfind("\n")
        if newline >= 0:
            prefix = prefix[newline + 1 :]
        return len(w) <= 2 and prefix.strip() == ""
    if len(w) == 1 and w.isalpha():
        return True
    if buf[end_start : end_start + 1] == ".":
        rest = buf[end_start + 1 :].lstrip()
        if _context_abbreviation(w.lower(), rest):
            return True
    return w.lower() in _ABBREVIATIONS


def _stop_run_start(text: str, stops: frozenset[str]) -> int:
    index = len(text) - 1
    while index > 0 and text[index - 1] in stops:
        index -= 1
    return index


def _strip_sentence_closers(text: str) -> tuple[str, bool]:
    # Closers stay with the stop, so "Done.\"" still ends the sentence.
    saw_space = text[-1:].isspace()
    body = text
    while True:
        trimmed = body.rstrip()
        if trimmed != body:
            saw_space = True
        if not trimmed or trimmed[-1] not in SENTENCE_CLOSERS:
            return trimmed, saw_space
        body = trimmed[:-1]


def ends_sentence(text: str) -> bool:
    """Return whether text ends on a real sentence boundary.

    Parameters
    ----------
    text : str
        Buffered model text.

    Returns
    -------
    bool
        True when the last stop ends a sentence. False for ``Dr.``,
        a one-letter initial, ``1.``, and a stop that still needs its
        following space. A closing quote or parenthesis after the stop still
        counts.
    """
    # The Arabic, Devanagari, and Urdu stops are in _SENT_END. They count
    # only after the space. "Dr..." is still an abbreviation: skip from the
    # first mark in the run, not the last.
    stripped, saw_space = _strip_sentence_closers(text)
    if not stripped:
        return False
    last = stripped[-1]
    if last in IDEOGRAPHIC_STOPS:
        return True
    if last in SPACED_STOPS:
        if not saw_space:
            return False
        return not _skip_abbreviation(stripped, _stop_run_start(stripped, frozenset(SPACED_STOPS)))
    if last not in ASCII_STOPS:
        return False
    return not _skip_abbreviation(stripped, _stop_run_start(stripped, frozenset(ASCII_STOPS)))


def _sentence_cut(buf: str) -> int:
    search_from = 0
    while True:
        m = _SENT_END.search(buf, search_from)
        if not m:
            return -1
        if _skip_abbreviation(buf, m.start()):
            search_from = m.end()
            continue
        return m.end()


def _partial_cut(buf: str, partial_chars: int) -> int:
    # A zero window returns a zero-length piece and the same buffer. The
    # caller would spin. No window means keep buffering until flush.
    if partial_chars < 1 or len(buf) < partial_chars:
        return -1
    # Search only the window. The last space in the whole buffer would send
    # the entire clause once it passes the threshold. A newline is a word
    # boundary too: a line-broken reply was cut inside "golf". A non-breaking
    # space is the same boundary. Leaving it out cut "door" in half.
    cut = -1
    for index, ch in enumerate(buf[:partial_chars]):
        if ch.isspace():
            cut = index
    if cut >= _MIN_WORD_CUT:
        return cut + 1
    return partial_chars


def _cut_outside_brackets(buf: str, cut: int) -> int:
    # The 40-character cut can land inside [whispering]. Fish then hears
    # "[whi" and "spering]" as two pieces and speaks the markup.
    if cut <= 0:
        return cut
    depth = 0
    start = -1
    for index, ch in enumerate(buf):
        if ch == "[":
            if depth == 0:
                start = index
            depth += 1
        elif ch == "]" and depth:
            depth -= 1
            if depth == 0 and start >= 0 and start < cut <= index:
                if start >= _MIN_WORD_CUT:
                    return start
                return index + 1
            if depth == 0:
                start = -1
        if index >= cut and depth == 0:
            break
    if depth and start >= 0 and start < cut:
        if start >= _MIN_WORD_CUT:
            return start
        return -1
    return cut


def next_tts_cut(buf: str, *, partial_chars: int = _PARTIAL_CHARS) -> int:
    """Return how many leading characters are ready for one Fish TTS send.

    Parameters
    ----------
    buf : str
        Text buffered from the model so far.
    partial_chars : int, optional
        Flush near this length when no sentence end is in the buffer.
        Default is about 40 characters.

    Returns
    -------
    int
        End index of the piece to send, or -1 to keep buffering.

    Notes
    -----
    A sentence end wins, except ``Dr.``, a one-letter initial, and ``1.``.
    An ideographic full stop ends a sentence with no space required. A space
    that is already after it stays on that sentence, so the next cue is not
    glued to the stop. A stop can also be glued to the next word. A finished
    sentence longer than the window still cuts early, or the first audio
    would wait for the whole sentence. Otherwise the cut is the last space
    or newline before ``partial_chars``, or ``partial_chars`` itself when
    there is no usable break. A cut that would land inside ``[whispering]``
    or ``[hello. there]`` moves to the bracket instead.

    Examples
    --------
    >>> text = "Hello there. How are you? Fine."
    >>> next_tts_cut(text)
    13
    >>> text[:13]
    'Hello there. '
    >>> next_tts_cut("Dr. Smith is here. Okay.")
    19
    """
    if not buf:
        return -1
    end = _sentence_cut(buf)
    # The final period of a long reply is a sentence end. Using it as the
    # only cut sends the whole sentence before any audio starts.
    if end == len(buf) and len(buf) > partial_chars:
        end = -1
    if end >= 0:
        # "Note [hello. there]" is one cue. The period is not a sentence end
        # until the bracket closes, or the words before the bracket go first.
        adjusted = _cut_outside_brackets(buf, end)
        if adjusted > 0:
            return adjusted
    partial = _partial_cut(buf, partial_chars)
    if partial < 0:
        return -1
    return _cut_outside_brackets(buf, partial)


def split_tts_piece(buf: str, partial_chars: int, *, flush_rest: bool) -> tuple[str, str] | None:
    """Split one ready TTS piece from the unsent tail.

    Parameters
    ----------
    buf : str
        Buffered model text.
    partial_chars : int
        Passed to ``next_tts_cut``.
    flush_rest : bool
        When True and no cut is ready, return the whole buffer as the piece.
        When False, return None so the stream keeps buffering.

    Returns
    -------
    tuple of str and str or None
        ``(piece, tail)``, or None while the caller should wait.
    """
    cut = next_tts_cut(buf, partial_chars=partial_chars)
    if cut < 0:
        if not flush_rest:
            return None
        return buf, ""
    return buf[:cut], buf[cut:]
