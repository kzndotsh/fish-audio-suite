"""Sentence cuts for one Fish TTS send."""

from __future__ import annotations

import re

from fish_audio_suite_kit.defaults import SuiteDefaults

# Flush a long clause when no sentence end has arrived. Default is 40.
_PARTIAL_CHARS = SuiteDefaults().tts_partial_chars
# Do not cut a word shorter than this just to hit the window.
_MIN_WORD_CUT = 12

# Sentence end, including Arabic, Devanagari, and Urdu stops. Trailing quotes
# stay on the sentence so the cut does not split "end." from the closer.
# A CJK stop does not need a space, but a space that is already there belongs
# to this sentence. Leaving it on the next sentence made strip() eat it, so
# "你好。 Excited" was spoken as "你好。[excited]".
# A stop can be glued to the next word. "Done.Excited" is a new sentence.
# A space that is already there still belongs to this sentence. Arabic
# stops keep the space requirement. A decimal or "Dr." is not a stop.
# A corner quote or a guillemet after the stop is still that sentence.
# Leaving it out spoke the next mood as a word.
_SENT_CLOSER_CLASS = "\"'”’)」』»›〉》"
_SENT_END = re.compile(
    rf"(?:[.!?…]+[{_SENT_CLOSER_CLASS}]*\s?"
    rf"|[؟।۔]+[{_SENT_CLOSER_CLASS}]*\s"
    rf"|[。！？]+[{_SENT_CLOSER_CLASS}]*\s?)"
)
# The token immediately before a period: abbreviation, initial, or "1.".
_TRAIL_WORD = re.compile(r"(\d+|[A-Za-z]+)\s*$")

# Periods that are not sentence ends. A one-letter initial is handled separately.
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
        "co",
        "no",
        "vol",
        "fig",
        "approx",
        "est",
    }
)


def _skip_abbreviation(buf: str, end_start: int) -> bool:
    """Return whether the period belongs to ``Dr.``, a one-letter initial, or ``1.``.

    A sentence cut must not flush ``Dr.`` as its own TTS request.
    """
    before = buf[:end_start]
    m = _TRAIL_WORD.search(before)
    if not m:
        return False
    w = m.group(1)
    if w.isdigit():
        # "3.14" is one number. A leading "1." is a list marker, so it is
        # not its own sentence. "page 12." and "10:30." do end the sentence,
        # or the next mood is spoken as a word.
        index = end_start
        while index < len(buf) and buf[index] in ".!?…\"'”’)":
            index += 1
        if index < len(buf) and buf[index].isdigit():
            return True
        prefix = before[: m.start()]
        newline = prefix.rfind("\n")
        if newline >= 0:
            prefix = prefix[newline + 1 :]
        return len(w) <= 2 and prefix.strip() == ""
    if len(w) == 1 and w.isalpha():
        return True
    return w.lower() in _ABBREVIATIONS


def _stop_run_start(text: str, stops: frozenset[str]) -> int:
    index = len(text) - 1
    while index > 0 and text[index - 1] in stops:
        index -= 1
    return index


# Closers the sentence cut keeps after the stop. "Done." still ends;
# looking only at the quote made the next mood stay spoken.
_SENT_CLOSERS = frozenset(_SENT_CLOSER_CLASS)
_SENT_STOPS = frozenset(".!?…。！？؟।۔")


def _strip_sentence_closers(text: str) -> tuple[str, bool]:
    saw_space = text[-1:].isspace()
    body = text
    while True:
        trimmed = body.rstrip()
        if trimmed != body:
            saw_space = True
        if not trimmed or trimmed[-1] not in _SENT_CLOSERS:
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
        True when the last stop ends a sentence. False for ``Dr.``, ``No.``,
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
    if last in "。！？":
        return True
    if last in "؟।۔":
        if not saw_space:
            return False
        return not _skip_abbreviation(stripped, _stop_run_start(stripped, frozenset("؟।۔")))
    if last not in ".!?…":
        return False
    return not _skip_abbreviation(stripped, _stop_run_start(stripped, frozenset(".!?…")))


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
