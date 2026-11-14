"""Estimate how much of a reply the listener actually heard.

History after a barge-in must hold only that prefix. The estimate is a rate
(about 16 characters a second at 1x speed) applied to the bytes that reached
the device, minus what was still buffered, cut back to a finished word.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

from fish_audio_suite_kit import (
    CLEAR_CUE,
    SENTENCE_AND_CLAUSE_MARKS,
    SENTENCE_CLOSERS,
    cue_pattern,
    is_cjk,
    strip_cue_tags,
)
from fish_audio_suite_voice.capture.pcm import SAMPLE_BYTES

__all__ = [
    "Unheard",
    "heard_written",
    "spoken_prefix",
    "unheard",
    "unspoken_text",
]

# Conversational English at speed 1.0. Other languages and voices drift, so
# this is an estimate and the cut is always rounded down to a word.
CHARS_PER_S: Final = 16
_MIN_SPEED: Final = 0.1
# A cue is a short bracketed stage direction. Brackets are never speech here.
_CUE_SPAN_RE: Final = re.compile(cue_pattern(80, flat=True))
# The short [clear] tag stays out of the next turn's history. It was once the default lead
# the app added to every reply, and a stored one taught the model to open with it. Other
# cues stay, so the model keeps seeing the style it used.
_CLEAR: Final = f"[{CLEAR_CUE}]"


def _without_clear(text: str) -> str:
    return _CUE_SPAN_RE.sub(lambda m: " " if m.group().lower() == _CLEAR else m.group(), text)


def _pcm_chars(bytes_played: int, sample_rate: int, speed: float, latency_s: float) -> int:
    # bytes_played counts bytes handed to the device. The last output_latency_s
    # of them are still in its buffer, so the listener has not heard them.
    secs = bytes_played / max(sample_rate * SAMPLE_BYTES, 1) - max(0.0, latency_s)
    return max(0, int(secs * CHARS_PER_S * max(speed, _MIN_SPEED)))


def _word_char(ch: str) -> bool:
    return ch.isalnum() and not is_cjk(ch)


def _extends_tail(tail: str, nxt: str) -> bool:
    # "there" is finished when the next character is "." or a space.
    # "ther" is not finished when the next character is "e". A Cyrillic
    # letter continues the word too. "there.Friend" has finished "there":
    # the period is not part of the next word.
    if not tail or is_cjk(nxt):
        return False
    last = tail[-1]
    if nxt.isalnum():
        if _word_char(last):
            return True
        # "3.14" is one number. "there.Friend" is two words.
        if last == "." and nxt.isdigit() and len(tail) > 1 and tail[-2].isdigit():
            return True
        return last in "'’-" and len(tail) > 1 and _word_char(tail[-2])
    return nxt in "'’-" and _word_char(last)


def _boundary_finished(prefix: str, nxt: str) -> bool:
    if not prefix or _extends_tail(prefix, nxt):
        return False
    # "3" before "." is the start of "3.14", not a finished word.
    return not (nxt == "." and prefix[-1].isdigit())


def _finished_prefix(text: str, nxt: str) -> str:
    if _boundary_finished(text, nxt):
        return text
    for index in range(len(text) - 1, 0, -1):
        if _boundary_finished(text[:index], text[index]):
            return text[:index]
    return ""


def _word_prefix(text: str, n: int) -> str:
    if n >= len(text):
        return text.strip()
    cut = text[:n]
    # A word dropped here is spoken again next turn, and a half word kept is
    # recorded as heard. Any whitespace is a boundary, so a newline or the
    # non-breaking space in "Hello\u00a0there" still separates two words.
    last = -1
    for index, ch in enumerate(cut):
        if ch.isspace():
            last = index
    if last < 0:
        # A cut with no space is the middle of the first English word. History
        # should not record that fragment. CJK has no spaces, including after
        # an English word ("Hello你好"), so there the cut is kept.
        stripped = cut.strip()
        if any(is_cjk(ch) for ch in stripped):
            if is_cjk(stripped[-1]) or _boundary_finished(stripped, text[n]):
                return stripped
            # The cut is inside a Latin word that follows CJK text. Keep the
            # CJK and drop the half word, or history records "你好H".
            end = len(stripped)
            while end > 0 and not is_cjk(stripped[end - 1]):
                end -= 1
            return stripped[:end].strip()
        # "Hello there" cut on the last letter of "Hello" has no space yet.
        # The next character is the space, so that word was played.
        if _boundary_finished(stripped, text[n]):
            return stripped
        # "Hello.Friend" has no space yet. The cut is inside "Friend", but
        # "Hello." was played.
        return _finished_prefix(stripped, text[n])
    head = cut[:last].strip()
    tail = cut[last + 1 :]
    if not tail:
        return head
    # "Hello there" has finished "there" when the next character is a space
    # or a period. "Hello 你" has finished "你" even when "好" is still coming.
    if _boundary_finished(tail, text[n]):
        return f"{head} {tail}".strip()
    # "there.Frien" still extends into "d", but "there." was already played.
    for index in range(len(tail) - 1, 0, -1):
        if _boundary_finished(tail[:index], tail[index]):
            return f"{head} {tail[:index]}".strip()
    extra = []
    for ch in tail:
        if not is_cjk(ch):
            break
        extra.append(ch)
    if extra:
        return f"{head} {''.join(extra)}".strip()
    return head


def spoken_prefix(
    sent_text: str,
    *,
    bytes_played: int,
    sample_rate: int,
    audio_format: str,
    got_audio: bool,
    cancelled: bool,
    failed: bool = False,
    speed: float = 1.0,
    output_latency_s: float = 0.0,
) -> str:
    """Return the part of ``sent_text`` the listener heard.

    Parameters
    ----------
    sent_text : str
        Text sent to Fish for the turn, cues included.
    bytes_played : int
        Bytes the sink accepted.
    sample_rate : int
        PCM sample rate of ``bytes_played``.
    audio_format : str
        Fish output format. Only ``pcm`` can be cut by length.
    got_audio : bool
        True after the first Fish audio chunk.
    cancelled : bool
        True when barge-in or Ctrl+C stopped the turn.
    failed : bool, optional
        True when the socket dropped or Fish returned an error.
    speed : float, optional
        Prosody speed. Faster speech covers more text per second.
    output_latency_s : float, optional
        Device buffer in seconds. That much of ``bytes_played`` is not heard yet.

    Returns
    -------
    str
        The full text without ``[clear]`` when the turn finished. After a
        cancel or failure, a word-aligned prefix with every cue removed. Empty
        when nothing was played, or when the format cannot be cut by length,
        so history never records a reply the listener did not hear.
    """
    if not sent_text.strip() or not got_audio or bytes_played <= 0:
        # Fish can deliver a chunk the sink then drops, such as a dead mpv
        # pipe. History should not record a reply the speaker never played.
        return ""
    if not (cancelled or failed):
        return " ".join(_without_clear(sent_text).split())
    # Encoded audio has no fixed bytes-per-second, so a partial turn cannot be
    # cut. Recording the whole reply would tell the next turn it was heard.
    if audio_format != "pcm":
        return ""
    chars = _pcm_chars(bytes_played, sample_rate, speed, output_latency_s)
    if chars <= 0:
        return ""
    return _word_prefix(strip_cue_tags(sent_text), chars)


# Punctuation and closers that stay with the word they follow: kit's stops, clause marks and
# sentence closers, plus closing brackets (a cue or an aside ends on one), the two-dot leader
# and the ideographic comma, which kit's sentence sets leave out.
_CLOSERS: Final = SENTENCE_AND_CLAUSE_MARKS | SENTENCE_CLOSERS | frozenset("]}）】〕〗‥、")
# Fewer letters than this are not worth speaking again: "right?" after a barge-in.
_MIN_RESUME_CHARS: Final = 8


def _folded(text: str) -> tuple[str, list[int]]:
    """Return ``text`` as case-folded letters and digits, with where each came from."""
    cues = [(m.start(), m.end()) for m in _CUE_SPAN_RE.finditer(text)]
    keys: list[str] = []
    origin: list[int] = []
    cue = 0
    for index, ch in enumerate(text):
        while cue < len(cues) and index >= cues[cue][1]:
            cue += 1
        if cue < len(cues) and cues[cue][0] <= index < cues[cue][1]:
            continue
        for folded in ch.casefold():
            if folded.isalnum():
                keys.append(folded)
                origin.append(index)
    return "".join(keys), origin


def _heard_end(text: str, origin: list[int], count: int) -> int:
    """Return where the ``count`` heard letters of ``text`` end, with the stops after them."""
    cut = origin[count - 1] + 1
    while cut < len(text) and text[cut] in _CLOSERS:
        cut += 1
    return cut


def unspoken_text(sent: str, heard: str, *, lead: bool = True) -> str:
    """Return the part of a reply that was cut off, ready to be spoken again.

    Parameters
    ----------
    sent : str
        The reply as sent to Fish, cues included.
    heard : str
        What the listener probably heard of it, from ``spoken_prefix``: a
        word-aligned prefix with the cues removed.
    lead : bool, optional
        Lead the rest with the last cue before the cut, so the mood carries on. Default
        True. False gives the rest as written.

    Returns
    -------
    str
        The text after ``heard``. Empty when ``heard`` is not a prefix of ``sent``
        (the estimate cannot be trusted), or when too little is left to be
        worth speaking.

    Examples
    --------
    >>> unspoken_text("[calm] Hello there. [warm] How was your day?", "Hello there.")
    '[warm] How was your day?'
    >>> unspoken_text("[calm] Hello there. [warm] How was your day?", "Hello")
    '[calm] there. [warm] How was your day?'
    >>> unspoken_text("[calm] Hello there. How was your day?", "Hello there.", lead=False)
    'How was your day?'
    """
    keys, origin = _folded(sent)
    heard_key, _ = _folded(strip_cue_tags(heard))
    count = len(heard_key)
    if not keys.startswith(heard_key) or len(keys) - count < _MIN_RESUME_CHARS:
        return ""
    cut = 0
    if count:
        if origin[count] == origin[count - 1]:
            return ""  # the cut falls inside one character's folding, such as "ß"
        cut = _heard_end(sent, origin, count)
    rest = sent[cut:].lstrip()
    if lead and not rest.startswith("["):
        before = [m for m in _CUE_SPAN_RE.finditer(sent) if m.end() <= cut]
        if before:
            rest = f"{before[-1].group()} {rest}"
    return rest


def heard_written(written: str, heard: str) -> str:
    """Return what the model wrote of the part the listener heard.

    Parameters
    ----------
    written : str
        The reply as the model wrote it, shaped for Fish but before any cue was added for
        Fish: the mood a sentence carries over, or the default lead.
    heard : str
        What was heard of the text Fish was sent, from ``spoken_prefix``.

    Returns
    -------
    str
        The prefix of ``written`` with the same words as ``heard``, with the cues the model
        wrote in it and none of the cues added for Fish, its spaces collapsed and without
        ``[clear]``. All of ``written`` when every word was heard. When ``heard`` is not a
        prefix of ``written`` (the stream shaped a piece differently), ``heard`` with every
        cue removed, so no added cue is kept.

    Examples
    --------
    >>> heard_written(
    ...     "[calm] A one. B two. C three.", "[calm] A one. [calm] B two. [calm] C three."
    ... )
    '[calm] A one. B two. C three.'
    >>> heard_written("[calm] A one. B two. C three.", "A one. B")
    '[calm] A one. B'
    """
    keys, origin = _folded(written)
    heard_key, _ = _folded(heard)
    count = len(heard_key)
    if not count:
        return ""
    if not keys.startswith(heard_key):
        return strip_cue_tags(heard)
    cut = len(written) if count == len(keys) else _heard_end(written, origin, count)
    return " ".join(_without_clear(written[:cut]).split())


@dataclass(frozen=True, slots=True)
class Unheard:
    """The rest of a cut-off reply, spoken again if the interrupt was only noise.

    Attributes
    ----------
    sent : str
        The rest as Fish is sent it, led by the mood cue from before the cut.
    written : str
        The same rest as the model wrote it, which the conversation records once heard.
    """

    sent: str
    written: str


def unheard(sent: str, written: str, heard: str) -> Unheard | None:
    """Return the rest of a reply the listener did not hear, if it is worth speaking.

    Parameters
    ----------
    sent : str
        The reply as sent to Fish, cues included.
    written : str
        The same reply before any cue was added for Fish (see ``heard_written``).
    heard : str
        What was heard of ``sent``, from ``spoken_prefix``.

    Returns
    -------
    Unheard or None
        None when ``unspoken_text`` finds nothing to speak again. When ``heard`` is not a
        prefix of ``written``, the written rest is the sent rest with every cue removed.
    """
    rest = unspoken_text(sent, heard)
    if not rest:
        return None
    return Unheard(rest, unspoken_text(written, heard, lead=False) or strip_cue_tags(rest))
