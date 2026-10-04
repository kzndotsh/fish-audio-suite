"""Turn Fish's word-level ASR segments into the timed phrase cues a caption file needs."""

from __future__ import annotations

import bisect
import math
import unicodedata
from typing import Any, Final

from fish_audio_suite_kit import (
    AsrBody,
    CaptionCue,
    ends_sentence,
    is_caption_watermark,
    parse_number,
    scrub_asr,
)

__all__ = [
    "caption_cues",
    "word_cues",
]

# A Fish word is looked for in the transcript's letters and digits, at most this
# many of them ahead of the last match. A word the transcript lacks is used as
# Fish sent it, and the search does not move.
_ALIGN_WINDOW: Final = 24
_CUE_MAX: Final = 64
_EDGE_MAX: Final = 8
# A ``<|speaker:0|>`` marker or ``[laughter]`` cue in the transcript is not
# speech. Both are bounded so an unclosed opener cannot start a long scan.
_MARKER_MAX: Final = 200
_OPENERS: Final = frozenset("\"'“‘«‹([{「『（【〈《〔〖¿¡")
# About two 42-character caption lines, and a few seconds on screen.
_PHRASE_MAX_CHARS: Final = 84
_PHRASE_MAX_S: Final = 6.0
# Fish ``segments`` are single words. Caption cues and OpenAI ``segments`` are
# phrases, so words are grouped: a new cue starts after a sentence end, a pause,
# a speaker change, or when the cue would get too long to read.
_PHRASE_PAUSE_S: Final = 0.7
# Characters a word keeps from the transcript: punctuation and closers right
# after it, and an opening quote or bracket right before it. A cue's brackets
# never touch a matched word, because the cue is not in the stream, so "[" and
# "]" here only ever belong to speech such as "[5]".
_TRAILING: Final = frozenset(".,!?;:…‥。！？，、；：\"'”’»›)]}」』）】〉》〕〗")
# A turn that starts this close after a word's start still begins at that word.
_TURN_SLACK_S: Final = 1e-6


def _seconds(value: Any) -> float:
    return parse_number(value or 0, 0.0, float)


def _duration_s(value: Any) -> float:
    """Fish ``duration`` is a number of seconds. A string or boolean is not a caption length."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    if not math.isfinite(value):
        return 0.0
    return float(value)


def _segment_cue(seg: Any, *, strip_speakers: bool, strip_cues: bool) -> CaptionCue | None:
    if not isinstance(seg, dict):
        return None
    raw_text = seg.get("text", "")
    if not isinstance(raw_text, str):
        return None
    body = scrub_asr(raw_text, strip_speakers=strip_speakers, strip_cues=strip_cues)
    if not body:
        return None
    start = max(0.0, _seconds(seg.get("start", 0)))
    end = max(start, _seconds(seg.get("end", 0)))
    return CaptionCue(start, end, body)


def word_cues(data: AsrBody, *, strip_speakers: bool, strip_cues: bool) -> list[CaptionCue]:
    """Return one timed cue per word of a Fish ASR body.

    Parameters
    ----------
    data : AsrBody
        The Fish ASR response. Its segments are word-level.
    strip_speakers : bool
        Drop inline ``<|speaker:N|>`` marks from the words.
    strip_cues : bool
        Drop inline emotion cues such as ``[calm]`` from the words.

    Returns
    -------
    list of CaptionCue
        Words with their start and end in milliseconds. Segments that are not
        well-formed are skipped.
    """
    # Fish JSON is checked only for ``text``, so a segment list can hold anything.
    raw_segments: object = dict(data).get("segments") or []
    if not isinstance(raw_segments, list):
        return []
    words: list[CaptionCue] = []
    for seg in raw_segments:
        cue = _segment_cue(seg, strip_speakers=strip_speakers, strip_cues=strip_cues)
        if cue is not None:
            words.append(cue)
    return words


def _folded(text: str) -> str:
    # Case-folded letters and digits only: "3.5" and "35" fold the same, and so
    # do "Hello," and "hello".
    return "".join(ch for ch in text.casefold() if ch.isalnum())


def _skipped_span(text: str, i: int) -> int:
    """Return the end of a speaker marker or cue that starts at ``i``, or ``i``."""
    if text.startswith("<|", i):
        close = text.find("|>", i + 2, i + 4 + _MARKER_MAX)
        return close + 2 if close != -1 else i
    if text[i] == "[":
        close = text.find("]", i + 1, i + 2 + _CUE_MAX)
        if close != -1 and _is_cue(text[i + 1 : close]):
            return close + 1
    return i


def _is_cue(inner: str) -> bool:
    # A bracket with only digits, such as "[5]", is something the speaker said.
    return "[" not in inner and "\n" not in inner and any(ch.isalpha() for ch in inner)


# A Latin word shows its whole space-delimited token from the transcript, so
# "$3.5," stays whole when Fish splits it into "3" and "5". Bounded for long runs.
_TOKEN_MAX: Final = 48


class _Transcript:
    """The transcript as a stream of folded letters and digits, mapped back to it."""

    def __init__(self, text: str) -> None:
        self.text = text
        keys: list[str] = []
        self.origin: list[int] = []
        i = 0
        while i < len(text):
            skip = _skipped_span(text, i)
            if skip != i:
                i = skip
                continue
            for ch in _folded(text[i]):
                keys.append(ch)
                self.origin.append(i)
            i += 1
        self.keys = "".join(keys)

    def find(self, key: str, at: int, *, spaced: bool) -> int | None:
        """Return the stream index where ``key`` starts as a whole word near ``at``."""
        limit = min(len(self.keys), at + _ALIGN_WINDOW + len(key))
        found = self.keys.find(key, at, limit)
        while found != -1:
            if self._whole(found, len(key), spaced=spaced):
                return found
            found = self.keys.find(key, found + 1, limit)
        return None

    def _whole(self, found: int, size: int, *, spaced: bool) -> bool:
        text, origin = self.text, self.origin
        first, last = origin[found], origin[found + size - 1]
        # A match must not start or end inside one character's folding ("ß").
        if found > 0 and origin[found - 1] == first:
            return False
        if found + size < len(origin) and origin[found + size] == last:
            return False
        span = text[first : last + 1]
        if any(ch in "[<" or (ch.isspace() and not spaced) for ch in span):
            return False
        # A Latin word must not match inside a longer one ("um" in "drum").
        # CJK is written without spaces, so a single character stands alone.
        if not _is_wide(text[first]) and first > 0 and text[first - 1].isalnum():
            return False
        return _is_wide(text[last]) or last + 1 >= len(text) or not text[last + 1].isalnum()

    def start_of(self, found: int) -> int:
        """Return where the stream index ``found`` sits in ``text``."""
        return self.origin[found]

    def display(self, found: int, size: int, floor: int, *, keep_cues: bool) -> tuple[str, int]:
        """Return the transcript text for a match and where it ends in ``text``."""
        text = self.text
        first, last = self.origin[found], self.origin[found + size - 1]
        if not _is_wide(text[first]):
            return self._token(first, last, floor, keep_cues=keep_cues)
        stop = last + 1
        while stop < len(text) and stop - last <= _EDGE_MAX and text[stop] in _TRAILING:
            stop += 1
        begin = first
        while begin > floor and first - begin < _EDGE_MAX and text[begin - 1] in _OPENERS:
            begin -= 1
        lead = self._lead_cue(begin, floor) if keep_cues else ""
        return lead + text[begin:stop], stop

    def _token(self, first: int, last: int, floor: int, *, keep_cues: bool) -> tuple[str, int]:
        # The space-delimited token around a Latin match. It stops at a cue or
        # marker edge ("]" or ">") so a cue glued to the word is not shown as text.
        text = self.text
        begin = first
        while (
            begin > floor
            and first - begin < _TOKEN_MAX
            and not text[begin - 1].isspace()
            and text[begin - 1] not in "]>"
        ):
            begin -= 1
        stop = last + 1
        while (
            stop < len(text)
            and stop - last <= _TOKEN_MAX
            and not text[stop].isspace()
            and text[stop] not in "[<"
            and not _is_wide(text[stop])
        ):
            stop += 1
        lead = self._lead_cue(begin, floor) if keep_cues else ""
        return lead + text[begin:stop], stop

    def _lead_cue(self, begin: int, floor: int) -> str:
        # A cue right before the word, such as "[laughs] Well", stays with it.
        text = self.text
        end = begin
        while end > floor and begin - end < _EDGE_MAX and text[end - 1].isspace():
            end -= 1
        if end <= floor or text[end - 1] != "]":
            return ""
        opener = text.rfind("[", max(floor, end - 2 - _CUE_MAX), end - 1)
        if opener == -1 or not _is_cue(text[opener + 1 : end - 1]):
            return ""
        return text[opener:end] + (" " if end < begin else "")


def _punctuated(words: list[CaptionCue], text: str, *, keep_cues: bool) -> list[CaptionCue]:
    """Give each Fish word the punctuation and case it has in the transcript.

    Words are aligned on the transcript's letters and digits, not on its
    whitespace, so a CJK character finds the "。" after it and "35" finds "3.5".
    """
    stream = _Transcript(text)
    out: list[CaptionCue] = []
    at = 0
    floor = 0
    for cue in words:
        key = _folded(cue.text)
        found = stream.find(key, at, spaced=any(ch.isspace() for ch in cue.text)) if key else None
        if found is None:
            out.append(cue)
            continue
        if out and stream.start_of(found) < floor:
            # Fish split one transcript token ("3.5") into several words. The
            # first already shows the whole token, so this one only extends it.
            out[-1] = CaptionCue(out[-1].start, cue.end, out[-1].text)
            at = found + len(key)
            continue
        shown, floor = stream.display(found, len(key), floor, keep_cues=keep_cues)
        out.append(CaptionCue(cue.start, cue.end, shown))
        at = found + len(key)
    return out


def _is_wide(ch: str) -> bool:
    return unicodedata.east_asian_width(ch) in {"W", "F"}


def _join_words(left: str, right: str) -> str:
    if not left:
        return right
    # CJK words are written without spaces between them. A cue in front of a
    # word, as in "[高兴]很", is joined by the character after it.
    head = right
    if right.startswith("[") and "]" in right[:-1]:
        head = right[right.index("]") + 1 :]
    if _is_wide(left[-1]) and _is_wide(head[0]):
        return left + right
    return f"{left} {right}"


def _turn_starts(data: AsrBody) -> list[float]:
    turns: object = dict(data).get("speaker_turns") or []
    if not isinstance(turns, list):
        return []
    starts = [_seconds(turn.get("start", 0)) for turn in turns if isinstance(turn, dict)]
    return sorted(start for start in starts if start > 0)


def _starts_new_turn(turn_starts: list[float], previous: CaptionCue, word: CaptionCue) -> bool:
    i = bisect.bisect_right(turn_starts, previous.start)
    return i < len(turn_starts) and turn_starts[i] <= word.start + _TURN_SLACK_S


def _phrases(words: list[CaptionCue], turn_starts: list[float]) -> list[CaptionCue]:
    phrases: list[CaptionCue] = []
    current: list[CaptionCue] = []
    text = ""

    def close() -> None:
        if current:
            phrases.append(CaptionCue(current[0].start, current[-1].end, text))

    for word in words:
        if current:
            joined = _join_words(text, word.text)
            breaks = (
                ends_sentence(f"{text} ")
                or word.start - current[-1].end >= _PHRASE_PAUSE_S
                or len(joined) > _PHRASE_MAX_CHARS
                or word.end - current[0].start > _PHRASE_MAX_S
                or _starts_new_turn(turn_starts, current[-1], word)
            )
            if breaks:
                close()
                current, text = [], ""
        current.append(word)
        text = _join_words(text, word.text)
    close()
    return phrases


def caption_cues(
    data: AsrBody,
    text: str,
    *,
    strip_speakers: bool,
    strip_cues: bool = False,
) -> list[CaptionCue]:
    """Build phrase cues from Fish word segments, or one cue for the whole transcript.

    Fish ``segments`` are word-level: each holds one word with ``start`` and
    ``end`` in seconds, with no punctuation, and a CJK word is usually one
    character. Words are grouped into phrases, taking punctuation and case
    from ``text``: each word is found among the transcript's letters and
    digits a little ahead of the last match, so "35" finds "3.5" and a CJK
    character finds the "。" after it. A word the transcript lacks is used as
    Fish sent it. Speaker markers are never shown. A cue ends after a sentence
    end, a pause of 0.7 s or more, a ``speaker_turns`` boundary, or before it
    would pass 84 characters (two caption lines) or 6 seconds.

    Parameters
    ----------
    data : AsrBody
        Decoded Fish ASR JSON.
    text : str
        Scrubbed full transcript. It supplies punctuation, and is the one cue
        when ``segments`` is missing or empty.
    strip_speakers : bool
        Drop speaker labels inside each word.
    strip_cues : bool, optional
        Drop ``[cue]`` annotations inside each word. Default False, which also
        keeps a cue that ``text`` puts right before a word in front of that
        word (``[laughs] Well,``), as Fish's own ``speaker_turns`` text does.

    Returns
    -------
    list of CaptionCue
        Phrase cues when any word has text. A phrase that is a known caption
        watermark is omitted. Otherwise one cue from 0 to ``duration`` (seconds)
        covering ``text``, or an empty list when ``text`` is empty.
    """
    words = word_cues(data, strip_speakers=strip_speakers, strip_cues=strip_cues)
    phrases = _phrases(_punctuated(words, text, keep_cues=not strip_cues), _turn_starts(data))
    cues = [cue for cue in phrases if not is_caption_watermark(cue.text)]
    if cues:
        return cues
    if not text:
        return []
    return [CaptionCue(0.0, max(0.0, _duration_s(data.get("duration"))), text)]
