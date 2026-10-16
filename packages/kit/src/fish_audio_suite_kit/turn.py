"""Judge whether a transcript looks like a finished turn, so a short silence can end it.

A listener that waits a fixed second of quiet answers late every time. A listener that
waits less cuts people off mid-thought. Looking at the words tells the two apart: a
sentence that ends on "and", "because" or a comma is not over, whatever the pause. The
judgement is a guess from English text, kept cautious, and a wrong "not finished" only
costs the wait the listener would have had anyway.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

from fish_audio_suite_kit.literals import TurnReason

__all__ = [
    "TurnJudgement",
    "judge_turn",
]

# Only the end of the text matters, and a bound keeps hostile input linear.
_TAIL_CHARS: Final = 160
# Quotes and brackets that close after the last real character.
_CLOSERS: Final = "\"')]}\u201d\u2019\u300d\u300f\uff09"
# Punctuation that leaves a thought open: a comma, a colon, a dash, a trailing-off ellipsis.
_OPEN_ENDS: Final = (*",;:-\u2013\u2014\u2026\u3001\uff0c\uff1b\uff1a", "...")
_TERMINALS: Final = ".?!\u3002\uff1f\uff01"
_WORD_RE: Final = re.compile(r"[^\W_]+(?:['\u2019][^\W_]+)*")
_SHORT_ANSWER_WORDS: Final = 2

# Words a finished sentence does not end on, even when the transcript adds a full stop.
_NEVER_LAST: Final = frozenset(
    {"and", "but", "or", "because", "cuz", "if", "the", "a", "an", "um", "uh", "er", "erm", "uhm"}
)
# Words a spoken sentence rarely ends on when no punctuation closes it. Fish ASR punctuates,
# so text that reaches this list without a full stop is usually a thought that stopped short.
_OPEN_LAST: Final = frozenset(
    {
        "to", "of", "with", "for", "from", "in", "on", "at", "by", "about", "into", "onto",
        "over", "under", "between", "through", "like", "as", "than", "which", "who", "whose",
        "where", "when", "while", "although", "though", "unless", "until", "since", "whether",
        "is", "are", "was", "were", "am", "be", "been", "being", "i", "i'm", "i've", "i'd",
        "i'll", "we", "we're", "my", "your", "our", "their", "its", "also", "then",
        "so",
    }
)  # fmt: skip
# "so" and "then" close a sentence ("I think so") unless a comma or "and" leads into them.
_LEAD_IN_ONLY: Final = frozenset({"so", "then"})
_NUMBER_WORDS: Final = frozenset(
    {
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
        "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
        "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
        "eighty", "ninety", "hundred", "thousand",
    }
)  # fmt: skip
_LEAD_IN_RE: Final = re.compile(r"(?:,|\band)\s+\w+$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class TurnJudgement:
    """Whether a transcript looks like a finished turn, and why.

    Attributes
    ----------
    finished : bool
        True when the speaker looks done. False when the text stops mid-thought.
    reason : str
        What decided it: ``empty``, ``open_punctuation``, ``dangling_word``,
        ``number_list``, ``terminal_punctuation``, ``short_answer`` or ``complete``.
    """

    finished: bool
    reason: TurnReason


def _is_number(word: str) -> bool:
    return word.isdigit() or word in _NUMBER_WORDS


def judge_turn(text: str) -> TurnJudgement:
    """Judge whether a transcript looks like the end of what the speaker meant to say.

    Parameters
    ----------
    text : str
        What speech recognition heard. Only the last 160 characters are read.

    Returns
    -------
    TurnJudgement
        ``finished`` is False when the text ends on a comma, a colon, a dash or an
        ellipsis, on a word that cannot end a sentence ("and", "because", "the",
        "um", a preposition), on ", so", or on a comma-separated run of numbers with no full stop.
        A full stop, question mark or exclamation mark finishes a turn, and so does a
        short answer of one or two words. English only. Other languages are judged
        by their punctuation alone.

    Examples
    --------
    >>> judge_turn("I was going to say that, but").finished
    False
    >>> judge_turn("Can you tell me what time it is?").finished
    True
    >>> judge_turn("Okay,").reason
    'open_punctuation'
    >>> judge_turn("The numbers are 4, 8, 15").reason
    'number_list'
    >>> judge_turn("I think so").finished
    True
    """
    tail = text.rstrip()[-_TAIL_CHARS:]
    core = tail.rstrip(_CLOSERS)
    if not core.strip():
        return TurnJudgement(finished=True, reason="empty")
    if core.endswith(_OPEN_ENDS):
        return TurnJudgement(finished=False, reason="open_punctuation")
    words = [word.casefold() for word in _WORD_RE.findall(core)]
    if not words:
        return TurnJudgement(finished=True, reason="complete")
    last = words[-1]
    closed = core[-1] in _TERMINALS
    short = len(words) <= _SHORT_ANSWER_WORDS and len(text) <= _TAIL_CHARS
    if closed:
        if last in _NEVER_LAST and not short:
            return TurnJudgement(finished=False, reason="dangling_word")
        return TurnJudgement(finished=True, reason="terminal_punctuation")
    if short and last not in _NEVER_LAST:
        return TurnJudgement(finished=True, reason="short_answer")
    if last in _NEVER_LAST:
        return TurnJudgement(finished=False, reason="dangling_word")
    if last in _OPEN_LAST and (last not in _LEAD_IN_ONLY or _LEAD_IN_RE.search(core)):
        return TurnJudgement(finished=False, reason="dangling_word")
    if (
        _is_number(last)
        and "," in core
        and sum(_is_number(word) for word in words) >= _SHORT_ANSWER_WORDS
    ):
        return TurnJudgement(finished=False, reason="number_list")
    return TurnJudgement(finished=True, reason="complete")
