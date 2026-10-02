"""Holds that keep an unfinished span out of a streamed TTS send.

Each check returns the index where an unfinished span starts, or None when
the buffer is stable. ``hold_tts`` takes the earliest. Every hold mirrors a
scrub rule in ``scrub_markdown``: releasing the span early would speak markup
the scrubber removes only once it is complete.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from functools import partial

from fish_audio_suite_kit._charsets import (
    SENTENCE_CLOSERS,
    SENTENCE_STOPS,
    THOUGHT_CLOSE_RE,
    THOUGHT_OPEN_RE,
    THOUGHT_WORDS,
)
from fish_audio_suite_kit.cues import mood_lead_hold_at, spoken_mood_span
from fish_audio_suite_kit.cuts import next_tts_cut
from fish_audio_suite_kit.scrub_markdown import (
    HTML_BLOCK_CLOSE_RE,
    HTML_BLOCK_OPEN_RE,
    HTML_ENTITY_RE,
    html_char,
    unclosed_span_start,
)

_Hold = Callable[[str], int | None]

_WHISPER_OPEN_RE = re.compile(r"<\s*whisper\s*>", re.IGNORECASE)
_WHISPER_CLOSE_RE = re.compile(r"<\s*/\s*whisper\s*>", re.IGNORECASE)
# The address ends at a closing parenthesis or angle bracket, as in scrub.
# Holding past either would release the "<" or "[label](" in front of it.
_URL_AT_END_RE = re.compile(r"https?://[^\s)>]*$", re.IGNORECASE)
# A hash run at the start of a line is a heading only once a space follows.
_OPEN_HEADING_RE = re.compile(r"(?m)^[ \t]{0,3}#{1,6}$")
# "* item" is a list mark only once a space follows. A lone "*" is not speech.
_OPEN_LIST_RE = re.compile(r"(?m)^[ \t]{0,3}[-*_~+]+$")
_LT_ENTITY_RE = re.compile(r"&(?:lt|#0*60|#x0*3c);", re.IGNORECASE)
_GT_ENTITY_RE = re.compile(r"&(?:gt|#0*62|#x0*3e);|>", re.IGNORECASE)
# A split "<thi" is not a tag yet. The tag names that open a removed span.
_TAG_WORDS = (*THOUGHT_WORDS, "whisper")
# Backticks stay with the fence holds, so they are not trailing emphasis.
_TRAIL_MARKS = frozenset("*_~")


def _first_unclosed_pair(
    text: str,
    open_re: re.Pattern[str],
    close_re: re.Pattern[str],
    *,
    same_name: bool = False,
) -> int | None:
    # A pair is removed (or rewritten) only when whole, so an opener without
    # its closer holds. With same_name, a closer of another name is skipped.
    idx = 0
    while True:
        opened = open_re.search(text, idx)
        if opened is None:
            return None
        search_at = opened.end()
        while True:
            closed = close_re.search(text, search_at)
            if closed is None:
                return opened.start()
            if not same_name or closed.group(1).lower() == opened.group(1).lower():
                idx = closed.end()
                break
            search_at = closed.end()


def _unclosed_fence(text: str, mark: str) -> int | None:
    # An odd count means the last fence is open: releasing it speaks the code.
    # A run of three tildes only. "~~word~~" is a deletion, not a fence.
    if text.count(mark) % 2 == 0:
        return None
    return text.rfind(mark)


def _finished_tilde_fence(text: str) -> int | None:
    # The closer is the last character, so release waits for the next one.
    # Scrubbing before the closer arrives in the same chunk would speak the code.
    if text.count("~~~") % 2 or not text.endswith("~~~"):
        return None
    end = text.rfind("~~~")
    opener = text.rfind("~~~", 0, end)
    return opener if opener >= 0 else None


def _incomplete_fence_mark(text: str) -> int | None:
    # One or two trailing backticks may still become a fence.
    count = 0
    index = len(text)
    while index > 0 and text[index - 1] == "`":
        count += 1
        index -= 1
    if count == 0 or count % 3 == 0:
        return None
    return index


def _unclosed_strike(text: str) -> int | None:
    # "~~Tuesday" has no closer yet. Releasing it speaks the word.
    if text.count("~~") % 2 == 0:
        return None
    return text.rfind("~~")


def _trailing_emphasis(text: str) -> int | None:
    # "2*" is not emphasis until the next character arrives. Releasing the
    # star strips it, and "2*3" would be spoken as "23".
    if not text or text[-1] not in _TRAIL_MARKS:
        return None
    index = len(text)
    while index > 0 and text[index - 1] in _TRAIL_MARKS:
        index -= 1
    marks = text[index:]
    # A finished ~~Tuesday~~ ends on its closer. Holding those tildes would
    # scrub "~~Tuesday" alone and speak the deleted word.
    if set(marks) == {"~"} and len(marks) >= 2 and text.count("~~") % 2 == 0:
        return None
    return index


def _partial_scheme_start(text: str, schemes: tuple[str, ...]) -> int | None:
    # The buffer ends on a prefix of a scheme ("htt", "<http"). Releasing it
    # would speak the address once the rest arrives in a later token.
    low = text.lower()
    best: int | None = None
    for scheme in schemes:
        for length in range(1, len(scheme)):
            if not low.endswith(scheme[:length]):
                continue
            start = len(text) - length
            if start and text[start - 1].isalnum():
                continue
            if best is None or start < best:
                best = start
    return best


def _incomplete_autolink(text: str) -> int | None:
    best = _partial_scheme_start(text, ("<https://", "<http://"))
    opened = re.search(r"<https?://[^>\s]*$", text, re.IGNORECASE)
    if opened is not None and (best is None or opened.start() < best):
        return opened.start()
    return best


def _incomplete_url_scheme(text: str) -> int | None:
    return _partial_scheme_start(text, ("https://", "http://"))


def _url_at_end(text: str) -> int | None:
    found = _URL_AT_END_RE.search(text)
    return found.start() if found is not None else None


def _incomplete_tag(text: str) -> int | None:
    start = text.rfind("<")
    if start < 0:
        return None
    tail = text[start + 1 :]
    if ">" in tail:
        return None
    body = tail.lstrip()
    if body.startswith("/"):
        body = body[1:].lstrip()
    if not body:
        return start
    low = body.lower()
    if any(word.startswith(low) for word in _TAG_WORDS):
        return start
    return None


def _incomplete_angle_token(text: str) -> int | None:
    # "<|ACT smile|>" is one stage token. Releasing "<|" speaks the direction.
    start = text.rfind("<|")
    if start < 0 or "|>" in text[start + 2 :]:
        return None
    return start


def _incomplete_comment(text: str) -> int | None:
    mark = text.rfind("<")
    if mark < 0:
        return None
    tail = text[mark:]
    if tail.startswith("<!--"):
        return None if "-->" in tail[4:] else mark
    return mark if "<!--".startswith(tail) else None


def _incomplete_html(text: str) -> int | None:
    # "<b" is not done. Releasing it speaks the tag name.
    start = text.rfind("<")
    if start < 0 or ">" in text[start + 1 :]:
        return None
    tail = text[start + 1 :]
    if tail[:1] == "/":
        tail = tail[1:]
    # A space inside the tag ("</ p") does not end it.
    body = tail.lstrip(" \t")
    if body[:1] and not body[0].isalpha():
        return None
    if len(body) > 80:
        return None
    return start


def _incomplete_escaped_tag(text: str) -> int | None:
    # "&lt;br" is a tag still being written. Releasing it speaks "br".
    found: int | None = None
    for match in _LT_ENTITY_RE.finditer(text):
        rest = text[match.end() :]
        if _GT_ENTITY_RE.search(rest):
            continue
        body = rest[1:] if rest[:1] == "/" else rest
        body = body.lstrip(" \t")
        if body[:1] and not body[0].isalpha():
            continue
        if len(body) > 80:
            continue
        found = match.start() if found is None else min(found, match.start())
    return found


def _incomplete_entity(text: str) -> int | None:
    # "&amp" is not "&" yet. Releasing it speaks the letters.
    mark = text.rfind("&")
    if mark < 0 or ";" in text[mark:]:
        return None
    tail = text[mark + 1 :]
    if len(tail) > 12 or not re.fullmatch(r"#?x?[0-9A-Za-z]*", tail):
        return None
    return mark


def _incomplete_image(text: str) -> int | None:
    # "![alt](url)" is one span. Holding at "[" would release the bang first.
    mark = text.rfind("![")
    if mark < 0:
        return None
    tail = text[mark:]
    close = tail.find("](")
    if close >= 0 and ")" in tail[close + 2 :]:
        return None
    return mark


def _incomplete_table_row(text: str) -> int | None:
    # "| name | score" is one row. A cut before the last pipe leaves the bars.
    line_at = text.rfind("\n") + 1
    line = text[line_at:]
    stripped = line.lstrip(" \t")
    if not stripped.startswith("|"):
        return None
    if stripped.rstrip().endswith("|") and stripped.count("|") >= 3:
        return None
    return line_at + (len(line) - len(stripped))


def _open_heading(text: str) -> int | None:
    match = _OPEN_HEADING_RE.search(text)
    if match is None or match.end() != len(text):
        return None
    return match.start()


def _open_list_marker(text: str) -> int | None:
    match = _OPEN_LIST_RE.search(text)
    if match is None or match.end() != len(text):
        return None
    return match.start()


def _mark_on_its_line(found: int | None, *, line_start: bool) -> int | None:
    # A "*" that begins a mid-line chunk is the multiply sign. A marker after a
    # newline inside the chunk is a bullet either way.
    if found is None or (found == 0 and not line_start):
        return None
    return found


def _open_markdown_link(text: str) -> int | None:
    # [label](https://... is one span. Releasing the label at "(" would speak
    # [label] as a cue and could strand the closing parenthesis.
    search = 0
    while True:
        mark = text.find("](", search)
        if mark < 0:
            return None
        if text.find(")", mark + 2) < 0:
            bracket = text.rfind("[", 0, mark)
            if bracket >= 0:
                return bracket
        search = mark + 2


def _finished_link_hold(text: str) -> int | None:
    # A finished link waits for the next character, or "docs" and "today"
    # would join on the closing parenthesis.
    mark = text.rfind("](")
    if mark < 0:
        return None
    close = text.find(")", mark + 2)
    if close < 0 or close != len(text) - 1:
        return None
    bracket = text.rfind("[", 0, mark)
    if bracket < 0:
        return None
    # "![alt](url)" includes the bang.
    if bracket > 0 and text[bracket - 1] == "!":
        return bracket - 1
    return bracket


def _open_ref_link(text: str) -> int | None:
    # [docs][ref] is one span, including once finished, until the next
    # character shows whether the trailing-bracket hold applies.
    search = 0
    while True:
        mark = text.find("][", search)
        if mark < 0:
            return None
        bracket = text.rfind("[", 0, mark)
        close = text.find("]", mark + 2)
        if bracket >= 0 and (close < 0 or close == len(text) - 1):
            return bracket
        search = mark + 2


def _after_closed_bracket(text: str) -> int | None:
    # "]" at the end may still be a markdown link. A letter right after it is
    # the next word, as in "[pause 1s]Excited", and goes in the next piece.
    search = 0
    while True:
        mark = text.find("]", search)
        if mark < 0 or mark + 1 >= len(text):
            return None
        if text[mark + 1].isalnum():
            return mark + 1
        search = mark + 1


def _trailing_bracket(text: str) -> int | None:
    # [docs] at the end of the buffer may still become [docs](https://...).
    if not text.endswith("]"):
        return None
    depth = 0
    for index in range(len(text) - 1, -1, -1):
        ch = text[index]
        if ch == "]":
            depth += 1
        elif ch == "[":
            depth -= 1
            if depth == 0:
                return index
    return None


def _ref_label_before_id(text: str) -> int | None:
    # "[the docs][" is still one reference. The id arrives too late to join a
    # label already released as a cue.
    mark = text.rfind("][")
    if mark < 0 or "]" in text[mark + 2 :]:
        return None
    start_label = text.rfind("[", 0, mark)
    if start_label < 0 or " " not in text[start_label + 1 : mark]:
        return None
    return start_label


def _trailing_ref_link(text: str) -> int | None:
    # [the docs][ref]: the second bracket must not be held alone.
    if not text.endswith("]"):
        return None
    start_id = text.rfind("[", 0, len(text) - 1)
    if start_id <= 0 or text[start_id - 1] != "]":
        return None
    start_label = text.rfind("[", 0, start_id - 1)
    if start_label < 0 or " " not in text[start_label + 1 : start_id - 1]:
        return None
    return start_label


def _hold_closes_paren(text: str, index: int) -> bool:
    depth = 0
    for ch in text[:index]:
        if ch == "(":
            depth += 1
        elif ch == ")" and depth:
            depth -= 1
    return depth > 0


def sentence_closer_hold_at(text: str, *, lead: bool = False) -> int | None:
    """Return where a stop is waiting for a possible closing quote.

    Parameters
    ----------
    text : str
        Unsent model text.
    lead : bool, optional
        True when mood leads are enabled. A stop that completes a mood lead
        (``Anxious!``) is then not a sentence stop. Default False.

    Returns
    -------
    int or None
        Index of the trailing stop when the buffer ends on that stop or on a
        closing quote or parenthesis after it. None when a later character
        shows the stop is finished, including a space that already followed it.

    Notes
    -----
    Releasing an ideographic stop before a closing quote arrives puts the
    quote on the next sentence. A parenthesis that closes an aside is not a
    sentence closer: holding the period inside ``(aside.)`` would speak the
    leftover ``.)``.
    """
    if not text:
        return None
    index = len(text)
    while index > 0 and text[index - 1] in SENTENCE_CLOSERS:
        if text[index - 1] == ")" and _hold_closes_paren(text, index - 1):
            break
        index -= 1
    if index == 0 or text[index - 1] not in SENTENCE_STOPS:
        return None
    while index > 0 and text[index - 1] in SENTENCE_STOPS:
        index -= 1
    span = spoken_mood_span(text) if lead else None
    if span is not None and span[1] > index:
        if span[1] == len(text):
            return None
        return span[0]
    return index


def skip_empty_delta(piece: str) -> bool:
    """Return whether a TTS delta has no speakable characters.

    Parameters
    ----------
    piece : str
        One split piece. Whitespace-only is empty.

    Returns
    -------
    bool
        True when the websocket sender should not emit a ``TextEvent``.
        An empty turn must not be followed by a bare ``FlushEvent``.
    """
    return not (piece or "").strip()


def _entity_stop_hold(text: str) -> int | None:
    # "&#33; Ex" is an exclamation mark plus the start of a mood, and "&#10;Ex"
    # a new line. Until decoded, the mood looks mid-sentence.
    for match in HTML_ENTITY_RE.finditer(text):
        decoded = html_char(match)
        if decoded != "\n" and decoded not in SENTENCE_STOPS:
            continue
        held = mood_lead_hold_at(text[match.end() :], sentence_start=True)
        if held is not None:
            return match.end() + held
    return None


def _sentence_lead_hold(text: str, *, sentence_start: bool, before: str = "") -> int | None:
    # The mood lead can follow a sentence inside this same buffer ("Hello. Ex"),
    # and the stop can land in the next chunk ("Don" then "e.Ex"). Hold the last
    # sentence, taken over the already-accepted text plus this buffer.
    combined = before + text
    start = 0
    pos = 0
    buf = combined
    found = False
    while buf:
        cut = next_tts_cut(buf, partial_chars=len(buf) + 1)
        if cut < 0:
            break
        pos += cut
        buf = combined[pos:]
        start = pos
        found = True
    if found:
        held = mood_lead_hold_at(combined[start:], sentence_start=True)
        if held is None:
            return None
        index = start + held
        return index - len(before) if index >= len(before) else 0
    if not sentence_start or before:
        return None
    return mood_lead_hold_at(text, sentence_start=True)


# Text-only checks. Order does not matter: the earliest start wins.
_TEXT_HOLDS: tuple[_Hold, ...] = (
    partial(_first_unclosed_pair, open_re=THOUGHT_OPEN_RE, close_re=THOUGHT_CLOSE_RE),
    partial(_first_unclosed_pair, open_re=_WHISPER_OPEN_RE, close_re=_WHISPER_CLOSE_RE),
    partial(
        _first_unclosed_pair,
        open_re=HTML_BLOCK_OPEN_RE,
        close_re=HTML_BLOCK_CLOSE_RE,
        same_name=True,
    ),
    _incomplete_tag,
    _incomplete_angle_token,
    partial(_unclosed_fence, mark="```"),
    partial(_unclosed_fence, mark="~~~"),
    _finished_tilde_fence,
    _incomplete_fence_mark,
    _unclosed_strike,
    _trailing_emphasis,
    _incomplete_autolink,
    _incomplete_url_scheme,
    _incomplete_comment,
    _incomplete_html,
    _incomplete_escaped_tag,
    _incomplete_entity,
    _incomplete_image,
    _incomplete_table_row,
    _ref_label_before_id,
    _trailing_ref_link,
    _url_at_end,
    partial(unclosed_span_start, open_ch="(", close_ch=")"),
    partial(unclosed_span_start, open_ch="（", close_ch="）"),
    partial(unclosed_span_start, open_ch="[", close_ch="]"),
    _open_markdown_link,
    _finished_link_hold,
    _open_ref_link,
    _trailing_bracket,
    _after_closed_bracket,
)


def hold_tts(
    text: str,
    *,
    line_start: bool,
    sentence_start: bool,
    before: str = "",
    lead: bool = False,
) -> int:
    """Return the index where a stream must keep buffering.

    Parameters
    ----------
    text : str
        Unscrubbed chunk, including any unfinished span.
    line_start : bool
        True when this chunk begins a line. A mid-line star is not a bullet.
    sentence_start : bool
        True when a mood word here can become a lead cue. Used only when
        ``lead`` is True.
    before : str, optional
        Scrubbed text already accepted on this line. Default empty.
    lead : bool, optional
        True when mood leads are enabled in ``normalize_cues``. The mood-word
        holds (an unfinished ``Exc`` before ``ited,``) are then active.
        Default False, so no mood-related text is held.

    Returns
    -------
    int
        Start of the unfinished span, or ``len(text)`` when the chunk is stable.
    """
    marks = [check(text) for check in _TEXT_HOLDS]
    marks.append(sentence_closer_hold_at(text, lead=lead))
    marks.append(_mark_on_its_line(_open_heading(text), line_start=line_start))
    marks.append(_mark_on_its_line(_open_list_marker(text), line_start=line_start))
    if lead:
        marks.append(_entity_stop_hold(text))
        marks.append(mood_lead_hold_at(text, sentence_start=sentence_start))
        marks.append(_sentence_lead_hold(text, sentence_start=sentence_start, before=before))
    starts = [mark for mark in marks if mark is not None]
    return min(starts) if starts else len(text)
