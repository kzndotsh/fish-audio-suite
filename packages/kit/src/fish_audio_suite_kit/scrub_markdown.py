"""Strip markdown, HTML, thoughts, and stage markup from text bound for Fish TTS."""

from __future__ import annotations

import re
from collections.abc import Callable
from functools import partial

from fish_audio_suite_kit._charsets import (
    ANGLE_TOKEN_RE,
    BREAKS_RE,
    SPACE_BEFORE_STOP_RE,
    THOUGHT_CLOSE_RE,
    THOUGHT_OPEN_RE,
    plain_breaks,
    utf8_text,
)
from fish_audio_suite_kit.cues import is_paren_cue, rewrite_s1_parens

# (neighbor before the chunk, neighbor after the chunk). Empty means the call
# is the whole string, so the string's own edges are the edges.
_Edge = tuple[str, str]
_Repl = str | Callable[[re.Match[str]], str]
_Replacement = tuple[tuple[re.Pattern[str], _Repl], ...]

# Every pattern below that scans for a closer is bounded, or is anchored so it
# starts once per run. Reply text is model output, and an unbounded scan from each
# of thousands of openers is quadratic. A label, tag, or aside longer than the bound
# is left as text, which is what a real reply never contains.
_LABEL = 200
_TAG = 300
_ASIDE = 500

# --- Blocks that are removed whole -------------------------------------------

# Chain-of-thought is scratch work. An unclosed block means the model was cut
# off, so the rest of the reply is scratch too. A closed span is removed by
# _erase_spans, which scans once. The tail patterns drop what is left after an
# opener that never closed.
_THOUGHT_TAIL_RE = re.compile(f"{THOUGHT_OPEN_RE.pattern}.*", re.IGNORECASE | re.DOTALL)
# Fenced code (backticks or tildes) is source, not speech. An unclosed fence of
# either kind swallows the rest of the reply.
_FENCE_OPEN_RE = re.compile(r"```|~~~")
_FENCE_CLOSERS = {"```": re.compile("```"), "~~~": re.compile("~~~")}
_FENCE_TAIL_RE = re.compile(r"(?:```|~~~).*", re.DOTALL)
_HTML_COMMENT_OPEN_RE = re.compile(r"<!--")
_HTML_COMMENT_CLOSE_RE = re.compile(r"-->")
_HTML_COMMENT_TAIL_RE = re.compile(r"<!--.*", re.DOTALL)
# <script>, <style>, and <ref> hold source or notes. The generic tag rule would
# keep their contents, so they are removed with the tags.
_HTML_BLOCK_NAMES = r"script|style|ref"
_HTML_BLOCK_TAIL_RE = re.compile(
    rf"<\s*(?:{_HTML_BLOCK_NAMES})\b[^>]{{0,{_TAG}}}>.*", re.IGNORECASE | re.DOTALL
)
HTML_BLOCK_OPEN_RE = re.compile(rf"<\s*({_HTML_BLOCK_NAMES})\b[^>]{{0,{_TAG}}}>", re.IGNORECASE)
HTML_BLOCK_CLOSE_RE = re.compile(rf"<\s*/\s*({_HTML_BLOCK_NAMES})\s*>", re.IGNORECASE)
_HTML_BLOCK_CLOSERS = {
    name: re.compile(rf"<\s*/\s*{name}\s*>", re.IGNORECASE) for name in _HTML_BLOCK_NAMES.split("|")
}
# CosyVoice stage directions. Phoneme tokens use a different shape and stay.
_STAGE_TOKEN_RE = re.compile(r"<\|(?:ACT|DELAY|CALL)\b[^|]{0,200}\|>", re.IGNORECASE)
# MOSS duration pause. Fish [pause] has no number and is an alias, not this.
_MOSS_PAUSE_RE = re.compile(r"\[pause\s+\d+(?:\.\d+)?s\]", re.IGNORECASE)
# TTSD speaker labels [S1] through [S5]. A Fish [cue] is words, not S plus a digit.
_TTSD_SPEAKER_RE = re.compile(r"\[S[1-5]\]")

# TTS only. Fish [cue] and <|phoneme|> tokens are not in this list.
_TTS_ERASE: _Replacement = (
    (_STAGE_TOKEN_RE, " "),
    (_MOSS_PAUSE_RE, " "),
    (_TTSD_SPEAKER_RE, " "),
)

# --- Links and URLs ----------------------------------------------------------

# The address is ASCII. Inside a parenthesis it stops at the first non-ASCII
# character, so CJK text after the address is the sentence, not the link.
_URL_ASCII = r"[A-Za-z0-9\-._~:/?#\[\]@!$&'*+=%,;]"
# Only a URL target makes a Markdown link. [happy](softly) is a cue plus an aside.
# A title after the URL belongs to the link.
# The address is possessive: it can only end at a character the class excludes,
# so giving characters back never helps and would cost a scan per position.
_MD_LINK_RE = re.compile(
    rf"\[([^\]]{{1,{_LABEL}}})\]\(\s*<?https?://"
    + _URL_ASCII
    + rf"++>?(?:\s+[\"'][^\"']{{0,{_LABEL}}}[\"'])?([^)]{{0,{_ASIDE}}})\)",
    re.IGNORECASE,
)
# A link whose closing parenthesis never arrived: keep the label, drop the URL.
_OPEN_MD_LINK_RE = re.compile(
    rf"\[([^\]]{{1,{_LABEL}}})\]\(\s*<?https?://" + _URL_ASCII + r"++[ \t]*",
    re.IGNORECASE,
)
_OPEN_URL_PAREN_RE = re.compile(
    r"\(\s*<?https?://" + _URL_ASCII + r"++[ \t]*",
    re.IGNORECASE,
)
# "(https://example.com)." The space before the parenthesis belongs to the
# link, so the stop attaches to the word before it. The lookbehind starts the
# match at the first space of a run.
_CLOSED_URL_PAREN_RE = re.compile(
    r"(?<![ \t])[ \t]*\(\s*<?https?://" + _URL_ASCII + rf"++>?([^)]{{0,{_ASIDE}}})\)",
    re.IGNORECASE,
)
# ![alt](url) is an image: keep the alt text, drop the bang with it.
_MD_IMAGE_RE = re.compile(rf"!\[([^\]]{{0,{_LABEL}}})\]\([^)\n]{{0,{_ASIDE}}}\)")
# A bare URL. The address is ASCII and ends at whitespace or a closer.
# A trailing period, question mark, or comma is the sentence. A comma, bang,
# or colon before a letter starts the next word. A colon before a digit is a
# port. The space before the address belongs to the link.
_URL_RE = re.compile(
    "(?<![ \t])[ \t]*https?://(?:[A-Za-z0-9\\-_~/?#\\[\\]@$&'*+=%]"
    "|\\.(?!\\s|$)"
    "|[,!?;:](?!\\s|$|[A-Za-z]))+"
    "(?:[,!:;](?=[A-Za-z]))?",
    re.IGNORECASE,
)
# <https://example.com> is one link, not a less-than sign plus a URL.
_MD_AUTOLINK_RE = re.compile(rf"<https?://[^>\s]{{1,{_ASIDE}}}>", re.IGNORECASE)
# [the door](softly) is the words. [happy](softly) is a cue and stays.
_MD_PLAIN_LINK_RE = re.compile(rf"\[([^\]]{{1,{_LABEL}}})\]\([^)\n]{{0,{_ASIDE}}}\)")
# [docs][ref] is the word "docs". [happy][whispering] is two cues. a[i][j] is
# an index. The spaced-label form is handled by _MD_REF_LINK_RE.
_MD_BARE_REF_RE = re.compile(rf"(?<!\w)\[([^\]]{{1,{_LABEL}}})\]\[([^\]]{{0,{_LABEL}}})\]")

# --- Line-level markdown -----------------------------------------------------

_MD_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+", re.MULTILINE)
# A list marker needs a space after it and sits at the line start: "-5" and
# "2 - 3" are not bullets.
_MD_LIST_RE = re.compile(r"(?m)^[ \t]{0,3}[-*_~+]+[ \t]+")
# A line of only ---, ***, or ___ is a rule, not speech.
_MD_RULE_RE = re.compile(r"(?m)^[ \t]{0,3}([-*_])(?:[ \t]*\1){2,}[ \t]*$")
# "> text" is a quote mark. "2 > 1" stays.
_MD_QUOTE_RE = re.compile(r"(?m)^[ \t]{0,3}>+[ \t]?")
# A task checkbox after the bullet. [x] would be read as a Fish cue.
_MD_TASK_RE = re.compile(r"(?m)^[ \t]{0,3}\[[ \t]*(?:[xX][ \t]*)?\][ \t]+")
# Emphasis marks. A mark between word characters is an identifier or a
# product (fish_audio, 5*5), and a mark spaced on both sides is an operator.
# A closer may follow a comma ("*Excited,*") or a "]" (`a[i]`).
_MD_WRAP_RE = re.compile(
    r"(?<=\w)[*_`~]{2,}(?=\w)"
    r"|(?<![\w*_`~])[*_`~]+(?=\w)"
    r"|(?<=\w)[*_`~]+(?!\w)"
    r"|(?<=\w[,.!?;:，！：])[*_`~]+(?!\w)"
    r"|(?<=\])[*_`~]+(?!\w)"
)
# ~~retracted~~ is a deletion. Dropping only the marks would speak the word.
_MD_STRIKE_RE = re.compile(r"~~[^~\n]+~~")
_MD_FOOTNOTE_RE = re.compile(rf"\[\^[^\]]{{1,{_LABEL}}}\]")
_MD_FOOTNOTE_DEF_RE = re.compile(rf"(?m)^[ \t]*\[\^[^\]]{{1,{_LABEL}}}\]:[ \t]*")
# [the docs][ref] is a reference link. The label must contain a space, or it
# could be a cue stack like [sad][whispering]. The lookahead finds that space
# first, so the label itself is scanned once.
_MD_REF_LINK_RE = re.compile(
    rf"\[(?=[^\]\s]{{0,{_LABEL}}}\s)([^\]]{{1,{_LABEL}}})\]\[[^\]]{{1,{_LABEL}}}\]"
)
_MD_REF_DEF_RE = re.compile(rf"(?m)^[ \t]{{0,3}}\[[^\]]{{1,{_LABEL}}}\]:[ \t]*\n?")
# Block tags are word boundaries. Inline tags (<sup>, <span>) are not.
# <|phoneme|> starts with a bar, so it is not a tag. <whisper> is a cue.
_MD_BREAK_RE = re.compile(
    r"<[ \t]*(?:/[ \t]*)?"
    r"(?:br|p|div|hr|li|tr|td|th|table|thead|tbody|tfoot|ul|ol|pre|details|summary"
    r"|h[1-6]|blockquote|section|article|header|footer|nav|figure|figcaption)\b"
    rf"[^>\n]{{0,{_TAG}}}>",
    re.IGNORECASE,
)
_MD_HTML_RE = re.compile(rf"<(?!/?\s*whisper\b)/?[A-Za-z][^>\n]{{0,{_TAG}}}>", re.IGNORECASE)
HTML_ENTITY_RE = re.compile(r"&(#x[0-9A-Fa-f]+|#\d+|[A-Za-z]+);")
# A separator row is pipes and dashes. A line needs a pipe, so a prose hyphen stays.
_MD_TABLE_SEP_RE = re.compile(r"(?m)^(?=[ \t|:-]*\|)[ \t|:-]+$")
_MD_TABLE_ROW_RE = re.compile(r"(?m)^[ \t]*\|(.+\|.+)[ \t]*$")
_HTML_NAMED = {"amp": "&", "lt": "<", "gt": ">", "quot": '"', "apos": "'", "nbsp": " "}

# --- Asides ------------------------------------------------------------------

# Stage asides. Runs after S1 (happy) becomes [happy], so a Fish paren tag
# survives. A digit means a phone number or code, so (555) stays. The space
# before the aside goes with it, so removal leaves a single gap.
_PARENS_RE = re.compile(rf"(?<![ \t])[ \t]*\([^)\d]{{0,{_ASIDE}}}\)")
_FW_PARENS_RE = re.compile(rf"（[^）\d]{{0,{_ASIDE}}}）")


def _edge_char(match: re.Match[str], *, end: bool, edge: _Edge) -> str:
    if end:
        nxt = match.string[match.end() : match.end() + 1]
        return nxt or edge[1][:1]
    if match.start():
        return match.string[match.start() - 1]
    return edge[0][:1]


def _with_word_gap(match: re.Match[str], body: str, edge: _Edge) -> str:
    # A replacement must not glue to a word on either side of it.
    if _edge_char(match, end=False, edge=edge).isalnum():
        body = f" {body}"
    if _edge_char(match, end=True, edge=edge).isalnum():
        body = f"{body} "
    return body


def _keep_link_label(match: re.Match[str], edge: _Edge) -> str:
    return _with_word_gap(match, f"{match.group(1)}{match.group(2) or ''}", edge)


def _keep_image_alt(match: re.Match[str], edge: _Edge) -> str:
    return _with_word_gap(match, match.group(1), edge)


def _plain_md_link(match: re.Match[str], edge: _Edge) -> str:
    label = match.group(1).strip()
    if is_paren_cue(label):
        return match.group(0)
    return _with_word_gap(match, label, edge)


def _ref_link(match: re.Match[str], edge: _Edge) -> str:
    # A letter right before the bracket makes this an index (a[i][j]), even
    # when the chunk starts at the bracket and the letter is the neighbor.
    if _edge_char(match, end=False, edge=edge).isalnum():
        return match.group(0)
    label = match.group(1).strip()
    ref = match.group(2).strip()
    if is_paren_cue(label) and (not ref or is_paren_cue(ref)):
        return match.group(0)
    if is_paren_cue(label):
        return f"[{label}]"
    return _with_word_gap(match, label, edge)


def _unwrap_mark(match: re.Match[str], edge: _Edge) -> str:
    # Two or more marks between words are bold: "door**today" is two words.
    # One mark between words stays: 5*5 and fish_audio are single tokens.
    before = _edge_char(match, end=False, edge=edge)
    after = _edge_char(match, end=True, edge=edge)
    if before.isalnum() and after.isalnum():
        if len(match.group(0)) == 1:
            return match.group(0)
        return " "
    return ""


def _markdown_subs(edge: _Edge) -> _Replacement:
    # Link text is kept and the URL is dropped. Parens run last, after S1 tags
    # have become brackets.
    return (
        (_MD_IMAGE_RE, partial(_keep_image_alt, edge=edge)),
        (_MD_LINK_RE, partial(_keep_link_label, edge=edge)),
        (_CLOSED_URL_PAREN_RE, r"\1"),
        (_OPEN_MD_LINK_RE, r"\1 "),
        (_OPEN_URL_PAREN_RE, ""),
        (_MD_AUTOLINK_RE, " "),
        (_URL_RE, " "),
        (_MD_PLAIN_LINK_RE, partial(_plain_md_link, edge=edge)),
        (_MD_BARE_REF_RE, partial(_ref_link, edge=edge)),
        (_PARENS_RE, ""),
        (_FW_PARENS_RE, " "),
    )


def _erase_spans(
    text: str,
    open_re: re.Pattern[str],
    close_for: Callable[[re.Match[str]], re.Pattern[str]],
) -> str:
    """Replace each opener-to-closer span with a space, scanning the text once.

    The first opener that has no closer ends the scan. A later opener cannot have
    one either, and a ``*_TAIL_RE`` pattern drops that unclosed tail afterwards.
    A lazy ``.*?`` pattern would rescan the rest of the text from every opener.
    """
    out: list[str] = []
    pos = 0
    while True:
        opener = open_re.search(text, pos)
        if opener is None:
            break
        closer = close_for(opener).search(text, opener.end())
        if closer is None:
            break
        out.append(text[pos : opener.start()])
        out.append(" ")
        pos = closer.end()
    out.append(text[pos:])
    return "".join(out)


def _replace(text: str, pairs: _Replacement) -> str:
    for pattern, repl in pairs:
        text = pattern.sub(repl, text)
    return text


def _line_sub(pattern: re.Pattern[str], repl: _Repl, text: str, *, line_start: bool) -> str:
    # A chunk that continues a line has no heading or bullet at index 0, but a
    # marker after a newline inside the chunk still is one.
    if line_start:
        return pattern.sub(repl, text)

    def keep_start(match: re.Match[str]) -> str:
        if match.start() == 0:
            return match.group(0)
        if callable(repl):
            return repl(match)
        return match.expand(repl)

    return pattern.sub(keep_start, text)


def html_char(match: re.Match[str]) -> str:
    """Decode one ``HTML_ENTITY_RE`` match to its character.

    Parameters
    ----------
    match : re.Match
        A match of ``HTML_ENTITY_RE``.

    Returns
    -------
    str
        The decoded character. A tab, newline, or carriage return becomes a
        space or newline, a non-breaking space becomes a space, and a control
        character or surrogate becomes empty. An unknown name is left as is.
    """
    body = match.group(1)
    named = _HTML_NAMED.get(body.lower())
    if named is not None:
        return named
    try:
        code = int("0" + body[1:], 0) if body[1:2].lower() == "x" else int(body[1:], 10)
        char = chr(code)
    except (ValueError, OverflowError):
        return match.group(0)
    # Whitespace controls are word breaks, so deleting them would join words.
    if char in "\t\n\r":
        return "\n" if char != "\t" else " "
    if char == "\u00a0":
        return " "
    # Another control, or a surrogate that UTF-8 cannot encode, is not speech.
    if ord(char) < 32 or 0xD800 <= ord(char) <= 0xDFFF:
        return ""
    return char


def _table_cells(match: re.Match[str]) -> str:
    cells = [cell.strip() for cell in match.group(1).split("|")]
    return " ".join(cell for cell in cells if cell)


def _unwrap_edge_marks(text: str, *, line_start: bool, before: str, after: str) -> str:
    # A chunk of only emphasis marks: decide from its neighbors whether it is
    # an opener, a closer, or a literal operator such as "2*3".
    marks = "*_`~"
    prev = before[-1:]
    nxt = after[:1]
    # "]" closes an index, so a backtick after `a[i]` is still a closer.
    attached = prev.isalnum() or prev == "]"
    if text and all(ch in marks for ch in text):
        if len(text) == 1 and attached and nxt.isalnum():
            return text
        if attached and nxt.isalnum():
            return " "
        if nxt.isalnum() and not attached:
            return ""
        if attached and not nxt.isalnum():
            return ""
        if line_start and not prev.isalnum() and nxt.isspace():
            return ""
        return text
    # A closer plus the word space after "]": drop one mark only. "~~" and
    # "```" stay, so a strike and a fence are still one span.
    if (
        attached
        and text[:1] in marks
        and text[1:2] not in marks
        and text[1:2]
        and not text[1:2].isalnum()
    ):
        return text[1:]
    return text


def _strip_marks(text: str, *, line_start: bool, edge: _Edge) -> str:
    # Entities decode first, so "&lt;br&gt;" is handled as the tag it spells.
    stripped = HTML_ENTITY_RE.sub(html_char, text)
    stripped = _line_sub(_MD_TABLE_SEP_RE, "", stripped, line_start=line_start)
    stripped = _line_sub(_MD_TABLE_ROW_RE, _table_cells, stripped, line_start=line_start)
    stripped = _MD_STRIKE_RE.sub(" ", stripped)
    stripped = _line_sub(_MD_FOOTNOTE_DEF_RE, "", stripped, line_start=line_start)
    stripped = _MD_FOOTNOTE_RE.sub(" ", stripped)
    stripped = _MD_REF_LINK_RE.sub(r"\1", stripped)
    stripped = _line_sub(_MD_REF_DEF_RE, "", stripped, line_start=line_start)
    stripped = _MD_BREAK_RE.sub(" ", stripped)
    stripped = _MD_HTML_RE.sub("", stripped)
    stripped = _line_sub(_MD_LIST_RE, "", stripped, line_start=line_start)
    stripped = _line_sub(_MD_TASK_RE, "", stripped, line_start=line_start)
    stripped = _line_sub(_MD_RULE_RE, "", stripped, line_start=line_start)
    stripped = _line_sub(_MD_QUOTE_RE, "", stripped, line_start=line_start)
    return _MD_WRAP_RE.sub(partial(_unwrap_mark, edge=edge), stripped)


def _strip_markdownish(text: str, *, line_start: bool, edge: _Edge) -> str:
    cleaned = rewrite_s1_parens(text)
    cleaned = _line_sub(_MD_HEADING_RE, "", cleaned, line_start=line_start)
    cleaned = _replace(cleaned, _markdown_subs(edge))
    # <|token|> spans are kept verbatim (phonemes). Only the text between them
    # is stripped, and only the first piece can be at the start of a line.
    parts: list[str] = []
    last = 0
    for m in ANGLE_TOKEN_RE.finditer(cleaned):
        parts.append(
            _strip_marks(cleaned[last : m.start()], line_start=line_start and last == 0, edge=edge)
        )
        parts.append(m.group(0))
        last = m.end()
    parts.append(_strip_marks(cleaned[last:], line_start=line_start and last == 0, edge=edge))
    return "".join(parts)


# Trailing spaces before a newline. Horizontal runs collapse; newlines stay.
_TRAIL_SPACE_RE = re.compile(r"(?<![ \t])[ \t]+\n")


def _tidy_breaks(text: str, *, before: str = "", after: str = "") -> str:
    text = SPACE_BEFORE_STOP_RE.sub(r"\1", text)
    text = BREAKS_RE.sub("\n\n", text)
    if not before and not after:
        return text.strip()
    # A chunk that has a neighbor keeps the one edge space a replacement
    # inserted, or the next word would join. Spaces strip; a decoded line
    # break stays.
    lead = text[:1] == " "
    trail = text[-1:] == " "
    body = text.strip(" ")
    if not body and (lead or trail):
        return " "
    if lead and body[:1] != " ":
        body = f" {body}"
    if trail and not body.endswith(" "):
        body = f"{body} "
    return body


def unclosed_span_start(text: str, open_ch: str, close_ch: str) -> int | None:
    """Return where the outermost unclosed span starts.

    Parameters
    ----------
    text : str
        Text that may contain nested pairs of ``open_ch`` and ``close_ch``.
    open_ch : str
        One opening character.
    close_ch : str
        The matching closing character.

    Returns
    -------
    int or None
        Index of the opener that never closed. None when every pair closes.
    """
    depth = 0
    start: int | None = None
    for index, ch in enumerate(text):
        if ch == open_ch:
            if depth == 0:
                start = index
            depth += 1
        elif ch == close_ch and depth:
            depth -= 1
            if depth == 0:
                start = None
    return start


def _drop_unclosed(
    text: str,
    open_ch: str,
    close_ch: str,
    *,
    keep_digit: bool,
    continued: bool,
) -> str:
    # A cutoff never sends the closer, so the unfinished tail would be spoken.
    start = unclosed_span_start(text, open_ch, close_ch)
    if start is None:
        return text
    tail = text[start:]
    # A digit means a phone number or code, which stays.
    if keep_digit and any(ch.isdigit() for ch in tail):
        return text
    inner = tail[1:].strip()
    # An unclosed paren that opens the reply and runs four or more words is
    # the reply itself. A cutoff after other words ("Hello (she smiles") is an aside.
    if not continued and not text[:start].strip() and len(inner.split()) >= 4:
        return inner
    return text[:start]


def _unwrap_spoken_paren(text: str, *, continued: bool) -> str:
    # A reply wholly wrapped in one pair of parens with four or more words is
    # speech, not a stage note. A short aside such as "(she smiles.)" still goes.
    if continued:
        return text
    stripped = text.strip()
    if len(stripped) < 2:
        return text
    open_ch = stripped[0]
    close_ch = stripped[-1]
    if (open_ch, close_ch) not in (("(", ")"), ("（", "）")):
        return text
    inner = stripped[1:-1].strip()
    if open_ch in inner or close_ch in inner or any(ch.isdigit() for ch in inner):
        return text
    if len(inner.split()) < 4:
        return text
    return inner


def scrub_tts(
    text: str,
    *,
    line_start: bool = True,
    continued: bool = False,
    before: str = "",
    after: str = "",
) -> str:
    """Strip thoughts, markdown, and stage markup. Keep Fish cues.

    Parameters
    ----------
    text : str
        One reply, or one stable chunk of a streamed reply.
    line_start : bool, optional
        True when ``text`` begins a line. Default True, the whole-string call.
        A mid-line star is not a bullet.
    continued : bool, optional
        True when text was already sent before this chunk. Default False.
        A leading parenthesis is then an aside, not the whole reply.
    before : str, optional
        One character already spoken before this chunk. Default empty.
    after : str, optional
        One character still held after this chunk. Default empty.

    Returns
    -------
    str
        Spoken text. The no-argument call is the whole string, stripped.
        A neighbor character keeps one edge space the replacement inserted.
    """
    if not text:
        return ""
    edge = (before[-1:], after[:1])
    text = _unwrap_edge_marks(text, line_start=line_start, before=before, after=after)
    if not text:
        return ""
    cleaned = _unwrap_spoken_paren(plain_breaks(text), continued=continued)
    cleaned = _erase_spans(cleaned, THOUGHT_OPEN_RE, lambda _opener: THOUGHT_CLOSE_RE)
    cleaned = _replace(cleaned, _TTS_ERASE)
    cleaned = _THOUGHT_TAIL_RE.sub("", cleaned)
    cleaned = _erase_spans(cleaned, _FENCE_OPEN_RE, lambda opener: _FENCE_CLOSERS[opener.group(0)])
    cleaned = _FENCE_TAIL_RE.sub("", cleaned)
    # Blocks run after fences, so a <!-- or <script> inside a code block does
    # not eat the reply. Script runs before comments, so a comment inside a
    # script does not either.
    cleaned = _erase_spans(
        cleaned,
        HTML_BLOCK_OPEN_RE,
        lambda opener: _HTML_BLOCK_CLOSERS[opener.group(1).lower()],
    )
    cleaned = _HTML_BLOCK_TAIL_RE.sub("", cleaned)
    cleaned = _erase_spans(cleaned, _HTML_COMMENT_OPEN_RE, lambda _opener: _HTML_COMMENT_CLOSE_RE)
    cleaned = _HTML_COMMENT_TAIL_RE.sub("", cleaned)
    cleaned = _strip_markdownish(cleaned, line_start=line_start, edge=edge)
    cleaned = _drop_unclosed(cleaned, "(", ")", keep_digit=True, continued=continued)
    cleaned = _drop_unclosed(cleaned, "（", "）", keep_digit=True, continued=continued)
    cleaned = _drop_unclosed(cleaned, "[", "]", keep_digit=False, continued=continued)
    cleaned = _TRAIL_SPACE_RE.sub("\n", cleaned)
    # A lone surrogate cannot be encoded as UTF-8, and httpx would fail the request.
    return utf8_text(_tidy_breaks(cleaned, before=before, after=after))
