"""TTS scrubbers and the holds that keep an unfinished span out of a send."""

from __future__ import annotations

import contextvars
import re
from collections.abc import Callable

from fish_audio_suite_kit.cues import (
    is_paren_cue,
    mood_lead_hold_at,
    rewrite_s1_parens,
    spoken_mood_span,
)
from fish_audio_suite_kit.cuts import next_tts_cut

# A quote whose only content is a [cue] in this window is not speech.
_CUE_LOOKAHEAD = 24

# Chain-of-thought blocks. Speaking them reads the model's scratch work aloud.
# A missing closer still means the rest of the reply is scratch: the model was
# cut off, and that tail must not be spoken.
_THOUGHTS_RE = re.compile(
    r"<\s*(?:Thoughts?|thinking|reasoning|think)\s*>.*?"
    r"<\s*/\s*(?:Thoughts?|thinking|reasoning|think)\s*>",
    re.IGNORECASE | re.DOTALL,
)
_THOUGHT_TAIL_RE = re.compile(
    r"<\s*(?:Thoughts?|thinking|reasoning|think)\s*>.*",
    re.IGNORECASE | re.DOTALL,
)
# A fenced block is source, not speech. The closing ``` after punctuation
# is not a word mark, so it was left in the sentence and spoken.
# A tilde fence is the same kind of block. Leaving ~~~ spoke the code.
_FENCE_RE = re.compile(r"```.*?```|~~~.*?~~~", re.DOTALL)
_FENCE_TAIL_RE = re.compile(r"```.*", re.DOTALL)
# <!-- note --> is not speech. The tag rule requires a letter after "<",
# so the comment, and the words inside it, were spoken.
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_HTML_COMMENT_TAIL_RE = re.compile(r"<!--.*", re.DOTALL)
# <script> and <style> are source. The tag rule deletes the tags and
# leaves the code, so the speaker reads the program. A reference note
# was spoken and glued on, so "friend.secret" was one word.
_HTML_BLOCK_RE = re.compile(
    r"<\s*(script|style|ref)\b[^>]*>.*?</\s*\1\s*>",
    re.IGNORECASE | re.DOTALL,
)
_HTML_BLOCK_TAIL_RE = re.compile(
    r"<\s*(?:script|style|ref)\b[^>]*>.*",
    re.IGNORECASE | re.DOTALL,
)
# ASR diarization token. Not a phoneme, so the transcript drops it.
# CosyVoice stage directions. Phoneme tokens use a different shape and stay.
_STAGE_TOKEN_RE = re.compile(r"<\|(?:ACT|DELAY|CALL)\b[^|]*\|>", re.IGNORECASE)
# Any <|...|> token. TTS keeps these (phonemes). ASR replaces them with a space.
_ANGLE_TOKEN_RE = re.compile(r"<\|[^|>]*\|>")
# MOSS duration pause. Fish [pause] has no number and is an alias, not this.
_MOSS_PAUSE_RE = re.compile(r"\[pause\s+\d+(?:\.\d+)?s\]", re.IGNORECASE)
# TTSD speaker labels [S1] through [S5]. A Fish [cue] is words, not S plus a digit.
_TTSD_SPEAKER_RE = re.compile(r"\[S[1-5]\]")
# Markdown the model copies from a chat reply. Headings and URLs are not speech.
_MD_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+", re.MULTILINE)
# Only a URL target is a Markdown link. [happy](softly) is a cue plus an aside.
# A title after the URL is still the link. Leaving the brackets made Fish
# treat "the docs" as a cue, so that label was not spoken.
# Inside a parenthesis the address stops at the first non-ASCII character.
# Chinese after the address used to be deleted with the link.
_URL_ASCII = r"[A-Za-z0-9\-._~:/?#\[\]@!$&'*+=%,!;]"
_MD_LINK_RE = re.compile(
    r"\[([^\]]+)\]\(\s*<?https?://" + _URL_ASCII + r"+>?(?:\s+[\"'][^\"']*[\"'])?([^)]*)\)",
    re.IGNORECASE,
)
# The closer never arrived, and the URL was removed first. The words after
# it were then an unclosed paren, so "today friend" was not spoken.
_OPEN_MD_LINK_RE = re.compile(
    r"\[([^\]]+)\]\(\s*<?https?://" + _URL_ASCII + r"+[ \t]*",
    re.IGNORECASE,
)
_OPEN_URL_PAREN_RE = re.compile(
    r"\(\s*<?https?://" + _URL_ASCII + r"+[ \t]*",
    re.IGNORECASE,
)
# "(https://example.com)." left "door ." so the period was spoken apart
# from the word. The space before the parenthesis belongs to the link.
# A non-ASCII tail inside the parenthesis is the sentence, not the address.
_CLOSED_URL_PAREN_RE = re.compile(
    r"[ \t]*\(\s*<?https?://" + _URL_ASCII + r"+>?([^)]*)\)",
    re.IGNORECASE,
)
# ![alt](url) is an image. The link rule keeps the alt and leaves the bang,
# so the speaker says "exclamation" before the description.
_MD_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)\n]*\)")
# A closing paren ends an aside. Including it in the URL leaves "(" and the
# following sentence is then treated as an unclosed span.
# ">" closes a markdown autolink. Including it leaves the opening "<", which
# is then spoken and glues onto the next word.
# A period, question mark, or comma after the address is the sentence, not
# the link. Eating it left "Excited" as a spoken word. The address is ASCII.
# A fullwidth comma or the next Chinese word is not part of the link, or
# every word after the address was dropped.
# The space before the address belongs to the link. Leaving it spoke
# "See ." and the period was its own word.
# A comma, bang, or colon before a letter is the next word. Treating it
# like a path deleted "today" from "example.com/a!today friend".
# A colon before a digit is a port, so ":8080" stays in the address.
_URL_RE = re.compile(
    "[ \t]*https?://(?:[A-Za-z0-9\\-_~/?#\\[\\]@$&'*+=%]"
    "|\\.(?!\\s|$)"
    "|[,!?;:](?!\\s|$|[A-Za-z]))+"
    "(?:[,!:;](?=[A-Za-z]))?",
    re.IGNORECASE,
)
# <https://example.com> is one link, not a less-than sign plus a URL.
_MD_AUTOLINK_RE = re.compile(r"<https?://[^>\s]+>", re.IGNORECASE)
# A list marker at the start of a line. "-", "+", and "*" are bullets.
# "2 - 3" and "-5" are not: the marker has to be the start of the line
# and a space has to follow it.
_MD_LIST_RE = re.compile(r"(?m)^[ \t]{0,3}[-*_~+]+[ \t]+")
# A line of only ---, ***, or ___ is a break. Speaking it reads "dash dash dash".
# "wait---then" is not a line of dashes, so those marks stay.
_MD_RULE_RE = re.compile(r"(?m)^[ \t]{0,3}([-*_])(?:[ \t]*\1){2,}[ \t]*$")
# "> Open the door" is a quote. The mark is not a word. "2 > 1" stays.
_MD_QUOTE_RE = re.compile(r"(?m)^[ \t]{0,3}>+[ \t]?")
# A task checkbox left after the bullet is a Fish cue. [x] would change
# the voice, and [] is not a word. [happy] is a real cue, so it stays.
_MD_TASK_RE = re.compile(r"(?m)^[ \t]{0,3}\[[ \t]*[xX]?[ \t]*\][ \t]+")
# Emphasis around a word. A mark between word characters is an identifier
# or a product, so fish_audio and 5*5 stay speakable. A mark with a space on
# both sides is an operator, so it stays too.
# The closer may sit after the comma: "*Excited,*" is one cue, not a spoken star.
# A closer after "]" is still a mark. "`a[i]`" used to keep the last
# backtick, so the speaker read it after the index.
_MD_WRAP_RE = re.compile(
    r"(?<=\w)[*_`~]{2,}(?=\w)"
    r"|(?<!\w)[*_`~]+(?=\w)"
    r"|(?<=\w)[*_`~]+(?!\w)"
    r"|(?<=\w[,.!?;:\uff0c\uff01\uff1a])[*_`~]+(?!\w)"
    r"|(?<=\])[*_`~]+(?!\w)"
)


def _unwrap_edge_marks(
    text: str,
    *,
    line_start: bool,
    before: str,
    after: str,
) -> str:
    # A chunk of only marks. "2*3" keeps one star. An opener before a word,
    # and a closer after one, are the same marks the whole string drops.
    marks = "*_`~"
    prev = before[-1:]
    nxt = after[:1]
    # "]" closes an index. The backtick after `a[i]` is still a closer.
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
    # "` " after "]" is the closer plus the word space. One mark only.
    # "~~" and "```" stay, so a strike and a fence are still one span.
    if (
        attached
        and text[:1] in marks
        and text[1:2] not in marks
        and text[1:2]
        and not text[1:2].isalnum()
    ):
        return text[1:]
    return text


def _unwrap_mark(match: re.Match[str]) -> str:
    # "door**today" is bold, not one word. Removing the marks with no
    # space spoke "doortoday". One mark stays, so 5*5 and fish_audio do.
    before = _edge_char(match, end=False)
    after = _edge_char(match, end=True)
    # "2*3" is one mark between words. "**" in the same place is bold.
    if before.isalnum() and after.isalnum():
        if len(match.group(0)) == 1:
            return match.group(0)
        return " "
    return ""


# ~~retracted~~ is a deletion. Stripping only the marks speaks the word the
# model took back: "meant ~~Tuesday~~ Wednesday" was spoken as both days.
_MD_STRIKE_RE = re.compile(r"~~[^~\n]+~~")
# [^1] is a footnote marker. The speaker reads the brackets and the caret.
_MD_FOOTNOTE_RE = re.compile(r"\[\^[^\]]+\]")
# "[^1]: the note" is the note. Leaving the colon spoke "colon".
_MD_FOOTNOTE_DEF_RE = re.compile(r"(?m)^[ \t]*\[\^[^\]]+\]:[ \t]*")
# [the docs][ref] is a reference link. [sad][whispering] is a cue stack, so
# this only matches when the label itself contains a space.
_MD_REF_LINK_RE = re.compile(r"\[([^\]]*\s[^\]]*)\]\[[^\]]+\]")
_MD_REF_DEF_RE = re.compile(r"(?m)^[ \t]{0,3}\[[^\]]+\]:[ \t]*\n?")
# <b> and <br> are markup. <|phoneme|> starts with a bar, so it is not a tag.
# <whisper> is a cue, rewritten later. Stripping it here drops [whispering].
# A block tag is a word boundary. Deleting <p> or <hr> spoke "doorthen".
# <sup> and <span> are inline, so "5th" stays one word.
_MD_BREAK_RE = re.compile(
    r"<[ \t]*/?[ \t]*(?:br|p|div|hr|li|tr|td|th|table|thead|tbody|tfoot|ul|ol|pre|details|summary"
    r"|h[1-6]|blockquote|section|article|header|footer|nav|figure|figcaption)\b[^>\n]*>",
    re.IGNORECASE,
)
_MD_HTML_RE = re.compile(r"<(?!/?\s*whisper\b)/?[A-Za-z][^>\n]*>", re.IGNORECASE)
_HTML_ENTITY_RE = re.compile(r"&(#x[0-9A-Fa-f]+|#\d+|[A-Za-z]+);")
# A separator row is pipes and dashes. Speaking it reads "dash dash dash".
# A line needs a pipe, so a prose hyphen is left alone.
_MD_TABLE_SEP_RE = re.compile(r"(?m)^(?=[ \t|:-]*\|)[ \t|:-]+$")
_MD_TABLE_ROW_RE = re.compile(r"(?m)^[ \t]*\|(.+\|.+)[ \t]*$")
_HTML_NAMED = {"amp": "&", "lt": "<", "gt": ">", "quot": '"', "apos": "'", "nbsp": " "}
# Stage asides. Runs after S1 (happy) is rewritten, so a Fish paren tag survives.
# A digit is a phone number or a code, not a stage note: (555) stays speakable.
_PARENS_RE = re.compile(r"\([^)\d]*\)")
# A fullwidth parenthesis is the same aside. Leaving it spoke the stage
# direction, and the closer glued onto the next word.
_FW_PARENS_RE = re.compile(r"（[^）\d]*）")
# Optional [cue] glued to a quote, so dialogue extraction does not drop the cue.
_LEAD_CUE = r"\[[^\]\n]{1,80}\]\s*"
# Cue with an empty body too, so "[ ]" alone still counts as no speech.
_EMPTY_CUE_RE = re.compile(r"\[[^\]\n]{0,80}\]")
_SPACE_RE = re.compile(r"\s+")
_LEAD_CUE_RE = re.compile(rf"({_LEAD_CUE})")
# Closed quotes, straight or curly. The cue in front is optional.
# A wrapped quote still closes on the next quote mark. Stopping at the
# newline left the stage direction in the spoken text.
# A digit before the mark is inches (5"), not dialogue. Pairing it with the
# next quote dropped the spoken line.
# Guillemets and corner brackets are quotes too. Leaving them out made
# "She smiles. «Hello there.»" look like narration, so the reply was silence.
# Either order is a quote. French uses «…» and German uses »…«.
# A one-way class left the other order as narration, so the line was silence.
_ANGLE_QUOTES = "\u00ab\u00bb\u2039\u203a\u3008\u3009\u300a\u300b"
_QUOTE_OPEN = '"\u201c\u300c\u300e' + _ANGLE_QUOTES
_QUOTE_CLOSE = '"\u201d\u300d\u300f' + _ANGLE_QUOTES
_DIALOGUE_RE = re.compile(
    rf"(?:{_LEAD_CUE})?"
    rf"(?<!\d)[{_QUOTE_OPEN}]([^{_QUOTE_CLOSE}]+)[{_QUOTE_CLOSE}]"
)
# A quote opened on this line and not closed yet. Streaming replies arrive this way.
_OPEN_DIALOGUE_RE = re.compile(
    rf"(?:{_LEAD_CUE})?"
    rf"[{_QUOTE_OPEN}](.+)$",
    re.DOTALL,
)
# Unquoted stage direction: a pronoun or article plus a body verb. Quoted speech
# is already kept, so this only drops narration that has no quotes.
_NARRATION_RE = re.compile(r"^(She|He|They|Her|His|The|A|An)\b.*", re.IGNORECASE)
_NARRATION_VERB_RE = re.compile(
    r"\b(shifts|leans|smiles|laughs|settles|tilts|watches|murmurs|"
    r"reaches|moves|looks|dropping|rustle|stretches)\b",
    re.IGNORECASE,
)


# Straight, curly, guillemet, and corner quotes. Used to tell narration from dialogue.
_QUOTE_RE = re.compile(r'["“”«»「」『』‹›〈〉《》]')
# Closers stripped from an unclosed quote before the speakable check.
_CLOSE_QUOTES = '"”」』' + _ANGLE_QUOTES


# Hiragana/katakana, CJK ext A, unified ideographs, compatibility, Hangul.
# A single CJK character is too thin; two or more count as a real sentence.
_CJK_RANGES = (
    range(0x3040, 0x3100),
    range(0x3400, 0x4DC0),
    range(0x4E00, 0xA000),
    range(0xF900, 0xFB00),
    range(0xAC00, 0xD7B0),
)


def _is_cjk(ch: str) -> bool:
    code = ord(ch)
    return any(code in span for span in _CJK_RANGES)


def _cjk_latin_counts(s: str) -> tuple[int, int]:
    cjk = latin = 0
    for ch in s:
        if _is_cjk(ch):
            cjk += 1
        elif ch.isalpha():
            # "Été" and "Привет" are speech. Counting only ASCII dropped them.
            latin += 1
    return cjk, latin


def _too_thin(s: str, *, min_latin: int) -> bool:
    stripped = s.strip()
    if not stripped:
        return True
    cjk, latin = _cjk_latin_counts(stripped)
    if cjk >= 2:
        return False
    # "100" and "3.14" are answers. The letter floor still drops "ok" and "hi".
    if sum(ch.isdigit() for ch in stripped) >= 3:
        return False
    if len(stripped) <= 2:
        return True
    return latin < min_latin


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
    if not _NARRATION_RE.match(piece):
        return False
    return bool(_NARRATION_VERB_RE.search(piece))


def _looks_like_narration(text: str) -> bool:
    s = text.strip()
    if not s:
        return False
    if _QUOTE_RE.search(s):
        return False
    # "She smiles. It's open today friend." matched once, so the sentence
    # after the stage direction was silence.
    pieces = _stage_pieces(s)
    return bool(pieces) and all(_one_stage_direction(piece) for piece in pieces)


_Repl = str | Callable[[re.Match[str]], str]
_Replacement = tuple[tuple[re.Pattern[str], _Repl], ...]
# TTS only. Fish [cue] and <|phoneme|> tokens are not in this list.
_TTS_ERASE: _Replacement = (
    (_THOUGHTS_RE, " "),
    (_STAGE_TOKEN_RE, " "),
    (_MOSS_PAUSE_RE, " "),
    (_TTSD_SPEAKER_RE, " "),
)
# Optional. The proxy leaves speaker labels when the client asks to keep them.
# [the door](softly) is the words. [happy](softly) is a cue. Leaving the
# brackets made Fish skip "the door".
_MD_PLAIN_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)\n]*\)")


def _spoken_cue(label: str) -> bool:
    word = label.strip().lower()
    return is_paren_cue(word)


# One neighbor character on each side of a chunk. Empty means the call is
# the whole string, so the edge is the string itself.
_SCRUB_EDGE: contextvars.ContextVar[tuple[str, str]] = contextvars.ContextVar(
    "scrub_edge",
    default=("", ""),
)


def _edge_char(match: re.Match[str], *, end: bool) -> str:
    if end:
        nxt = match.string[match.end() : match.end() + 1]
        if nxt:
            return nxt
        return _SCRUB_EDGE.get()[1][:1]
    if match.start():
        return match.string[match.start() - 1]
    return _SCRUB_EDGE.get()[0][:1]


def _with_word_gap(match: re.Match[str], body: str) -> str:
    # "[the docs](https://example.com/a)today" has no space. The label
    # then joins the next word, so Fish says "docstoday".
    # "door![the cat](photo.png)" joins the other way, so Fish says "doorthe".
    if _edge_char(match, end=False).isalnum():
        body = f" {body}"
    if _edge_char(match, end=True).isalnum():
        body = f"{body} "
    return body


def _keep_link_label(match: re.Match[str]) -> str:
    return _with_word_gap(match, f"{match.group(1)}{match.group(2) or ''}")


def _keep_image_alt(match: re.Match[str]) -> str:
    return _with_word_gap(match, match.group(1))


def _plain_md_link(match: re.Match[str]) -> str:
    label = match.group(1).strip()
    if _spoken_cue(label):
        return match.group(0)
    return _with_word_gap(match, label)


# [docs][ref] is the words "docs". [happy][whispering] is two cues.
# a[i][j] is an index. Unwrapping that joined the letters.
# A label with a space is already unwrapped later. This one is the
# single-word label the spaced pattern leaves as a cue.
_MD_BARE_REF_RE = re.compile(r"(?<!\w)\[([^\]]+)\]\[([^\]]*)\]")


def _ref_link(match: re.Match[str]) -> str:
    # a[i][j] is an index. The chunk can start at the bracket, so the
    # letter is the neighbor, not a character inside this string.
    if _edge_char(match, end=False).isalnum():
        return match.group(0)
    label = match.group(1).strip()
    ref = match.group(2).strip()
    if _spoken_cue(label) and (not ref or _spoken_cue(ref)):
        return match.group(0)
    if _spoken_cue(label):
        return f"[{label}]"
    return label


# Link text is kept; the URL is not. Parens run after S1 tags are brackets.
_MARKDOWN_SUBS: _Replacement = (
    (_MD_HEADING_RE, ""),
    (_MD_IMAGE_RE, _keep_image_alt),
    (_MD_LINK_RE, _keep_link_label),
    (_CLOSED_URL_PAREN_RE, r"\1"),
    (_OPEN_MD_LINK_RE, r"\1 "),
    (_OPEN_URL_PAREN_RE, ""),
    (_MD_AUTOLINK_RE, " "),
    (_URL_RE, " "),
    (_MD_PLAIN_LINK_RE, _plain_md_link),
    (_MD_BARE_REF_RE, _ref_link),
    (_PARENS_RE, ""),
    (_FW_PARENS_RE, " "),
)


def _replace(text: str, pairs: _Replacement) -> str:
    for pattern, repl in pairs:
        text = pattern.sub(repl, text)
    return text


def _line_sub(pattern: re.Pattern[str], repl: _Repl, text: str, *, line_start: bool) -> str:
    # A chunk that continues a line is not a heading or a bullet. Skipping
    # only a match at index 0 leaves a marker after a newline inside the chunk.
    if line_start:
        return pattern.sub(repl, text)

    def keep_start(match: re.Match[str]) -> str:
        if match.start() == 0:
            return match.group(0)
        if callable(repl):
            return repl(match)
        return match.expand(repl)

    return pattern.sub(keep_start, text)


def _html_char(match: re.Match[str]) -> str:
    body = match.group(1)
    named = _HTML_NAMED.get(body.lower())
    if named is not None:
        return named
    try:
        code = int("0" + body[1:], 0) if body[1:2].lower() == "x" else int(body[1:], 10)
        char = chr(code)
    except (ValueError, OverflowError):
        return match.group(0)
    # A control or a surrogate is not a letter. Leaving &#0; would be spoken
    # as the entity, and chr of a surrogate cannot be encoded as UTF-8.
    # A newline or tab is a word break. Deleting it spoke "hellothere".
    if char in "\t\n\r":
        return "\n" if char != "\t" else " "
    # &#160; is the same break as &nbsp;. Left as a nbsp, a chunk that is
    # only that character is skipped and the words join.
    if char == "\u00a0":
        return " "
    if ord(char) < 32 or 0xD800 <= ord(char) <= 0xDFFF:
        return ""
    return char


def _table_cells(match: re.Match[str]) -> str:
    cells = [cell.strip() for cell in match.group(1).split("|")]
    return " ".join(cell for cell in cells if cell)


def _strip_marks(text: str, *, line_start: bool = True) -> str:
    # "&lt;br&gt;" is still a break. Decoding after the tag pass left "<br>"
    # in the words, so Fish said "br" between "door" and "today".
    stripped = _HTML_ENTITY_RE.sub(_html_char, text)
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
    return _MD_WRAP_RE.sub(_unwrap_mark, stripped)


def _strip_markdownish(text: str, *, line_start: bool = True) -> str:
    cleaned = rewrite_s1_parens(text)
    cleaned = _line_sub(_MD_HEADING_RE, "", cleaned, line_start=line_start)
    cleaned = _replace(cleaned, _MARKDOWN_SUBS[1:])
    parts: list[str] = []
    last = 0
    for m in _ANGLE_TOKEN_RE.finditer(cleaned):
        parts.append(_strip_marks(cleaned[last : m.start()], line_start=line_start and last == 0))
        parts.append(m.group(0))
        last = m.end()
    parts.append(_strip_marks(cleaned[last:], line_start=line_start and last == 0))
    return "".join(parts)


# Trailing spaces before a newline. Horizontal runs collapse; newlines stay.
_TRAIL_SPACE_RE = re.compile(r"[ \t]+\n")
# "(a)" and "(about five)" are removed and leave the space. Fish then says
# "door" and "dot". The stop belongs on the word.
_SPACE_BEFORE_STOP_RE = re.compile(r"[ \t]+([.!?…。！？,;:，；：])")
# Three or more blank lines become one paragraph break.
_BREAKS_RE = re.compile(r"\n{3,}")


def _tidy_breaks(text: str, *, before: str = "", after: str = "") -> str:
    text = _SPACE_BEFORE_STOP_RE.sub(r"\1", text)
    text = _BREAKS_RE.sub("\n\n", text)
    if not before and not after:
        return text.strip()
    # A chunk scrub inserts a gap at the neighbor. strip() would eat it
    # and the next word would join. Either neighbor means this is a chunk.
    # Spaces strip. A decoded break stays, so the next mood is still a cue.
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


def _unclosed_tail(
    text: str,
    open_ch: str,
    close_ch: str,
    *,
    keep_digit: bool,
    continued: bool = False,
) -> str:
    # A closer removes the span. A cutoff never sends the closer, so the tail
    # would be spoken. A digit means a phone number or a code, which stays.
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
    if start is None:
        return text
    tail = text[start:]
    if keep_digit and any(ch.isdigit() for ch in tail):
        return text
    inner = tail[1:].strip()
    # "(Please open the door today" is the whole reply and never closed.
    # Dropping it made the turn silence. A cutoff after real words, such as
    # "Hello (she smiles", is still an aside.
    if not continued and not text[:start].strip() and len(inner.split()) >= 4:
        return inner
    return text[:start]


def _unwrap_spoken_paren(text: str, *, continued: bool = False) -> str:
    # "(I can help with that today friend.)" is the reply, not a stage note.
    # The paren rule would delete it and the turn would be silence. A short
    # aside such as "(with that)" or "(she smiles.)" still has to go.
    if continued:
        return text
    stripped = text.strip()
    if len(stripped) < 2:
        return text
    pairs = (("(", ")"), ("（", "）"))
    open_ch = stripped[0]
    close_ch = stripped[-1]
    if (open_ch, close_ch) not in pairs:
        return text
    inner = stripped[1:-1].strip()
    if open_ch in inner or close_ch in inner or any(ch.isdigit() for ch in inner):
        return text
    # "(Please open the door today)" is the reply. Requiring a period
    # deleted it, so the turn was silence. A short aside still stays.
    if len(inner.split()) < 4:
        return text
    return inner


def _plain_breaks(text: str) -> str:
    # "\r\n" is one line break. A line or paragraph separator is too.
    # A zero-width space, a word joiner, a BOM, or a soft hyphen is not a
    # sound. Left in a transcript, "bye" did not quit and an echo started
    # another turn.
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
    token = _SCRUB_EDGE.set((before[-1:], after[:1]))
    try:
        text = _unwrap_edge_marks(text, line_start=line_start, before=before, after=after)
        if not text:
            return ""
        cleaned = _unwrap_spoken_paren(_plain_breaks(text), continued=continued)
        cleaned = _replace(cleaned, _TTS_ERASE)
        cleaned = _THOUGHT_TAIL_RE.sub("", cleaned)
        cleaned = _FENCE_RE.sub(" ", cleaned)
        cleaned = _FENCE_TAIL_RE.sub("", cleaned)
        # After fences, so a <!-- or <script> inside a code block does not eat
        # the reply. Script runs before comments so a comment inside it does not
        # either.
        cleaned = _HTML_BLOCK_RE.sub(" ", cleaned)
        cleaned = _HTML_BLOCK_TAIL_RE.sub("", cleaned)
        cleaned = _HTML_COMMENT_RE.sub(" ", cleaned)
        cleaned = _HTML_COMMENT_TAIL_RE.sub("", cleaned)
        cleaned = _strip_markdownish(cleaned, line_start=line_start)
        cleaned = _unclosed_tail(cleaned, "(", ")", keep_digit=True, continued=continued)
        cleaned = _unclosed_tail(cleaned, "（", "）", keep_digit=True, continued=continued)
        cleaned = _unclosed_tail(cleaned, "[", "]", keep_digit=False, continued=continued)
        cleaned = _TRAIL_SPACE_RE.sub("\n", cleaned)
        # A lone surrogate cannot be encoded as UTF-8. httpx then fails the request.
        return utf8_text(_tidy_breaks(cleaned, before=before, after=after))
    finally:
        _SCRUB_EDGE.reset(token)


def _enough_speech(text: str) -> bool:
    # Two Latin letters skip a one-letter quote. Two CJK characters are a
    # sentence: a quoted Chinese line has no Latin letters at all.
    # "100" is an answer. A one-letter quote and "42" stay too thin.
    cjk, latin = _cjk_latin_counts(text)
    if latin >= 2 or cjk >= 2:
        return True
    return sum(ch.isdigit() for ch in text) >= 3


def _speakable_quote(inner: str) -> bool:
    if not inner or (inner.startswith("[") and inner.endswith("]")):
        return False
    return _enough_speech(inner)


def _closed_quotes(text: str) -> list[str]:
    parts: list[str] = []
    for m in _DIALOGUE_RE.finditer(text):
        if not _speakable_quote((m.group(1) or "").strip()):
            continue
        chunk = m.group(0).strip()
        if chunk:
            parts.append(chunk)
    return parts


def _open_quote_line(stripped: str) -> str:
    m = _OPEN_DIALOGUE_RE.match(stripped)
    if m is None:
        return ""
    inner = (m.group(1) or "").strip().rstrip(_CLOSE_QUOTES).strip()
    cjk, latin = _cjk_latin_counts(inner)
    cue_only = inner.startswith("[") and "]" in inner[:_CUE_LOOKAHEAD] and latin < 3 and cjk < 2
    if not inner or not _enough_speech(inner) or cue_only:
        return ""
    cm = _LEAD_CUE_RE.match(stripped)
    cue = cm.group(1) if cm else ""
    return f'{cue}"{inner}"'


def extract_quoted_speech(text: str) -> str:
    """Keep quoted dialogue (optional leading [cue]); drop stage notes."""
    if not text:
        return ""
    parts = _closed_quotes(text)
    if parts:
        return " ".join(parts)
    opened = _open_quote_line(text.strip())
    if opened:
        return opened
    # A closed pair that is not speech must not fall through to the stage
    # direction. A lone inch mark is not a pair, so the sentence stays.
    if _DIALOGUE_RE.search(text):
        return ""
    return text


def is_tts_junk(text: str) -> bool:
    """Return whether scrubbed text should not be sent to Fish TTS.

    Parameters
    ----------
    text : str
        Text after ``scrub_tts``. Cue-only text, an unmatched ``[``, narration
        that is not speech, or fewer than three letters counts as junk.
        ``Été`` and a Russian sentence are spoken. Three digits are a number,
        so ``100`` is spoken.
        A lead cue is not speech: ``[clear] hi`` stays junk, and ``[clear]``
        does not hide a stage direction. A real CJK sentence is not junk.

    Returns
    -------
    bool
        True when the proxy should return silence instead of calling Fish.
    """
    s = (text or "").strip()
    if s.count("[") > s.count("]"):
        return True
    # The cue name is not speech. Counting "clear" would speak "hi", and a
    # leading [clear] would hide "She smiles" from the narration check.
    spoken = _SPACE_RE.sub(" ", _EMPTY_CUE_RE.sub(" ", s)).strip()
    bare = _SPACE_RE.sub(" ", _QUOTE_RE.sub(" ", spoken)).strip()
    if not bare:
        return True
    if _too_thin(spoken, min_latin=3):
        return True
    return _looks_like_narration(spoken)


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


# Same closer class the sentence cut uses. A mood bang is not one of these.
_HOLD_CLOSERS = frozenset("\"'”’)」』»›〉》")
_HOLD_STOPS = frozenset(".!?…。！？؟।۔")


def _hold_closes_paren(text: str, index: int) -> bool:
    depth = 0
    for ch in text[:index]:
        if ch == "(":
            depth += 1
        elif ch == ")" and depth:
            depth -= 1
    return depth > 0


def sentence_closer_hold_at(text: str) -> int | None:
    """Return where a stop is waiting for a possible closing quote.

    Parameters
    ----------
    text : str
        Unsent model text.

    Returns
    -------
    int or None
        Index of the trailing stop when the buffer ends on that stop or on a
        closing quote or parenthesis after it. None when a later character
        shows the stop is finished, including a space that already followed it.

    Notes
    -----
    Releasing an ideographic stop before a closing quote arrives puts the
    quote on the next sentence, so the following mood is spoken as a word.
    A bang that completes a mood lead is not that stop: holding it spoke
    ``Anxious`` and left the ``!`` behind. A parenthesis that closes an
    aside is not a sentence closer either: holding the period inside
    ``(aside.)`` spoke the leftover ``.)``.
    """
    if not text:
        return None
    index = len(text)
    while index > 0 and text[index - 1] in _HOLD_CLOSERS:
        if text[index - 1] == ")" and _hold_closes_paren(text, index - 1):
            break
        index -= 1
    if index == 0 or text[index - 1] not in _HOLD_STOPS:
        return None
    while index > 0 and text[index - 1] in _HOLD_STOPS:
        index -= 1
    span = spoken_mood_span(text)
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


_HTML_BLOCK_OPEN_RE = re.compile(r"<\s*(script|style|ref)\b[^>]*>", re.IGNORECASE)
_HTML_BLOCK_CLOSE_RE = re.compile(r"<\s*/\s*(script|style|ref)\s*>", re.IGNORECASE)

# A 40-character cut must not speak the inside of a span the closer removes.
_THOUGHT_OPEN_RE = re.compile(
    r"<\s*(?:Thoughts?|thinking|reasoning|think)\s*>",
    re.IGNORECASE,
)
_THOUGHT_CLOSE_RE = re.compile(
    r"<\s*/\s*(?:Thoughts?|thinking|reasoning|think)\s*>",
    re.IGNORECASE,
)
# A closing parenthesis or angle bracket ends the URL, matching scrub.
# Including either one holds only the URL and releases the "<" or "[label]("
# before it, so the next word glues on.
_URL_AT_END_RE = re.compile(r"https?://[^\s)>]*$", re.IGNORECASE)
_WHISPER_OPEN_RE = re.compile(r"<\s*whisper\s*>", re.IGNORECASE)
_WHISPER_CLOSE_RE = re.compile(r"<\s*/\s*whisper\s*>", re.IGNORECASE)
# A hash run at the start of a line is a heading only after the following space.
_OPEN_HEADING_RE = re.compile(r"(?m)^[ \t]{0,3}#{1,6}$")
# "* item" is a list mark only after the following space. A lone "*" is not speech.
_OPEN_LIST_RE = re.compile(r"(?m)^[ \t]{0,3}[-*_~+]+$")


def _unclosed_thought(text: str) -> int | None:
    idx = 0
    while True:
        opened = _THOUGHT_OPEN_RE.search(text, idx)
        if opened is None:
            return None
        closed = _THOUGHT_CLOSE_RE.search(text, opened.end())
        if closed is None:
            return opened.start()
        idx = closed.end()


def _unclosed_span(text: str, open_ch: str, close_ch: str) -> int | None:
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
    if depth:
        return start
    return None


def _unclosed_whisper(text: str) -> int | None:
    # <whisper>...</whisper> becomes a Fish cue only as a pair. Releasing the
    # open tag early speaks the markup.
    idx = 0
    while True:
        opened = _WHISPER_OPEN_RE.search(text, idx)
        if opened is None:
            return None
        closed = _WHISPER_CLOSE_RE.search(text, opened.end())
        if closed is None:
            return opened.start()
        idx = closed.end()


def _open_heading(text: str) -> int | None:
    # "#" at the start of a line is stripped once a space follows. Releasing
    # the hash first speaks it.
    match = _OPEN_HEADING_RE.search(text)
    if match is None or match.end() != len(text):
        return None
    return match.start()


def _open_list_marker(text: str) -> int | None:
    # "* item" and "- item" lose the marker once a space follows.
    # Releasing the marker first speaks it.
    match = _OPEN_LIST_RE.search(text)
    if match is None or match.end() != len(text):
        return None
    return match.start()


def _open_markdown_link(text: str) -> int | None:
    # [label](https://...) is one span. Releasing the label when "(" arrives
    # speaks [label] as a cue and can leave the closing parenthesis behind.
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
    # A finished link has to stay until the next character. Releasing it
    # on the closing parenthesis joins "docs" and "today".
    mark = text.rfind("](")
    if mark < 0:
        return None
    close = text.find(")", mark + 2)
    if close < 0 or close != len(text) - 1:
        return None
    bracket = text.rfind("[", 0, mark)
    if bracket < 0:
        return None
    # "![alt](url)" includes the bang. Releasing it first speaks "See!the".
    if bracket > 0 and text[bracket - 1] == "!":
        return bracket - 1
    return bracket


def _open_ref_link(text: str) -> int | None:
    # [docs][ref] is one span. Releasing [docs] when "[" arrives speaks it
    # as a cue, and the reference id never puts the word back. A finished
    # pair still has to stay together until the next character, or the
    # trailing-bracket hold sends the label first.
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
    # "]" at the end may still be a markdown link. Once the next character
    # is not "(", the bracket is finished. Keeping that character in the
    # same piece speaks the first letter of a mood that followed a pause.
    search = 0
    while True:
        mark = text.find("]", search)
        if mark < 0 or mark + 1 >= len(text):
            return None
        # A space or another "]" still belongs to this token. A letter is
        # the next word, as in "[pause 1s]Excited".
        if text[mark + 1].isalnum():
            return mark + 1
        search = mark + 1


def _trailing_bracket(text: str) -> int | None:
    # [docs] at the end of the buffer may still be [docs](https://...).
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


def _finished_tilde_fence(text: str) -> int | None:
    # The closer is the last character. Releasing there scrubs the code
    # before the closer arrives in the same chunk, so the code is spoken.
    if text.count("~~~") % 2 or not text.endswith("~~~"):
        return None
    end = text.rfind("~~~")
    opener = text.rfind("~~~", 0, end)
    if opener < 0:
        return None
    return opener


def _unclosed_tilde_fence(text: str) -> int | None:
    # ~~~code has no closer yet. Releasing it speaks the code. ~~word~~ is
    # a deletion, so this only holds a run of three tildes.
    if text.count("~~~") % 2 == 0:
        return None
    mark = text.rfind("~~~")
    if mark < 0:
        return None
    return mark


def _unclosed_fence(text: str) -> int | None:
    # An odd number of fences means the last ``` is still open. Releasing
    # it speaks the code and the leftover ticks.
    if text.count("```") % 2 == 0:
        return None
    mark = text.rfind("```")
    if mark < 0:
        return None
    return mark


# A split "<thi" is not a tag yet. Releasing it speaks the thought.
_TAG_WORDS = ("thinking", "thought", "thoughts", "reasoning", "think", "whisper")


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
    if start < 0:
        return None
    if "|>" in text[start + 2 :]:
        return None
    return start


# Backticks stay with the fence hold. Holding them here ate the words
# after a split ``` block.
_TRAIL_MARKS = frozenset("*_~")


def _unclosed_strike(text: str) -> int | None:
    # ~~Tuesday has no closer yet. Releasing it speaks the word, and the
    # later ~~ is then a bullet instead of the end of the deletion.
    if text.count("~~") % 2 == 0:
        return None
    return text.rfind("~~")


def _trailing_emphasis(text: str) -> int | None:
    # "2*" is not emphasis until the next character arrives. Releasing it
    # strips the star, and "2*3" is then spoken as "23".
    if not text or text[-1] not in _TRAIL_MARKS:
        return None
    index = len(text)
    while index > 0 and text[index - 1] in _TRAIL_MARKS:
        index -= 1
    marks = text[index:]
    # A finished ~~Tuesday~~ ends on the closer. Holding those tildes
    # scrubs "~~Tuesday" alone, so the deleted word is spoken.
    if set(marks) == {"~"} and len(marks) >= 2 and text.count("~~") % 2 == 0:
        return None
    return index


def _incomplete_fence_mark(text: str) -> int | None:
    # One or two trailing backticks may still become ```. Releasing them
    # first leaves the ticks in the sentence, and the later fence tail
    # then deletes the words that follow.
    count = 0
    index = len(text)
    while index > 0 and text[index - 1] == "`":
        count += 1
        index -= 1
    if count == 0 or count % 3 == 0:
        return None
    return index


def _incomplete_autolink(text: str) -> int | None:
    # "<https://example.com" is not done. Releasing the "<" speaks it, and
    # the closer is then removed with the URL, so the next word glues on.
    low = text.lower()
    best: int | None = None
    for scheme in ("<https://", "<http://"):
        for length in range(1, len(scheme)):
            if not low.endswith(scheme[:length]):
                continue
            start = len(text) - length
            if start and text[start - 1].isalnum():
                continue
            if best is None or start < best:
                best = start
    opened = re.search(r"<https?://[^>\s]*$", text, re.IGNORECASE)
    if opened is not None and (best is None or opened.start() < best):
        return opened.start()
    return best


def _incomplete_url_scheme(text: str) -> int | None:
    # "htt" is not a URL yet. Releasing it speaks the address once the
    # rest of the scheme arrives in a later token.
    low = text.lower()
    best: int | None = None
    for scheme in ("https://", "http://"):
        for length in range(1, len(scheme)):
            if not low.endswith(scheme[:length]):
                continue
            start = len(text) - length
            if start and text[start - 1].isalnum():
                continue
            if best is None or start < best:
                best = start
    return best


def _entity_stop_hold(text: str) -> int | None:
    # "&#33; Ex" is an exclamation mark plus the start of a mood. "&#10;Ex"
    # is a new line. Both are still entities, so the mood looks mid-sentence
    # and is spoken as a word.
    for match in _HTML_ENTITY_RE.finditer(text):
        decoded = _html_char(match)
        if decoded != "\n" and decoded not in ".!?…。！？؟।۔":
            continue
        rest = text[match.end() :]
        held = mood_lead_hold_at(rest, sentence_start=True)
        if held is not None:
            return match.end() + held
    return None


def _sentence_lead_hold(text: str, *, sentence_start: bool, before: str = "") -> int | None:
    # "Hello. Ex" can arrive in one token. The lead is the last sentence, so
    # holding only the start of the buffer speaks "Excited" before the comma.
    # The period can also land in the next chunk ("Don" then "e.Ex"). That
    # chunk looks like the initial "e." and the mood is spoken as a word.
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
        if index >= len(before):
            return index - len(before)
        return 0
    if not sentence_start or before:
        return None
    return mood_lead_hold_at(text, sentence_start=True)


def _mark_on_its_line(found: int | None, *, line_start: bool) -> int | None:
    # A "*" that begins this chunk after "2" is the multiply sign. A marker
    # after a newline in the same chunk is a bullet even when the previous
    # chunk did not end on a newline.
    if found is None or (found == 0 and not line_start):
        return None
    return found


def _unclosed_html_block(text: str) -> int | None:
    # "<script>alert" is not done. The tag scrubber would drop the tags
    # and speak the code before the closer arrives.
    idx = 0
    while True:
        opened = _HTML_BLOCK_OPEN_RE.search(text, idx)
        if opened is None:
            return None
        name = opened.group(1).lower()
        search_at = opened.end()
        while True:
            closed = _HTML_BLOCK_CLOSE_RE.search(text, search_at)
            if closed is None:
                return opened.start()
            if closed.group(1).lower() == name:
                idx = closed.end()
                break
            search_at = closed.end()


def _incomplete_comment(text: str) -> int | None:
    # "<!-- secret" is not done. Releasing it speaks the note, and the
    # closer never meets the opener in one scrub.
    mark = text.rfind("<")
    if mark < 0:
        return None
    tail = text[mark:]
    if tail.startswith("<!--"):
        if "-->" in tail[4:]:
            return None
        return mark
    if "<!--".startswith(tail):
        return mark
    return None


_LT_ENTITY_RE = re.compile(r"&(?:lt|#0*60|#x0*3c);", re.IGNORECASE)
_GT_ENTITY_RE = re.compile(r"&(?:gt|#0*62|#x0*3e);|>", re.IGNORECASE)


def _incomplete_escaped_tag(text: str) -> int | None:
    # "&lt;br" is not a tag yet. Releasing it speaks "br" between the words.
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


def _incomplete_html(text: str) -> int | None:
    # "<b" is not done. Releasing it speaks the tag, and the closer never
    # meets the opener in one scrub.
    start = text.rfind("<")
    if start < 0 or ">" in text[start + 1 :]:
        return None
    tail = text[start + 1 :]
    if tail[:1] == "/":
        tail = tail[1:]
    # "</ p" is still a tag. A space used to release it, so "door</" was spoken.
    body = tail.lstrip(" \t")
    if body[:1] and not body[0].isalpha():
        return None
    if len(body) > 80:
        return None
    return start


def _incomplete_entity(text: str) -> int | None:
    # "&amp" is not "&" yet. Releasing it speaks the letters amp.
    mark = text.rfind("&")
    if mark < 0 or ";" in text[mark:]:
        return None
    tail = text[mark + 1 :]
    if len(tail) > 12 or not re.fullmatch(r"#?x?[0-9A-Za-z]*", tail):
        return None
    return mark


def _incomplete_image(text: str) -> int | None:
    # "![alt](url)" is one span. Holding at "[" speaks the bang first.
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


def _ref_label_before_id(text: str) -> int | None:
    # "[the docs][" is still one reference. Releasing the label speaks it
    # as a cue, and the id arrives too late to join it.
    mark = text.rfind("][")
    if mark < 0 or "]" in text[mark + 2 :]:
        return None
    start_label = text.rfind("[", 0, mark)
    if start_label < 0 or " " not in text[start_label + 1 : mark]:
        return None
    return start_label


def _trailing_ref_link(text: str) -> int | None:
    # [the docs][ref] is a reference. The second bracket is held alone, so
    # the label is spoken as a cue before the id arrives.
    if not text.endswith("]"):
        return None
    start_id = text.rfind("[", 0, len(text) - 1)
    if start_id <= 0 or text[start_id - 1] != "]":
        return None
    start_label = text.rfind("[", 0, start_id - 1)
    if start_label < 0 or " " not in text[start_label + 1 : start_id - 1]:
        return None
    return start_label


def hold_tts(
    text: str,
    *,
    line_start: bool,
    sentence_start: bool,
    before: str = "",
) -> int:
    """Return the index where a stream must keep buffering.

    Parameters
    ----------
    text : str
        Unscrubbed chunk, including any unfinished span.
    line_start : bool
        True when this chunk begins a line. A mid-line star is not a bullet.
    sentence_start : bool
        True when a mood word here can become a lead cue.
    before : str, optional
        Scrubbed text already accepted on this line. Default empty.

    Returns
    -------
    int
        Start of the unfinished span, or ``len(text)`` when the chunk is stable.
    """
    url = _URL_AT_END_RE.search(text)
    marks = (
        _unclosed_thought(text),
        _unclosed_whisper(text),
        _incomplete_tag(text),
        _incomplete_angle_token(text),
        _unclosed_fence(text),
        _unclosed_tilde_fence(text),
        _finished_tilde_fence(text),
        _incomplete_fence_mark(text),
        _unclosed_strike(text),
        _trailing_emphasis(text),
        _incomplete_autolink(text),
        _incomplete_url_scheme(text),
        _unclosed_html_block(text),
        _incomplete_comment(text),
        _incomplete_html(text),
        _incomplete_escaped_tag(text),
        _incomplete_entity(text),
        _incomplete_image(text),
        _incomplete_table_row(text),
        _ref_label_before_id(text),
        _trailing_ref_link(text),
        _mark_on_its_line(_open_heading(text), line_start=line_start),
        _mark_on_its_line(_open_list_marker(text), line_start=line_start),
        url.start() if url is not None else None,
        _unclosed_span(text, "(", ")"),
        _unclosed_span(text, "（", "）"),
        _unclosed_span(text, "[", "]"),
        _open_markdown_link(text),
        _finished_link_hold(text),
        _open_ref_link(text),
        _trailing_bracket(text),
        _after_closed_bracket(text),
        sentence_closer_hold_at(text),
        _entity_stop_hold(text),
        mood_lead_hold_at(text, sentence_start=sentence_start),
        _sentence_lead_hold(text, sentence_start=sentence_start, before=before),
    )
    starts = [mark for mark in marks if mark is not None]
    if not starts:
        return len(text)
    return min(starts)
