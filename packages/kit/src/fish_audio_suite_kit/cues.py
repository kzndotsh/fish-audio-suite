"""Fish [cue] tags: mood leads, lowercase, and third-party aliases."""

from __future__ import annotations

import re

from fish_audio_suite_kit.cuts import next_tts_cut

# Official single-word emotions (basic + advanced). Both the sentence-lead
# pattern and the S1 parenthesis rewriter use this set, so the lists cannot drift.
# https://docs.fish.audio/api-reference/emotion-reference.md
_FISH_EMOTIONS = frozenset(
    {
        "happy",
        "sad",
        "angry",
        "excited",
        "calm",
        "nervous",
        "confident",
        "surprised",
        "satisfied",
        "delighted",
        "scared",
        "worried",
        "upset",
        "frustrated",
        "depressed",
        "empathetic",
        "embarrassed",
        "disgusted",
        "moved",
        "proud",
        "relaxed",
        "grateful",
        "curious",
        "sarcastic",
        "disdainful",
        "unhappy",
        "anxious",
        "hysterical",
        "indifferent",
        "uncertain",
        "doubtful",
        "confused",
        "disappointed",
        "regretful",
        "guilty",
        "ashamed",
        "jealous",
        "envious",
        "hopeful",
        "optimistic",
        "pessimistic",
        "nostalgic",
        "lonely",
        "bored",
        "contemptuous",
        "sympathetic",
        "compassionate",
        "determined",
        "resigned",
    }
)
# The only Fish tone tags that are one word and open a sentence
# ("Whispering, come here"). "in a hurry tone" and "soft tone" are phrases.
# "emphasis" marks a word mid-sentence. Sound effects are not leads: "Pause, wait"
# is speech.
_SPOKEN_TONE = frozenset({"whispering", "shouting", "screaming"})
# "Excited, hello" -> [excited] hello. Longest word first so a shorter emotion
# cannot take a prefix. Punctuation is required, so "I am anxious today" stays speech.
_SPOKEN_MOOD_LEAD = re.compile(
    r"^\s*("
    + "|".join(
        re.escape(word) for word in sorted(_FISH_EMOTIONS | _SPOKEN_TONE, key=len, reverse=True)
    )
    + r")\s*[,:!\-–—，！：]+\s*",
    re.IGNORECASE,
)
# The same marks the lead regex consumes. A stream can end on the first "!"
# of "Anxious !!", so that bang stays buffered until a space or a word.
# Fullwidth comma, bang, and colon are the same marks in mixed CJK text.
_LEAD_PUNCT = frozenset(",:!-\u2013\u2014\uff0c\uff01\uff1a")
# Closing emphasis still belongs to the mood word: "Excited*" is not done.
_LEAD_WRAP = frozenset("*_~`")

# Any [cue], including free-form S2 text. A newline ends the tag.
_CUE_RE = re.compile(r"\[([^\]\n]+)\]")
# Some models write <whisper>...</whisper> instead of a Fish cue.
_WHISPER_XML_RE = re.compile(
    r"<\s*whisper\s*>(.*?)<\s*/\s*whisper\s*>",
    re.IGNORECASE | re.DOTALL,
)

# Spellings models write that are not the Fish tag. [cough] is not aliased.
_CUE_ALIASES = {
    "laugh": "laughing",
    "laughs": "laughing",
    "whisper": "whispering",
    "whispers": "whispering",
    "pause": "break",
    "sigh": "sighing",
    "chuckle": "chuckling",
}

# (happy) from the old S1 model. Emotions and the three sentence tones are
# unioned above. Unknown (asides) are left for the scrubber.
_S1_PAREN_TAGS = (
    _FISH_EMOTIONS
    | _SPOKEN_TONE
    | frozenset(
        {
            # Tone tags that do not fit the one-word sentence lead.
            "emphasis",
            "in a hurry tone",
            "soft tone",
            # Official sound effects, plus breath/cough/lip-smacking from older prompts.
            "break",
            "long-break",
            "laughing",
            "chuckling",
            "sighing",
            "sobbing",
            "crying loudly",
            "groaning",
            "panting",
            "gasping",
            "yawning",
            "snoring",
            "clear throat",
            "audience laughing",
            "background laughter",
            "crowd laughing",
            "breath",
            "cough",
            "lip-smacking",
            # Alias spellings, so (laugh) and (pause) match before the alias map.
            "pause",
            "laugh",
            "laughs",
            "chuckle",
            "sigh",
            "whisper",
            "whispers",
        }
    )
)
# Longest tag first, so "laughing" wins over "laugh".
_S1_PAREN_RE = re.compile(
    r"\(\s*("
    + "|".join(re.escape(t) for t in sorted(_S1_PAREN_TAGS, key=len, reverse=True))
    + r")\s*\)",
    re.IGNORECASE,
)


# A heading, list marker, task box, or emphasis mark before a mood lead.
# "### Excited," and "*Sad," are cues. Releasing the word after the mark
# speaks the mood instead.
_LEAD_MARK_RE = re.compile(
    r"^(?:[ \t]{0,3}#{1,6}[ \t]+)?"
    r"(?:[ \t]{0,3}[-*_~+]+[ \t]+)?"
    r"(?:\[[ \t]*[xX]?[ \t]*\][ \t]+)?"
    r"(?:[*_~`]+(?=[A-Za-z]))?"
)
# One or more [cues] already at the start of a sentence. Group 2 is the spoken rest.
# A cue body cannot contain "[". "[[page]]" is not a stack, and treating the
# inner "[" as cue text inserted a space: "[[page] ]".
_LEAD_STACK_RE = re.compile(r"^((?:\[[^\[\]]+\]\s*)+)(.*)$", re.DOTALL)
# Each cue inside that stack, so [sad][whispering] stays two tags.
_INNER_CUE_RE = re.compile(r"\[([^\]]+)\]")
# Keep the newlines, so a blank line is not folded into the next sentence.
_LINE_SPLIT_RE = re.compile(r"(\n+)")


def paren_cue_names() -> frozenset[str]:
    """Return S1 cue names, including ``clear``.

    Returns
    -------
    frozenset of str
        Lowercase tags a parenthesis or bracket can carry without being speech.
    """
    return _S1_PAREN_TAGS | {"clear"}


def is_paren_cue(label: str) -> bool:
    """Return whether a bracket label is an S1 tag or ``clear``.

    Parameters
    ----------
    label : str
        Text inside one pair of brackets.

    Returns
    -------
    bool
        True for a known S1 tag or ``clear``.
    """
    return label.strip().lower() in paren_cue_names()


def spoken_mood_span(text: str) -> tuple[int, int] | None:
    """Return the span of a mood lead at the start of text.

    Parameters
    ----------
    text : str
        Unsent model text.

    Returns
    -------
    tuple of int and int or None
        Start and end of the lead. None when the text does not open with one.
    """
    lead = _SPOKEN_MOOD_LEAD.match(text)
    if lead is None:
        return None
    return lead.start(), lead.end()


def _bracket(inner: str) -> str:
    tag = inner.strip().lower()
    return f"[{_CUE_ALIASES.get(tag, tag)}]"


def _cue_word(inner: str) -> bool:
    word = inner.strip().lower()
    return word in _CUE_ALIASES or word in _FISH_EMOTIONS or word in _SPOKEN_TONE or word == "clear"


def rewrite_s1_parens(text: str) -> str:
    """`(happy)` / `(break)` → `[happy]` / `[break]`. Unknown `(asides)` left for scrubbers."""
    return _S1_PAREN_RE.sub(lambda m: _bracket(m.group(1)), text)


def _lead(tags: str, rest: str) -> str:
    return f"{tags} {rest}" if rest else tags


def mood_lead_hold_at(text: str, *, sentence_start: bool) -> int | None:
    """Return where an unfinished sentence-lead cue starts.

    Parameters
    ----------
    text : str
        Unsent model text.
    sentence_start : bool
        True when ``text`` begins a sentence. A later line in ``text`` is
        its own sentence either way.

    Returns
    -------
    int or None
        Index to keep buffered. None when a later character shows the cue
        is finished, the word has ended, or this is not the start of a sentence.

    Notes
    -----
    A token stream can emit ``Exc`` before ``ited,``. Scrubbing that prefix
    speaks the emotion. The comma has to arrive before the word is released.
    ``Happy - hello`` puts a space before the dash. Releasing ``Happy`` at
    that space speaks the mood before the dash arrives. A list marker is
    not the mood word: ``* E`` still has to wait for ``xcited,``.
    """
    line_at = text.rfind("\n") + 1
    if line_at == 0 and not sentence_start:
        return None
    line = text[line_at:]
    skipped = len(line) - len(line.lstrip(" \t"))
    marked = _LEAD_MARK_RE.match(line.lstrip(" \t"))
    mark_len = marked.end() if marked is not None else 0
    body = line.lstrip(" \t")[mark_len:]
    if not body:
        return None
    lead = _SPOKEN_MOOD_LEAD.match(body)
    if lead is not None:
        # "Anxious!" matches with one bang. "Anxious !!" is the same cue.
        # Releasing at the first bang speaks the second one. "*Excited,*"
        # still has the closing star; releasing at the comma speaks it.
        tail = body[lead.end() :]
        if not tail and body[-1] in _LEAD_PUNCT:
            return line_at + skipped
        if tail and all(ch in _LEAD_WRAP or ch.isspace() for ch in tail):
            return line_at + skipped
        return None
    idx = 0
    while idx < len(body) and body[idx].isalpha():
        idx += 1
    if idx == 0:
        return None
    token = body[:idx].lower()
    words = _FISH_EMOTIONS | _SPOKEN_TONE
    held = line_at + skipped + mark_len
    if idx < len(body):
        tail = body[idx:]
        if token in words and tail.strip() == "":
            return held
        # "**Excited**," is one cue. The closing stars are not the end of
        # the word, and the comma has not arrived yet.
        if token in words and all(ch in _LEAD_WRAP or ch.isspace() for ch in tail):
            return held
        return None
    if any(word.startswith(token) for word in words):
        return held
    return None


def _sentence_spans(part: str) -> list[tuple[str, str]]:
    spans: list[tuple[str, str]] = []
    rest = part
    while rest:
        # A huge window disables the partial-word cut. Only a real sentence
        # end splits, so "Dr. Happy" is not a new sentence.
        cut = next_tts_cut(rest, partial_chars=len(rest) + 1)
        if cut < 1:
            spans.append((rest, ""))
            break
        piece = rest[:cut]
        rest = rest[cut:]
        body = piece.rstrip()
        spans.append((body, piece[len(body) :]))
    return spans


def _one_sentence(chunk: str, *, lead: bool = True) -> str:
    chunk = chunk.strip()
    if not chunk:
        return chunk
    m = _LEAD_STACK_RE.match(chunk)
    if m:
        cues = _INNER_CUE_RE.findall(m.group(1))
        # [i][j] is an index. Spacing every bracket stack spoke "a[i] [j]".
        if cues and (len(cues) == 1 or all(_cue_word(c) for c in cues)):
            tags = " ".join(_bracket(c) for c in cues)
            return _lead(tags, m.group(2).lstrip())
    # A stream may cut before this word. It is a lead only at a real sentence start.
    if lead:
        m2 = _SPOKEN_MOOD_LEAD.match(chunk)
        if m2:
            return _lead(_bracket(m2.group(1)), chunk[m2.end() :].lstrip())
    return chunk


def normalize_cues(text: str, *, lead: bool = True) -> str:
    """Rewrite third-party mood markup into Fish ``[cue]`` tags.

    Parameters
    ----------
    text : str
        Model text that may contain ``[Tags]``, S1 ``(happy)``, a mood lead
        such as ``Excited, …``, or ``<whisper>…</whisper>``.
    lead : bool, optional
        When False, the first sentence is not rewritten as a mood lead.
        A later sentence in the same string still is. Default True.

    Returns
    -------
    str
        The same words with cues lowercased. Stacked leads stay stacked.
        ``laugh`` / ``sigh`` / ``chuckle`` / ``whisper`` / ``pause`` become
        ``laughing`` / ``sighing`` / ``chuckling`` / ``whispering`` / ``break``.
        ``[cough]`` is left as ``[cough]``.

    Notes
    -----
    A mood word only becomes a cue when it leads a sentence (``Happy, hello``).
    ``Dr. Happy`` and ``1. Excited`` are not new sentences. The lead list is
    Fish's single-word emotions, plus whispering, shouting, and screaming.
    The same word mid-sentence is spoken text. Sound effects such as
    ``Pause, wait`` stay spoken.
    """
    if not text:
        return text
    text = _WHISPER_XML_RE.sub(lambda m: f"[whispering] {m.group(1).strip()}", text)
    text = rewrite_s1_parens(text)
    text = _CUE_RE.sub(lambda m: _bracket(m.group(1)), text)
    parts = _LINE_SPLIT_RE.split(text)
    out: list[str] = []
    allow_lead = lead
    for part in parts:
        if not part or part.isspace():
            out.append(part)
            continue
        spoken: list[str] = []
        for sent, sep in _sentence_spans(part):
            if not sent.strip():
                if sep:
                    spoken.append(sep)
                continue
            spoken.append(_one_sentence(sent, lead=allow_lead) + sep)
            allow_lead = True
        out.append("".join(spoken))
    return "".join(out)


def ensure_lead_cue(text: str, *, default: str = "clear") -> str:
    """Prepend one cue when the reply has none.

    Parameters
    ----------
    text : str
        Already scrubbed reply.
    default : str, optional
        Cue name without brackets. Blank becomes ``clear``.

    Returns
    -------
    str
        ``text`` unchanged when any ``[cue]`` is already present, including
        one mid-reply. Empty or whitespace-only input is unchanged.

    Notes
    -----
    This does not invent a tag on every sentence. Prosody sticks until the
    next cue, so a second lead would reset the voice.
    """
    if not text.strip():
        return text
    if _CUE_RE.search(text):
        return text
    tag = default.strip().lower() or "clear"
    return f"[{tag}] {text.lstrip()}"
