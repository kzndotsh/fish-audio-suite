"""Fish [cue] tags: mood leads, lowercase, and third-party aliases."""

from __future__ import annotations

import re

from fish_audio_suite_kit.cuts import ends_sentence, next_tts_cut

__all__ = [
    "MoodCarry",
    "ensure_lead_cue",
    "is_paren_cue",
    "last_emotion",
    "mood_lead_hold_at",
    "normalize_cues",
    "official_cue",
    "paren_cue_names",
    "rewrite_s1_parens",
    "split_cues",
    "spoken_mood_span",
    "strip_cue_tags",
]

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
# A bare bracket ("Happy [chuckling]") is also accepted as the delimiter.
_SPOKEN_MOOD_LEAD = re.compile(
    r"^\s*("
    + "|".join(
        re.escape(word) for word in sorted(_FISH_EMOTIONS | _SPOKEN_TONE, key=len, reverse=True)
    )
    + r")\s*(?:[,:!\-–—，！：]+\s*|(?=\[))",
    re.IGNORECASE,
)
# The same marks the lead regex consumes. A stream can end on the first "!"
# of "Anxious !!", so that bang stays buffered until a space or a word.
# Fullwidth comma, bang, and colon are the same marks in mixed CJK text.
_LEAD_PUNCT = frozenset(",:!-\u2013\u2014\uff0c\uff01\uff1a")
# Closing emphasis still belongs to the mood word: "Excited*" is not done.
_LEAD_WRAP = frozenset("*_~`")

# Longest cue text and longest whispered span the patterns look at. Each pattern
# scans at most this far from a bracket or tag, so a long run of openers with no
# closer costs a fixed amount per opener instead of the rest of the text.
_MAX_CUE_CHARS = 200
_MAX_WHISPER_CHARS = 1000
# Any [cue], including free-form S2 text. A newline ends the tag.
_CUE_RE = re.compile(rf"\[([^\]\n]{{1,{_MAX_CUE_CHARS}}})\]")
# Some models write <whisper>...</whisper> instead of a Fish cue.
_WHISPER_XML_RE = re.compile(
    rf"<\s{{0,8}}whisper\s{{0,8}}>(.{{0,{_MAX_WHISPER_CHARS}}}?)<\s{{0,8}}/\s{{0,8}}whisper\s{{0,8}}>",
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
    "gasp": "gasping",
    "groan": "groaning",
    "yawn": "yawning",
    "sob": "sobbing",
    "cry": "sobbing",
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
            # Common S2 cues from Fish's models guide that are not in the official lists.
            "gasp",
            "inhale",
            "exhale",
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
# A cue body cannot contain "[", so "[[page]]" is not a stack.
_LEAD_STACK_RE = re.compile(r"^((?:\[[^\[\]]+\]\s*)+)(.*)$", re.DOTALL)
# Each cue inside that stack, so [sad][whispering] stays two tags.
_INNER_CUE_RE = re.compile(rf"\[([^\]]{{1,{_MAX_CUE_CHARS}}})\]")
# Keep the newlines, so a blank line is not folded into the next sentence.
_LINE_SPLIT_RE = re.compile(r"(\n+)")


_PAREN_CUE_NAMES = _S1_PAREN_TAGS | {"clear"}
# [name] for any known cue name, longest first so "laughing" beats "laugh".
# A tag inside a larger bracket expression ("[[happy]]") is literal text.
_KNOWN_CUE_TAG_RE = re.compile(
    r"(?<!\[)\[(?:"
    + "|".join(re.escape(tag) for tag in sorted(_PAREN_CUE_NAMES, key=len, reverse=True))
    + r")\](?!\])",
    re.IGNORECASE,
)


def split_cues(text: str) -> list[tuple[str, bool]]:
    """Split text into its ``[cue]`` tags and the words around them.

    Parameters
    ----------
    text : str
        Text that may contain bracketed cues such as ``[calm]`` or
        ``[smothering affection]``.

    Returns
    -------
    list of tuple of str and bool
        Pieces in order, each with ``True`` when it is a cue (brackets included).
        Joined back together they are exactly ``text``, so nothing is lost.
        Empty pieces are left out. A bracket longer than a cue can be, or one
        that spans a line break, is plain text.

    Notes
    -----
    A display uses this to style cues without passing the text through a markup
    parser, which would read the brackets as tags.

    Examples
    --------
    >>> split_cues("[calm] Hello [soft encouragement] there.")
    [('[calm]', True), (' Hello ', False), ('[soft encouragement]', True), (' there.', False)]
    >>> split_cues("no cues")
    [('no cues', False)]
    """
    pieces: list[tuple[str, bool]] = []
    last = 0
    for match in _CUE_RE.finditer(text):
        if match.start() > last:
            pieces.append((text[last : match.start()], False))
        pieces.append((match.group(0), True))
        last = match.end()
    if last < len(text):
        pieces.append((text[last:], False))
    return pieces


def paren_cue_names() -> frozenset[str]:
    """Return S1 cue names, including ``clear``.

    Returns
    -------
    frozenset of str
        Lowercase tags a parenthesis or bracket can carry without being speech.
    """
    return _PAREN_CUE_NAMES


def strip_cue_tags(text: str) -> str:
    """Remove known ``[cue]`` tags, leaving the words that are spoken.

    Parameters
    ----------
    text : str
        Text that may contain Fish cues such as ``[happy]`` or ``[clear]``.

    Returns
    -------
    str
        The text with each known cue removed and whitespace collapsed to
        single spaces. A bracket that is not a known cue, such as the index in
        ``a[i]``, is kept because the speaker said it. A cue nested in a larger
        bracket expression, such as ``[[happy]]``, is kept too.

    Examples
    --------
    >>> strip_cue_tags("[happy] Hello [laughing] there")
    'Hello there'
    >>> strip_cue_tags("keep [1-2] and [[happy]]")
    'keep [1-2] and [[happy]]'
    """
    return " ".join(_KNOWN_CUE_TAG_RE.sub(" ", text).split())


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
    return label.strip().lower() in _PAREN_CUE_NAMES


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


# A word a model writes in a cue that Fish's own list lacks, and the official cue that sounds
# nearest. Stems of four letters or more match the start of a word; shorter ones match a whole
# word, so "mad" does not catch "made". The earliest row with a matching word decides, so sounds
# and tones come before moods ("soft chuckle" is a chuckle, not a soft tone). A descriptive cue about a
# face, a voice or an action ("smiling wider", "echoing voice") is acted out as a sound by
# some voices, which is why it is mapped to a mood or dropped.
_CUE_STEMS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("chuckl", "giggl", "snicker", "titter"), "chuckling"),
    (("laugh", "cackl", "guffaw"), "laughing"),
    (("sigh",), "sighing"),
    (("gasp",), "gasping"),
    (("groan",), "groaning"),
    (("yawn",), "yawning"),
    (("sob", "weep", "cry", "crying", "tearful"), "sobbing"),
    (("pant", "breathless"), "panting"),
    (("throat",), "clear throat"),
    (("whisper", "hush", "murmur"), "whispering"),
    (("shout", "yell", "bellow"), "shouting"),
    (("scream", "shriek"), "screaming"),
    (("hurried", "rushed", "urgent"), "in a hurry tone"),
    (
        (
            *("soft", "softly", "sultry", "seduct", "breathy", "husky"),
            *("intimate", "sensual", "purr", "velvet", "silky", "smoky"),
        ),
        "soft tone",
    ),
    (("delight", "elat", "overjoy"), "delighted"),
    (
        ("excit", "eager", "enthusias", "thrill", "hyped", "ecstatic", "energetic", "giddy"),
        "excited",
    ),
    (("lively", "animated"), "excited"),
    (("proud",), "proud"),
    (("grate", "thank", "apprecia"), "grateful"),
    (("relie", "relax", "easygoing", "unhurried"), "relaxed"),
    (
        (
            *("smil", "grin", "cheer", "happy", "joy", "glad", "pleas", "warm", "friendly"),
            *("sunny", "bright", "upbeat", "playful", "teas", "mischiev", "flirt", "coy"),
            *("cheeky", "amus", "affection", "fond", "loving", "kind"),
        ),
        "happy",
    ),
    (("sarcas", "ironic", "dry", "deadpan", "wry", "mock"), "sarcastic"),
    (("curious", "intrigu", "interest", "inquisitive", "wonder"), "curious"),
    (("surpris", "amaz", "astonish", "impress", "shock", "stunned", "startl"), "surprised"),
    (("sad", "sorrow", "melanchol", "mourn", "heartbr", "gloom", "somber", "grief"), "sad"),
    (("regret", "remors", "sorry", "apolog"), "regretful"),
    (("disappoint",), "disappointed"),
    (("frustrat", "exasperat", "impatient"), "frustrated"),
    (("angry", "furious", "irate", "enrag", "mad", "seething", "irritat", "annoy"), "angry"),
    (("scar", "frighten", "afraid", "terrif", "fearful", "panic", "horrif"), "scared"),
    (("worr", "concern", "troubled", "fretful"), "worried"),
    (
        (
            "nervous",
            "anxious",
            "uneasy",
            "tense",
            "apprehens",
            "jittery",
            "hesitant",
            "timid",
            "shy",
        ),
        "nervous",
    ),
    (("embarrass", "awkward", "sheepish", "flustered", "bashful"), "embarrassed"),
    (
        ("empath", "sympath", "compassion", "caring", "tender", "comfort", "reassur", "understand"),
        "empathetic",
    ),
    (("support",), "empathetic"),
    (("hope", "optimis", "encourag"), "hopeful"),
    (
        ("confident", "assertive", "firm", "command", "authorit", "dominant", "stern", "decisive"),
        "confident",
    ),
    (("bold", "determined", "resolute"), "determined"),
    (("bore", "weary", "tired", "sleepy", "drowsy"), "bored"),
    (("calm", "gentle", "soothing", "peaceful", "serene", "tranquil", "patient", "steady"), "calm"),
    (("composed", "measured", "thoughtful", "pensive", "contemplat", "reflective"), "calm"),
    (("mysterious", "suspense", "ominous", "eerie", "sinister", "dramatic", "intense"), "calm"),
)
# "very excited", "slightly sad": Fish reads an intensity word before an emotion.
_INTENSITY_RE = re.compile(r"^(very|slightly|extremely|really|somewhat|a bit|a little) (\w+)$")


def _word_matches(word: str, stem: str) -> bool:
    return word.startswith(stem) if len(stem) >= 4 else word == stem


def official_cue(inner: str) -> str | None:
    """Return the Fish cue that a model's cue text stands for, or None to drop it.

    Parameters
    ----------
    inner : str
        Text inside one pair of brackets, such as ``smiling wider``.

    Returns
    -------
    str or None
        The cue name without brackets. An official cue, an alias, or an official emotion
        with an intensity word (``very excited``) is returned as it is, lowercased. Any other
        text becomes the nearest official cue by its words, or None when no word suggests
        one (``in a storytelling voice``, ``back to normal voice``).

    Examples
    --------
    >>> official_cue("Happy"), official_cue("very excited"), official_cue("smiling wider")
    ('happy', 'very excited', 'happy')
    >>> official_cue("soft chuckle"), official_cue("in a storytelling voice")
    ('chuckling', None)
    """
    tag = " ".join(inner.lower().split())
    tag = _CUE_ALIASES.get(tag, tag)
    if tag in _PAREN_CUE_NAMES:
        return tag
    modified = _INTENSITY_RE.match(tag)
    if modified and (_CUE_ALIASES.get(modified.group(2), modified.group(2)) in _FISH_EMOTIONS):
        return tag
    words = re.findall(r"[a-z][a-z-]*", tag)
    for stems, target in _CUE_STEMS:
        if any(_word_matches(word, stem) for word in words for stem in stems):
            return target
    return None


def last_emotion(text: str) -> str | None:
    """Return the cue of the last emotion in ``text``, or None when it has none.

    Parameters
    ----------
    text : str
        Text that may hold ``[cues]``.

    Returns
    -------
    str or None
        The last bracket that is an emotion, with its intensity word if it had one (``very
        excited``). That is an official emotion or free-form text such as ``smiling``; official
        sounds, tones and breaks do not count.

    Examples
    --------
    >>> last_emotion("[happy] Hi. [laughing] Ha. [very sad] Oh.")
    'very sad'
    >>> last_emotion("[laughing] Ha.") is None
    True
    """
    found: str | None = None
    for match in _INNER_CUE_RE.finditer(text):
        tag = " ".join(match.group(1).lower().split())
        modified = _INTENSITY_RE.match(tag)
        base = modified.group(2) if modified else tag
        base = _CUE_ALIASES.get(base, base)
        if base in _FISH_EMOTIONS or base not in _PAREN_CUE_NAMES:
            found = tag
    return found


def _bracket(inner: str) -> str:
    tag = inner.strip().lower()
    return f"[{_CUE_ALIASES.get(tag, tag)}]"


_DOUBLE_SPACE_RE = re.compile(r"(?<=\S)[ \t]{2,}(?=\S)")


def _official_bracket(inner: str) -> str:
    tag = official_cue(inner)
    return f"[{tag}]" if tag else ""


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
        # Hold through every bang, and through the closing star of "*Excited,*".
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
        # "**Excited**," is one cue: closing stars do not end the word.
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


def _one_sentence(chunk: str, *, lead: bool) -> str:
    chunk = chunk.strip()
    if not chunk:
        return chunk
    m = _LEAD_STACK_RE.match(chunk)
    if m:
        cues = _INNER_CUE_RE.findall(m.group(1))
        # Only a lone bracket or a stack of known cues is a cue stack: [i][j] is an index.
        if cues and (len(cues) == 1 or all(_cue_word(c) for c in cues)):
            tags = " ".join(_bracket(c) for c in cues)
            return _lead(tags, m.group(2).lstrip())
    # A stream may cut before this word. It is a lead only at a real sentence start.
    if lead:
        m2 = _SPOKEN_MOOD_LEAD.match(chunk)
        if m2:
            return _lead(_bracket(m2.group(1)), chunk[m2.end() :].lstrip())
    return chunk


def normalize_cues(
    text: str, *, lead: bool = False, continued: bool = False, official: bool = False
) -> str:
    """Rewrite third-party mood markup into Fish ``[cue]`` tags.

    Parameters
    ----------
    text : str
        Model text that may contain ``[Tags]``, S1 ``(happy)``, or
        ``<whisper>…</whisper>``, and, when ``lead`` is True, a mood lead
        such as ``Excited, …``.
    lead : bool, optional
        Rewrite a sentence-leading mood word (``Excited, hello``) into a cue.
        Default False: the mood word is deleted from the speech when
        rewritten, and it also matches ordinary English (``Curious, isn't
        it?``). Enable it for models that write moods instead of cues.
    continued : bool, optional
        True when ``text`` continues a sentence already sent. The first
        sentence is then not a mood lead, but later ones still are. Only
        meaningful with ``lead``. Default False.
    official : bool, optional
        Keep only Fish's own cues. Any other cue is replaced by the nearest official one
        (``[smiling]`` becomes ``[happy]``, ``[soft chuckle]`` becomes ``[chuckling]``) or
        removed when no word in it suggests one (``[in a storytelling voice]``). Some voices
        act a description out as a sound, such as a hum, instead of changing tone. Default
        False, which keeps free-form S2 cues as written.

    Returns
    -------
    str
        The same words with cues lowercased. Stacked leads stay stacked.
        ``laugh`` / ``sigh`` / ``chuckle`` / ``whisper`` / ``pause`` become
        ``laughing`` / ``sighing`` / ``chuckling`` / ``whispering`` / ``break``.
        ``[cough]`` is left as ``[cough]``.

    Notes
    -----
    With ``lead``, a mood word becomes a cue only when it leads a sentence
    (``Happy, hello``). ``Dr. Happy`` and ``1. Excited`` are not new
    sentences. The lead list is Fish's single-word emotions, plus whispering,
    shouting, and screaming. The same word mid-sentence is spoken text, and
    sound effects such as ``Pause, wait`` stay spoken.

    Examples
    --------
    >>> normalize_cues("<whisper>shh</whisper> and [Happy] day")
    '[whispering] shh and [happy] day'
    >>> normalize_cues("[excited] Yes! [  calm  ] ok")
    '[excited] Yes! [calm] ok'
    """
    if not text:
        return text
    text = _WHISPER_XML_RE.sub(lambda m: f"[whispering] {m.group(1).strip()}", text)
    text = rewrite_s1_parens(text)
    if official:
        text = _CUE_RE.sub(lambda m: _official_bracket(m.group(1)), text)
        text = _DOUBLE_SPACE_RE.sub(" ", text)
    else:
        text = _CUE_RE.sub(lambda m: _bracket(m.group(1)), text)
    parts = _LINE_SPLIT_RE.split(text)
    out: list[str] = []
    allow_lead = lead and not continued
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
            allow_lead = lead
        out.append("".join(spoken))
    return "".join(out)


_LEADING_CUES_RE = re.compile(rf"^\s*((?:\[[^\[\]\n]{{1,{_MAX_CUE_CHARS}}}\]\s*)*)")


class MoodCarry:
    """Give each sentence Fish speaks the emotion cue it would otherwise start without.

    Fish reads a cue for the sentence it opens, and each piece of a streamed reply goes to it
    on its own, so a sentence the model wrote without a cue starts cold: some voices begin it
    with a breath or a hum. This remembers the last emotion cue sent and puts it at the front
    of the next piece that starts a sentence and has no emotion of its own. A piece that
    continues a sentence is left as it is.

    Examples
    --------
    >>> carry = MoodCarry()
    >>> carry.apply("[happy] You sound relaxed. ")
    '[happy] You sound relaxed. '
    >>> carry.apply("Just perfect. ")
    '[happy] Just perfect. '
    >>> carry.apply("[laughing] Ha. ")
    '[happy] [laughing] Ha. '
    """

    def __init__(self) -> None:
        self._mood: str | None = None
        self._before = ""

    def apply(self, piece: str) -> str:
        """Return ``piece``, with the carried emotion in front when it starts a sentence without one.

        Parameters
        ----------
        piece : str
            The next text sent to Fish, already scrubbed.

        Returns
        -------
        str
            The piece to send.
        """
        body = piece.lstrip(" \t")
        out = piece
        starts = not self._before.strip() or ends_sentence(self._before)
        if starts and self._mood and body:
            lead = _LEADING_CUES_RE.match(body)
            if not (lead and last_emotion(lead.group(1))):
                out = f"{piece[: len(piece) - len(body)]}[{self._mood}] {body}"
        found = last_emotion(piece)
        if found:
            self._mood = found
        if piece.strip():
            self._before = piece
        return out


def ensure_lead_cue(text: str, *, default: str | None = None) -> str:
    """Prepend one cue when the reply has none at the start.

    Parameters
    ----------
    text : str
        Already scrubbed reply.
    default : str or None, optional
        Cue name without brackets. When given, the reply is prepended with
        ``[default]`` unless it already *starts* with a cue. When omitted or
        blank, the reply is left unchanged regardless of cue presence.

    Returns
    -------
    str
        ``text`` unchanged when it already starts with a ``[cue]``, or when
        no default cue is set. Without a default, also unchanged when any
        ``[cue]`` is present anywhere (existing behaviour). Empty or
        whitespace-only input is unchanged.

    Notes
    -----
    Fish speaks ``[clear throat]`` as ahem, and a short ``[clear]`` tag does
    the same. A missing cue stays as written words.
    """
    if not text.strip():
        return text
    tag = (default or "").strip().lower()
    if not tag:
        # No default: legacy behaviour — unchanged when any cue present.
        if _CUE_RE.search(text):
            return text
        return text
    # With a default: prepend unless the reply already *starts* with a cue.
    if _CUE_RE.match(text.lstrip()):
        return text
    return f"[{tag}] {text.lstrip()}"
