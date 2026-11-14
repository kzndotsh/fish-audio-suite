"""Fish ``[cue]`` tags.

Rewrites third-party markup and aliases into lowercase Fish cues, maps free-form
cues to official ones, turns a sentence-leading mood word into a cue, and carries
an emotion cue across the pieces of a streamed reply.
"""

from __future__ import annotations

import re
from typing import Final

from fish_audio_suite_kit.tts.cuts import ABBREVIATION_LOOKBACK, ends_sentence, sentence_cuts

__all__ = [
    "CLEAR_CUE",
    "MoodCarry",
    "ensure_lead_cue",
    "last_emotion",
    "normalize_cues",
    "official_cue",
    "split_cues",
    "strip_cue_tags",
]

# Official single-word emotions (basic + advanced), the common ones first. The default
# system prompt lists them in this order, and the sentence-lead pattern and the S1
# parenthesis rewriter read the same tuple.
# https://docs.fish.audio/api-reference/emotion-reference.md
FISH_EMOTIONS: Final = (
    *("happy", "sad", "angry", "excited", "calm", "nervous", "confident", "surprised"),
    *("delighted", "scared", "worried", "frustrated", "empathetic", "embarrassed", "proud"),
    *("relaxed", "grateful", "curious", "sarcastic", "hopeful", "disappointed", "determined"),
    *("satisfied", "optimistic", "moved", "nostalgic", "sympathetic", "compassionate"),
    *("confused", "uncertain", "doubtful", "anxious", "upset", "unhappy", "lonely", "bored"),
    *("indifferent", "resigned", "regretful", "guilty", "ashamed", "jealous", "envious"),
    *("disgusted", "disdainful", "contemptuous", "pessimistic", "depressed", "hysterical"),
)
_FISH_EMOTIONS: Final = frozenset(FISH_EMOTIONS)
# The only Fish tone tags that are one word and open a sentence
# ("Whispering, come here"). "in a hurry tone" and "soft tone" are phrases.
# "emphasis" marks a word mid-sentence. Sound effects are not leads: "Pause, wait"
# is speech.
_SPOKEN_TONE: Final = frozenset({"whispering", "shouting", "screaming"})
# Every word that can open a sentence as a mood lead.
LEAD_WORDS: Final = _FISH_EMOTIONS | _SPOKEN_TONE
# Tone tags that do not fit the one-word sentence lead.
_PHRASE_TONES: Final = frozenset({"emphasis", "in a hurry tone", "soft tone"})
# Fish's pauses, the short one first.
FISH_BREAKS: Final = ("break", "long-break")
# Fish's official sound effects.
FISH_SOUNDS: Final = frozenset(
    {
        *("laughing", "chuckling", "sighing", "sobbing", "crying loudly", "groaning"),
        *("panting", "gasping", "yawning", "snoring", "clear throat", "audience laughing"),
        *("background laughter", "crowd laughing"),
    }
)
# breath/cough/lip-smacking from older prompts, and common S2 cues from Fish's models
# guide that are not in the official lists.
_OTHER_SOUNDS: Final = frozenset({"breath", "cough", "lip-smacking", "inhale", "exhale"})

# Spellings models write that are not the Fish tag. [cough] is not aliased.
_CUE_ALIASES: Final = {
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

# Every tag name a (paren) or [bracket] is known by. Each alias and the tag it stands
# for are in it by construction, so an alias is never an unknown tag: (sob) is
# rewritten and [groan] is stripped like the tags they stand for.
_KNOWN_TAGS: Final = (
    _FISH_EMOTIONS
    | _SPOKEN_TONE
    | _PHRASE_TONES
    | frozenset(FISH_BREAKS)
    | FISH_SOUNDS
    | _OTHER_SOUNDS
    | frozenset(_CUE_ALIASES)
    | frozenset(_CUE_ALIASES.values())
)
# (happy) from the old S1 model, longest tag first so "laughing" wins over "laugh".
# Unknown (asides) are left for the scrubber.
_S1_PAREN_RE: Final = re.compile(
    r"\(\s*("
    + "|".join(re.escape(t) for t in sorted(_KNOWN_TAGS, key=len, reverse=True))
    + r")\s*\)",
    re.IGNORECASE,
)

# "Excited, hello" -> [excited] hello. Longest word first so a shorter emotion
# cannot take a prefix. Punctuation is required, so "I am anxious today" stays speech.
# A bare bracket ("Happy [chuckling]") is also accepted as the delimiter.
LEAD_PUNCT: Final = ",:!-\u2013\u2014\uff0c\uff01\uff1a"
SPOKEN_MOOD_LEAD_RE: Final = re.compile(
    r"^\s*("
    + "|".join(re.escape(word) for word in sorted(LEAD_WORDS, key=len, reverse=True))
    + rf")\s*(?:[{re.escape(LEAD_PUNCT)}]+\s*|(?=\[))",
    re.IGNORECASE,
)


def cue_pattern(
    max_chars: int, *, min_chars: int = 1, capture: bool = False, flat: bool = False
) -> str:
    """Return the regex source for one ``[cue]``: a bracket with no newline or ``]`` inside.

    Parameters
    ----------
    max_chars : int
        The longest cue text it matches. The bound keeps a scan from each ``[`` short, so
        a long run of openers with no closer costs a fixed amount per opener.
    min_chars : int, optional
        The shortest cue text. Default 1; 0 also matches ``[]``.
    capture : bool, optional
        Put the cue text in a group. Default False, to embed in a larger pattern.
    flat : bool, optional
        Refuse a ``[`` inside the cue too, so ``[[calm]`` matches only ``[calm]``.
        Default False.

    Returns
    -------
    str
        The pattern source.

    Examples
    --------
    >>> import re
    >>> re.findall(cue_pattern(80, flat=True), "[[calm] hi [a [sad]")
    ['[calm]', '[sad]']
    """
    body = (
        rf"[^\[\]\n]{{{min_chars},{max_chars}}}" if flat else rf"[^\]\n]{{{min_chars},{max_chars}}}"
    )
    return rf"\[({body})\]" if capture else rf"\[{body}\]"


# Longest cue text and longest whispered span the patterns look at.
_MAX_CUE_CHARS: Final = 200
_MAX_WHISPER_CHARS: Final = 1000
# Any [cue], including free-form S2 text. A newline ends the tag. Group 1 is the text.
_CUE_RE: Final = re.compile(cue_pattern(_MAX_CUE_CHARS, capture=True))
WHISPER_TAG: Final = "whisper"


def whisper_tag(*, close: bool, space: str = r"\s{0,8}") -> str:
    """Return the regex source for ``<whisper>`` or ``</whisper>`` as models write it.

    Parameters
    ----------
    close : bool
        True for the closing tag.
    space : str, optional
        The pattern for the space allowed around the name and the slash. Default a
        bounded run, so the rewrite stays linear.

    Returns
    -------
    str
        A pattern with no group.
    """
    slash = rf"/{space}" if close else ""
    return rf"<{space}{slash}{WHISPER_TAG}{space}>"


# Some models write <whisper>...</whisper> instead of a Fish cue.
_WHISPER_XML_RE: Final = re.compile(
    rf"{whisper_tag(close=False)}(.{{0,{_MAX_WHISPER_CHARS}}}?){whisper_tag(close=True)}",
    re.IGNORECASE | re.DOTALL,
)

# One or more [cues] already at the start of a sentence. Group 2 is the spoken rest.
# A cue body cannot contain "[", so "[[page]]" is not a stack.
_LEAD_STACK_RE: Final = re.compile(r"^((?:\[[^\[\]]+\]\s*)+)(.*)$", re.DOTALL)
# Each cue inside that stack, so [sad][whispering] stays two tags. A newline does not end
# it: the stack it reads from may span lines.
_INNER_CUE_RE: Final = re.compile(rf"\[([^\]]{{1,{_MAX_CUE_CHARS}}})\]")
# Keep the newlines, so a blank line is not folded into the next sentence.
_LINE_SPLIT_RE: Final = re.compile(r"(\n+)")


# The short [clear] tag. Fish's list names only [clear throat], yet Fish speaks [clear] as
# an ahem too, so it counts as a cue. It was once the default lead cue.
CLEAR_CUE: Final = "clear"
_PAREN_CUE_NAMES: Final = _KNOWN_TAGS | {CLEAR_CUE}
# [name] for any known cue name, longest first so "laughing" beats "laugh".
# A tag inside a larger bracket expression ("[[happy]]") is literal text.
_KNOWN_CUE_TAG_RE: Final = re.compile(
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
    return word in _CUE_ALIASES or word in LEAD_WORDS or word == CLEAR_CUE


def rewrite_s1_parens(text: str) -> str:
    """Rewrite S1 ``(happy)`` tags as ``[happy]``. An unknown ``(aside)`` is left for the scrubber."""
    return _S1_PAREN_RE.sub(lambda m: _bracket(m.group(1)), text)


def _lead(tags: str, rest: str) -> str:
    return f"{tags} {rest}" if rest else tags


def _sentence_spans(part: str, before: str = "") -> list[tuple[str, str]]:
    # Only a real sentence end splits, so "Dr. Happy" is not a new sentence. A stop at
    # the start of a streamed chunk ends a sentence only by what came before it.
    spans: list[tuple[str, str]] = []
    start = 0
    for cut in sentence_cuts(before + part):
        end = cut - len(before)
        if end <= 0:
            continue
        piece = part[start:end]
        body = piece.rstrip()
        spans.append((body, piece[len(body) :]))
        start = end
    if start < len(part):
        spans.append((part[start:], ""))
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
        m2 = SPOKEN_MOOD_LEAD_RE.match(chunk)
        if m2:
            return _lead(_bracket(m2.group(1)), chunk[m2.end() :].lstrip())
    return chunk


def normalize_cues(
    text: str,
    *,
    lead: bool = False,
    continued: bool = False,
    before: str = "",
    official: bool = False,
    directed: bool = False,
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
    before : str, optional
        Text already sent on the line ``text`` continues. It is read only to tell
        whether a stop at the start of ``text`` ends a sentence (``Dr`` then
        ``. Happy`` does not). Default empty.
    official : bool, optional
        Keep only Fish's own cues. Any other cue is replaced by the nearest official one
        (``[smiling]`` becomes ``[happy]``, ``[soft chuckle]`` becomes ``[chuckling]``) or
        removed when no word in it suggests one (``[in a storytelling voice]``). Some voices
        act a description out as a sound, such as a hum, instead of changing tone. Default
        False, which keeps free-form S2 cues as written.
    directed : bool, optional
        True for a model that is directed in plain language (``takes_directions``). Every
        ``[direction]`` is then left exactly as written, ``official`` and ``lead`` are
        ignored, and ``<whisper>…</whisper>`` stays a paired tag. Default False.

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
    >>> normalize_cues("[Barely holding it together] <whisper>fine</whisper>", directed=True)
    '[Barely holding it together] <whisper>fine</whisper>'
    """
    if not text:
        return text
    if directed:
        return rewrite_s1_parens(text)
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
    # Whether a stop ends a sentence depends only on the word before it, so a short tail
    # of the line is enough, and a long single-line reply is not rescanned per chunk.
    context = before.rsplit("\n", 1)[-1][-2 * ABBREVIATION_LOOKBACK :]
    for part in parts:
        if not part or part.isspace():
            out.append(part)
            context = ""
            # A new line starts a sentence even when the chunk continued one, as it does
            # in the whole reply.
            if "\n" in part:
                allow_lead = lead
            continue
        spoken: list[str] = []
        for sent, sep in _sentence_spans(part, context):
            if not sent.strip():
                if sep:
                    spoken.append(sep)
                continue
            spoken.append(_one_sentence(sent, lead=allow_lead) + sep)
            allow_lead = lead
        out.append("".join(spoken))
        context = ""
    return "".join(out)


_LEADING_CUES_RE: Final = re.compile(rf"^\s*((?:{cue_pattern(_MAX_CUE_CHARS, flat=True)}\s*)*)")


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
        Cue name without brackets, lowercased before use. When omitted or
        blank, the reply is returned unchanged.

    Returns
    -------
    str
        ``text`` unchanged when it is blank, when no default is set, or when
        it already starts with a ``[cue]``. Otherwise ``[default]``, a space,
        and ``text`` without its leading whitespace.

    Notes
    -----
    Pick an emotion such as ``calm`` as the default. ``[clear]`` is not
    neutral, because Fish speaks it as an ahem, the same as ``[clear throat]``.

    Examples
    --------
    >>> ensure_lead_cue("  hi", default="Calm")
    '[calm] hi'
    >>> ensure_lead_cue("[sad] hi", default="calm")
    '[sad] hi'
    """
    if not text.strip():
        return text
    tag = (default or "").strip().lower()
    if not tag or _CUE_RE.match(text.lstrip()):
        return text
    return f"[{tag}] {text.lstrip()}"
