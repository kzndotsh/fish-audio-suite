"""Fish [cue] tags: mood leads, lowercase, and third-party aliases."""

from __future__ import annotations

import re

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
    + r")\s*[,:!\-–—]+\s*",
    re.IGNORECASE,
)

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


# One or more [cues] already at the start of a sentence. Group 2 is the spoken rest.
_LEAD_STACK_RE = re.compile(r"^((?:\[[^\]]+\]\s*)+)(.*)$", re.DOTALL)
# Each cue inside that stack, so [sad][whispering] stays two tags.
_INNER_CUE_RE = re.compile(r"\[([^\]]+)\]")
# Split after a sentence ender and keep the ender on the sentence.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+")
# Keep the newlines, so a blank line is not folded into the next sentence.
_LINE_SPLIT_RE = re.compile(r"(\n+)")


def _bracket(inner: str) -> str:
    tag = inner.strip().lower()
    return f"[{_CUE_ALIASES.get(tag, tag)}]"


def rewrite_s1_parens(text: str) -> str:
    """`(happy)` / `(break)` → `[happy]` / `[break]`. Unknown `(asides)` left for scrubbers."""
    return _S1_PAREN_RE.sub(lambda m: _bracket(m.group(1)), text)


def _lead(tags: str, rest: str) -> str:
    return f"{tags} {rest}" if rest else tags


def _one_sentence(chunk: str) -> str:
    chunk = chunk.strip()
    if not chunk:
        return chunk
    m = _LEAD_STACK_RE.match(chunk)
    if m:
        cues = _INNER_CUE_RE.findall(m.group(1))
        if cues:
            tags = " ".join(_bracket(c) for c in cues)
            return _lead(tags, m.group(2).lstrip())
    m2 = _SPOKEN_MOOD_LEAD.match(chunk)
    if m2:
        return _lead(_bracket(m2.group(1)), chunk[m2.end() :].lstrip())
    return chunk


def normalize_cues(text: str) -> str:
    """Rewrite third-party mood markup into Fish ``[cue]`` tags.

    Parameters
    ----------
    text : str
        Model text that may contain ``[Tags]``, S1 ``(happy)``, a mood lead
        such as ``Excited, …``, or ``<whisper>…</whisper>``.

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
    The lead list is Fish's single-word emotions, plus whispering, shouting,
    and screaming. The same word mid-sentence is spoken text. Sound effects
    such as ``Pause, wait`` stay spoken.
    """
    if not text:
        return text
    text = _WHISPER_XML_RE.sub(lambda m: f"[whispering] {m.group(1).strip()}", text)
    text = rewrite_s1_parens(text)
    text = _CUE_RE.sub(lambda m: _bracket(m.group(1)), text)
    parts = _LINE_SPLIT_RE.split(text)
    out: list[str] = []
    for part in parts:
        if not part or part.isspace():
            out.append(part)
            continue
        sents = _SENTENCE_SPLIT_RE.split(part)
        out.append(" ".join(_one_sentence(s) for s in sents if s.strip()))
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
