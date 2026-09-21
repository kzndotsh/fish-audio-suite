"""Fish [cue] tags: mood leads, lowercase, and third-party aliases."""

from __future__ import annotations

import re

_SPOKEN_MOOD_LEAD = re.compile(
    r"^\s*("
    r"seductive|playful|excited|happy|delighted|calm|relaxed|confident|"
    r"curious|surprised|sad|empathetic|nervous|sarcastic|hopeful|"
    r"determined|embarrassed|jealous|lonely|nostalgic|whispering|"
    r"soft|warm|intimate|flirty|teasing|breathless|angry|scared|"
    r"worried|grateful|proud|shy"
    r")\s*[,:!\-–—]+\s*",
    re.IGNORECASE,
)

_CUE_RE = re.compile(r"\[([^\]\n]+)\]")
_WHISPER_XML_RE = re.compile(
    r"<\s*whisper\s*>(.*?)<\s*/\s*whisper\s*>",
    re.IGNORECASE | re.DOTALL,
)

_CUE_ALIASES = {
    "laugh": "laughing",
    "laughs": "laughing",
    "whisper": "whispering",
    "whispers": "whispering",
    "pause": "break",
    "sigh": "sighing",
    "chuckle": "chuckling",
}

# S1 / V1.6 used parentheses. Rewrite known tags to S2 [brackets] before asides are stripped.
_S1_PAREN_TAGS = frozenset(
    {
        "break",
        "long-break",
        "breath",
        "cough",
        "lip-smacking",
        "pause",
        "laugh",
        "laughs",
        "laughing",
        "chuckle",
        "chuckling",
        "sigh",
        "sighing",
        "whisper",
        "whispers",
        "whispering",
        "emphasis",
        "shouting",
        "screaming",
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
        "in a hurry tone",
        "soft tone",
        "crying loudly",
        "clear throat",
        "audience laughing",
        "background laughter",
        "crowd laughing",
        "sobbing",
        "panting",
        "gasping",
        "yawning",
        "snoring",
        "groaning",
    }
)
_S1_PAREN_RE = re.compile(
    r"\(\s*("
    + "|".join(re.escape(t) for t in sorted(_S1_PAREN_TAGS, key=len, reverse=True))
    + r")\s*\)",
    re.IGNORECASE,
)


def _alias_cue(inner: str) -> str:
    tag = inner.strip().lower()
    return _CUE_ALIASES.get(tag, tag)


def _rewrite_cues(text: str) -> str:
    def repl(m: re.Match[str]) -> str:
        return f"[{_alias_cue(m.group(1))}]"

    return _CUE_RE.sub(repl, text)


def rewrite_s1_parens(text: str) -> str:
    """`(happy)` / `(break)` → `[happy]` / `[break]`. Unknown `(asides)` left for scrubbers."""
    return _S1_PAREN_RE.sub(lambda m: f"[{_alias_cue(m.group(1))}]", text)


def _one_sentence(chunk: str) -> str:
    chunk = chunk.strip()
    if not chunk:
        return chunk
    m = re.match(r"^((?:\[[^\]]+\]\s*)+)(.*)$", chunk, flags=re.DOTALL)
    if m:
        cues = re.findall(r"\[([^\]]+)\]", m.group(1))
        if cues:
            tags = " ".join(f"[{_alias_cue(c)}]" for c in cues)
            rest = m.group(2).lstrip()
            return f"{tags} {rest}" if rest else tags
    m2 = _SPOKEN_MOOD_LEAD.match(chunk)
    if m2:
        tag = _alias_cue(m2.group(1))
        rest = chunk[m2.end() :].lstrip()
        return f"[{tag}] {rest}" if rest else f"[{tag}]"
    return chunk


def normalize_cues(text: str) -> str:
    """Lowercase [Tags]; keep stacked leads; map aliases; convert 'Excited, …' → '[excited] …'."""
    if not text:
        return text
    text = _WHISPER_XML_RE.sub(lambda m: f"[whispering] {m.group(1).strip()}", text)
    text = rewrite_s1_parens(text)
    text = _rewrite_cues(text)
    parts = re.split(r"(\n+)", text)
    out: list[str] = []
    for part in parts:
        if not part or part.isspace() or set(part) <= {"\n"}:
            out.append(part)
            continue
        sents = re.split(r"(?<=[.!?…])\s+", part)
        out.append(" ".join(_one_sentence(s) for s in sents if s.strip()))
    return "".join(out)


def ensure_lead_cue(text: str, *, default: str = "clear") -> str:
    """If the model emitted no [cue] at all, prepend one. Does not tag every sentence."""
    if not text.strip():
        return text
    if _CUE_RE.search(text):
        return text
    tag = default.strip().lower() or "clear"
    return f"[{tag}] {text.lstrip()}"


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+")
CUE_BEAT_MIN_SENTS = 4
CUE_BEAT_GAP_SENTS = 2
CUE_BEAT_TAGS = ("calm", "curious", "soft tone", "hopeful")


def spread_cues(
    text: str,
    *,
    min_sents: int = CUE_BEAT_MIN_SENTS,
    gap_sents: int = CUE_BEAT_GAP_SENTS,
    beats: tuple[str, ...] = CUE_BEAT_TAGS,
) -> str:
    """On long replies, insert a beat cue every `gap_sents` untagged sentences.

    Short replies are unchanged besides `ensure_lead_cue`. Existing cues reset the gap.
    """
    text = ensure_lead_cue(text)
    if not text.strip():
        return text
    sents = [s for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    if len(sents) < min_sents:
        return text
    out: list[str] = []
    since = 0
    beat_i = 0
    last_tag = ""
    for sent in sents:
        if _CUE_RE.search(sent):
            since = 0
            found = _CUE_RE.findall(sent)
            if found:
                last_tag = found[-1].strip().lower()
            out.append(sent)
            continue
        since += 1
        piece = sent
        if since >= gap_sents:
            tag = beats[beat_i % len(beats)]
            beat_i += 1
            if tag == last_tag and len(beats) > 1:
                tag = beats[beat_i % len(beats)]
                beat_i += 1
            last_tag = tag
            since = 0
            piece = f"[{tag}] {sent.lstrip()}"
        out.append(piece)
    return " ".join(out)
