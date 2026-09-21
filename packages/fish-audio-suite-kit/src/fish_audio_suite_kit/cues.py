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
    re.I,
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
}


def _alias_cue(inner: str) -> str:
    tag = inner.strip().lower()
    return _CUE_ALIASES.get(tag, tag)


def _rewrite_cues(text: str) -> str:
    def repl(m: re.Match[str]) -> str:
        return f"[{_alias_cue(m.group(1))}]"

    return _CUE_RE.sub(repl, text)


def _one_sentence(chunk: str) -> str:
    chunk = chunk.strip()
    if not chunk:
        return chunk
    m = re.match(r"^((?:\[[^\]]+\]\s*)+)(.*)$", chunk, flags=re.S)
    if m:
        cues = re.findall(r"\[([^\]]+)\]", m.group(1))
        if cues:
            tag = _alias_cue(cues[0])
            rest = m.group(2).lstrip()
            return f"[{tag}] {rest}" if rest else f"[{tag}]"
    m2 = _SPOKEN_MOOD_LEAD.match(chunk)
    if m2:
        tag = _alias_cue(m2.group(1))
        rest = chunk[m2.end() :].lstrip()
        return f"[{tag}] {rest}" if rest else f"[{tag}]"
    return chunk


def normalize_cues(text: str) -> str:
    """Lowercase [Tags]; map aliases; convert 'Excited, …' → '[excited] …'."""
    if not text:
        return text
    text = _WHISPER_XML_RE.sub(lambda m: f"[whispering] {m.group(1).strip()}", text)
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
