"""TTS/ASR scrubbers, junk gates, and sentence-or-40-char cuts."""

from __future__ import annotations

import re
import unicodedata

from fish_audio_suite_kit.defaults import DEFAULT_TTS_PARTIAL_CHARS

_THOUGHTS_RE = re.compile(
    r"<\s*(?:Thoughts?|thinking|reasoning|think)\s*>.*?"
    r"<\s*/\s*(?:Thoughts?|thinking|reasoning|think)\s*>",
    re.IGNORECASE | re.DOTALL,
)
_SPEAKER_RE = re.compile(r"<\|speaker:\d+\|>")
_STAGE_TOKEN_RE = re.compile(r"<\|(?:ACT|DELAY|CALL)\b[^|]*\|>", re.IGNORECASE)
_ANGLE_TOKEN_RE = re.compile(r"<\|[^|>]*\|>")
_MOSS_PAUSE_RE = re.compile(r"\[pause\s+\d+(?:\.\d+)?s\]", re.IGNORECASE)
_TTSD_SPEAKER_RE = re.compile(r"\[S[1-5]\]")
_MD_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+", re.MULTILINE)
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_MD_WRAP_RE = re.compile(r"[*_`~]+")
_PARENS_RE = re.compile(r"\([^)]*\)")
_VIBEVOICE_TS_RE = re.compile(r"\[\d+(?:\.\d+)?\s*[-–]\s*\d+(?:\.\d+)?\]\s*")
_SPEAKER_N_RE = re.compile(r"\bSpeaker\s+\d+\s*:\s*", re.IGNORECASE)
_DIALOGUE_RE = re.compile(
    r"(?:\[[^\]\n]{1,80}\]\s*)?"
    r"[\"\u201c]([^\"\u201d\n]+)[\"\u201d]"
)
_OPEN_DIALOGUE_RE = re.compile(
    r"(?:\[[^\]\n]{1,80}\]\s*)?"
    r'["\u201c](.+)$',
    re.DOTALL,
)
_NARRATION_RE = re.compile(r"^(She|He|They|Her|His|The|A|An)\b.*", re.IGNORECASE)
_NARRATION_VERB_RE = re.compile(
    r"\b(shifts|leans|smiles|laughs|settles|tilts|watches|murmurs|"
    r"reaches|moves|looks|dropping|rustle|stretches)\b",
    re.IGNORECASE,
)
_SENT_END = re.compile(r"[.!?؟।۔]+[\"'”’)]*\s")
_TRAIL_WORD = re.compile(r"(\d+|[A-Za-z]+)\s*$")

_ABBREVIATIONS = frozenset(
    {
        "dr",
        "mr",
        "mrs",
        "ms",
        "prof",
        "sr",
        "jr",
        "vs",
        "etc",
        "st",
        "ave",
        "inc",
        "ltd",
        "co",
        "no",
        "vol",
        "fig",
        "approx",
        "est",
    }
)

_BACKCHANNELS = frozenset(
    {
        "yeah",
        "yep",
        "yup",
        "ya",
        "sure",
        "ok",
        "okay",
        "k",
        "mhmm",
        "mhm",
        "mmhmm",
        "mm hmm",
        "mm-hmm",
        "mm hm",
        "mm-hm",
        "uh huh",
        "uh-huh",
        "uhhuh",
        "huh",
        "mm",
        "hmm",
        "right",
        "uh",
        "um",
        "ah",
    }
)

_QUIT = frozenset({"quit", "exit", "stop", "goodbye", "good bye", "bye"})

_EN_HALLUCINATION_PHRASES = frozenset(
    {
        "",
        ".",
        "...",
        "…",
        "thanks for watching",
        "thank you for watching",
        "please subscribe",
        "subscribe",
        "like and subscribe",
    }
)

_EMOJI_RE = re.compile(
    "[\U0001f300-\U0001faff\U00002700-\U000027bf\U0001f1e0-\U0001f1ff]+",
    flags=re.UNICODE,
)


def cjk_latin_counts(s: str) -> tuple[int, int]:
    cjk = latin = 0
    for ch in s:
        o = ord(ch)
        if (
            0x3040 <= o <= 0x30FF
            or 0x3400 <= o <= 0x4DBF
            or 0x4E00 <= o <= 0x9FFF
            or 0xF900 <= o <= 0xFAFF
            or 0xAC00 <= o <= 0xD7AF
        ):
            cjk += 1
        elif ch.isascii() and ch.isalpha():
            latin += 1
    return cjk, latin


def _looks_like_narration(text: str) -> bool:
    s = text.strip()
    if not s:
        return False
    if '"' in s or "\u201c" in s or "\u201d" in s:
        return False
    if not _NARRATION_RE.match(s):
        return False
    return bool(_NARRATION_VERB_RE.search(s))


def _strip_markdownish(text: str) -> str:
    cleaned = _MD_HEADING_RE.sub("", text)
    cleaned = _MD_LINK_RE.sub(r"\1", cleaned)
    cleaned = _URL_RE.sub("", cleaned)
    cleaned = _PARENS_RE.sub("", cleaned)
    return _MD_WRAP_RE.sub("", cleaned)


def scrub_tts(text: str, *, dialogue_only: bool = False) -> str:
    """Strip thoughts, stage junk, markdown, MOSS/TTSD markup; keep Fish [cues]."""
    if not text:
        return ""
    cleaned = _THOUGHTS_RE.sub("", text)
    cleaned = _STAGE_TOKEN_RE.sub("", cleaned)
    cleaned = _ANGLE_TOKEN_RE.sub("", cleaned)
    cleaned = _MOSS_PAUSE_RE.sub("", cleaned)
    cleaned = _TTSD_SPEAKER_RE.sub("", cleaned)
    cleaned = _strip_markdownish(cleaned)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = cleaned.strip()
    if not cleaned:
        return ""

    if dialogue_only:
        parts: list[str] = []
        for m in _DIALOGUE_RE.finditer(cleaned):
            inner = (m.group(1) or "").strip()
            if not inner or (inner.startswith("[") and inner.endswith("]")):
                continue
            if sum(ch.isascii() and ch.isalpha() for ch in inner) < 2:
                continue
            chunk = m.group(0).strip()
            if chunk:
                parts.append(chunk)
        if parts:
            return " ".join(parts)
        m = _OPEN_DIALOGUE_RE.match(cleaned.strip())
        if m:
            inner = (m.group(1) or "").strip().rstrip('"”').strip()
            latin = sum(ch.isascii() and ch.isalpha() for ch in inner)
            cue_only = inner.startswith("[") and "]" in inner[:24] and latin < 3
            if inner and latin >= 2 and not cue_only:
                cue = ""
                cm = re.match(r"(\[[^\]\n]{1,80}\]\s*)", cleaned.strip())
                if cm:
                    cue = cm.group(1)
                return f'{cue}"{inner}"'
        if any(ch in cleaned for ch in ('"', "“", "”")):
            return ""
        return cleaned

    return cleaned


def is_tts_junk(text: str) -> bool:
    if not text or not text.strip():
        return True
    s = text.strip()
    if len(s) <= 2:
        return True
    if s.count("[") > s.count("]"):
        return True
    bare = re.sub(r"\[[^\]\n]{0,80}\]", " ", s)
    for q in ('"', "\u201c", "\u201d"):
        bare = bare.replace(q, " ")
    bare = re.sub(r"\s+", " ", bare).strip()
    if not bare:
        return True
    cjk, latin = cjk_latin_counts(s)
    if cjk and latin < 3:
        return True
    if latin < 3:
        return True
    return bool(_looks_like_narration(s))


def scrub_asr(text: str, *, strip_speakers: bool = True) -> str:
    if not text:
        return ""
    cleaned = text
    if strip_speakers:
        cleaned = _SPEAKER_RE.sub("", cleaned)
        cleaned = _SPEAKER_N_RE.sub("", cleaned)
    cleaned = _VIBEVOICE_TS_RE.sub("", cleaned)
    cleaned = _ANGLE_TOKEN_RE.sub(" ", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def is_asr_hallucination(text: str, language: str | None = None) -> bool:
    if not text or not text.strip():
        return True
    s = text.strip()
    if len(s) <= 2:
        return True
    folded = re.sub(r"[.!,?]+$", "", s.lower()).strip()
    if folded in _EN_HALLUCINATION_PHRASES:
        return True
    if "<|nospeech|>" in s.lower() or folded == "nospeech":
        return True
    lang = (language or "").strip().lower()
    if (
        lang
        and lang not in {"en", "english", "eng"}
        and any(
            x in lang for x in ("zh", "chinese", "ja", "japanese", "ko", "korean", "yue", "cmn")
        )
    ):
        return True
    cjk, latin = cjk_latin_counts(s)
    if cjk and latin == 0:
        return True
    if cjk >= 1 and latin < 3 and cjk >= latin:
        return True
    letters = [ch for ch in s if ch.isalpha()]
    if not letters and not any(ch.isdigit() for ch in s):
        return True
    no_emoji = _EMOJI_RE.sub("", s).strip()
    if not no_emoji:
        return True
    if _ANGLE_TOKEN_RE.fullmatch(s.replace(" ", "")):
        return True
    tagged = _ANGLE_TOKEN_RE.sub("", s).strip()
    return bool(not tagged)


def is_backchannel(text: str) -> bool:
    s = re.sub(r"[.!,?]+", "", (text or "").strip().lower())
    s = re.sub(r"\s+", " ", s)
    return s in _BACKCHANNELS


def is_quit_utterance(text: str) -> bool:
    s = re.sub(r"[.!,?]+", "", (text or "").strip().lower())
    s = re.sub(r"\s+", " ", s)
    return s in _QUIT


def _skip_abbreviation(buf: str, end_start: int) -> bool:
    before = buf[:end_start]
    m = _TRAIL_WORD.search(before)
    if not m:
        return False
    w = m.group(1)
    if w.isdigit() and len(w) <= 2:
        return True
    if len(w) == 1 and w.isalpha():
        return True
    return w.lower() in _ABBREVIATIONS


def next_tts_cut(buf: str, *, partial_chars: int = DEFAULT_TTS_PARTIAL_CHARS) -> int:
    """Index to flush into Fish TTS, or -1 to keep buffering."""
    if not buf:
        return -1
    search_from = 0
    while True:
        m = _SENT_END.search(buf, search_from)
        if not m:
            break
        if _skip_abbreviation(buf, m.start()):
            search_from = m.end()
            continue
        return m.end()
    if len(buf) >= partial_chars:
        cut = buf.rfind(" ", 0, len(buf))
        if cut >= 12:
            return cut + 1
        return partial_chars
    return -1


def skip_empty_delta(piece: str) -> bool:
    return not (piece or "").strip()


def is_punctuation_only(text: str) -> bool:
    s = (text or "").strip()
    if not s:
        return True
    return all(unicodedata.category(ch).startswith("P") or ch.isspace() for ch in s)
