"""TTS/ASR scrubbers, junk gates, and sentence-or-40-char cuts."""

from __future__ import annotations

import re
import zlib

from fish_audio_suite_kit.cues import rewrite_s1_parens
from fish_audio_suite_kit.defaults import SuiteDefaults

_PARTIAL_CHARS = SuiteDefaults().tts_partial_chars
_MIN_WORD_CUT = 12
_CUE_LOOKAHEAD = 24

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
_LEAD_CUE = r"\[[^\]\n]{1,80}\]\s*"
_EMPTY_CUE_RE = re.compile(r"\[[^\]\n]{0,80}\]")
_LEAD_CUE_RE = re.compile(rf"({_LEAD_CUE})")
_DIALOGUE_RE = re.compile(
    rf"(?:{_LEAD_CUE})?"
    r"[\"\u201c]([^\"\u201d\n]+)[\"\u201d]"
)
_OPEN_DIALOGUE_RE = re.compile(
    rf"(?:{_LEAD_CUE})?"
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
        "嗯",
        "嗯嗯",
        "啊",
        "啊啊",
        "哦",
        "噢",
        "喔",
        "唔",
        "呃",
        "唉",
        "哎",
        "诶",
        "欸",
        "哼",
        "呵",
        "呀",
        "哇",
        "呐",
        "うん",
        "えっと",
        "음",
        "응",
        "어",
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
        "please like and subscribe",
        "subtitles by the amaraorg community",
        "transcribed by",
        "i hope you enjoyed the video",
    }
)
_CJK_HALLUCINATION_PHRASES = frozenset({"谢谢观看", "感谢观看", "请订阅", "字幕"})
_ASR_PUNCT_RE = re.compile(r"[\s.。、，,!?！？…·・~～'\"“”‘’]+")
_FOLD_PUNCT_RE = re.compile(r"[.。、，,!?！？…·・~～]+")
_SPACE_RE = re.compile(r"\s+")
_GZIP_MIN_BYTES = 48
_GZIP_RATIO = 2.4

_EMOJI_RE = re.compile(
    "[\U0001f300-\U0001faff\U00002700-\U000027bf\U0001f1e0-\U0001f1ff]+",
    flags=re.UNICODE,
)

_QUOTE_RE = re.compile('["“”]')
_CLOSE_QUOTES = '"”'


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
        elif ch.isascii() and ch.isalpha():
            latin += 1
    return cjk, latin


def _folded(text: str) -> str:
    s = _FOLD_PUNCT_RE.sub("", (text or "").strip().lower())
    return _SPACE_RE.sub(" ", s)


def _gzip_repetitive(text: str) -> bool:
    raw = text.encode("utf-8")
    if len(raw) < _GZIP_MIN_BYTES:
        return False
    compressed = zlib.compress(raw)
    return (len(raw) / max(len(compressed), 1)) >= _GZIP_RATIO


def _too_thin(s: str, *, min_latin: int) -> bool:
    stripped = s.strip()
    if not stripped:
        return True
    cjk, latin = _cjk_latin_counts(stripped)
    if cjk >= 2:
        return False
    if len(stripped) <= 2:
        return True
    return latin < min_latin


def _looks_like_narration(text: str) -> bool:
    s = text.strip()
    if not s:
        return False
    if _QUOTE_RE.search(s):
        return False
    if not _NARRATION_RE.match(s):
        return False
    return bool(_NARRATION_VERB_RE.search(s))


_Replacement = tuple[tuple[re.Pattern[str], str], ...]
_TTS_ERASE: _Replacement = (
    (_THOUGHTS_RE, ""),
    (_STAGE_TOKEN_RE, ""),
    (_MOSS_PAUSE_RE, ""),
    (_TTSD_SPEAKER_RE, ""),
)
_ASR_SPEAKERS: _Replacement = (
    (_SPEAKER_RE, ""),
    (_SPEAKER_N_RE, ""),
)
_MARKDOWN_SUBS: _Replacement = (
    (_MD_HEADING_RE, ""),
    (_MD_LINK_RE, r"\1"),
    (_URL_RE, ""),
    (_PARENS_RE, ""),
)


def _replace(text: str, pairs: _Replacement) -> str:
    for pattern, repl in pairs:
        text = pattern.sub(repl, text)
    return text


def _strip_markdownish(text: str) -> str:
    cleaned = _replace(rewrite_s1_parens(text), _MARKDOWN_SUBS)
    parts: list[str] = []
    last = 0
    for m in _ANGLE_TOKEN_RE.finditer(cleaned):
        parts.append(_MD_WRAP_RE.sub("", cleaned[last : m.start()]))
        parts.append(m.group(0))
        last = m.end()
    parts.append(_MD_WRAP_RE.sub("", cleaned[last:]))
    return "".join(parts)


_TRAIL_SPACE_RE = re.compile(r"[ \t]+\n")
_H_SPACE_RE = re.compile(r"[ \t]+")
_BREAKS_RE = re.compile(r"\n{3,}")


def _tidy_breaks(text: str) -> str:
    return _BREAKS_RE.sub("\n\n", text).strip()


def scrub_tts(text: str) -> str:
    """Strip thoughts, CosyVoice stage junk, markdown, MOSS/TTSD markup; keep Fish [cues] and <|…|> control tokens."""
    if not text:
        return ""
    cleaned = _replace(text, _TTS_ERASE)
    cleaned = _strip_markdownish(cleaned)
    cleaned = _TRAIL_SPACE_RE.sub("\n", cleaned)
    return _tidy_breaks(cleaned)


def _speakable_quote(inner: str) -> bool:
    if not inner or (inner.startswith("[") and inner.endswith("]")):
        return False
    return _cjk_latin_counts(inner)[1] >= 2


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
    latin = _cjk_latin_counts(inner)[1]
    cue_only = inner.startswith("[") and "]" in inner[:_CUE_LOOKAHEAD] and latin < 3
    if not inner or latin < 2 or cue_only:
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
    if _QUOTE_RE.search(text):
        return ""
    return text


def is_tts_junk(text: str) -> bool:
    s = (text or "").strip()
    if s.count("[") > s.count("]"):
        return True
    bare = _QUOTE_RE.sub(" ", _EMPTY_CUE_RE.sub(" ", s))
    bare = _SPACE_RE.sub(" ", bare).strip()
    if not bare:
        return True
    if _too_thin(s, min_latin=3):
        return True
    return _looks_like_narration(s)


def utf8_text(text: str) -> str:
    return text.encode("utf-8", "replace").decode("utf-8")


def scrub_asr(text: str, *, strip_speakers: bool = True) -> str:
    if not text:
        return ""
    cleaned = text
    if strip_speakers:
        cleaned = _replace(cleaned, _ASR_SPEAKERS)
    cleaned = _VIBEVOICE_TS_RE.sub("", cleaned)
    cleaned = _ANGLE_TOKEN_RE.sub(" ", cleaned)
    cleaned = _H_SPACE_RE.sub(" ", cleaned)
    return utf8_text(_tidy_breaks(cleaned))


def _known_hallucination(s: str) -> bool:
    if _known_phrase(s, _EN_HALLUCINATION_PHRASES) or "nospeech" in s.lower():
        return True
    return _ASR_PUNCT_RE.sub("", s) in _CJK_HALLUCINATION_PHRASES


def _without_marks(s: str) -> str | None:
    if not _EMOJI_RE.sub("", s).strip():
        return None
    if _ANGLE_TOKEN_RE.fullmatch(s.replace(" ", "")):
        return None
    tagged = _ANGLE_TOKEN_RE.sub("", s).strip()
    return tagged or None


def is_asr_hallucination(text: str) -> bool:
    if not text or not text.strip():
        return True
    s = text.strip()
    if _known_hallucination(s):
        return True
    tagged = _without_marks(s)
    if tagged is None or _gzip_repetitive(tagged):
        return True
    cjk, latin = _cjk_latin_counts(s)
    if latin == 0 and cjk == 1:
        return True
    if cjk:
        return False
    return latin < 3


def _known_phrase(text: str, phrases: frozenset[str]) -> bool:
    return _folded(text) in phrases


def is_backchannel(text: str) -> bool:
    return _known_phrase(text, _BACKCHANNELS)


def is_quit_utterance(text: str) -> bool:
    return _known_phrase(text, _QUIT)


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


def _sentence_cut(buf: str) -> int:
    search_from = 0
    while True:
        m = _SENT_END.search(buf, search_from)
        if not m:
            return -1
        if _skip_abbreviation(buf, m.start()):
            search_from = m.end()
            continue
        return m.end()


def _partial_cut(buf: str, partial_chars: int) -> int:
    if len(buf) < partial_chars:
        return -1
    cut = buf.rfind(" ", 0, len(buf))
    if cut >= _MIN_WORD_CUT:
        return cut + 1
    return partial_chars


def next_tts_cut(buf: str, *, partial_chars: int = _PARTIAL_CHARS) -> int:
    """Index to flush into Fish TTS, or -1 to keep buffering."""
    if not buf:
        return -1
    end = _sentence_cut(buf)
    if end >= 0:
        return end
    return _partial_cut(buf, partial_chars)


def split_tts_piece(buf: str, partial_chars: int, *, flush_rest: bool) -> tuple[str, str] | None:
    """One TTS piece and the unsent tail. None means keep buffering."""
    cut = next_tts_cut(buf, partial_chars=partial_chars)
    if cut < 0:
        if not flush_rest:
            return None
        return buf, ""
    return buf[:cut], buf[cut:]


def skip_empty_delta(piece: str) -> bool:
    return not (piece or "").strip()
