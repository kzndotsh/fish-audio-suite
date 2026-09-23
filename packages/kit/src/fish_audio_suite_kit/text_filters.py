"""TTS/ASR scrubbers, junk gates, and sentence-or-40-char cuts."""

from __future__ import annotations

import re
import zlib

from fish_audio_suite_kit.cues import rewrite_s1_parens
from fish_audio_suite_kit.defaults import SuiteDefaults

# Flush a long clause when no sentence end has arrived. Default is 40.
_PARTIAL_CHARS = SuiteDefaults().tts_partial_chars
# A space cut shorter than this is a fragment, so the buffer waits or hard-cuts.
_MIN_WORD_CUT = 12
# A quote whose only content is a [cue] in this window is not speech.
_CUE_LOOKAHEAD = 24

# Chain-of-thought blocks. Speaking them reads the model's scratch work aloud.
_THOUGHTS_RE = re.compile(
    r"<\s*(?:Thoughts?|thinking|reasoning|think)\s*>.*?"
    r"<\s*/\s*(?:Thoughts?|thinking|reasoning|think)\s*>",
    re.IGNORECASE | re.DOTALL,
)
# ASR diarization token. Not a phoneme, so the transcript drops it.
_SPEAKER_RE = re.compile(r"<\|speaker:\d+\|>")
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
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_MD_WRAP_RE = re.compile(r"[*_`~]+")
# Stage asides. Runs after S1 (happy) is rewritten, so a Fish paren tag survives.
_PARENS_RE = re.compile(r"\([^)]*\)")
# VibeVoice timestamp [0.00-1.23]. The hyphen class includes an en dash.
_VIBEVOICE_TS_RE = re.compile(r"\[\d+(?:\.\d+)?\s*[-–]\s*\d+(?:\.\d+)?\]\s*")
# "Speaker 1:" labels from engines that do not use <|speaker:N|>.
_SPEAKER_N_RE = re.compile(r"\bSpeaker\s+\d+\s*:\s*", re.IGNORECASE)
# Optional [cue] glued to a quote, so dialogue extraction does not drop the cue.
_LEAD_CUE = r"\[[^\]\n]{1,80}\]\s*"
# Cue with an empty body too, so "[ ]" alone still counts as no speech.
_EMPTY_CUE_RE = re.compile(r"\[[^\]\n]{0,80}\]")
_LEAD_CUE_RE = re.compile(rf"({_LEAD_CUE})")
# Closed quotes, straight or curly. The cue in front is optional.
_DIALOGUE_RE = re.compile(
    rf"(?:{_LEAD_CUE})?"
    r"[\"\u201c]([^\"\u201d\n]+)[\"\u201d]"
)
# A quote opened on this line and not closed yet. Streaming replies arrive this way.
_OPEN_DIALOGUE_RE = re.compile(
    rf"(?:{_LEAD_CUE})?"
    r'["\u201c](.+)$',
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
# Sentence end, including Arabic, Devanagari, and Urdu stops. Trailing quotes
# stay on the sentence so the cut does not split "end." from the closer.
_SENT_END = re.compile(r"[.!?؟।۔]+[\"'”’)]*\s")
# The token immediately before a period: abbreviation, initial, or "1.".
_TRAIL_WORD = re.compile(r"(\d+|[A-Za-z]+)\s*$")

# Periods that are not sentence ends. A one-letter initial is handled separately.
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

# Listener noises. Duplex hears one of these and keeps listening instead of answering.
# English spellings plus the CJK, Japanese, and Korean fillers models actually emit.
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

# Whole utterance, after folding. "stop" mid-sentence is not a quit.
_QUIT = frozenset({"quit", "exit", "stop", "goodbye", "good bye", "bye"})

# Whisper and caption-tool watermarks on silence. Empty and "..." are in here
# because a folded blank transcript is the same shape.
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
# Same watermarks in Chinese. Punctuation is stripped before this lookup.
_CJK_HALLUCINATION_PHRASES = frozenset({"谢谢观看", "感谢观看", "请订阅", "字幕"})
# Strip spaces and quotes too, so "谢谢观看。" matches the phrase with no marks.
_ASR_PUNCT_RE = re.compile(r"[\s.。、，,!?！？…·・~～'\"“”‘’]+")
# Fold for phrase lists. Spaces stay, so "uh huh" does not become "uhhuh".
_FOLD_PUNCT_RE = re.compile(r"[.。、，,!?！？…·・~～]+")
_SPACE_RE = re.compile(r"\s+")
# Below this size, gzip ratio is noise. A stuck caption loop compresses past the ratio.
_GZIP_MIN_BYTES = 48
_GZIP_RATIO = 2.4

# Emoji-only transcripts are not speech. Misc symbols, dingbats, and flags.
_EMOJI_RE = re.compile(
    "[\U0001f300-\U0001faff\U00002700-\U000027bf\U0001f1e0-\U0001f1ff]+",
    flags=re.UNICODE,
)

# Straight and curly double quotes. Used to tell narration from dialogue.
_QUOTE_RE = re.compile('["“”]')
# Closers stripped from an unclosed quote before the speakable check.
_CLOSE_QUOTES = '"”'


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
        elif ch.isascii() and ch.isalpha():
            latin += 1
    return cjk, latin


def _folded(text: str) -> str:
    s = _FOLD_PUNCT_RE.sub("", (text or "").strip().lower())
    return _SPACE_RE.sub(" ", s)


def _gzip_repetitive(text: str) -> bool:
    """Return whether gzip shrinkage looks like a stuck loop.

    Short strings are never repetitive. ASR models sometimes emit the same
    caption many times; the compression ratio catches that without a phrase list.
    """
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
# TTS only. Fish [cue] and <|phoneme|> tokens are not in this list.
_TTS_ERASE: _Replacement = (
    (_THOUGHTS_RE, ""),
    (_STAGE_TOKEN_RE, ""),
    (_MOSS_PAUSE_RE, ""),
    (_TTSD_SPEAKER_RE, ""),
)
# Optional. The proxy leaves speaker labels when the client asks to keep them.
_ASR_SPEAKERS: _Replacement = (
    (_SPEAKER_RE, ""),
    (_SPEAKER_N_RE, ""),
)
# Link text is kept; the URL is not. Parens run after S1 tags are brackets.
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


# Trailing spaces before a newline. Horizontal runs collapse; newlines stay.
_TRAIL_SPACE_RE = re.compile(r"[ \t]+\n")
_H_SPACE_RE = re.compile(r"[ \t]+")
# Three or more blank lines become one paragraph break.
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
    """Return whether scrubbed text should not be sent to Fish TTS.

    Parameters
    ----------
    text : str
        Text after ``scrub_tts``. Cue-only text, an unmatched ``[``, narration
        that is not speech, or fewer than three Latin letters counts as junk.
        A real CJK sentence is not junk.

    Returns
    -------
    bool
        True when the proxy should return silence instead of calling Fish.
    """
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


def scrub_asr(text: str, *, strip_speakers: bool = True) -> str:
    """Strip ASR markup Fish and other engines leave in the transcript.

    Parameters
    ----------
    text : str
        Raw transcript.
    strip_speakers : bool, optional
        Drop ``Speaker 1:`` style labels. Default True. The proxy turns this
        off unless ``FISH_ASR_STRIP_SPEAKERS`` or the client asks.

    Returns
    -------
    str
        Transcript without timestamps or ``<|…|>`` tokens. Fish ``[cue]`` tags
        are not expected here and are not specially kept.
    """
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
    """Return whether an ASR string is silence, a caption watermark, or too thin.

    Parameters
    ----------
    text : str
        Transcript, usually after ``scrub_asr``.

    Returns
    -------
    bool
        True for empty text, ``nospeech``, YouTube-caption boilerplate
        (thanks-for-watching, Amara, 谢谢观看), gzip-repetitive text, emoji-only
        or angle-token-only text, a single CJK character, or fewer than three
        Latin letters. A longer CJK sentence is kept.

    Notes
    -----
    Duplex uses this to drop the turn before the LLM. A false positive skips
    real speech; the Latin floor is three letters so ``ok`` is not speech and
    ``okay`` is.
    """
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
    """Return whether the utterance is only a listener noise.

    Parameters
    ----------
    text : str
        Short transcript such as ``yeah``, ``uh huh``, or ``嗯``.

    Returns
    -------
    bool
        True for a known backchannel after case-folding. Duplex then listens
        again instead of answering.
    """
    return _known_phrase(text, _BACKCHANNELS)


def is_quit_utterance(text: str) -> bool:
    """Return whether the utterance asks the duplex loop to stop.

    Parameters
    ----------
    text : str
        Transcript such as ``bye`` or ``quit``.

    Returns
    -------
    bool
        True for a known quit phrase after case-folding.
    """
    return _known_phrase(text, _QUIT)


def _skip_abbreviation(buf: str, end_start: int) -> bool:
    """Return whether the period belongs to ``Dr.``, a one-letter initial, or ``1.``.

    A sentence cut must not flush ``Dr.`` as its own TTS request.
    """
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
    """Return how many leading characters are ready for one Fish TTS send.

    Parameters
    ----------
    buf : str
        Text buffered from the model so far.
    partial_chars : int, optional
        Flush near this length when no sentence end is in the buffer.
        Default is about 40 characters.

    Returns
    -------
    int
        End index of the piece to send, or -1 to keep buffering.

    Notes
    -----
    A sentence end wins, except ``Dr.``, a one-letter initial, and ``1.``.
    Otherwise the cut is the last space before ``partial_chars``, or
    ``partial_chars`` itself when there is no usable space.
    """
    if not buf:
        return -1
    end = _sentence_cut(buf)
    if end >= 0:
        return end
    return _partial_cut(buf, partial_chars)


def split_tts_piece(buf: str, partial_chars: int, *, flush_rest: bool) -> tuple[str, str] | None:
    """Split one ready TTS piece from the unsent tail.

    Parameters
    ----------
    buf : str
        Buffered model text.
    partial_chars : int
        Passed to ``next_tts_cut``.
    flush_rest : bool
        When True and no cut is ready, return the whole buffer as the piece.
        When False, return None so the stream keeps buffering.

    Returns
    -------
    tuple of str and str or None
        ``(piece, tail)``, or None while the caller should wait.
    """
    cut = next_tts_cut(buf, partial_chars=partial_chars)
    if cut < 0:
        if not flush_rest:
            return None
        return buf, ""
    return buf[:cut], buf[cut:]


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
