"""ASR transcript scrubbers and the gates that drop a turn."""

from __future__ import annotations

import re
import unicodedata
import zlib
from typing import cast

from fish_audio_suite_kit.text_filters import utf8_text

_SPEAKER_RE = re.compile(r"<\|speaker:\d+\|>")
_ANGLE_TOKEN_RE = re.compile(r"<\|[^|>]*\|>")
_H_SPACE_RE = re.compile(r"[ \t]+")
_SPACE_BEFORE_STOP_RE = re.compile(r"[ \t]+([.!?…。！？,;:，；：])")
_BREAKS_RE = re.compile(r"\n{3,}")
_CJK_RANGES = (
    range(0x3040, 0x3100),
    range(0x3400, 0x4DC0),
    range(0x4E00, 0xA000),
    range(0xF900, 0xFB00),
    range(0xAC00, 0xD7B0),
)

# VibeVoice timestamp [0.00-1.23]. A decimal marks a clock. [1-2] is a range
# the speaker said, so an integer pair stays in the transcript.
_VIBEVOICE_TS_RE = re.compile(
    r"\[(?:"
    r"\d+\.\d+\s*[-–]\s*\d+(?:\.\d+)?"
    r"|\d+(?:\.\d+)?\s*[-–]\s*\d+\.\d+"
    r")\]\s*"
)
# "Speaker 1:" labels from engines that do not use <|speaker:N|>.
# A digit after the colon is a clock ("Speaker 1:00"), not a label.
# A fullwidth colon is the same label. Leaving it in sent "Speaker 1" to the model.
_SPEAKER_N_RE = re.compile(r"\bSpeaker\s+\d+\s*[:：](?!\d)\s*", re.IGNORECASE)

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
_QUIT = frozenset(
    {
        "quit",
        "exit",
        "stop",
        "goodbye",
        "good bye",
        "good-bye",
        "bye",
        "bye bye",
        "bye-bye",
        "please stop",
        "stop please",
        "stop now",
    }
)

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
        "nospeech",
    }
)
# Same watermarks in Chinese. Punctuation is stripped before this lookup.
_CJK_HALLUCINATION_PHRASES = frozenset({"谢谢观看", "感谢观看", "请订阅", "字幕"})
# Strip spaces and quotes too, so "谢谢观看。" matches the phrase with no marks.
_ASR_PUNCT_RE = re.compile(r"[\s.。、，,!?！？…·・~～'\"“”‘’]+")
# Fold for phrase lists. Spaces stay, so "uh huh" does not become "uhhuh".
# Arabic comma, semicolon, and question mark, plus the danda and Urdu stop.
# The sentence cutter already treats those as stops. Leaving them on the
# line made the same utterance look new, so duplex answered it again.
_FOLD_PUNCT_RE = re.compile(r"[.。、，,!?！？…·・~～؟،؛।۔]+")
_SPACE_RE = re.compile(r"\s+")
# Below this size, gzip ratio is noise. A stuck caption loop compresses past the ratio.
_GZIP_MIN_BYTES = 48
_GZIP_RATIO = 2.4

# Emoji-only transcripts are not speech. Misc symbols, dingbats, and flags.
_EMOJI_RE = re.compile(
    "[\U0001f300-\U0001faff\U00002700-\U000027bf\U0001f1e0-\U0001f1ff]+",
    flags=re.UNICODE,
)

_ASR_SPEAKERS = (
    # An empty replacement joined the words on either side, so "door" and
    # "today" were heard as one word.
    (_SPEAKER_RE, " "),
    (_SPEAKER_N_RE, ""),
)


def _plain_breaks(text: str) -> str:
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


def _replace(text: str, pairs: tuple[tuple[re.Pattern[str], str], ...]) -> str:
    for pattern, repl in pairs:
        text = pattern.sub(repl, text)
    return text


def _tidy_asr(text: str) -> str:
    text = _SPACE_BEFORE_STOP_RE.sub(r"\1", text)
    return _BREAKS_RE.sub("\n\n", text).strip()


def _cjk_latin_counts(text: str) -> tuple[int, int]:
    cjk = latin = 0
    for ch in text:
        code = ord(ch)
        if any(code in span for span in _CJK_RANGES):
            cjk += 1
        elif ch.isalpha():
            latin += 1
    return cjk, latin


def _folded(text: str) -> str:
    # A curly apostrophe is the same word. Leaving it made the two spellings
    # of "don't" two lines, so the assistant answered the echo again.
    # "cafe" plus a combining acute is the same word as the composed letter.
    straight = (
        unicodedata.normalize("NFC", (text or "").strip())
        .lower()
        .replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u02bc", "'")
        # A kashida only stretches a letter. Leaving it made the same line
        # a new utterance, so the assistant answered it again.
        .replace("\u0640", "")
    )
    s = _FOLD_PUNCT_RE.sub("", straight)
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
    cleaned = _plain_breaks(text)
    if strip_speakers:
        cleaned = _replace(cleaned, _ASR_SPEAKERS)
    cleaned = _VIBEVOICE_TS_RE.sub(" ", cleaned)
    cleaned = _ANGLE_TOKEN_RE.sub(" ", cleaned)
    cleaned = _H_SPACE_RE.sub(" ", cleaned)
    return utf8_text(_tidy_asr(cleaned))


def _known_hallucination(s: str) -> bool:
    if _known_phrase(s, _EN_HALLUCINATION_PHRASES):
        return True
    return _ASR_PUNCT_RE.sub("", s) in _CJK_HALLUCINATION_PHRASES


def is_caption_watermark(text: str) -> bool:
    """Return whether text is a known silence caption.

    Parameters
    ----------
    text : str
        One ASR segment or a full transcript.

    Returns
    -------
    bool
        True for a YouTube-caption watermark such as ``thanks for watching``
        or ``谢谢观看``. A short utterance such as ``ok`` is not a watermark.

    Notes
    -----
    ``is_asr_hallucination`` also drops short text. Using that on each
    segment would delete a real ``ok`` that shares a clip with other words.
    """
    if not text or not text.strip():
        return False
    return _known_hallucination(text.strip())


# Same stops that end a sentence, plus an ellipsis. A caption after one of
# those was left in the transcript and answered on the next turn.
_CLAUSE_END_RE = re.compile(r"[.!?。！？…؟।۔][\"'”’)\]]*\s*$")
_WATERMARK_STOP_RE = re.compile(r"[.!?。！？…؟।۔]")


def _drop_watermark_clause(text: str, phrase: str) -> str:
    pattern = _watermark_pattern(phrase)
    if pattern is None:
        return text
    cursor = 0
    while cursor <= len(text):
        match = pattern.search(text, cursor)
        if match is None:
            return text
        before = text[: match.start()]
        after = text[match.end() :]
        matched = text[match.start() : match.end()]
        # A trailing "thanks for watching" is the caption. "Thanks for watching
        # the door" is the sentence: the match ate a space, not a period.
        # A later copy is still a caption. Stopping at the first hit left it.
        trailing = not after.strip()
        stopped = _WATERMARK_STOP_RE.search(matched) is not None
        at_edge = not before.strip() or _CLAUSE_END_RE.search(before) is not None
        if trailing or (stopped and at_edge):
            text = f"{before} {after}"
            cursor = len(before)
            continue
        cursor = match.end()
    return text


def _watermark_pattern(phrase: str) -> re.Pattern[str] | None:
    words = _folded(phrase).split()
    if not words:
        return None
    body = r"\W+".join(re.escape(word) for word in words)
    # The watermark's own period is not part of the previous sentence.
    return re.compile(rf"(?i)(?<!\w){body}(?!\w)\W*")


def without_watermark_segments(
    text: str,
    segments: object,
    *,
    strip_speakers: bool = True,
) -> str:
    """Remove watermark segment phrases from a transcript that also has speech.

    Parameters
    ----------
    text : str
        Scrubbed full transcript.
    segments : object
        Fish ``segments``. A non-list leaves ``text`` unchanged.
    strip_speakers : bool, optional
        Passed to ``scrub_asr`` for each segment. Default True.

    Returns
    -------
    str
        ``text`` with each watermark segment removed. A transcript that has
        no watermark segment is returned unchanged.

    Notes
    -----
    Rebuilding the transcript from every segment drops words that exist only
    in the top-level text. Only the watermark phrase is removed.
    """
    if not isinstance(segments, list):
        return text
    cleaned = text
    dropped = False
    for seg in cast(list[object], segments):
        if not isinstance(seg, dict):
            continue
        raw_text = cast(dict[str, object], seg).get("text")
        if not isinstance(raw_text, str):
            continue
        body = scrub_asr(raw_text, strip_speakers=strip_speakers)
        if not body or not is_caption_watermark(body):
            continue
        dropped = True
        # "Thanks for watching." does not equal "thanks for watching".
        # The phrase stayed in the transcript and the next turn answered it.
        updated = _drop_watermark_clause(cleaned, body)
        if updated != cleaned:
            cleaned = updated
            continue
        pattern = _watermark_pattern(body)
        if pattern is not None and pattern.search(cleaned):
            continue
        for needle in {raw_text.strip(), body}:
            if needle:
                cleaned = cleaned.replace(needle, " ")
    if not dropped:
        return text
    return " ".join(cleaned.split()).strip()


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
        letters. ``ok`` stays too thin. ``Été`` and a Russian sentence are
        speech. Three digits are a number, so ``100`` is kept. A longer CJK
        sentence is kept.

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
    # "100" is an answer. "ok" and "42" stay below the floor.
    if sum(ch.isdigit() for ch in s) >= 3:
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


def same_utterance(text: str, previous: str) -> bool:
    """Return whether two transcripts are the same spoken line.

    Parameters
    ----------
    text : str
        The new transcript.
    previous : str
        The line already accepted.

    Returns
    -------
    bool
        True when case and punctuation fold to the same words. Empty text
        is not a repeat, so a blank transcript does not match a blank one.
    """
    folded = _folded(text)
    return bool(folded) and folded == _folded(previous)
