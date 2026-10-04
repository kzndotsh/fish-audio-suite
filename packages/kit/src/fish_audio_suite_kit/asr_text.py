"""ASR transcript scrubbers and the gates that drop a turn."""

from __future__ import annotations

import re
import unicodedata
import zlib
from collections.abc import Sequence, Set
from typing import Final, cast

from fish_audio_suite_kit._charsets import (
    ANGLE_TOKEN_RE,
    BREAKS_RE,
    SENTENCE_STOPS,
    cjk_latin_counts,
    plain_breaks,
    utf8_text,
)
from fish_audio_suite_kit._single_pass import collapse_space_before_stop
from fish_audio_suite_kit.junk import DEFAULT_MIN_LETTERS, DEFAULT_SHORT_WORDS
from fish_audio_suite_kit.payloads import AsrSegment

__all__ = [
    "DEFAULT_BACKCHANNELS",
    "DEFAULT_QUIT_PHRASES",
    "asr_language_hint",
    "is_asr_hallucination",
    "is_backchannel",
    "is_caption_watermark",
    "is_quit_utterance",
    "is_same_utterance",
    "scrub_asr",
    "without_watermark_segments",
]

_SPEAKER_RE = re.compile(r"<\|speaker:\d+\|>")
_H_SPACE_RE = re.compile(r"[ \t]+")

# VibeVoice timestamp [0.00-1.23]. A decimal marks a clock. [1-2] is a range
# the speaker said, so an integer pair stays in the transcript.
_VIBEVOICE_TS_RE = re.compile(
    r"\[(?:"
    r"\d+\.\d+\s*[-–]\s*\d+(?:\.\d+)?"
    r"|\d+(?:\.\d+)?\s*[-–]\s*\d+\.\d+"
    r")\]\s*"
)
# "Speaker 1:" labels from engines that do not use <|speaker:N|>. A digit
# after the colon is a clock ("Speaker 1:00"), not a label. A fullwidth colon
# is the same label.
_SPEAKER_N_RE = re.compile(r"\bSpeaker\s+\d+\s*[:：](?!\d)\s*", re.IGNORECASE)

# transcribe-1-pro annotations such as [laughter] or [高兴]. Digits-only
# brackets ("[1-2]", "[5]") are something the speaker said, so they stay.
_ASR_CUE_RE = re.compile(r"\[(?!\d+(?:\s*[-–]\s*\d+)?\])[^\[\]\n]{1,40}\]")

# Listener noises. A caller that hears one keeps listening instead of answering.
# English spellings plus the Chinese, Japanese, and Korean fillers models emit.
# "right" and "sure" are real answers, so they are not here. A caller can pass
# its own set to is_backchannel.
DEFAULT_BACKCHANNELS: Final = frozenset(
    {
        "yeah",
        "yep",
        "yup",
        "ya",
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

# Whole utterance, after folding. These end the session, so they are explicit
# goodbyes. "stop" is not here: people say it to interrupt speech. A caller can
# pass its own set to is_quit_utterance.
DEFAULT_QUIT_PHRASES: Final = frozenset(
    {
        "quit",
        "exit",
        "goodbye",
        "good bye",
        "good-bye",
        "bye",
        "bye bye",
        "bye-bye",
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
# Includes the Arabic comma, semicolon, and question mark, plus the danda and
# Urdu stop, so the same utterance folds the same way in every script.
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

# A speaker marker is replaced by a space, not nothing: removing it would join
# the words on either side.
_ASR_SPEAKERS = (
    (_SPEAKER_RE, " "),
    (_SPEAKER_N_RE, ""),
)


def _tidy_asr(text: str) -> str:
    text = collapse_space_before_stop(text)
    return BREAKS_RE.sub("\n\n", text).strip()


def _folded(text: str) -> str:
    # Fold spellings of the same word: curly and straight apostrophes, and a
    # composed letter against a letter plus a combining accent.
    straight = (
        unicodedata.normalize("NFC", (text or "").strip())
        .lower()
        .replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u02bc", "'")
        # A kashida only stretches an Arabic letter.
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


_LANGUAGE_SUBTAG_RE: Final = re.compile(r"[-_]")


def asr_language_hint(value: str) -> str:
    """Reduce a language tag to the ISO 639-1 code Fish ASR accepts.

    Parameters
    ----------
    value : str
        A language hint as a client or the environment wrote it, such as
        ``en``, ``EN``, ``en-US`` or ``zh_CN``.

    Returns
    -------
    str
        The primary subtag in lowercase when it is exactly two ASCII letters,
        otherwise ``""``. Fish may answer 400 to ``en-US`` or ``English``, and
        an empty hint lets it detect the language instead.

    Examples
    --------
    >>> asr_language_hint(" en-US ")
    'en'
    >>> asr_language_hint("zh_CN")
    'zh'
    >>> asr_language_hint("English")
    ''
    """
    primary = _LANGUAGE_SUBTAG_RE.split(value.strip().lower(), maxsplit=1)[0]
    if len(primary) == 2 and primary.isascii() and primary.isalpha():
        return primary
    return ""


def scrub_asr(text: str, *, strip_speakers: bool = True, strip_cues: bool = False) -> str:
    """Strip ASR markup Fish and other engines leave in the transcript.

    Parameters
    ----------
    text : str
        Raw transcript.
    strip_speakers : bool, optional
        Drop ``Speaker 1:`` style labels. Default True. The proxy follows
        ``FISH_PROXY_ASR_STRIP_SPEAKERS``, which is also on by default.
    strip_cues : bool, optional
        Drop ``[laughter]`` style annotations that ``transcribe-1-pro`` adds.
        Default False. A bracket with only digits, such as ``[1-2]``, stays.

    Returns
    -------
    str
        Transcript without timestamps or ``<|…|>`` tokens. Annotations are
        kept unless ``strip_cues`` is True.
    """
    if not text:
        return ""
    cleaned = plain_breaks(text)
    if strip_speakers:
        for pattern, repl in _ASR_SPEAKERS:
            cleaned = pattern.sub(repl, cleaned)
    cleaned = _VIBEVOICE_TS_RE.sub(" ", cleaned)
    cleaned = ANGLE_TOKEN_RE.sub(" ", cleaned)
    if strip_cues:
        cleaned = _ASR_CUE_RE.sub(" ", cleaned)
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


# A caption after any sentence stop is a separate clause, so it is dropped.
_STOP_CLASS = re.escape("".join(sorted(SENTENCE_STOPS)))
_CLAUSE_END_RE = re.compile(rf"[{_STOP_CLASS}][\"'”’)\]]*\s*$")
_WATERMARK_STOP_RE = re.compile(rf"[{_STOP_CLASS}]")


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
        # A trailing "thanks for watching" is the caption, but "Thanks for
        # watching the door" is a sentence: the match consumed no stop. Keep
        # scanning, since a later copy can still be a caption.
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
    segments: Sequence[AsrSegment] | None,
    *,
    strip_speakers: bool = True,
    strip_cues: bool = False,
) -> str:
    """Remove watermark segment phrases from a transcript that also has speech.

    Parameters
    ----------
    text : str
        Scrubbed full transcript.
    segments : Sequence of AsrSegment, or None
        Fish ``segments``. None, or anything that is not a list at run time (the
        body is untrusted JSON), leaves ``text`` unchanged, and entries that are
        not objects with a string ``text`` are skipped.
    strip_speakers : bool, optional
        Passed to ``scrub_asr`` for each segment. Default True.
    strip_cues : bool, optional
        Passed to ``scrub_asr`` for each segment. Default False.

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
        body = scrub_asr(raw_text, strip_speakers=strip_speakers, strip_cues=strip_cues)
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
    if ANGLE_TOKEN_RE.fullmatch(s.replace(" ", "")):
        return None
    tagged = ANGLE_TOKEN_RE.sub("", s).strip()
    return tagged or None


def is_asr_hallucination(
    text: str,
    *,
    min_letters: int = DEFAULT_MIN_LETTERS,
    short_words: Set[str] | None = None,
) -> bool:
    """Return whether an ASR string is silence, a caption watermark, or too thin.

    Parameters
    ----------
    text : str
        Transcript, usually after ``scrub_asr``.
    min_letters : int, optional
        Fewest letters that count as speech. Default 2, so ``no``, ``ok``,
        and ``hi`` are kept and a lone letter is dropped.
    short_words : AbstractSet of str or None, optional
        Lowercase words kept even below ``min_letters``. Default
        ``DEFAULT_SHORT_WORDS``. Matters when a caller raises the floor.

    Returns
    -------
    bool
        True for empty text, ``nospeech``, YouTube-caption boilerplate
        (thanks-for-watching, Amara, 谢谢观看), gzip-repetitive text, emoji-only
        or angle-token-only text, a single CJK character, or fewer than
        ``min_letters`` letters. ``Été`` and a Russian sentence are speech.
        Three digits are a number, so ``100`` is kept, and a longer CJK
        sentence is kept.

    Notes
    -----
    A caller uses this to drop a turn before the LLM, so a false positive
    discards real speech. Callers who want a stricter gate can raise
    ``min_letters`` and rely on ``short_words`` for the answers they keep.
    """
    if not text or not text.strip():
        return True
    s = text.strip()
    if _known_hallucination(s):
        return True
    tagged = _without_marks(s)
    if tagged is None or _gzip_repetitive(tagged):
        return True
    cjk, letters = cjk_latin_counts(s)
    if letters == 0 and cjk == 1:
        return True
    if cjk:
        return False
    if sum(ch.isdigit() for ch in s) >= 3:
        return False
    words = DEFAULT_SHORT_WORDS if short_words is None else short_words
    if "".join(ch for ch in s.lower() if ch.isalnum()) in words:
        return False
    return letters < min_letters


def _known_phrase(text: str, phrases: Set[str]) -> bool:
    return _folded(text) in phrases


def _folded_phrases(phrases: Set[str]) -> frozenset[str]:
    return frozenset(_folded(phrase) for phrase in phrases)


def is_backchannel(text: str, *, phrases: Set[str] | None = None) -> bool:
    """Return whether the utterance is only a listener noise.

    Parameters
    ----------
    text : str
        Short transcript such as ``yeah``, ``uh huh``, or ``嗯``.
    phrases : AbstractSet of str or None, optional
        Phrases that count as noise. Folded (lowercase, punctuation removed)
        before matching. Default ``DEFAULT_BACKCHANNELS``.

    Returns
    -------
    bool
        True for a known backchannel after case-folding. A caller then
        listens again instead of answering.
    """
    known = DEFAULT_BACKCHANNELS if phrases is None else _folded_phrases(phrases)
    return _known_phrase(text, known)


def is_quit_utterance(text: str, *, phrases: Set[str] | None = None) -> bool:
    """Return whether the utterance asks the loop to stop.

    Parameters
    ----------
    text : str
        Transcript such as ``bye`` or ``quit``.
    phrases : AbstractSet of str or None, optional
        Whole-utterance phrases that quit. Folded before matching. Default
        ``DEFAULT_QUIT_PHRASES``.

    Returns
    -------
    bool
        True when the whole utterance is a quit phrase after case-folding.
    """
    known = DEFAULT_QUIT_PHRASES if phrases is None else _folded_phrases(phrases)
    return _known_phrase(text, known)


def is_same_utterance(text: str, previous: str) -> bool:
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
